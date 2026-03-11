import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .MotionSqueeze import MS
import torch.utils.checkpoint as checkpoint
from .dat_blocks import DATSwinLayer,DATSwinTransformerBlock
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import visualization_utils as vis_utils
except ImportError:
    class DummyVisUtils:
        VISUALIZE = False
    vis_utils = DummyVisUtils()

## input: (b,c_in,t,h,w)
## output:(b,memory_size,48,48)
## TODO 使用mask之后的分割图进行运动估计？ SimVP?
class MotionEncoder2D(nn.Module):
    def __init__(self,in_len=12,out_c=400):
        super(MotionEncoder2D,self).__init__()
        self.conv2d_1 = nn.Sequential(
                                      nn.Conv2d(1,32,kernel_size=(7,7),stride=(2,2),padding=(3,3)),
                                      nn.SiLU())
        self.conv2d_2 = nn.Sequential(nn.Conv2d(32,64,kernel_size=(3,3),stride=(2,2),padding=(1,1)),
                                      nn.SiLU())
        self.conv2d_3 = nn.Sequential(nn.Conv2d(64,128,kernel_size=(3,3),stride=(2,2),padding=(1,1)),
                                      nn.SiLU())
        self.conv2d_4 = nn.Sequential(nn.Conv2d(128,256,kernel_size=(3,3),stride=(2,2),padding=(1,1)),
                                      nn.SiLU())
        self.conv2d_5 = nn.Sequential(nn.Conv2d(256,out_c,kernel_size=(3,3),stride=(2,2),padding=(1,1)),
                                      nn.SiLU())        
        self.motion_squeeze = MS(c_out=out_c)
        # self.linear = nn.Sequential(nn.Linear(256*in_len,out_c),
        #                             nn.SiLU() )## 240, memory_size)
        # self.avg_pool = nn.AdaptiveAvgPool2d([16,16])

        # self.conv2d_6 = nn.Sequential(nn.Conv2d(512*in_len,out_c,kernel_size=(1,1)),
        #                               nn.SiLU())
        self.conv3d_1 = nn.Sequential(nn.Conv3d(out_c,out_c,kernel_size=(3,3,3),stride=(2,2,2),padding=(0,1,1)),
                                      nn.SiLU())    
        self.conv3d_2 = nn.Sequential(nn.Conv3d(out_c,out_c,kernel_size=(3,3,3),stride=(2,1,1),padding=(0,1,1)),
                                      nn.SiLU())     
        self.avg_pool_3d = nn.AdaptiveAvgPool3d([1,None,None])
             
    def forward(self, x, chunk_size=20):
        b, t, c, h, w = x.shape

        # 分块处理: 当帧数超过 chunk_size 时, 使用末端重叠分块
        # 非末块多取 1 帧 (overlap), MS 能正确计算边界帧对, 然后丢弃末帧 (padding帧)
        # flow_refine_conv 全是 2D 卷积, 每帧独立处理, 因此结果与全量处理 100% 一致
        if t > chunk_size:
            chunk_outputs = []
            pos = 0
            while pos < t:
                remaining = t - pos
                if remaining <= chunk_size:
                    # 最后一块: 处理所有剩余帧
                    x_chunk = x[:, pos:]
                    keep_all = True
                elif remaining <= chunk_size + 1:
                    # 边界情况: 剩余 chunk_size+1 帧, 一次处理 (拆开下一块只剩1帧, MS无法工作)
                    x_chunk = x[:, pos:]
                    keep_all = True
                else:
                    # 非末块: 多取 1 帧作为 overlap, 确保边界帧对 (i, i+1) 被计算
                    x_chunk = x[:, pos:pos + chunk_size + 1]
                    keep_all = False

                b_c, t_c, c_c, h_c, w_c = x_chunk.shape

                # 2D convs
                x_flat = x_chunk.contiguous().view(-1, c_c, h_c, w_c)
                x_flat = self.conv2d_1(x_flat)
                x_flat = self.conv2d_2(x_flat)
                x_flat = self.conv2d_3(x_flat)
                x_flat = self.conv2d_4(x_flat)
                x_flat = self.conv2d_5(x_flat)
                x_3d = x_flat.contiguous().view(b_c, -1, t_c, h_c // 32, w_c // 32)

                # MotionSqueeze (帧间相关矩阵, 显存瓶颈)
                x_3d = self.motion_squeeze(x_3d)

                if not keep_all:
                    # 丢弃末帧 (MS padding 复制帧, 不是真实光流), 只保留前 chunk_size 帧
                    x_3d = x_3d[:, :, :chunk_size]
                    pos += chunk_size
                else:
                    pos = t  # 已处理完

                chunk_outputs.append(x_3d)

            x = torch.cat(chunk_outputs, dim=2)
        else:
            x = x.contiguous().view(-1, c, h, w)
            x = self.conv2d_1(x)
            x = self.conv2d_2(x)
            x = self.conv2d_3(x)
            x = self.conv2d_4(x)
            x = self.conv2d_5(x)
            x = x.contiguous().view(b, -1, t, h // 32, w // 32)
            x = self.motion_squeeze(x)

        x = self.conv3d_1(x)
        x = self.conv3d_2(x)
        x = self.avg_pool_3d(x)

        return x.squeeze(2)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 128, dropout = 0., use_flash=True):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.head_dim = dim_head
        self.dropout_p = dropout

        self.norm = nn.LayerNorm(dim)

        # 始终保留手动 attention 组件，避免运行时切换 use_flash 后属性缺失
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        # Flash Attention (SDPA) 支持
        self.use_flash = use_flash and hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        self.to_q = nn.Linear(dim, inner_dim * 1, bias = False)
        self.to_k = nn.Linear(dim, inner_dim * 1, bias = False)
        self.to_v = nn.Linear(dim, inner_dim * 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x,y,z):
        if x.equal(y):
            x = self.norm(x)
            y = self.norm(y)
            z = self.norm(z)
        else:
            x = self.norm(x)
            y = self.norm(y)     

        qkv = [self.to_q(x),self.to_k(y),self.to_v(z)]
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        if self.use_flash:
            # Flash Attention / SDPA — O(N) 显存, 自动选择最优内核
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,  # cross-attention, 非因果
            )
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = self.attend(dots)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)             
            ]))

    def forward(self, x,y,z):
        # out = []
        for attn_1,attn_2, ff in self.layers:
            x = attn_1(x=x,y=x,z=x) + x
            x = attn_2(x=x,y=y,z=z) + x
            x = ff(x) + x
            # out.append(x)
        # out = torch.cat(out,-1)

        return torch.sigmoid(self.norm(x))

class Memory(nn.Module):
    def __init__(self, args,memory_channel_size=256,memory_slot_size=100,short_len=12,long_len=24):
        super(Memory,self).__init__()
        self.args = args
        self.memory_size = memory_channel_size
        self.memory_slot_size = memory_slot_size
        self.short_len = short_len
        self.long_len = long_len
        self.motion_encoder_2d = MotionEncoder2D(in_len=self.short_len,out_c=self.memory_size)
        self.motion_context_encoder_2d = MotionEncoder2D(in_len=self.long_len,out_c=self.memory_size)
        
        # [修复] 动态计算维度，不再硬编码
        num_layers = len(args.depths_down)
        # SwinLSTM 最后一层 cell 的 hidden state 维度
        # 注意：每层经过 PatchMerging 后维度翻倍，但 cell 返回的是 merging 前的维度
        # 所以最后一层的 hidden dim = embed_dim * 2^(num_layers-1)
        self.hidden_dim = args.embed_dim * (2 ** (num_layers - 1))  # 如 embed_dim=128, num_layers=2 -> 256
        
        # MotionEncoder2D 输出分辨率: img_size / 32 / 2 / 1 = img_size / 64
        # 对于 384x384: 384/64=6, 输出 6x6
        self.motion_spatial_size = args.input_img_size // 64
        self.motion_seq_len = self.motion_spatial_size * self.motion_spatial_size
        
        # SwinLSTM 下采样后的分辨率
        # patches_resolution = img_size / patch_size, 然后每层 /2
        patches_per_side = args.input_img_size // args.patch_size  # 如 384/4=96
        # states_down[-1] 是在最后一层的 PatchMerging 前，所以分辨率是 /2^(num_layers-1)
        self.swin_spatial_size = patches_per_side // (2 ** (num_layers - 1))  # 如 96/2=48
        self.swin_seq_len = self.swin_spatial_size * self.swin_spatial_size  # 如 48*48=2304
        
        # pos_embedding 匹配 motion encoder 输出
        self.pos_embedding = nn.Parameter(torch.randn(1, self.motion_seq_len, self.memory_size))
        
        # additional_linear: 将 memory_size 映射到合适大小用于后续处理
        self.additional_linear = nn.Linear(self.memory_size, 512)  # 固定为 512 用于 embedder
        
        # embedder: 从 motion_spatial_size 扩展到 swin_spatial_size
        # 输出维度映射到 hidden_dim 以便与 states_down[-1] 匹配
        embedder_layers = []
        current_res = self.motion_spatial_size  # 6
        current_dim = 512  # additional_linear 的输出
        target_res = self.swin_spatial_size  # 48
        
        while current_res < target_res:
            embedder_layers.append(PatchExpanding(input_resolution=[current_res, current_res], 
                                                   dim=current_dim, dim_scale=2))
            current_res *= 2
            current_dim //= 2
        
        # 最终映射到 hidden_dim (与 states_down[-1][1] 的通道数相同)
        embedder_layers.append(nn.Linear(current_dim, self.hidden_dim))
        self.embedder = nn.Sequential(*embedder_layers)
        self.matched_memory_dim = self.hidden_dim

        self.attention_block = Transformer(dim=self.memory_size, depth=1, heads=8, dim_head=64, mlp_dim=self.memory_size, dropout = 0.1)

        self.predictor = SwinLSTM(img_size=args.input_img_size, patch_size=args.patch_size,
                         in_chans=args.input_channels, embed_dim=args.embed_dim,
                         depths_downsample=args.depths_down, depths_upsample=args.depths_up,
                         num_heads=args.heads_number, window_size=args.window_size)

        self.target_len = args.out_len    
        self.memory_shape = [self.memory_slot_size,self.memory_size]
        self.memory_bank = nn.init.uniform_(torch.empty(self.memory_shape), a=0.0, b=1.0)
        self.memory_bank = nn.Parameter(self.memory_bank, requires_grad=True)
        
        # [修复] attention_func 输入维度: hidden_dim + matched_memory_dim = 2 * hidden_dim
        self.attention_input_dim = self.hidden_dim + self.matched_memory_dim  # 256 + 256 = 512
        self.attention_size = self.hidden_dim
        self.attention_func = nn.Sequential(
            nn.Linear(self.attention_input_dim, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, self.attention_size),
            nn.Sigmoid())
        
        # [修复] fuse 输入维度: hidden_dim + matched_memory_dim = 2 * hidden_dim
        # fuse 输出维度: hidden_dim (保持与 states_down 一致)
        self.fuse = nn.Linear(self.hidden_dim + self.matched_memory_dim, self.hidden_dim)

    def forward(self,inputs,memory_x,phase):
        ## 不考虑像素值<31的
        # memory_x[memory_x < 31/255.] = 0.
        # memory_x = torch.clamp(memory_x,31./255)
        # ## 用帧间差代表运动特征
        # memory_x = memory_x[:,1:,...] - memory_x[:,:-1,...] # B,T,C,H,W
        b,t,c,h,w = memory_x.shape
        # memory_query_3d = self.motion_context_encoder_3d(memory_x) if phase==1 else self.motion_encoder_3d(memory_x)
        memory_query = self.motion_context_encoder_2d(memory_x) if phase==1 else self.motion_encoder_2d(memory_x) #b,(t-1)*c,h,w
        b_,c_,h_,w_ = memory_query.shape
        memory_query = memory_query.contiguous().permute(0,2,3,1).view(b_,-1,c_)
        b__,l,c__ = memory_query.shape      
        memory_bank = self.memory_bank.unsqueeze(0).repeat(b_,1,1).contiguous() ## B x N x C, N个特征,每个特征维度为C

        memory_query += self.pos_embedding
        matched_memory = self.attention_block(memory_query,memory_bank,memory_bank)
        # [修复] 使用动态维度而不是硬编码 512
        if not matched_memory.shape[-1] == self.hidden_dim:
            matched_memory = self.additional_linear(matched_memory)
        matched_memory = self.embedder(matched_memory)
        
        ## matched_memory与h_t,c_t融合
        outputs = []
        inputs_len = inputs.shape[1]
        last_input = inputs[:, -1]

        states_down = [None] * len(self.args.depths_down)
        states_up = [None] * len(self.args.depths_down)  

        for i in range(inputs_len - 1):
            # print(i)
            output, states_down, states_up = self.predictor(inputs[:, i], states_down, states_up)
            outputs.append(output)

        # [修复] 在循环开始前，先获取第一个状态来确定目标尺寸
        # 先做一次前向传播获取 states_down 的形状信息
        output, states_down, states_up = self.predictor(last_input, states_down, states_up)
        outputs.append(output)
        last_input = output
        
        # 计算 matched_memory 需要的目标分辨率（只计算一次）
        state_seq_len = states_down[-1][1].shape[1]
        memory_seq_len = matched_memory.shape[1]
        
        if memory_seq_len != state_seq_len:
            # 使用插值调整空间分辨率
            B, L, C = matched_memory.shape
            H = W = int(L ** 0.5)
            matched_memory_2d = matched_memory.view(B, H, W, C).permute(0, 3, 1, 2)
            target_H = target_W = int(state_seq_len ** 0.5)
            matched_memory_2d = torch.nn.functional.interpolate(
                matched_memory_2d, size=(target_H, target_W), mode='bilinear', align_corners=False
            )
            matched_memory = matched_memory_2d.permute(0, 2, 3, 1).view(B, -1, C)
        
        # 第一步的注意力融合
        attention = self.attention_func(torch.cat([states_down[-1][1], matched_memory], dim=2))
        memory_feature_att = matched_memory * attention
        tmp = torch.cat((states_down[-1][1], memory_feature_att), 2)
        states_down[-1][1] = self.fuse(tmp)

        # 剩余步骤
        for i in range(1, self.target_len):
            output, states_down, states_up = self.predictor(last_input, states_down, states_up)
            outputs.append(output)
            last_input = output
            
            attention = self.attention_func(torch.cat([states_down[-1][1], matched_memory], dim=2))
            
            if vis_utils.VISUALIZE:
                 pass # Skip for now as shape is ambiguous without running.
            
            memory_feature_att = matched_memory * attention
            tmp = torch.cat((states_down[-1][1],memory_feature_att),2)
            states_down[-1][1] = self.fuse(tmp)        
        return outputs
    
    def set_memory_bank_requires_grad(self, requires_grad: bool):
        """设置 memory_bank 的梯度需求"""
        self.memory_bank.requires_grad = requires_grad

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.contiguous().view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.contiguous().view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., use_flash=True):

        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.head_dim = head_dim
        self.attn_drop_p = attn_drop

        # Flash Attention (SDPA) 支持
        self.use_flash = use_flash and hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        if self.use_flash:
            # === Flash Attention / SDPA 路径 ===
            # 将 relative position bias 作为 attn_bias 传入 SDPA
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.contiguous().view(-1)
            ].view(N, N, -1).permute(2, 0, 1).contiguous()  # nH, N, N
            attn_bias = relative_position_bias.unsqueeze(0).expand(B_, -1, -1, -1)  # B_, nH, N, N

            if mask is not None:
                nW = mask.shape[0]
                # mask: (nW, N, N) → expand to (B_, nH, N, N)
                attn_bias = attn_bias.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn_bias = attn_bias.view(-1, self.num_heads, N, N)

            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_bias,
                dropout_p=self.attn_drop_p if self.training else 0.0,
                is_causal=False,
            )
        else:
            # === 原始手动 Attention 路径 ===
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.contiguous().view(-1)
            ].view(N, N, -1).permute(2, 0, 1).contiguous()  # nH, N, N
            attn = attn + relative_position_bias.unsqueeze(0)

            if mask is not None:
                nW = mask.shape[0]
                attn = attn.contiguous().view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.contiguous().view(-1, self.num_heads, N, N)
                attn = self.softmax(attn)
            else:
                attn = self.softmax(attn)

            attn = self.attn_drop(attn)
            x = (attn @ v)

        x = x.transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

       Args:
           dim (int): Number of input channels.
           input_resolution (tuple[int]): Input resulotion.
           num_heads (int): Number of attention heads.
           window_size (int): Window size.
           shift_size (int): Shift size for SW-MSA.
           mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
           qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
           qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
           drop (float, optional): Dropout rate. Default: 0.0
           attn_drop (float, optional): Attention dropout rate. Default: 0.0
           drop_path (float, optional): Stochastic depth rate. Default: 0.0
           act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
           norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
       """

    def __init__(self, dim, input_resolution, num_heads, window_size=2, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.red = nn.Linear(2 * dim, dim)
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.contiguous().view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, hx=None):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        if hx is not None:
            hx = self.norm1(hx)
            x = torch.cat((x, hx), -1)
            x = self.red(x)
        x = x.contiguous().view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.contiguous().view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.contiguous().view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        # FFN
        x = x.contiguous().view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.contiguous().view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.contiguous().view(B, -1, 4 * C)  # B H/2*W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)
        return x


# cite the 'PatchExpand' code from Swin-Unet, Thanks!
# https://github.com/HuCaoFighting/Swin-Unet/blob/main/networks/
# swin_transformer_unet_skip_expand_decoder_sys.py, line 333-355
class PatchExpanding(nn.Module):
    r""" Patch Expanding Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super(PatchExpanding, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.contiguous().view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = x.contiguous().view(B, -1, C // 4)
        x = self.norm(x)

        return x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.
        patch_size (int): Patch token size.
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of linear projection output channels.
    """

    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super(PatchEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class PatchInflated(nn.Module):
    r""" Tensor to Patch Inflating

    Args:
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of linear projection output channels.
        input_resolution (tuple[int]): Input resulotion.
    """

    def __init__(self, in_chans, embed_dim, input_resolution, stride=2, padding=1, output_padding=1):
        super(PatchInflated, self).__init__()

        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        output_padding = to_2tuple(output_padding)
        self.input_resolution = input_resolution
        
        self.Conv_ = nn.Sequential(
                    nn.ConvTranspose2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(3, 3), stride=stride, padding=padding, output_padding=output_padding),
                    nn.GroupNorm(16, embed_dim),
                    nn.LeakyReLU(0.2, inplace=True)
        )
        self.Conv = nn.ConvTranspose2d(in_channels=embed_dim, out_channels=in_chans, kernel_size=(3, 3),
                                       stride=stride, padding=padding, output_padding=output_padding)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2)
        x = self.Conv_(x)
        x = self.Conv(x)

        return x


class SwinLSTMCell(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size, depth, n_group=4,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, flag=None):
        super(SwinLSTMCell, self).__init__()

        # self.Swin = SwinTransformer(dim=dim, input_resolution=input_resolution, depth=depth,
        #                             num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
        #                             qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
        #                             drop_path=drop_path, norm_layer=norm_layer, flag=flag)
        self.DATSwin = DATSwinTransformer(dim=dim, input_resolution=input_resolution, depth=depth,
                                          n_head=num_heads, window_size=window_size, n_group=n_group,
                                          mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                                          norm_layer=norm_layer, downsample=None, use_checkpoint=False,
                                          use_pe=True, dwc_pe=False, no_off=False, fixed_pe=False

        )
    def forward(self, xt, hidden_states):
        if hidden_states is None:
            B, L, C = xt.shape
            hx = torch.zeros(B, L, C).to(xt.device)
            cx = torch.zeros(B, L, C).to(xt.device)

        else:
            hx, cx = hidden_states

        # Ft = self.Swin(xt, hx) #B,HW,C
        Ft = self.DATSwin(xt,hx)

        gate = torch.sigmoid(Ft)
        cell = torch.tanh(Ft)

        cy = gate * (cx + cell)
        hy = gate * torch.tanh(cy)
        hx = hy
        cx = cy

        return hx, [hx, cx]

class DATSwinTransformer(nn.Module):
    r""" Swin Deformable Attention Layer.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        n_head (int): Number of attention heads.
        window_size (int): Local window size.
        n_group (int): Number of groups for offset.
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool, optional): Whether to use checkpointing to save memory. Default: False.
        use_pe (bool, optional): Use position encoding. Default: True
        dwc_pe (bool, optional): Use DWC position encoding. Default: False
        no_off (bool, optional): DO NOT use offset (Set True to turn off offset). Default False
        fixed_pe (bool, optional): Use Fix position encoding. Default: False
    """

    def __init__(self, dim, input_resolution, depth, n_head, window_size, n_group,
                 mlp_ratio=2., drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 use_pe=True, dwc_pe=False, no_off=False, fixed_pe=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.n_group = n_group
        self.use_checkpoint = use_checkpoint

        self.use_pe = use_pe
        self.dwc_pe = dwc_pe
        self.no_off = no_off
        self.fixed_pe = fixed_pe

        # build blocks
        self.layers = nn.ModuleList([
            DATSwinTransformerBlock(dim=dim,
                                   input_resolution=input_resolution,
                                   n_head=n_head,
                                   n_group=n_group,
                                   window_size=window_size,
                                   shift_size=0 if (i % 2 == 0) else window_size // 2,
                                   mlp_ratio=mlp_ratio,
                                   drop=drop,
                                   attn_drop=attn_drop,
                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                   norm_layer=norm_layer,
                                   use_pe=use_pe,
                                   dwc_pe=dwc_pe,
                                   no_off=no_off,
                                   fixed_pe=fixed_pe,
                                   )
            for i in range(depth)])

        # patch merging/expend layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, xt,hx):
        outputs = []
        # assert xt.shape[1] == x_size[0] * x_size[1], "input feature has wrong size"
        for idx, layer in enumerate(self.layers):
            if self.use_checkpoint:
                if idx == 0:
                    x = checkpoint.checkpoint(layer, x,hx)
                    outputs.append(x)
                else:
                    if idx % 2 == 0:
                        x = checkpoint.checkpoint(layer, outputs[-1],xt)
                        outputs.append(x)
                    if idx % 2 == 1:
                        x = checkpoint.checkpoint(layer, outputs[-1],None)
                        outputs.append(x)                
            else:
                if idx == 0:
                    x = layer(xt,hx)
                    outputs.append(x)
                else:
                    if idx % 2 == 0:
                        x = layer(outputs[-1],xt)
                        outputs.append(x)
                    if idx % 2 == 1:
                        x = layer(outputs[-1],None)
                        outputs.append(x)
        return outputs[-1]
        # position = []
        # # reference = []
        # # attn_map = []
        # for blk in self.blocks:
        #     if self.use_checkpoint:
        #         x = checkpoint.checkpoint(blk, x, x_size)
        #         # position.append(pos)
        #         # reference.append(ref)
        #         # attn_map.append(att)
        #     else:
        #         x = blk(x, x_size)
        #         # position.append(pos)
        #         # reference.append(ref)
        #         # attn_map.append(att)

        # if self.downsample is not None:
        #     x = self.downsample(x)
        # # return x, position, reference, attn_map
        # return x#, 0, 0, 0

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class SwinTransformer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, flag=None):
        super(SwinTransformer, self).__init__()

        self.layers = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[depth - i - 1] if (flag == 0) else drop_path[i],
                                 norm_layer=norm_layer)
            for i in range(depth)]) # l w, l w, l w, l w

    def forward(self, xt, hx):

        outputs = []

        for index, layer in enumerate(self.layers):
            if index == 0:
                x = layer(xt, hx)
                outputs.append(x)

            else:
                if index % 2 == 0:
                    x = layer(outputs[-1], xt)
                    outputs.append(x)

                if index % 2 == 1:
                    x = layer(outputs[-1], None)
                    outputs.append(x)

        return outputs[-1]


class DownSample(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depths_downsample, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm):
        super(DownSample, self).__init__()

        self.num_layers = len(depths_downsample)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        patches_resolution = self.patch_embed.patches_resolution

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_downsample))]

        self.layers = nn.ModuleList()
        self.downsample = nn.ModuleList()

        for i_layer in range(self.num_layers):
            downsample = PatchMerging(input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                        patches_resolution[1] // (2 ** i_layer)),
                                      dim=int(embed_dim * 2 ** i_layer))

            layer = SwinLSTMCell(dim=int(embed_dim * 2 ** i_layer),
                                 input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                   patches_resolution[1] // (2 ** i_layer)),
                                 depth=depths_downsample[i_layer],
                                 num_heads=num_heads[i_layer],
                                 window_size=window_size,
                                 mlp_ratio=self.mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=dpr[sum(depths_downsample[:i_layer]):sum(depths_downsample[:i_layer + 1])],
                                 norm_layer=norm_layer)

            self.layers.append(layer)
            self.downsample.append(downsample)

    def forward(self, x, y):

        x = self.patch_embed(x) #B,C,H,W-->B,(H//patch_size)^2,embed_dim

        hidden_states_down = []

        for index, layer in enumerate(self.layers):
            x, hidden_state = layer(x, y[index])
            x = self.downsample[index](x)
            hidden_states_down.append(hidden_state)

        return hidden_states_down, x


class UpSample(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depths_upsample, num_heads, window_size, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, flag=0):
        super(UpSample, self).__init__()

        self.img_size = img_size
        self.num_layers = len(depths_upsample)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        patches_resolution = self.patch_embed.patches_resolution
        self.Unembed = PatchInflated(in_chans=in_chans, embed_dim=embed_dim, input_resolution=patches_resolution)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_upsample))]

        self.layers = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for i_layer in range(self.num_layers):
            resolution1 = (patches_resolution[0] // (2 ** (self.num_layers - i_layer)))
            resolution2 = (patches_resolution[1] // (2 ** (self.num_layers - i_layer)))

            dimension = int(embed_dim * 2 ** (self.num_layers - i_layer))
            upsample = PatchExpanding(input_resolution=(resolution1, resolution2), dim=dimension)

            layer = SwinLSTMCell(dim=dimension, input_resolution=(resolution1, resolution2),
                                 depth=depths_upsample[(self.num_layers - 1 - i_layer)],
                                 num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                 window_size=window_size,
                                 mlp_ratio=self.mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=dpr[sum(depths_upsample[:(self.num_layers - 1 - i_layer)]):
                                               sum(depths_upsample[:(self.num_layers - 1 - i_layer) + 1])],
                                 norm_layer=norm_layer, flag=flag)

            self.layers.append(layer)
            self.upsample.append(upsample)

    def forward(self, x, y):
        hidden_states_up = []

        for index, layer in enumerate(self.layers):
            x, hidden_state = layer(x, y[index])
            x = self.upsample[index](x)
            hidden_states_up.append(hidden_state)

        x = torch.sigmoid(self.Unembed(x))

        return hidden_states_up, x


class SwinLSTM(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depths_downsample, depths_upsample, num_heads,
                 window_size):
        super(SwinLSTM, self).__init__()

        self.Downsample = DownSample(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                     embed_dim=embed_dim, depths_downsample=depths_downsample,
                                     num_heads=num_heads, window_size=window_size)

        self.Upsample = UpSample(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                 embed_dim=embed_dim, depths_upsample=depths_upsample,
                                 num_heads=num_heads, window_size=window_size)

    def forward(self, input, states_down, states_up):
        states_down, x = self.Downsample(input, states_down)

        states_up, output = self.Upsample(x, states_up)

        return output, states_down, states_up

if __name__ == '__main__':
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5678'    
    import argparse
    device = torch.device('cuda:0')
    import datetime
    import numpy as np
    from dataloader.dataloader import SEVIR
    from torch.utils.data import DataLoader


    parser = argparse.ArgumentParser('SwinLSTM training and evaluation script', add_help=False)

    # Setup parameters
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--seed', default=1234, type=int)

    # Moving_MNIST dataset parameters
    parser.add_argument('--num_frames_input', default=12, type=int, help='Input sequence length')
    parser.add_argument('--num_frames_output', default=12, type=int, help='Output sequence length')
    parser.add_argument('--input_size', default=(384, 384), help='Input resolution')
    parser.add_argument('--train_data_dir', default='')
    parser.add_argument('--validation_data_dir', default='')
    parser.add_argument('--test_data_dir', default='')
    parser.add_argument('--short_len', default=12)
    parser.add_argument('--long_len', default=24)
    parser.add_argument('--out_len', default=12)
    ## 多卡训练
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    # model parameters
    parser.add_argument('--model', default='SwinLSTM-D', type=str, choices=['SwinLSTM-B', 'SwinLSTM-D'],
                        help='Model type')
    parser.add_argument('--input_channels', default=1, type=int, help='Number of input image channels')
    parser.add_argument('--input_img_size', default=384, type=int, help='Input image size')
    parser.add_argument('--patch_size', default=4, type=int, help='Patch size of input images')
    parser.add_argument('--embed_dim', default=128, type=int, help='Patch embedding dimension')
    parser.add_argument('--depths', default=[12], type=int, help='Depth of Swin Transformer layer for SwinLSTM-B')
    parser.add_argument('--depths_down', default=[3, 2], type=int, help='Downsample of SwinLSTM-D')
    parser.add_argument('--depths_up', default=[2,3], type=int, help='Upsample of SwinLSTM-D')
    parser.add_argument('--heads_number', default=[4,8], type=int,
                        help='Number of attention heads in different layers')
    parser.add_argument('--window_size', default=4, type=int, help='Window size of Swin Transformer layer')
    parser.add_argument('--drop_rate', default=0., type=float, help='Dropout rate')
    parser.add_argument('--attn_drop_rate', default=0., type=float, help='Attention dropout rate')
    parser.add_argument('--drop_path_rate', default=0.1, type=float, help='Stochastic depth rate')

    # Training parameters
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training,validation and test')
    # parser.add_argument('--valid_batch_size', default=16, type=int, help='Batch size for validation')
    # parser.add_argument('--test_batch_size', default=16, type=int, help='Batch size for testing')
    parser.add_argument('--workers', default=2, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--epoch_valid', default=1, type=int)
    parser.add_argument('--log_train', default=200, type=int)
    parser.add_argument('--log_valid', default=200, type=int)
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')

    args = parser.parse_args()

    model = Memory(args,memory_channel_size=512,short_len=12,long_len=24)
    print('\n Model is loaded!!!!!!!!!!!!!!!!!!!!!')

    outputs = model(inputs,memory_x=inputs,phase=2)
