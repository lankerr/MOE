# from torch import nn
# import torch
# from config import cfg
# import numpy as np
# from util.utils import make_layers
# import math


# def scheduled_sampling(shape, eta):
#     S, B, C, H, W = shape
#     # 随机种子已固定, 生成[0,1)随机数，形状 = (pre_len-1行，batch_size列)
#     random_flip = np.random.random_sample((S - 1, B))  # outS-1 * B
#     true_token = (random_flip < eta)  # 若eta为1，true_token[t, i]全部为True，mask元素全为1
#     one = torch.FloatTensor(1, C, H, W).fill_(1.0).cuda()  # 1*C*H*W
#     zero = torch.FloatTensor(1, C, H, W).fill_(0.0).cuda()  # 1*C*H*W
#     masks = []
#     for t in range(S - 1):
#         masks_b = []  # B*C*H*W
#         for i in range(B):
#             if true_token[t, i]:
#                 masks_b.append(one)
#             else:
#                 masks_b.append(zero)
#         mask = torch.cat(masks_b, 0)  # along batch size
#         masks.append(mask)  # outS-1 * B*C*H*W
#     return masks


# def reverse_scheduled_sampling(shape_r, epoch):
#     start_epoch = cfg.epoch / 3
#     end_epoch = cfg.epoch * 2 / 3
#     step = start_epoch / 5
#     if epoch < start_epoch:
#         eta_r = 0.5
#     elif epoch < end_epoch:
#         eta_r = 1.0 - 0.5 * math.exp(-float(epoch - start_epoch) / step)
#     else:
#         eta_r = 1.0
#     if epoch < start_epoch:
#         eta = 0.5
#     elif epoch < end_epoch:
#         eta = 0.5 - 0.5 * (epoch - start_epoch) / (end_epoch - start_epoch)
#     else:
#         eta = 0.0
#     S, B, C, H, W = shape_r
#     random_flip_r = np.random.random_sample((cfg.in_len - 1, B))  # inS-1 * B
#     random_flip = np.random.random_sample((S - cfg.in_len - 1, B))  # outS-1 * B
#     true_token_r = (random_flip_r < eta_r)  # 若eta为1，true_token[t, i]全部为True，mask元素全为1
#     true_token = (random_flip < eta)  # 若eta为0，true_token[t, i]全部为False，mask元素全为0
#     one = torch.FloatTensor(1, C, H, W).fill_(1.0).cuda()  # 1*C*H*W
#     zero = torch.FloatTensor(1, C, H, W).fill_(0.0).cuda()  # 1*C*H*W
#     masks = []
#     for t in range(S - 2):
#         if t < cfg.in_len - 1:
#             masks_b = []  # B*C*H*W
#             for i in range(B):
#                 if true_token_r[t, i]:
#                     masks_b.append(one)
#                 else:
#                     masks_b.append(zero)
#             mask = torch.cat(masks_b, 0)  # along batch size
#             masks.append(mask)  # inS-1 * B*C*H*W
#         else:
#             masks_b = []  # B*C*H*W
#             for i in range(B):
#                 if true_token[t - (cfg.in_len - 1), i]:
#                     masks_b.append(one)
#                 else:
#                     masks_b.append(zero)
#             mask = torch.cat(masks_b, 0)  # along batch size
#             masks.append(mask)  # outS-1 * B*C*H*W
#     return masks


# class Model(nn.Module):
#     def __init__(self, embed, rnn, fc):
#         super().__init__()
#         self.embed = make_layers(embed)
#         self.rnns = rnn  # 可以是原始 RNN，也可以是包含 encoder/decoder 的组合结构
#         self.fc = make_layers(fc)
#         self.use_ss = cfg.scheduled_sampling
#         self.use_rss = cfg.reverse_scheduled_sampling

#     def forward(self, inputs, mode=''):
#         x, eta, epoch = inputs  # x: [T, B, C, H, W]
#         in_len = cfg.in_len
#         out_len = cfg.out_len

#         if cfg.model_name == 'my_ed_ConvLSTM':
#             return self.forward_ed_convlstm(x, eta, epoch)
#         else:
#             return self.forward_autoregressive(x, eta, epoch, mode)

#     def forward_ed_convlstm(self, x, eta, epoch):
#         """
#         Encoder-Decoder ConvLSTM Forward Logic
#         """
#         input_seq = x[:cfg.in_len].permute(1, 0, 2, 3, 4)      # [T_in, B, C, H, W]
#         gt_future = x[cfg.in_len:]      # [T_out, B, C, H, W]

#         # if self.use_rss:
#         #     mask = reverse_scheduled_sampling([cfg.in_len + cfg.out_len] + list(x.shape)[1:], epoch)
#         # elif self.use_ss:
#         #     mask = scheduled_sampling([cfg.out_len] + list(x.shape)[1:], eta)

#         # Step 1: 编码
#         # encoder_states = self.rnns.encoder(input_seq.permute(1, 0, 2, 3, 4))  # [B, T, C, H, W]

#         # # Step 2: 构造 decoder 输入
#         # if cfg.decoder_seed == 'zero':
#         #     decoder_input = torch.zeros_like(gt_future)
#         # elif cfg.decoder_seed == 'last':
#         #     decoder_input = torch.zeros_like(gt_future)
#         #     decoder_input[0] = input_seq[-1]

#         # decoder_input = decoder_input.permute(1, 0, 2, 3, 4)  # [B, T, C, H, W]

#         # # Step 3: 解码
#         # output = self.rnns.decoder(decoder_input, encoder_states)  # [B, T, C, H, W]
#         # output = output.permute(1, 0, 2, 3, 4)  # → [T, B, C, H, W]

#         decouple_losses = torch.zeros(cfg.out_len, cfg.LSTM_layers, cfg.batch, cfg.lstm_hidden_state).to(x.device)
#         # return output, decouple_losses
#         output = self.rnns(input_seq)
#         output = output.permute(1, 0, 2, 3, 4)
#         # print(output.shape)

#         return output, decouple_losses

#     def forward_autoregressive(self, x, eta, epoch, mode):
#         """
#         原始的自回归递推逻辑
#         """
#         if 'kth' in cfg.dataset:
#             out_len = cfg.eval_len if mode != 'train' else cfg.out_len
#         else:
#             out_len = cfg.out_len

#         shape = [out_len] + list(x.shape)[1:]
#         shape_r = [cfg.in_len + out_len] + list(x.shape)[1:]
#         if self.use_rss:
#             mask = reverse_scheduled_sampling(shape_r, epoch)
#         elif self.use_ss:
#             mask = scheduled_sampling(shape, eta)

#         outputs = []
#         decouple_losses = []
#         layer_hiddens = None
#         output = None
#         m = None

#         for t in range(x.shape[0] - 1):
#             if self.use_rss:
#                 input_t = x[t] if t == 0 else mask[t - 1] * x[t] + (1 - mask[t - 1]) * output
#             else:
#                 if t < cfg.in_len:
#                     input_t = x[t]
#                 else:
#                     if self.use_ss:
#                         input_t = mask[t - cfg.in_len] * x[t] + (1 - mask[t - cfg.in_len]) * output
#                     else:
#                         # print("no ss")
#                         input_t = output

#             output, m, layer_hiddens, decouple_loss = self.rnns(input_t, m, layer_hiddens, self.embed, self.fc)
#             outputs.append(output)
#             decouple_losses.append(decouple_loss)

#         outputs = torch.stack(outputs)  # [T, B, C, H, W]
#         decouple_losses = torch.stack(decouple_losses)  # [T, L, B, C]
#         return outputs, decouple_losses

from torch import nn
import torch
from config import cfg
import numpy as np
from util.utils import make_layers
import math


def scheduled_sampling(shape, eta):
    S, B, C, H, W = shape
    # 随机种子已固定, 生成[0,1)随机数，形状 = (pre_len-1行，batch_size列)
    random_flip = np.random.random_sample((S - 1, B))  # outS-1 * B
    true_token = (random_flip < eta)  # 若eta为1，true_token[t, i]全部为True，mask元素全为1
    one = torch.FloatTensor(1, C, H, W).fill_(1.0).cuda()  # 1*C*H*W
    zero = torch.FloatTensor(1, C, H, W).fill_(0.0).cuda()  # 1*C*H*W
    masks = []
    for t in range(S - 1):
        masks_b = []  # B*C*H*W
        for i in range(B):
            if true_token[t, i]:
                masks_b.append(one)
            else:
                masks_b.append(zero)
        mask = torch.cat(masks_b, 0)  # along batch size
        masks.append(mask)  # outS-1 * B*C*H*W
    return masks


def reverse_scheduled_sampling(shape_r, epoch):
    start_epoch = cfg.epoch / 3
    end_epoch = cfg.epoch * 2 / 3
    step = start_epoch / 5
    if epoch < start_epoch:
        eta_r = 0.5
    elif epoch < end_epoch:
        eta_r = 1.0 - 0.5 * math.exp(-float(epoch - start_epoch) / step)
    else:
        eta_r = 1.0
    if epoch < start_epoch:
        eta = 0.5
    elif epoch < end_epoch:
        eta = 0.5 - 0.5 * (epoch - start_epoch) / (end_epoch - start_epoch)
    else:
        eta = 0.0
    S, B, C, H, W = shape_r
    random_flip_r = np.random.random_sample((cfg.in_len - 1, B))  # inS-1 * B
    random_flip = np.random.random_sample((S - cfg.in_len - 1, B))  # outS-1 * B
    true_token_r = (random_flip_r < eta_r)  # 若eta为1，true_token[t, i]全部为True，mask元素全为1
    true_token = (random_flip < eta)  # 若eta为0，true_token[t, i]全部为False，mask元素全为0
    one = torch.FloatTensor(1, C, H, W).fill_(1.0).cuda()  # 1*C*H*W
    zero = torch.FloatTensor(1, C, H, W).fill_(0.0).cuda()  # 1*C*H*W
    masks = []
    for t in range(S - 2):
        if t < cfg.in_len - 1:
            masks_b = []  # B*C*H*W
            for i in range(B):
                if true_token_r[t, i]:
                    masks_b.append(one)
                else:
                    masks_b.append(zero)
            mask = torch.cat(masks_b, 0)  # along batch size
            masks.append(mask)  # inS-1 * B*C*H*W
        else:
            masks_b = []  # B*C*H*W
            for i in range(B):
                if true_token[t - (cfg.in_len - 1), i]:
                    masks_b.append(one)
                else:
                    masks_b.append(zero)
            mask = torch.cat(masks_b, 0)  # along batch size
            masks.append(mask)  # outS-1 * B*C*H*W
    return masks


class Model(nn.Module):
    def __init__(self, embed, rnn, fc):
        super().__init__()
        self.embed = make_layers(embed)
        self.rnns = rnn  # 可以是原始 RNN，也可以是包含 encoder/decoder 的组合结构
        self.fc = make_layers(fc)
        self.use_ss = cfg.scheduled_sampling
        self.use_rss = cfg.reverse_scheduled_sampling

    def forward(self, inputs, mode='train'):
        x, eta, epoch = inputs  # x: [T, B, C, H, W]
        in_len = cfg.in_len
        out_len = cfg.out_len

        if cfg.model_name == 'my_ed_ConvLSTM':
            return self.forward_ed_convlstm(x, eta, epoch)
        else:
            return self.forward_autoregressive(x, eta, epoch, mode)

    def forward_ed_convlstm(self, x, eta, epoch):
        """
        Encoder-Decoder ConvLSTM Forward Logic
        """
        input_seq = x[:cfg.in_len].permute(1, 0, 2, 3, 4)      # [T_in, B, C, H, W]
        gt_future = x[cfg.in_len:]      # [T_out, B, C, H, W]

        decouple_losses = torch.zeros(cfg.out_len, cfg.LSTM_layers, cfg.batch, cfg.lstm_hidden_state).to(x.device)
        output = self.rnns(input_seq)
        output = output.permute(1, 0, 2, 3, 4)

        return output, decouple_losses

    def forward_autoregressive(self, x, eta, epoch, mode):
        """
        修正的自回归递推逻辑
        关键修复：测试模式下只使用输入帧进行预测，不使用目标帧
        """
        if 'kth' in cfg.dataset:
            out_len = cfg.eval_len if mode != 'train' else cfg.out_len
        else:
            out_len = cfg.out_len

        shape = [out_len] + list(x.shape)[1:]
        shape_r = [cfg.in_len + out_len] + list(x.shape)[1:]
        
        # 只在训练模式下使用 scheduled sampling
        if mode == 'train':
            if self.use_rss:
                mask = reverse_scheduled_sampling(shape_r, epoch)
            elif self.use_ss:
                mask = scheduled_sampling(shape, eta)
        else:
            mask = None  # 测试模式下不使用mask

        outputs = []
        decouple_losses = []
        layer_hiddens = None
        output = None
        m = None
        
        # ========== 关键修复：区分训练和测试模式 ==========
        if mode == 'test':
            print(f"测试模式：输入帧数={x.shape[0]}, 预测帧数={out_len}")
            
            # 第一步：处理所有输入帧（建立隐藏状态）
            for t in range(cfg.in_len):
                input_t = x[t]
                output, m, layer_hiddens, decouple_loss = self.rnns(input_t, m, layer_hiddens, self.embed, self.fc)
                # 不保存这些输出，它们只是建立隐藏状态
            
            # 第二步：进行out_len步自回归预测
            for t in range(out_len):
                # 使用上一步的预测作为输入
                input_t = output
                output, m, layer_hiddens, decouple_loss = self.rnns(input_t, m, layer_hiddens, self.embed, self.fc)
                outputs.append(output)
                decouple_losses.append(decouple_loss)
            
            print(f"测试完成：生成了{len(outputs)}帧预测")
        
        else:
            # 训练模式：使用原来的逻辑
            print(f"训练模式：总帧数={x.shape[0]}, 输入帧数={cfg.in_len}, 预测帧数={out_len}")
            
            for t in range(x.shape[0] - 1):
                if self.use_rss:
                    input_t = x[t] if t == 0 else mask[t - 1] * x[t] + (1 - mask[t - 1]) * output
                else:
                    if t < cfg.in_len:
                        input_t = x[t]
                    else:
                        if self.use_ss:
                            input_t = mask[t - cfg.in_len] * x[t] + (1 - mask[t - cfg.in_len]) * output
                        else:
                            input_t = output
                
                output, m, layer_hiddens, decouple_loss = self.rnns(input_t, m, layer_hiddens, self.embed, self.fc)
                outputs.append(output)
                decouple_losses.append(decouple_loss)
        
        # 堆叠输出
        if outputs:
            outputs = torch.stack(outputs)  # [T, B, C, H, W]
        else:
            outputs = torch.empty(0).to(x.device)
            
        if decouple_losses:
            decouple_losses = torch.stack(decouple_losses)  # [T, L, B, C]
        else:
            decouple_losses = torch.empty(0).to(x.device)
            
        return outputs, decouple_losses



