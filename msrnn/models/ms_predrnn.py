from torch import nn
import torch
import sys
sys.path.append("..")
from config import cfg


class PredRNN_Cell(nn.Module):
    def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
        super().__init__()
        self._batch_size, self._state_height, self._state_width = b_h_w
        self._conv_x2h = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel * 4,
                                       kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_h2h = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel * 4,
                                       kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_c2o = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel,
                                       kernel_size=kernel_size, stride=stride, padding=padding)

        self._conv_x2h_m = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel * 3,
                                         kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_m2h_m = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel * 3,
                                         kernel_size=kernel_size, stride=stride, padding=padding)
        self._conv_m2o = cfg.LSTM_conv(in_channels=input_channel, out_channels=output_channel,
                                       kernel_size=kernel_size, stride=stride, padding=padding)

        self._conv_c_m = cfg.LSTM_conv(in_channels=2 * input_channel, out_channels=output_channel,
                                       kernel_size=1, stride=1, padding=0)

        self._input_channel = input_channel
        self._output_channel = output_channel

    def forward(self, x, m, hiddens):
        if hiddens is None:
            c = torch.zeros((x.shape[0], self._input_channel, self._state_height, self._state_width),
                            dtype=torch.float).cuda()
            h = torch.zeros((x.shape[0], self._input_channel, self._state_height, self._state_width),
                            dtype=torch.float).cuda()
        else:
            h, c = hiddens
        if m is None:
            m = torch.zeros((x.shape[0], self._output_channel, self._state_height, self._state_width),
                            dtype=torch.float).cuda()
        x2h = self._conv_x2h(x)
        h2h = self._conv_h2h(h)
        i, f, g, o = torch.chunk((x2h + h2h), 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        next_c = f * c + i * g

        x2h_m = self._conv_x2h_m(x)
        m2h_m = self._conv_m2h_m(m)
        i_m, f_m, g_m = torch.chunk((x2h_m + m2h_m), 3, dim=1)
        i_m = torch.sigmoid(i_m)
        f_m = torch.sigmoid(f_m)
        g_m = torch.tanh(g_m)
        next_m = f_m * m + i_m * g_m

        o = torch.sigmoid(o + self._conv_c2o(next_c) + self._conv_m2o(next_m))
        next_h = o * torch.tanh(self._conv_c_m(torch.cat([next_c, next_m], dim=1)))

        ouput = next_h
        next_hiddens = [next_h, next_c]
        return ouput, next_m, next_hiddens


class MS_PredRNN(nn.Module):
    def __init__(self, input_channel, output_channel, b_h_w, kernel_size, stride, padding):
        super().__init__()
        self.n_layers = cfg.LSTM_layers
        B, H, W = b_h_w
        lstm = [PredRNN_Cell(input_channel, output_channel, [B, H, W], kernel_size, stride, padding),
                PredRNN_Cell(input_channel, output_channel, [B, H // 2, W // 2], kernel_size, stride, padding),
                PredRNN_Cell(input_channel, output_channel, [B, H // 4, W // 4], kernel_size, stride, padding),
                PredRNN_Cell(input_channel, output_channel, [B, H // 4, W // 4], kernel_size, stride, padding),
                PredRNN_Cell(input_channel, output_channel, [B, H // 2, W // 2], kernel_size, stride, padding),
                PredRNN_Cell(input_channel, output_channel, [B, H, W], kernel_size, stride, padding)]
        self.lstm = nn.ModuleList(lstm)

        self.downs = nn.ModuleList([nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2)])
        self.ups = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear'), nn.Upsample(scale_factor=2, mode='bilinear')])

        self.downs_m = nn.ModuleList([nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2)])
        self.ups_m = nn.ModuleList([nn.Upsample(scale_factor=2, mode='bilinear'), nn.Upsample(scale_factor=2, mode='bilinear')])

        print('This is MS-PredRNN!')

    def forward(self, x, m, layer_hiddens, embed, fc):
        x = embed(x)
        next_layer_hiddens = []
        out = []
        for l in range(self.n_layers):
            if layer_hiddens is not None:
                hiddens = layer_hiddens[l]
            else:
                hiddens = None
            x, m, next_hiddens = self.lstm[l](x, m, hiddens)
            out.append(x)
            if l == 0:
                x = self.downs[0](x)
                m = self.downs_m[0](m)
            elif l == 1:
                x = self.downs[1](x)
                m = self.downs_m[1](m)
            elif l == 3:
                x = self.ups[0](x) + out[1]
                m = self.ups_m[0](m)
            elif l == 4:
                x = self.ups[1](x) + out[0]
                m = self.ups_m[1](m)
            next_layer_hiddens.append(next_hiddens)
        x = fc(x)
        decouple_loss = torch.zeros([cfg.LSTM_layers, cfg.batch, cfg.lstm_hidden_state]).cuda()
        return x, m, next_layer_hiddens, decouple_loss


def main():
    # 输入序列的尺寸：S = 40, B = 1, C = 1, H = 480, W = 560
    S, B, C, H, W = 40, 1, 1, 480, 560

    # 根据 cfg 初始化模型参数
    input_channel = 64  # 取自 cfg 中定义的输入通道数
    output_channel = 64  # 取自 cfg 中定义的输出通道数
    kernel_size = 3  # 卷积核大小
    stride = 1  # 步幅
    padding = 1  # 填充
    b_h_w = [B, H, W]  # 高度和宽度

    # 创建模型
    model = MS_PredRNN(input_channel, output_channel, b_h_w, kernel_size, stride, padding).cuda()
    
    # 随机生成输入序列
    x = torch.randn(S, B, C, H, W).cuda()  # 输入序列尺寸为 [S, B, C, H, W]
    m = torch.randn(B, output_channel, H, W).cuda()  # 随机生成初始记忆单元 m
    layer_hiddens = None  # 初始时刻没有隐藏层状态
    embed = cfg.embed_layer().cuda()  # 从 cfg 中加载嵌入层
    fc = cfg.fc_layer().cuda()  # 从 cfg 中加载全连接层

    # 执行前向传播，测试模型输出
    for t in range(S):
        output, m, next_layer_hiddens, decouple_loss = model(x[t], m, layer_hiddens, embed, fc)
        layer_hiddens = next_layer_hiddens  # 更新隐藏层状态
        print(f"Time step {t+1}/{S}: Output shape: {output.shape}, Memory shape: {m.shape}")

if __name__ == "__main__":
    main()