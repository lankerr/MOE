# import sys
# sys.path.append("/data_8t/WSG/code/MS-RNN-main/models/encoder_decoder")
# import numpy as np
# import torch
# import torch.nn as nn
# from ConvLSTM import *
# import ConvSeq as conv_seq
# from torch.autograd import Variable

# # class Encode_Decode_ConvLSTM(nn.Module):
# #     def __init__(self,
# #                  encoder,
# #                  decoder,
# #                  info,
# #                  ):
# #         super(Encode_Decode_ConvLSTM, self).__init__()
# #         self.encoder = encoder
# #         self.decoder = decoder
# #         self.info = info
# #         self.models = nn.ModuleList([self.encoder,self.decoder])

# #     def forward(self, input):
# #         in_decode_frame_dat = Variable(torch.zeros(
# #             self.info['TRAIN']['BATCH_SIZE'],
# #             self.info['DATA']['OUTPUT_SEQ_LEN'],
# #             self.info['MODEL_NETS']['ENCODE_CELLS'][-1][1],
# #             self.info['MODEL_NETS']['DESHAPE'][0],
# #             self.info['MODEL_NETS']['DESHAPE'][0],
# #         ).cuda())
# #         encode_states = self.models[0](input)
# #         output = self.models[1](in_decode_frame_dat, encode_states)
# #         return output

# def get_cell_param(parameter):
#     param = {}
#     param['input_channels'] = parameter[0]
#     param['output_channels'] = parameter[1]
#     param['input_to_state_kernel_size'] = (parameter[2], parameter[2])
#     param['state_to_state_kernel_size'] = (parameter[3], parameter[3])
#     if len(parameter) == 5:
#         param['input_to_input_kernel_size'] = (parameter[4], parameter[4])
#     return param


# def get_conv_param(parameter, padding, activate='relu', reset=False):
#     param = {}
#     if reset:
#         param['in_channel'] = parameter[1]
#         param['out_channel'] = parameter[0]
#     else:
#         param['in_channel'] = parameter[0]
#         param['out_channel'] = parameter[1]
#     param['kernel_size'] = (parameter[2], parameter[2])
#     param['stride'] = parameter[3] if len(parameter) >= 4 else 1
#     if len(padding) == 2:
#         param['output_padding'] = padding[1]
#     param['padding'] = padding[0]
#     param['activate'] = activate
#     return param


# class Decoder_ConvLSTM(nn.Module):
#     def __init__(self,
#                  conv_rnn_cells,
#                  conv_cells,
#                  output_cells,
#                  ):
#         super(Decoder_ConvLSTM, self).__init__()
#         self.conv_rnn_cells = conv_rnn_cells
#         self.conv_cells = conv_cells
#         self.models = []
#         self.output_cells = output_cells
#         self.layer_num = len(self.conv_rnn_cells)
#         for idx in range(self.layer_num):
#             self.models.append(
#                 ConvLSTM(
#                     cell_param=self.conv_rnn_cells[idx],
#                     return_sequence=True,
#                     return_state=False,
#                 )
#             )
#             self.models.append(
#                 conv_seq.DeConv2D(
#                     cell_param=self.conv_cells[idx]
#                 ).cuda()
#             )
#         for output_cell in output_cells:
#             self.models.append(
#                 conv_seq.Conv2D(
#                     cell_param=output_cell
#                 ).cuda()
#             )
#         self.models = nn.ModuleList(self.models)

#     def forward(self, input,state = None):
#         assert state is not None
#         for layer_idx in range(self.layer_num):
#             current_conv_rnn_output = self.models[2*layer_idx](input,state[self.layer_num-1-layer_idx])
#             current_conv_output = self.models[2*layer_idx+1](current_conv_rnn_output)

#             if layer_idx == self.layer_num-1:
#                 pass
#             else:
#                 input = current_conv_output

#         output = current_conv_output
#         for out_layer_idx in range(len(self.output_cells)):
#             output = self.models[2*len(self.conv_rnn_cells)+out_layer_idx](output)
#         # print('the size of output is:', output.size(), layer_idx)
#         return output

# class Encoder_ConvLSTM(nn.Module):
#     def __init__(self,
#                  conv_rnn_cells,
#                  conv_cells,
#     ):
#         super(Encoder_ConvLSTM, self).__init__()
#         self.conv_rnn_cells = conv_rnn_cells
#         self.conv_cells = conv_cells
#         self.models = []
#         self.layer_num = len(self.conv_rnn_cells)
#         for idx in range(self.layer_num):
#             self.models.append(
#                 conv_seq.Conv2D(
#                     cell_param=self.conv_cells[idx]
#                 ).cuda()
#             )
#             self.models.append(
#                 ConvLSTM(
#                     cell_param=self.conv_rnn_cells[idx],
#                     return_sequence=True,
#                     return_state=True,
#                 )
#             )
#         self.models = nn.ModuleList(self.models)


#     def forward(self, input,state = None):
#         encode_state = []
#         for layer_idx in range(self.layer_num):

#             current_conv_output = self.models[2*layer_idx](input)

#             current_conv_rnn_output,state = self.models[2*layer_idx+1](current_conv_output)
#             encode_state.append(state)
#             if layer_idx == self.layer_num-1:
#                 pass
#             else:
#                 input = current_conv_rnn_output
#         return encode_state

# class Encode_Decode_ConvLSTM(nn.Module):
#     def __init__(self):
#         super(Encode_Decode_ConvLSTM, self).__init__()
#         print("This is my_ed_ConvLSTM")

#         # === 等效于原YAML配置内容 ===
#         # self.info = {
#         #     'TRAIN': {'BATCH_SIZE': 1},
#         #     'DATA': {
#         #         'INPUT_SEQ_LEN': 13,
#         #         'OUTPUT_SEQ_LEN': 12,
#         #         'HEIGHT': 384,
#         #         'WIDTH': 384,
#         #     },
#         #     'MODEL_NETS': {
#         #         'ENCODE_CELLS': [[16, 64, 3, 3], [32, 32, 3, 3], [16, 16, 3, 3]],
#         #         'ENCODE_PADDING': [[1], [1], [1]],
#         #         'DECODE_CELLS': [[16, 16, 3, 3], [32, 32, 3, 3], [64, 64, 3, 3]],
#         #         'DECODE_PADDING': [[1, 1], [1, 0], [1, 0]],
#         #         'DESHAPE': [13, 26, 51, 101],
#         #         'DOWNSAMPLE_CONVS': [[1, 16, 3, 2], [64, 32, 3, 2], [32, 16, 3, 2]],
#         #         'UPSAMPLE_CONVS': [[16, 32, 3, 2], [32, 64, 3, 2], [64, 16, 3, 2]],
#         #         'OUTPUT_CONV': [[16, 1, 1, 1]],
#         #         'OUTPUT_PADDING': [[0]]
#         #     }
#         # }
#         self.info = {
#             'DATA': {
#                 'INPUT_SEQ_LEN': 13,
#                 'OUTPUT_SEQ_LEN': 12,
#                 'HEIGHT': 384,
#                 'WIDTH': 384,
#             },
#             'TRAIN': {
#                 'BATCH_SIZE': 1
#             },
#             'MODEL_NETS': {
#                 'DOWNSAMPLE_CONVS': [[1, 8, 7, 4], [64, 64, 5, 3], [192, 192, 3, 2]],
#                 'ENCODE_CELLS': [[8, 64, 3, 1], [64, 192, 3, 1], [192, 192, 3, 1]],
#                 'ENCODE_PADDING': [[3], [2], [1]],

#                 'DECODE_CELLS': [[192, 192, 3, 1], [192, 192, 3, 1], [192, 64, 3, 1]],
#                 'UPSAMPLE_CONVS': [[192, 192, 3, 2], [192, 192, 5, 3], [64, 8, 7, 4]],
#                 'DECODE_PADDING': [[1, 1], [1, 0], [2, 1]],

#                 'OUTPUT_CONV': [[8, 1, 1, 1]],
#                 'OUTPUT_PADDING': [[0]],
#                 'DESHAPE': [16, 32, 96, 384]  # 对应每一层上采样后的尺寸
#             }
#         }

#         # === 构建 encoder ===
#         encode_conv_rnn_cells = [
#             get_cell_param(cell) for cell in self.info['MODEL_NETS']['ENCODE_CELLS']
#         ]

#         downsample_cells = []
#         for idx, cell in enumerate(self.info['MODEL_NETS']['DOWNSAMPLE_CONVS']):
#             pad = self.info['MODEL_NETS']['ENCODE_PADDING'][idx]
#             act = 'tanh' if idx == len(self.info['MODEL_NETS']['DOWNSAMPLE_CONVS']) - 1 else None
#             downsample_cells.append(get_conv_param(cell, padding=pad, activate=act))

#         self.encoder = Encoder_ConvLSTM(
#             conv_rnn_cells=encode_conv_rnn_cells,
#             conv_cells=downsample_cells
#         )

#         # === 构建 decoder ===
#         decode_conv_rnn_cells = [
#             get_cell_param(cell) for cell in self.info['MODEL_NETS']['DECODE_CELLS']
#         ]

#         upsample_cells = []
#         for idx, cell in enumerate(self.info['MODEL_NETS']['UPSAMPLE_CONVS']):
#             pad = self.info['MODEL_NETS']['DECODE_PADDING'][idx]
#             act = 'tanh' if idx == len(self.info['MODEL_NETS']['UPSAMPLE_CONVS']) - 1 else None
#             upsample_cells.append(get_conv_param(cell, padding=pad, activate=act))

#         output_conv_cells = []
#         for idx, cell in enumerate(self.info['MODEL_NETS']['OUTPUT_CONV']):
#             pad = self.info['MODEL_NETS']['OUTPUT_PADDING'][idx]
#             act = 'tanh' if idx == len(self.info['MODEL_NETS']['OUTPUT_CONV']) - 1 else None
#             output_conv_cells.append(get_conv_param(cell, padding=pad, activate=act))

#         self.decoder = Decoder_ConvLSTM(
#             conv_rnn_cells=decode_conv_rnn_cells,
#             conv_cells=upsample_cells,
#             output_cells=output_conv_cells
#         )

#         self.models = nn.ModuleList([self.encoder, self.decoder])

#     def forward(self, input):  # input: [B, T, C, H, W]
#         B = self.info['TRAIN']['BATCH_SIZE']
#         T_out = self.info['DATA']['OUTPUT_SEQ_LEN']
#         C_out = self.info['MODEL_NETS']['ENCODE_CELLS'][-1][1]
#         H = W = self.info['MODEL_NETS']['DESHAPE'][0]

#         in_decode_frame_dat = torch.zeros(B, T_out, C_out, H, W, device=input.device)
#         encode_states = self.encoder(input)
#         output = self.decoder(in_decode_frame_dat, encode_states)
#         # print(output.shape)
#         return output




# def main():
#     # 设置设备
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # 创建模型
#     model = Encode_Decode_ConvLSTM().to(device)
#     model.eval()  # 测试模式

#     # 模拟输入：[B, T_in, C, H, W]
#     B = 1
#     T_in = 13
#     C = 1
#     H = W = 384
#     input_tensor = torch.randn(B, T_in, C, H, W).to(device)

#     # 前向推理
#     with torch.no_grad():
#         output = model(input_tensor)  # [B, T_out, 1, H, W]


#     # 输出结果信息
#     print(f"\nModel structure:\n{model}")
#     print(f"\nInput shape: {input_tensor.shape}")
#     print(f"Output shape: {output.shape}")


# if __name__ == '__main__':
#     main()

# ✅ 已完成修改：清理所有 .cuda()
# ✅ 所有设备管理统一交由外部 `.to(device)` 控制
# ✅ forward 中动态使用 `input.device`
# ✅ 完全兼容多卡训练（如 DDP）环境

import sys
sys.path.append("/data_8t/WSG/code/MS-RNN-main/models/encoder_decoder")
import numpy as np
import torch
import torch.nn as nn
import ConvSeq as conv_seq
from ConvLSTM import *


def get_cell_param(parameter):
    param = {
        'input_channels': parameter[0],
        'output_channels': parameter[1],
        'input_to_state_kernel_size': (parameter[2], parameter[2]),
        'state_to_state_kernel_size': (parameter[3], parameter[3])
    }
    if len(parameter) == 5:
        param['input_to_input_kernel_size'] = (parameter[4], parameter[4])
    return param


def get_conv_param(parameter, padding, activate='relu', reset=False):
    in_ch, out_ch = (parameter[1], parameter[0]) if reset else (parameter[0], parameter[1])
    param = {
        'in_channel': in_ch,
        'out_channel': out_ch,
        'kernel_size': (parameter[2], parameter[2]),
        'stride': parameter[3] if len(parameter) >= 4 else 1,
        'padding': padding[0],
        'activate': activate
    }
    if len(padding) == 2:
        param['output_padding'] = padding[1]
    return param


class Decoder_ConvLSTM(nn.Module):
    def __init__(self, conv_rnn_cells, conv_cells, output_cells):
        super().__init__()
        self.conv_rnn_cells = conv_rnn_cells
        self.conv_cells = conv_cells
        self.output_cells = output_cells
        self.models = []
        self.layer_num = len(conv_rnn_cells)

        for idx in range(self.layer_num):
            self.models.append(ConvLSTM(conv_rnn_cells[idx], return_sequence=True, return_state=False))
            self.models.append(conv_seq.DeConv2D(cell_param=conv_cells[idx]))

        for output_cell in output_cells:
            self.models.append(conv_seq.Conv2D(cell_param=output_cell))

        self.models = nn.ModuleList(self.models)

    def forward(self, input, state=None):
        assert state is not None
        for layer_idx in range(self.layer_num):
            current_conv_rnn_output = self.models[2 * layer_idx](input, state[self.layer_num - 1 - layer_idx])
            current_conv_output = self.models[2 * layer_idx + 1](current_conv_rnn_output)
            if layer_idx < self.layer_num - 1:
                input = current_conv_output

        output = current_conv_output
        for out_layer_idx in range(len(self.output_cells)):
            output = self.models[2 * self.layer_num + out_layer_idx](output)
        return output


class Encoder_ConvLSTM(nn.Module):
    def __init__(self, conv_rnn_cells, conv_cells):
        super().__init__()
        self.conv_rnn_cells = conv_rnn_cells
        self.conv_cells = conv_cells
        self.models = []
        self.layer_num = len(conv_rnn_cells)

        for idx in range(self.layer_num):
            self.models.append(conv_seq.Conv2D(cell_param=conv_cells[idx]))
            self.models.append(ConvLSTM(conv_rnn_cells[idx], return_sequence=True, return_state=True))

        self.models = nn.ModuleList(self.models)

    def forward(self, input, state=None):
        encode_state = []
        for layer_idx in range(self.layer_num):
            current_conv_output = self.models[2 * layer_idx](input)
            current_conv_rnn_output, state = self.models[2 * layer_idx + 1](current_conv_output)
            encode_state.append(state)
            if layer_idx < self.layer_num - 1:
                input = current_conv_rnn_output
        return encode_state


class Encode_Decode_ConvLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        print("This is my_ed_ConvLSTM")

        self.info = {
            'DATA': {
                'INPUT_SEQ_LEN': 13,
                'OUTPUT_SEQ_LEN': 12,
                'HEIGHT': 384,
                'WIDTH': 384,
            },
            'TRAIN': {'BATCH_SIZE': 1},
            'MODEL_NETS': {
                'DOWNSAMPLE_CONVS': [[1, 8, 7, 4], [64, 64, 5, 3], [192, 192, 3, 2]],
                'ENCODE_CELLS': [[8, 64, 3, 1], [64, 192, 3, 1], [192, 192, 3, 1]],
                'ENCODE_PADDING': [[3], [2], [1]],
                'DECODE_CELLS': [[192, 192, 3, 1], [192, 192, 3, 1], [192, 64, 3, 1]],
                'UPSAMPLE_CONVS': [[192, 192, 3, 2], [192, 192, 5, 3], [64, 8, 7, 4]],
                'DECODE_PADDING': [[1, 1], [1, 0], [2, 1]],
                'OUTPUT_CONV': [[8, 1, 1, 1]],
                'OUTPUT_PADDING': [[0]],
                'DESHAPE': [16, 32, 96, 384]
            }
        }

        encode_conv_rnn_cells = [get_cell_param(c) for c in self.info['MODEL_NETS']['ENCODE_CELLS']]
        downsample_cells = [get_conv_param(c, p, activate='tanh' if i == 2 else None)
                            for i, (c, p) in enumerate(zip(self.info['MODEL_NETS']['DOWNSAMPLE_CONVS'],
                                                          self.info['MODEL_NETS']['ENCODE_PADDING']))]
        self.encoder = Encoder_ConvLSTM(encode_conv_rnn_cells, downsample_cells)

        decode_conv_rnn_cells = [get_cell_param(c) for c in self.info['MODEL_NETS']['DECODE_CELLS']]
        upsample_cells = [get_conv_param(c, p, activate='tanh' if i == 2 else None)
                          for i, (c, p) in enumerate(zip(self.info['MODEL_NETS']['UPSAMPLE_CONVS'],
                                                        self.info['MODEL_NETS']['DECODE_PADDING']))]
        output_conv_cells = [get_conv_param(c, p, activate='tanh' if i == 0 else None)
                             for i, (c, p) in enumerate(zip(self.info['MODEL_NETS']['OUTPUT_CONV'],
                                                           self.info['MODEL_NETS']['OUTPUT_PADDING']))]
        self.decoder = Decoder_ConvLSTM(decode_conv_rnn_cells, upsample_cells, output_conv_cells)

        # self.models = nn.ModuleList([self.encoder, self.decoder])

    def forward(self, input):
        B = self.info['TRAIN']['BATCH_SIZE']
        T_out = self.info['DATA']['OUTPUT_SEQ_LEN']
        C_out = self.info['MODEL_NETS']['ENCODE_CELLS'][-1][1]
        H = W = self.info['MODEL_NETS']['DESHAPE'][0]

        in_decode_frame_dat = torch.zeros(B, T_out, C_out, H, W, device=input.device)
        encode_states = self.encoder(input)
        output = self.decoder(in_decode_frame_dat, encode_states)
        return output


# import torch
# import os

# def print_gpu_usage(tag=""):
#     print(f"[{tag}] PID={os.getpid()} GPU memory allocated: {torch.cuda.memory_allocated(0)//1024//1024} MiB")

# device = torch.device("cuda:0")
# print_gpu_usage("before model init")
# model = Encode_Decode_ConvLSTM().to(device)
# print_gpu_usage("after model init")

# input = torch.randn(1, 13, 1, 384, 384, device=device)
# print_gpu_usage("after input")
# output = model(input)
# print_gpu_usage("after forward")
def main():
    import torch
    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = Encode_Decode_ConvLSTM().to(device)
    model.eval()

    B, T_in, C, H, W = 1, 13, 1, 384, 384
    input_tensor = torch.randn(B, T_in, C, H, W).to(device)

    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal Parameters: {total:,} ({total / 1e6:.2f} Million)")
        return total

    count_parameters(model)

    # === 计算 FLOPs ===
    try:
        flops = FlopCountAnalysis(model, input_tensor)
        print("\nFLOPs (in Giga): {:.2f} G".format(flops.total() / 1e9))
    except Exception as e:
        print("FLOPs 统计失败:", e)

    # === 模型推理 ===
    with torch.no_grad():
        output = model(input_tensor)

    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")



if __name__ == '__main__':
    main()