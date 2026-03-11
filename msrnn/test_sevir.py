import os
import torch
import numpy as np
from config_2 import cfg
from model_new_2 import Model
from net_params_2 import nets
from util.load_sevir import load_sevir
from torch import nn
from torch.utils.data.distributed import DistributedSampler
import argparse
from tqdm import tqdm
import pandas as pd


def remove_module_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
    return new_state_dict


def normalize_data_cuda(batch):
    if batch.dim() > 5:
        # Handle the case where batch has more dimensions than expected
        # For example, if batch is [1, 1, 25, 1, 384, 384], permute might not be necessary
        batch = batch.squeeze(0)  # Remove singleton dimensions
        batch = batch.permute(1, 0, 2, 3, 4)
        # batch = batch / 255.0
    else:
        batch = batch
        batch = batch.permute(1, 0, 2, 3, 4)  # S x B x C x H x W
        # batch = batch / 255.0
    return batch.cuda()

def compute_metrics(pred, target, thresholds):
    pred = (pred * 255).astype(np.uint8)
    target = (target * 255).astype(np.uint8)
    eps = 1e-6

    results = {}
    for th in thresholds:
        th = int(th)
        pred_bin = (pred >= th).astype(np.uint8)
        target_bin = (target >= th).astype(np.uint8)

        TP = np.logical_and(pred_bin == 1, target_bin == 1).sum()
        FP = np.logical_and(pred_bin == 1, target_bin == 0).sum()
        FN = np.logical_and(pred_bin == 0, target_bin == 1).sum()
        TN = np.logical_and(pred_bin == 0, target_bin == 0).sum()

        POD = TP / (TP + FN + eps)
        FAR = FP / (TP + FP + eps)
        CSI = TP / (TP + FP + FN + eps)
        HSS = 2 * (TP * TN - FN * FP) / ((TP + FN)*(FN + TN) + (TP + FP)*(FP + TN) + eps)

        results[f'CSI_{th}'] = CSI
        results[f'POD_{th}'] = POD
        results[f'FAR_{th}'] = FAR
        results[f'HSS_{th}'] = HSS
    return results


def is_master():
    return torch.distributed.get_rank() == 0

def reshape_patch_TBCHW(x, patch_size):
    """
    输入 x: [T, B, C, H, W]
    输出:  [T, B, patch_size²*C, H//patch_size, W//patch_size]
    """
    T, B, C, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0
    x = x.reshape(T, B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 1, 3, 5, 4, 6, 2)  # T, B, h, w, p, p, C
    x = x.reshape(T, B, H // patch_size, W // patch_size, patch_size * patch_size * C)
    x = x.permute(0, 1, 4, 2, 3)  # T, B, C', H', W'
    return x


def reshape_patch_back(x, patch_size):
    """
    输入 x: [T, B, patch_size²*C, H', W']
    输出:  [T, B, C, H, W]
    """
    T, B, C_p, H_p, W_p = x.shape
    C = C_p // (patch_size * patch_size)
    x = x.permute(0, 1, 3, 4, 2)  # T, B, H', W', C'
    x = x.reshape(T, B, H_p, W_p, patch_size, patch_size, C)
    x = x.permute(0, 1, 2, 4, 3, 5, 6)  # T, B, H', p, W', p, C
    x = x.reshape(T, B, H_p * patch_size, W_p * patch_size, C)
    x = x.permute(0, 1, 4, 2, 3)  # T, B, C, H, W
    return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--resume", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--save_path", type=str, default="results/test_metrics.xlsx", help="Relative path to save metrics")
    args = parser.parse_args()

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl')

    model = Model(nets[0], nets[1], nets[2])
    model = model.cuda()
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    checkpoint = torch.load(args.resume, map_location=map_location)
    state_dict = remove_module_prefix(checkpoint['model_state_dict'])
    model.load_state_dict(state_dict)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    _, _, test_loader, _ = load_sevir()

    thresholds = [16, 74, 132, 160, 181]
    all_metrics = []
    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc='Testing', disable=not is_master())):
            print(batch.shape)
            test_batch = normalize_data_cuda(batch)
            print(test_batch.shape)
            if cfg.reshape_patch == True :
                x_patch_t = reshape_patch_TBCHW(test_batch, cfg.patch_size)
            else :
                x_patch_t = test_batch
            
            y_patch_t, decouple_loss = model([x_patch_t, 0, cfg.epoch], mode='test')
            if cfg.reshape_patch == True :
                output = reshape_patch_back(y_patch_t, cfg.patch_size)
            else :
                output = y_patch_t
            output = output[-12:].detach().cpu().numpy()
            target = test_batch[-12:].detach().cpu().numpy()

            output = np.transpose(output, (1, 0, 2, 3, 4))  # [B, T, C, H, W]
            target = np.transpose(target, (1, 0, 2, 3, 4))

            output = output.squeeze(2)  # remove channel dim
            target = target.squeeze(2)

            for b in range(output.shape[0]):
                pred_sample = output[b]
                target_sample = target[b]
                metrics = compute_metrics(pred_sample, target_sample, thresholds)

                for metric_type in ['CSI', 'POD', 'FAR', 'HSS']:
                    avg = np.mean([metrics[f'{metric_type}_{th}'] for th in thresholds])
                    metrics[f'avg_{metric_type}'] = avg

                all_metrics.append(metrics)

    if is_master():
        save_dir = os.path.dirname(args.save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)
        df = pd.DataFrame(all_metrics)
        df.index.name = 'case_id'
        save_full_path = os.path.abspath(args.save_path)
        df.to_excel(save_full_path)
        print(f"✅ Saved metrics to: {save_full_path}")


if __name__ == "__main__":
    main()


#torchrun --nproc_per_node=4 --master_port=29500 test_ms_rnn.py 
# --resume /data/WSG/code/MS_RNN/experiments/save/Sevir/ConvLSTM/20250521/models/checkpoint_epoch_22.pth 
# --save_path test/ConvLSTM_1h/epoch_1_metrics.xlsx
#/data/WSG/code/MS_RNN/experiments/save/Sevir/my_convlstm/20250530/models/checkpoint_epoch_0.pth

#python -m torch.distributed.launch --nproc_per_node=1 --master_port 10086 test_sevir.py 
# --resume /data_8t/WSG/code/MS-RNN-main/experiments/save/Sevir/my_ed_ConvLSTM/20250620/models/checkpoint_epoch_2.pth
# --save_path test/ConvLSTM_1h/epoch_33_metrics.xlsx