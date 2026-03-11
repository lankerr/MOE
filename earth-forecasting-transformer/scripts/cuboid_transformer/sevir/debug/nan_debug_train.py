"""
NaN 调试训练脚本
================
用最小化的方式运行 SwiGLU+MoE Earthformer，
在每个前向/反向传播中逐层检查 NaN/Inf，
第一时间定位崩溃来源并输出详细报告到 debug/ 目录。

用法:
    cd sevir目录
    python debug/nan_debug_train.py --precision bf16-mixed --steps 3000
    python debug/nan_debug_train.py --precision 32 --steps 3000
"""
import os
import sys
import json
import time
import datetime
import argparse
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 路径设置
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEVIR_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SEVIR_DIR)
sys.path.insert(0, r"c:\Users\97290\Desktop\MOE\datswinlstm_memory")
sys.path.insert(0, os.path.join(SEVIR_DIR, "..", "..", "..", ".."))

DEBUG_DIR = SCRIPT_DIR  # debug/ 目录

# ========== NaN 检测器 ==========

class NaNDetector:
    """在每个 nn.Module 的 forward 输出和 backward 梯度上检查 NaN/Inf"""
    
    def __init__(self, model, log_path):
        self.model = model
        self.log_path = log_path
        self.hooks = []
        self.first_nan_report = None
        self.step_count = 0
        self.nan_history = []  # 所有 NaN 事件
        self._installing = False
        
    def install(self):
        """给所有模块注册 hook"""
        self._installing = True
        for name, module in self.model.named_modules():
            # forward hook: 检查输出
            h = module.register_forward_hook(
                self._make_forward_hook(name)
            )
            self.hooks.append(h)
            # backward hook: 检查梯度
            h2 = module.register_full_backward_hook(
                self._make_backward_hook(name)
            )
            self.hooks.append(h2)
        self._installing = False
        print(f"[NaN检测器] 已安装 {len(self.hooks)} 个 hooks (forward+backward)")
        
    def _make_forward_hook(self, module_name):
        def hook(module, input, output):
            self._check_tensors(output, module_name, "forward_output")
            # 也检查输入
            self._check_tensors(input, module_name, "forward_input")
        return hook
    
    def _make_backward_hook(self, module_name):
        def hook(module, grad_input, grad_output):
            self._check_tensors(grad_output, module_name, "grad_output")
            self._check_tensors(grad_input, module_name, "grad_input")
        return hook
    
    def _check_tensors(self, data, module_name, stage):
        """递归检查 tensor 中的 NaN/Inf"""
        if isinstance(data, torch.Tensor):
            self._check_single(data, module_name, stage, "")
        elif isinstance(data, (tuple, list)):
            for i, item in enumerate(data):
                if isinstance(item, torch.Tensor):
                    self._check_single(item, module_name, stage, f"[{i}]")
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    self._check_single(v, module_name, stage, f".{k}")
    
    def _check_single(self, tensor, module_name, stage, suffix):
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        
        if has_nan or has_inf:
            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()
            total = tensor.numel()
            
            # 统计
            finite_mask = torch.isfinite(tensor)
            if finite_mask.any():
                finite_vals = tensor[finite_mask]
                stats = {
                    "min": finite_vals.min().item(),
                    "max": finite_vals.max().item(),
                    "mean": finite_vals.float().mean().item(),
                    "std": finite_vals.float().std().item(),
                    "abs_max": finite_vals.abs().max().item(),
                }
            else:
                stats = {"min": "all_nan_inf", "max": "all_nan_inf"}
            
            report = {
                "timestamp": datetime.datetime.now().isoformat(),
                "step": self.step_count,
                "module": module_name,
                "stage": stage + suffix,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "nan_count": nan_count,
                "inf_count": inf_count,
                "total_elements": total,
                "nan_ratio": f"{nan_count/total*100:.2f}%",
                "finite_stats": stats,
            }
            
            self.nan_history.append(report)
            
            if self.first_nan_report is None:
                self.first_nan_report = report
                print(f"\n{'!'*70}")
                print(f"  [NaN检测] 首次 NaN/Inf 发现!")
                print(f"  Step: {self.step_count}")
                print(f"  模块: {module_name}")
                print(f"  阶段: {stage}{suffix}")
                print(f"  dtype: {tensor.dtype}, shape: {list(tensor.shape)}")
                print(f"  NaN: {nan_count}/{total} ({nan_count/total*100:.2f}%)")
                print(f"  Inf: {inf_count}/{total} ({inf_count/total*100:.2f}%)")
                print(f"  有限值统计: {stats}")
                print(f"{'!'*70}\n")
    
    def set_step(self, step):
        self.step_count = step
        
    def save_report(self):
        """保存完整报告到文件"""
        report = {
            "total_steps": self.step_count,
            "total_nan_events": len(self.nan_history),
            "first_nan": self.first_nan_report,
            "all_nan_events": self.nan_history[:200],  # 最多保存200条
        }
        path = os.path.join(self.log_path, f"nan_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[NaN检测器] 报告已保存: {path}")
        return path
    
    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ========== 权重检查 ==========

def check_weights(model, step, log_lines):
    """检查所有参数是否含NaN/Inf，返回问题参数列表"""
    problems = []
    for name, param in model.named_parameters():
        if param.data is not None:
            has_nan = torch.isnan(param.data).any().item()
            has_inf = torch.isinf(param.data).any().item()
            if has_nan or has_inf:
                info = f"Step {step} | 权重 NaN/Inf: {name} shape={list(param.shape)} nan={torch.isnan(param.data).sum().item()} inf={torch.isinf(param.data).sum().item()}"
                problems.append(info)
                log_lines.append(info)
                print(f"  [!] {info}")
        if param.grad is not None:
            has_nan = torch.isnan(param.grad).any().item()
            has_inf = torch.isinf(param.grad).any().item()
            if has_nan or has_inf:
                grad_abs_max = param.grad[torch.isfinite(param.grad)].abs().max().item() if torch.isfinite(param.grad).any() else "all_nan"
                info = f"Step {step} | 梯度 NaN/Inf: {name} shape={list(param.shape)} nan={torch.isnan(param.grad).sum().item()} inf={torch.isinf(param.grad).sum().item()} grad_abs_max={grad_abs_max}"
                problems.append(info)
                log_lines.append(info)
                print(f"  [!] {info}")
    return problems


# ========== 主流程 ==========

def build_model_and_data(precision_str):
    """构建模型和数据，和正式训练一模一样"""
    from omegaconf import OmegaConf
    from train_experiment_earthformer import CuboidSEVIRPLModule
    from experiment_factory_earthformer import apply_experiment
    from sevir_torch_wrap import SEVIRTorchDataset
    
    cfg_path = os.path.join(SEVIR_DIR, "cfg_sevir_20frame.yaml")
    oc = OmegaConf.load(cfg_path)
    dataset_oc = OmegaConf.to_object(oc.dataset)
    
    # 构建数据
    import datetime as dt
    sevir_catalog = r"c:\Users\97290\Desktop\datasets\sevir\CATALOG.csv"
    sevir_data_dir = r"c:\Users\97290\Desktop\datasets\sevir\data"
    train_dataset = SEVIRTorchDataset(
        sevir_catalog=sevir_catalog,
        sevir_data_dir=sevir_data_dir,
        seq_len=dataset_oc['in_len'] + dataset_oc['out_len'],
        batch_size=1,
        start_date=dt.datetime(*dataset_oc['start_date']),
        end_date=dt.datetime(*dataset_oc['train_val_split_date']),
        shuffle=True, verbose=True, layout="NTHWC"
    )
    
    # 构建模型 (用 PL module 保证和正式训练一样)
    total_num_steps = 10000
    pl_module = CuboidSEVIRPLModule(
        total_num_steps=total_num_steps,
        save_dir=os.path.join(DEBUG_DIR, "_debug_tmp"),
        oc_file=cfg_path)
    
    model = pl_module.torch_nn_module
    
    # 注入 SwiGLU+MoE+Flash (和 exp2 一样)
    apply_experiment(model, "exp2_swiglu_moe_flash")
    
    return model, train_dataset, dataset_oc


def run_debug_training(args):
    torch.set_float32_matmul_precision('medium')
    device = torch.device('cuda')
    
    precision_str = args.precision
    max_steps = args.steps
    
    print(f"\n{'='*70}")
    print(f"  NaN 调试训练")
    print(f"  精度: {precision_str}")
    print(f"  最大步数: {max_steps}")
    print(f"  实验: exp2_swiglu_moe_flash (SwiGLU + MoE + Flash)")
    print(f"{'='*70}\n")
    
    # 构建模型和数据
    model, train_dataset, dataset_oc = build_model_and_data(precision_str)
    model = model.to(device)
    
    # 优化器 (和正式训练一样)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    
    # 精度设置
    use_amp = precision_str != "32"
    if precision_str == "bf16-mixed":
        amp_dtype = torch.bfloat16
    elif precision_str in ("16-mixed", "16"):
        amp_dtype = torch.float16
    else:
        amp_dtype = None
    
    scaler = torch.amp.GradScaler('cuda', enabled=(precision_str == "16-mixed"))
    
    # 安装 NaN 检测器
    detector = NaNDetector(model, DEBUG_DIR)
    detector.install()
    
    # 训练循环
    log_lines = []
    in_len = dataset_oc['in_len']
    out_len = dataset_oc['out_len']
    
    log_lines.append(f"precision={precision_str}, amp_dtype={amp_dtype}, max_steps={max_steps}")
    log_lines.append(f"in_len={in_len}, out_len={out_len}")
    log_lines.append(f"model params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"开始训练循环... (每100步打印一次)")
    
    nan_found_step = None
    t0 = time.time()
    
    for step in range(max_steps):
        detector.set_step(step)
        
        # 取数据
        idx = step % len(train_dataset)
        sample = train_dataset[idx]
        if isinstance(sample, np.ndarray):
            sample = torch.from_numpy(sample)
        data_seq = sample.unsqueeze(0).to(device)  # (1, T, H, W, C)
        
        x = data_seq[:, :in_len].contiguous()
        y = data_seq[:, in_len:in_len+out_len].contiguous()
        
        # 检查输入数据
        if torch.isnan(x).any() or torch.isinf(x).any():
            msg = f"Step {step}: 输入数据本身包含 NaN/Inf!"
            print(f"  [!] {msg}")
            log_lines.append(msg)
        
        # 前向
        optimizer.zero_grad()
        
        if use_amp:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                output = model(x)
                loss = F.l1_loss(output, y) + F.mse_loss(output, y)
        else:
            output = model(x)
            loss = F.l1_loss(output, y) + F.mse_loss(output, y)
        
        # 检查 loss
        loss_val = loss.item()
        if np.isnan(loss_val) or np.isinf(loss_val):
            msg = f"Step {step}: loss = {loss_val} (NaN/Inf!)"
            print(f"  [!] {msg}")
            log_lines.append(msg)
            if nan_found_step is None:
                nan_found_step = step
        
        # 反向
        if use_amp and precision_str == "16-mixed":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # 检查权重
        weight_problems = check_weights(model, step, log_lines)
        
        # 定期打印
        if step % 100 == 0:
            elapsed = time.time() - t0
            speed = (step + 1) / elapsed if elapsed > 0 else 0
            vram = torch.cuda.max_memory_allocated() / 1024**3
            msg = f"Step {step:5d}/{max_steps} | loss={loss_val:.6f} | {speed:.1f} it/s | VRAM={vram:.2f}GB | NaN事件={len(detector.nan_history)}"
            print(msg)
            log_lines.append(msg)
        
        # 如果首次发现NaN，再多跑5步收集更多信息，然后停止
        if nan_found_step is not None and step > nan_found_step + 5:
            print(f"\n[停止] NaN 在 step {nan_found_step} 首次出现，已收集额外5步数据")
            break
    
    # 保存报告
    elapsed = time.time() - t0
    log_lines.append(f"\n总计: {step+1} 步, {elapsed:.1f}秒")
    log_lines.append(f"NaN事件总数: {len(detector.nan_history)}")
    
    report_path = detector.save_report()
    
    # 保存文本日志
    log_path = os.path.join(DEBUG_DIR, f"debug_log_{precision_str}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"[调试日志] {log_path}")
    
    # VRAM 峰值
    max_vram = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\nVRAM 峰值: {max_vram:.2f} GB")
    
    # 总结
    print(f"\n{'='*70}")
    if nan_found_step is not None:
        print(f"  结论: NaN 在第 {nan_found_step} 步首次出现")
        print(f"  首次出现位置: {detector.first_nan_report['module']}")
        print(f"  阶段: {detector.first_nan_report['stage']}")
        print(f"  详细报告: {report_path}")
    else:
        print(f"  结论: {step+1} 步内未发现 NaN (precision={precision_str})")
    print(f"  VRAM 峰值: {max_vram:.2f} GB")
    print(f"{'='*70}")
    
    detector.remove()
    return nan_found_step, max_vram


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", default="bf16-mixed", 
                        choices=["32", "16-mixed", "bf16-mixed"],
                        help="精度模式")
    parser.add_argument("--steps", default=3000, type=int,
                        help="最大训练步数 (约1.7个epoch)")
    args = parser.parse_args()
    run_debug_training(args)
