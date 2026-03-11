"""
SEVIR 评估指标
包含 CSI, POD, FAR 等降水预报评估指标
"""

import torch
import torch.nn as nn
from typing import List, Optional
import numpy as np


class SEVIRSkillScore(nn.Module):
    """
    SEVIR 技能评分计算
    
    支持的指标:
    - CSI (Critical Success Index): TP / (TP + FN + FP)
    - POD (Probability of Detection): TP / (TP + FN)
    - FAR (False Alarm Rate): FP / (TP + FP)
    - BIAS: (TP + FP) / (TP + FN)
    
    降水阈值 (VIL 归一化后的值):
    - 16: 轻度降水 (~0.063)
    - 74: 中度降水 (~0.290)
    - 133: 强降水 (~0.522)
    - 160: 暴雨 (~0.627)
    - 181: 大暴雨 (~0.710)
    - 219: 特大暴雨 (~0.859)
    """
    
    # 原始 VIL 阈值 (dBZ)
    DEFAULT_THRESHOLDS = [16, 74, 133, 160, 181, 219]
    
    def __init__(
        self,
        metrics_list: List[str] = None,
        threshold_list: List[float] = None,
        scale: float = 255.0,
        dist_sync_on_step: bool = False
    ):
        """
        初始化评分器
        
        Args:
            metrics_list: 要计算的指标列表 ['csi', 'pod', 'far', 'bias']
            threshold_list: 降水阈值列表 (原始 VIL 值)
            scale: 归一化缩放因子
            dist_sync_on_step: 分布式同步
        """
        super().__init__()
        
        self.metrics_list = metrics_list or ['csi', 'pod', 'far']
        self.threshold_list = threshold_list or self.DEFAULT_THRESHOLDS
        self.scale = scale
        self.dist_sync_on_step = dist_sync_on_step
        
        # 归一化阈值
        self.normalized_thresholds = [t / scale for t in self.threshold_list]
        
        # 注册缓冲区用于累积统计
        n_thresholds = len(self.threshold_list)
        self.register_buffer('hits', torch.zeros(n_thresholds))
        self.register_buffer('misses', torch.zeros(n_thresholds))
        self.register_buffer('fas', torch.zeros(n_thresholds))  # false alarms
        self.register_buffer('correct_negatives', torch.zeros(n_thresholds))
    
    def reset(self):
        """重置累积统计"""
        self.hits.zero_()
        self.misses.zero_()
        self.fas.zero_()
        self.correct_negatives.zero_()
    
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        更新统计量
        
        Args:
            pred: 预测值 (B, T, C, H, W) 或 (B, T, H, W)
            target: 目标值，形状与 pred 相同
        """
        # 确保是 tensor
        if not isinstance(pred, torch.Tensor):
            pred = torch.tensor(pred)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target)
        
        # 确保在同一设备
        device = self.hits.device
        pred = pred.to(device)
        target = target.to(device)
        
        # 展平为 (N,) 进行计算
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        for i, threshold in enumerate(self.normalized_thresholds):
            pred_binary = (pred_flat >= threshold).float()
            target_binary = (target_flat >= threshold).float()
            
            # TP: 预测和目标都为正
            hits = (pred_binary * target_binary).sum()
            # FN: 目标为正，预测为负
            misses = ((1 - pred_binary) * target_binary).sum()
            # FP: 预测为正，目标为负
            false_alarms = (pred_binary * (1 - target_binary)).sum()
            # TN: 预测和目标都为负
            correct_neg = ((1 - pred_binary) * (1 - target_binary)).sum()
            
            self.hits[i] += hits
            self.misses[i] += misses
            self.fas[i] += false_alarms
            self.correct_negatives[i] += correct_neg
    
    def compute(self) -> dict:
        """
        计算所有指标
        
        Returns:
            包含各阈值下各指标的字典
        """
        results = {}
        eps = 1e-10
        
        for i, threshold in enumerate(self.threshold_list):
            hits = self.hits[i]
            misses = self.misses[i]
            fas = self.fas[i]
            
            if 'csi' in self.metrics_list:
                csi = hits / (hits + misses + fas + eps)
                results[f'csi_{threshold}'] = csi.item()
            
            if 'pod' in self.metrics_list:
                pod = hits / (hits + misses + eps)
                results[f'pod_{threshold}'] = pod.item()
            
            if 'far' in self.metrics_list:
                far = fas / (hits + fas + eps)
                results[f'far_{threshold}'] = far.item()
            
            if 'bias' in self.metrics_list:
                bias = (hits + fas) / (hits + misses + eps)
                results[f'bias_{threshold}'] = bias.item()
        
        # 计算平均值
        if 'csi' in self.metrics_list:
            csi_values = [results[f'csi_{t}'] for t in self.threshold_list]
            results['csi_avg'] = np.mean(csi_values)
        
        if 'pod' in self.metrics_list:
            pod_values = [results[f'pod_{t}'] for t in self.threshold_list]
            results['pod_avg'] = np.mean(pod_values)
        
        return results
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        前向传播，计算指标
        
        Args:
            pred: 预测值
            target: 目标值
            
        Returns:
            指标字典
        """
        self.reset()
        self.update(pred, target)
        return self.compute()


def compute_csi(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.063) -> torch.Tensor:
    """
    计算单个阈值的 CSI
    
    Args:
        pred: 预测值
        target: 目标值
        threshold: 阈值 (归一化后的值)
        
    Returns:
        CSI 值
    """
    eps = 1e-10
    
    pred_binary = (pred >= threshold).float()
    target_binary = (target >= threshold).float()
    
    hits = (pred_binary * target_binary).sum()
    misses = ((1 - pred_binary) * target_binary).sum()
    false_alarms = (pred_binary * (1 - target_binary)).sum()
    
    csi = hits / (hits + misses + false_alarms + eps)
    return csi


def compute_pod(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.063) -> torch.Tensor:
    """
    计算单个阈值的 POD
    
    Args:
        pred: 预测值
        target: 目标值
        threshold: 阈值
        
    Returns:
        POD 值
    """
    eps = 1e-10
    
    pred_binary = (pred >= threshold).float()
    target_binary = (target >= threshold).float()
    
    hits = (pred_binary * target_binary).sum()
    misses = ((1 - pred_binary) * target_binary).sum()
    
    pod = hits / (hits + misses + eps)
    return pod


def test_metrics():
    """测试评估指标"""
    # 创建测试数据
    pred = torch.rand(2, 12, 1, 128, 128)
    target = torch.rand(2, 12, 1, 128, 128)
    
    # 创建评分器
    scorer = SEVIRSkillScore(metrics_list=['csi', 'pod', 'far'])
    
    # 计算指标
    results = scorer(pred, target)
    
    print("Metrics results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")


if __name__ == '__main__':
    test_metrics()
