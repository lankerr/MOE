"""
经典算法 VRAM 验证
==================
用 MNIST CNN + 简单 DCGAN 验证 RTX 3050Ti 真实可用显存。

测试项:
  1. MNIST 手写数字分类 (LeNet-5)
  2. DCGAN 生成 28×28 手写数字
  3. ResNet-18 猫狗分类 (假数据, 模拟 ImageNet)
  4. 逐步吃显存, 找到真正的 OOM 边界

用法:
  conda activate rtx3050ti_cu128
  python classic_vram_test.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gc
import time
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gpu_info():
    """打印 GPU 显存状态"""
    if device.type != 'cuda':
        print("⚠ 无 CUDA GPU, 跳过显存测试")
        return 0
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    alloc = torch.cuda.memory_allocated() / 1024**3
    peak  = torch.cuda.max_memory_allocated() / 1024**3
    print(f"GPU: {name}")
    print(f"  专用显存: {total:.2f} GB")
    print(f"  当前占用: {alloc:.2f} GB")
    print(f"  峰值占用: {peak:.2f} GB")
    return total

def reset_gpu():
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ============================================================
#  测试 1: MNIST LeNet-5
# ============================================================

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28→14
        x = self.pool(F.relu(self.conv2(x)))  # 14→7
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def test_mnist_cnn():
    print("\n" + "=" * 60)
    print("测试 1: MNIST LeNet-5 分类")
    print("=" * 60)
    reset_gpu()

    model = LeNet5().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {params/1e3:.1f}K")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    # 假 MNIST 数据
    for step in range(20):
        x = torch.randn(128, 1, 28, 28, device=device)
        y = torch.randint(0, 10, (128,), device=device)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  训练 20 step (batch=128)")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  峰值显存: {peak:.3f} GB")

    del model, opt
    return peak


# ============================================================
#  测试 2: DCGAN 生成手写数字
# ============================================================

class Generator(nn.Module):
    """DCGAN Generator: noise → 1×28×28"""
    def __init__(self, nz=100, ngf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),  # 1→4
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False),  # 4→7
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),  # 7→14
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),  # 14→28
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z.view(z.size(0), -1, 1, 1))


class Discriminator(nn.Module):
    """DCGAN Discriminator: 1×28×28 → real/fake"""
    def __init__(self, ndf=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),       # 28→14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # 14→7
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),  # 7→4
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),   # 4→1
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.main(x).view(-1)


def test_dcgan():
    print("\n" + "=" * 60)
    print("测试 2: DCGAN 生成 28×28 图像")
    print("=" * 60)
    reset_gpu()

    nz = 100
    netG = Generator(nz).to(device)
    netD = Discriminator().to(device)
    g_params = sum(p.numel() for p in netG.parameters())
    d_params = sum(p.numel() for p in netD.parameters())
    print(f"  Generator: {g_params/1e3:.1f}K params")
    print(f"  Discriminator: {d_params/1e3:.1f}K params")

    criterion = nn.BCELoss()
    optG = torch.optim.Adam(netG.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optD = torch.optim.Adam(netD.parameters(), lr=2e-4, betas=(0.5, 0.999))

    bs = 64
    for step in range(30):
        # Train D
        real = torch.randn(bs, 1, 28, 28, device=device) * 0.5 + 0.5
        label_real = torch.ones(bs, device=device)
        label_fake = torch.zeros(bs, device=device)

        optD.zero_grad()
        out_real = netD(real)
        loss_real = criterion(out_real, label_real)

        noise = torch.randn(bs, nz, device=device)
        fake = netG(noise)
        out_fake = netD(fake.detach())
        loss_fake = criterion(out_fake, label_fake)

        lossD = loss_real + loss_fake
        lossD.backward()
        optD.step()

        # Train G
        optG.zero_grad()
        out_g = netD(fake)
        lossG = criterion(out_g, label_real)
        lossG.backward()
        optG.step()

    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  训练 30 step (batch={bs})")
    print(f"  D_loss: {lossD.item():.4f}, G_loss: {lossG.item():.4f}")
    print(f"  峰值显存: {peak:.3f} GB")

    del netG, netD, optG, optD
    return peak


# ============================================================
#  测试 3: ResNet-18 猫狗分类 (假 ImageNet)
# ============================================================

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class MiniResNet18(nn.Module):
    """简化 ResNet-18: 用于猫/狗二分类"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


def test_resnet_catdog():
    """测试不同分辨率的 ResNet-18, 感受显存随分辨率的变化"""
    print("\n" + "=" * 60)
    print("测试 3: ResNet-18 猫狗分类 (不同分辨率)")
    print("=" * 60)

    resolutions = [
        (64,  32),   # 很小
        (128, 16),   # 小
        (224, 8),    # ImageNet 标准
        (384, 4),    # 高分辨率
        (512, 2),    # 更高
    ]

    results = []
    for res, bs in resolutions:
        reset_gpu()
        try:
            model = MiniResNet18(num_classes=2).to(device)
            opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            model.train()

            for step in range(10):
                x = torch.randn(bs, 3, res, res, device=device)
                y = torch.randint(0, 2, (bs,), device=device)
                loss = F.cross_entropy(model(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                del x, y, loss

            peak = torch.cuda.max_memory_allocated() / 1024**3
            tag = "OK" if peak < 3.8 else "SHARED"
            print(f"  {res}×{res} batch={bs:2d} | {peak:.3f} GB | {tag}")
            results.append((res, bs, peak, tag))
            del model, opt
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"  {res}×{res} batch={bs:2d} | OOM!")
                results.append((res, bs, 999, "OOM"))
            else:
                raise
        torch.cuda.empty_cache()
        gc.collect()

    return results


# ============================================================
#  测试 4: 显存边界探测 — 找到真正的 4GB 限制
# ============================================================

def test_vram_boundary():
    """逐步分配显存, 找到真实 OOM 边界"""
    print("\n" + "=" * 60)
    print("测试 4: 逐步分配显存, 探测真实 OOM 边界")
    print("=" * 60)
    reset_gpu()

    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  nvidia 报告的专用显存: {total:.2f} GB")

    # 每次分配 256MB, 看何时 OOM
    chunks = []
    chunk_mb = 256
    max_chunks = 40  # 最多 10 GB
    real_oom_at = None

    for i in range(1, max_chunks + 1):
        try:
            # 分配 chunk_mb MB (fp32, 每元素 4 bytes)
            n_elements = chunk_mb * 1024 * 1024 // 4
            t = torch.empty(n_elements, dtype=torch.float32, device=device)
            chunks.append(t)
            allocated = torch.cuda.memory_allocated() / 1024**3
            # 尝试做一次小计算确认内存可用
            _ = t.sum()
            speed_tag = ""
            if allocated > total:
                speed_tag = " ← 超出专用显存! 正在用共享内存(很慢)"
            print(f"  分配 {i * chunk_mb:5d} MB | GPU allocated: {allocated:.2f} GB{speed_tag}")
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                real_oom_at = (i - 1) * chunk_mb
                print(f"  ❌ OOM! 真实极限 ≈ {real_oom_at} MB ({real_oom_at/1024:.2f} GB)")
                break
            raise

    # 测速: 专用显存 vs 共享显存
    if len(chunks) > 0:
        # 清理
        del chunks
        torch.cuda.empty_cache()
        gc.collect()

        print("\n  --- 速度对比: 专用显存 vs 共享显存 ---")

        # 2GB 矩阵乘法 (在专用显存内)
        reset_gpu()
        n = 2048
        a = torch.randn(n, n, device=device)
        b = torch.randn(n, n, device=device)
        # warmup
        _ = a @ b
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            _ = a @ b
        torch.cuda.synchronize()
        t_fast = (time.perf_counter() - t0) / 10
        mem_fast = torch.cuda.memory_allocated() / 1024**3
        print(f"  2048×2048 matmul (在 {mem_fast:.2f} GB 内): {t_fast*1000:.1f} ms")

        del a, b
        torch.cuda.empty_cache()

        # 大矩阵 (可能溢出到共享)
        n2 = 8192
        try:
            a = torch.randn(n2, n2, device=device)
            b = torch.randn(n2, n2, device=device)
            mem_big = torch.cuda.memory_allocated() / 1024**3
            _ = a @ b
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                _ = a @ b
            torch.cuda.synchronize()
            t_slow = (time.perf_counter() - t0) / 3
            print(f"  8192×8192 matmul (在 {mem_big:.2f} GB 内): {t_slow*1000:.1f} ms")

            if mem_big > total:
                ratio = t_slow / (t_fast * (n2/n)**3)
                print(f"  ⚠ 溢出到共享显存后, 性能下降约 {ratio:.1f}x")
            del a, b
        except RuntimeError:
            print(f"  8192×8192 matmul: OOM (正常, 需要 ~1 GB)")

    return real_oom_at


# ============================================================
#  运行所有测试
# ============================================================

if __name__ == '__main__':
    print("🔬 经典算法 VRAM 验证")
    print("=" * 60)
    total = gpu_info()
    print()

    # 测试 1
    p1 = test_mnist_cnn()

    # 测试 2
    p2 = test_dcgan()

    # 测试 3
    r3 = test_resnet_catdog()

    # 测试 4
    oom_limit = test_vram_boundary()

    # 总结
    print("\n" + "=" * 60)
    print("📊 总结")
    print("=" * 60)
    print(f"  nvidia 报告: {total:.2f} GB 专用显存")
    if oom_limit:
        print(f"  实测 OOM 边界: {oom_limit/1024:.2f} GB")
        if oom_limit > total * 1024:
            print(f"  ⚠ Windows 共享显存机制让 PyTorch 能分配超过 {total:.0f}GB!")
            print(f"  ⚠ 但超出部分用系统内存, 速度极慢, 训练不可用!")
    else:
        print(f"  实测: 超过 {40*256/1024:.1f} GB 仍未 OOM (Windows 共享显存)")

    print(f"\n  经典算法显存占用:")
    print(f"    MNIST LeNet-5  (b=128):  {p1:.3f} GB")
    print(f"    DCGAN 28×28    (b=64):   {p2:.3f} GB")
    for res, bs, peak, tag in r3:
        if peak < 100:
            print(f"    ResNet-18 {res}×{res} (b={bs:2d}): {peak:.3f} GB  [{tag}]")

    print(f"\n  ✅ 结论: 实际可用专用显存 = {total:.2f} GB")
    print(f"  超出此值后 PyTorch 会用共享内存(系统RAM), 速度下降 10-50x")
    print(f"  训练时应确保峰值显存 < {total * 0.9:.1f} GB (留 10% 安全余量)")
