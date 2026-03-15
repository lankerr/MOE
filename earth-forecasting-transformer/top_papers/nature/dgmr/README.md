# DGMR - Deep Generative Model of Rain

## 论文信息
- **标题**: Skilful precipitation nowcasting using deep generative models of radar
- **作者**: Sørenby et al. (Google DeepMind)
- **期刊**: Nature
- **年份**: 2021
- **引用**: Nature 596, 261–266 (2021)

## 核心贡献
- 首个将生成式模型用于雷达降水外推
- 使用 GAN (Generative Adversarial Network) 架构
- 提供 90 分钟概率预报
- 超越传统数值天气预报方法

## 下载 PDF
```powershell
Invoke-WebRequest -Uri "https://arxiv.org/pdf/2104.00954.pdf" -OutFile "DGMR_Nature2021.pdf"
```

或直接点击: https://arxiv.org/pdf/2104.00954.pdf

## 架构要点
- 输入: 过去 20 分钟雷达数据
- 输出: 未来 90 分钟概率预报
- 关键: 时空一致性 + 物理约束

## 与我们工作的关系
- DGMR 是生成式外推的奠基工作
- 我们可以用类似思想改进 EarthFormer 的解码器
