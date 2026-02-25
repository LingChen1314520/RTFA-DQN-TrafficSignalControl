# RTFA-DQN-TrafficSignalControl

这是一个基于 SUMO 与深度强化学习（DQN）用于交通信号控制的研究/实验项目。

## 项目结构（节选）
- SUMO 配置/网络/流量文件：`.net.xml`、`.rou.xml`、`.sumocfg`
- 训练/测试脚本：`1x1_training.py`、`hangzhou_4x4_gudang_18041610_1h_testing.py` 等
- 训练模型：`model_1x1_dqn.pth`
- 结果/可视化：`testing_hangzhou_4x4_gudang_18041610_1h_results/`、`training_results/`

## 依赖（建议）
- SUMO（推荐 1.8 或以上，参见 https://sumo.dlr.de/）
- Python 3.8+
- 常用 Python 包：`numpy`, `pandas`, `torch`（如果使用 PyTorch）, `matplotlib` 等

## 使用说明
1. 安装并配置好 SUMO（确保 `sumo` 或 `sumo-gui` 可在终端运行）。
2. 在项目目录下运行训练/测试脚本，例如：

```bash
python 1x1_training.py
python hangzhou_4x4_gudang_18041610_1h_testing.py
```

具体脚本的参数和行为请查看脚本开头的注释或直接打开相应 `.py` 文件阅读。

## 模型与结果
- 训练得到的模型文件（如 `model_1x1_dqn.pth`）已包含在仓库
- 测试与可视化脚本位于 `testing_..._results/` 目录下。


