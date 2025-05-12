
# Yukino

<img src="https://github.com/user-attachments/assets/96cddbe6-882a-47e8-bf92-16f4994fdbf6" alt="Yukino AI" width="400"/>

## 项目简介

Yukino是一个基于 PyTorch 和 Transformer 架构的智能体项目，旨在模仿《我的青春恋爱物语果然有问题》轻小说中雪之下雪乃的语言风格。通过微调预训练的 GPT-2 模型，并结合 Live2D 和 Unity 平台，实现一个能够生成符合角色风格对话的智能体。目标不仅仅是通过简单的提示词让 AI 扮演角色，而是真正还原"原著"中的语言风格、思维方式、三观等，并通过 Unity 和 Live2D 实现类似斯坦福 AI 小镇中 AI 自主生活的有趣效果。

## 技术栈

- **语言模型**：PyTorch, Hugging Face Transformers (GPT-2)
- **动画与交互**：Live2D, Unity
- <img src="https://github.com/user-attachments/assets/cea1bccf-8ce2-45f5-8480-7b4680636195" alt="Transformers" width="400"/>

- **数据**：《我的青春恋爱物语果然有问题》轻小说

## 安装指南

### 1. 环境要求

- Python 3.8+
- PyTorch 1.10+
- Hugging Face Transformers 4.20+
- Datasets 2.0+
- Unity 2020.3+（用于 Live2D 集成）

### 2. 安装依赖

运行以下命令安装 Python 依赖：
```bash
pip install torch transformers datasets
```

### 3. Unity 环境
1. 安装 [Unity Hub](https://unity.com/)
2. 创建 Unity 2020.3+ 项目
3. 安装 [Live2D Cubism SDK](https://www.live2d.com/)

## 使用方法

### 1. 数据准备
创建 `yukino_dialogues.txt` 文件，格式示例：
```text
容我拒绝。看到这男生邪恶又下流的眼神，我感到非常危险。
什么事？
给你一个最明显的提示，我现在做的事就是社团的活动内容。
去死吧，笨蛋
```

### 2. 训练模型
```bash
python train.py --data_path yukino_dialogues.txt --output_dir ./yukino_gpt2_model
```

### 3. 生成文本
```bash
python generate.py --model_path ./yukino_gpt2_model --input_text "你好，雪乃"
```

### 4. Unity 集成
```python
模型转换脚本
import torch
model = torch.load("./yukino_gpt2_model/pytorch_model.bin")
dummy_input = torch.randn(1, 128)
torch.onnx.export(model, dummy_input, "yukino_model.onnx")
```

## 项目结构
```
Yukino_AI/
├── data/
│   └── yukino_dialogues.txt
├── models/
│   └── yukino_gpt2_model/
├── scripts/
│   ├── train.py
│   └── generate.py
├── unity_integration/
│   ├── YukinoLive2D.unity
│   └── scripts/
├── README.md
└── requirements.txt
```

## 贡献指南
1. Fork 本仓库
2. 创建功能分支 (`git checkout -b feature/xxx`)
3. 提交更改 (`git commit -m 'feat: xxx'`)
4. 推送分支 (`git push origin feature/xxx`)
5. 创建 Pull Request

## 致谢
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Live2D Cubism](https://www.live2d.com/)
- 《我的青春恋爱物语果然有问题》小说
