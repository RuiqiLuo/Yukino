# Yukino
这仅仅只是一个开始！！！
![Yukino AI](https://github.com/user-attachments/assets/96cddbe6-882a-47e8-bf92-16f4994fdbf6)

## 项目简介

Yukino AI 是一个基于 PyTorch 和 Transformer 架构的智能体项目，旨在模仿《我的青春恋爱物语果然有问题》轻小说中雪之下雪乃的语言风格(后面看看怎么结合故事的剧情能够在slm里去形成雪之下雪乃的三观)。通过微调预训练的 GPT-2 模型，并结合 Live2D 和 Unity 平台，实现一个能够生成符合角色风格对话的智能体。目标不仅仅是通过简单的提示词让 AI 扮演角色，而是真正还原“原著”中的语言风格、思维方式、三观等，并通过 Unity 和 Live2D 实现类似斯坦福 AI 小镇中 AI 自主生活的有趣效果。

## 技术栈

- **语言模型**：PyTorch, Hugging Face Transformers (GPT-2)  
  ![Transformers](https://github.com/user-attachments/assets/cea1bccf-8ce2-45f5-8480-7b4680636195)
- **动画与交互**：Live2D, Unity
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

- 安装 Unity Hub 并创建 Unity 2020.3+ 项目。
- 安装 Live2D Cubism SDK for Unity（官方下载）。

## 使用方法

### 1. 数据准备

将雪之下雪乃的对话文本整理成 `yukino_dialogues.txt` 文件，每行一句对话。

**示例**：

```text
容我拒绝。看到这男生邪恶又下流的眼神，我感到非常危险。
什么事？
给你一个最明显的提示，我现在做的事就是社团的活动内容。
去死吧，笨蛋
```

### 2. 训练模型

运行以下命令进行模型微调：

```bash
python train.py --data_path yukino_dialogues.txt --output_dir ./yukino_gpt2_model
```

训练参数可通过 `train.py` 脚本中的 `TrainingArguments` 调整。

### 3. 生成文本

训练完成后，使用以下命令生成文本：

```bash
python generate.py --model_path ./yukino_gpt2_model --input_text "你好，雪乃"
```

**示例输出**：

```text
你好，雪乃。我是雪之下雪乃，这种问候对我来说似乎没什么意义。你有什么具体的事吗？
```

### 4. Unity 集成

将训练好的模型转换为 ONNX 格式：

```python
import torch
dummy_input = torch.randn(1, 128)
torch.onnx.export(model, dummy_input, "yukino_model.onnx")
```

在 Unity 中使用 ONNX Runtime 加载模型，并通过 Live2D 控制角色动画（详见 `unity_integration/` 目录）。

## 项目结构

```
Yukino_AI/
├── data/                     # 存放对话数据
│   └── yukino_dialogues.txt
├── models/                   # 存放训练好的模型
│   └── yukino_gpt2_model/
├── scripts/                  # 训练和生成脚本
│   ├── train.py              # 模型微调脚本
│   └── generate.py           # 文本生成脚本
├── unity_integration/        # Unity 集成相关文件
│   ├── YukinoLive2D.unity    # Unity 项目文件
│   └── scripts/              # Unity C# 脚本
├── README.md                 # 项目说明
└── requirements.txt          # Python 依赖
```

- `train.py`：用于微调 GPT-2 模型。
- `generate.py`：用于生成模仿雪之下雪乃风格的文本。
- `unity_integration/`：包含 Unity 项目和脚本，用于将模型与 Live2D 角色集成。

## 贡献指南

欢迎对本项目进行贡献！你可以通过以下方式参与：

1. Fork 本仓库。
2. 创建你的功能分支（`git checkout -b feature/AmazingFeature`）。
3. 提交你的更改（`git commit -m 'Add some AmazingFeature'`）。
4. Push 到分支（`git push origin feature/AmazingFeature`）。
5. 提交 Pull Request。

请确保你的代码符合项目的编码规范，并通过所有测试。

## 致谢

- 感谢 Hugging Face 提供的 Transformers 库。
- 感谢 Live2D 和 Unity 平台的支持。
- 灵感来源于《我的青春恋爱物语果然有问题》轻小说及其角色雪之下雪乃。
```

