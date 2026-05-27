# EgoSimMatch

**Egocentric Similarity Matching** — 具身智能领域机器人数据与人手 Egocentric 数据的相似度计算与匹配。

## 📌 项目简介

本项目旨在解决具身智能（Embodied AI）领域中，**机器人数据**与**人手的 Egocentric（第一人称视角）数据**之间的跨模态相似度计算与匹配问题。

通过对图像和文本两种模态进行 Embedding 编码，将机器人数据与人类示教数据映射到共享的语义空间（如视觉-语言联合空间），从而实现高效的相似度检索与匹配。

## 🧠 核心思路

```
人类示教数据 (Ego)         机器人数据
     │                          │
     ├── Ego 图像 ──┐           ├── 机器人图像 ──┐
     ├── Ego 文本 ──┤───▶ Embd ──┤─── 机器人文本 ──┤───▶ Embd
     └── ...        │           └── ...         │
                    │                           │
                    ▼                           ▼
            Ego Embedding               Robot Embedding
                    │                           │
                    └─────────▶ 相似度计算 ◀────────┘
                                      │
                                      ▼
                              匹配 & 检索结果
```

## 🏗️ 项目结构

```
EgoSimMatch/
├── data/               # 数据加载与预处理
│   ├── ego_data.py     # 人类 Egocentric 数据接口
│   └── robot_data.py   # 机器人数据接口
├── embeddings/         # Embedding 提取模块
│   ├── image_emb.py    # 图像 Embedding
│   └── text_emb.py     # 文本 Embedding
├── matching/           # 相似度计算与匹配
│   ├── similarity.py   # 相似度度量（余弦相似度、对比学习等）
│   └── retrieval.py    # 检索与匹配逻辑
├── models/             # 模型定义
├── config/             # 配置文件
├── scripts/            # 训练与评估脚本
├── requirements.txt    # 依赖项
└── README.md           # 本文件
```

## 🚀 快速开始

```bash
# 克隆仓库
git clone https://github.com/CQU-finnalweapon/EgoSimMatch.git
cd EgoSimMatch

# 安装依赖
pip install -r requirements.txt
```

## 🧩 主要特性

- **多模态 Embedding**：分别对图像和文本进行特征提取
- **跨空间相似度计算**：支持在视觉空间、语言空间或联合视觉-语言空间中计算相似度
- **高效匹配**：支持大规模数据下的快速检索与匹配
- **可扩展**：模块化设计，便于接入不同的特征提取器与度量方式

## 📚 技术栈

- Python
- PyTorch / TensorFlow
- CLIP / ViT 等视觉-语言模型
- FAISS / Milvus 等向量检索工具（规划中）

## 📄 许可

MIT License

## 👤 作者

[CQU-finnalweapon](https://github.com/CQU-finnalweapon)
