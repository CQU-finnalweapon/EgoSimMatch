# EgoSimMatch

**Egocentric Similarity Matching** — 机器人遥操数据与人类第一视角 Egocentric 数据的相似度计算与匹配。

## 📌 项目简介

本项目旨在解决具身智能（Embodied AI）领域中，**机器人数据**与**人手的 Egocentric（第一人称视角）数据**之间的跨模态相似度计算与匹配问题。

通过对图像、文本、动作三种模态进行 Embedding 编码，将机器人数据与人类示教数据映射到共享的语义空间，从而实现高效的相似度检索、配对与训练 batch 构建。

## 🎯 核心功能

- **多模态编码**：视觉-语言（Qwen-VL）+ 动作（ACT VAE Encoder）联合编码
- **EgouDas × EgoTel 交叉匹配**：计算人类 ego 视角与远程操作视角的跨视角相似度
- **全量 Embedding 构建**：从 TOS webdataset 流式读取并编码 ~820K segment
- **训练 batch 配对**：基于 FAISS 聚类 + 匈牙利算法，或逐条采样贪心匹配
- **动作轨迹相似度**：基于 DTW / 插值欧氏距离的动作序列相似度
- **可视化**：相似度热力图、配对效果对比大图

## 🧠 核心思路

```
人类示教数据 (Ego)              机器人数据 (Robot)
     │                              │
     ├── 图像帧 ──┐                 ├── 图像帧 ──┐
     ├── 文本   ──┤── 多模态编码器 ──┤── 文本   ──┤── 多模态编码器
     └── 动作   ──┘    (VL+Action)  └── 动作   ──┘    (VL+Action)
                    │                           │
                    ▼                           ▼
            Ego Embedding               Robot Embedding
                    │                           │
                    └─────────▶ 相似度计算 ◀────────┘
                                      │
                                      ▼
                          匹配 / 检索 / batch 配对
```

## 🧩 多模态编码器

| 编码器 | 位置 | 输入 | 输出维度 |
|--------|------|------|----------|
| **VL Encoder** | `encoders/vl_encoder.py` | 多帧图像 + 文本 | 512 |
| **Action Encoder** | `encoders/action_encoder.py` | 动作序列 `[chunk_len, action_dim]` | 256 |
| **MultiModalEncoder** | `encoders/multimodal_encoder.py` | VL + Action 融合 | 768 |

> Action Encoder 基于 ACT 的 VAE Encoder 架构（从 mozbrain 移植，模型代码见 `models/act/`），支持加载 ACT 预训练 checkpoint。多模态融合支持 `concat` / `add` / `weighted`（可学习权重，推荐）三种方式。详见 [docs/MULTIMODAL_ENCODER.md](docs/MULTIMODAL_ENCODER.md)。

## 🏗️ 项目结构

```
EgoSimMatch/
├── encoders/                          # 多模态编码器
│   ├── vl_encoder.py                  # Qwen-VL 视觉-语言编码器 (512D)
│   ├── action_encoder.py              # ACT 动作编码器 (256D)
│   └── multimodal_encoder.py          # 多模态融合 (768D)
├── models/
│   └── act/                           # ACT 模型定义（移植自 mozbrain）
├── data/                              # 数据加载与预处理
│   ├── egoudas_loader.py              # EgouDas 数据加载
│   ├── egotel_loader.py               # EgoTel 数据加载
│   ├── webdataset_loader.py           # TOS webdataset 流式加载
│   └── pair_dataset.py                # 配对数据集
├── embeddings/                        # Embedding 提取模块（图像 / 文本）
├── matching/                          # 相似度计算与检索逻辑
├── scripts/                           # 流程脚本（见下）
├── examples/usage_example.py          # 使用示例
├── tests/test_encoders.py             # 编码器测试
├── docs/                              # 文档
├── requirements.txt                   # 依赖项
└── README.md
```

## 🛠️ 主要脚本

| 脚本 | 作用 |
|------|------|
| `build_full_embeddings.py` | 从 TOS webdataset 流式读取并编码全量 segment，存盘 embedding |
| `compute_similarity_from_embeddings.py` | 从已存 embedding 计算相似度（cross / self / topk），秒级完成 |
| `batch_pairing.py` | FAISS k-means 聚类 + 匈牙利算法，构建 16×16 训练 batch |
| `batch_pairing_greedy.py` | 每条 EgoTel 随机采样 N 条 EgouDas，贪心找最优匹配 |
| `action_trajectory_similarity.py` | 基于 DTW / 插值欧氏距离的动作轨迹相似度 |
| `visualize_batch.py` | 相似度热力图 + 配对效果可视化 |

## 🎬 可视化结果

### EgouDas × EgoTel 交叉匹配

**最高相似度配对 (sim=0.885)** — 同任务：

![top_01](docs/images/top_01_sim0.885.png)

**最低相似度配对 (sim=0.325)** — 不同任务：

![bottom_01](docs/images/bottom_01_sim0.325.png)

- 左侧：EgouDas（人类第一人称视角）
- 右侧：EgoTel（远程操作视角）
- 每个 segment：9 帧图像（4秒均匀采样）+ task 文本
- 相似度通过 Qwen-VL 视觉-语言联合编码计算

## 🚀 快速开始

```bash
# 克隆仓库
git clone https://github.com/CQU-finnalweapon/EgoSimMatch.git
cd EgoSimMatch

# 安装依赖
pip install -r requirements.txt

# 1) 构建全量 embedding（从 TOS webdataset 流式读取）
python scripts/build_full_embeddings.py --device cuda:0 --batch_size 24

# 2) 计算 EgouDas × EgoTel 交叉相似度
python scripts/compute_similarity_from_embeddings.py --mode cross

# 3) 构建训练 batch（贪心采样匹配）
python scripts/batch_pairing_greedy.py --n_sample 500 --viz_top 500
```

> 模型权重（`models/Qwen3-*`、`siglip2`）与生成产物（`outputs/`）体积较大，已通过 `.gitignore` 排除，不纳入版本管理。

## 📚 技术栈

- Python 3.8+
- PyTorch
- Qwen-VL（视觉-语言联合 Embedding）
- ACT VAE Encoder（动作编码）
- FAISS（聚类与检索）、SciPy（匈牙利算法、DTW）
- OpenCV、PIL（视频帧采样与可视化）

## 📄 许可

MIT License

## 👤 作者

[CQU-finnalweapon](https://github.com/CQU-finnalweapon)
