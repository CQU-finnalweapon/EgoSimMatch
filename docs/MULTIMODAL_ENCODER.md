# 多模态编码器使用指南
## 📦 已集成的模型
### 1. **Qwen-VL** (视觉-语言编码器)
- **位置**: `encoders/vl_encoder.py`
- **功能**: 编码图像帧序列 + 文本描述
- **输出**: 512 维 embedding

### 2. **ACT VAE Encoder** (动作编码器)
- **位置**: `encoders/action_encoder.py`
- **来源**: 从 mozbrain 移植
- **功能**: 编码动作序列 [chunk_len, action_dim]
- **输出**: 256 维 embedding

### 3. **MultiModalEncoder** (多模态融合)
- **位置**: `encoders/multimodal_encoder.py`
- **功能**: 融合 VL + Action 两个模态
- **输出**: 768 维 embedding

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本使用

```python
from encoders import MultiModalEncoder
import torch

# 1. 初始化编码器
encoder = MultiModalEncoder(
    vl_model_name="Qwen/Qwen-VL",
    vl_output_dim=512,
    action_dim=14,              # 你的动作维度
    action_chunk_len=100,       # chunk 长度
    action_output_dim=256,
    embedding_dim=768,
    fusion_type="concat",       # 融合方式: "concat", "add", "weighted"
    freeze_vl=True,
    freeze_action=True,
)

# 2. 准备数据
frames = ["frame_0.jpg", "frame_50.jpg", "frame_99.jpg"]  # 采样的帧
text = "pick up the cup and place it on the table"
actions = torch.randn(100, 14)  # [chunk_len, action_dim]

# 3. 编码
embedding = encoder(frames, text, actions, normalize=True)
print(f"Embedding shape: {embedding.shape}")  # [768]
```

### 加载 ACT 预训练权重

```python
# 如果你有训练好的 ACT checkpoint
encoder = MultiModalEncoder.from_pretrained(
    vl_model_name="Qwen/Qwen-VL",
    act_checkpoint_path="/path/to/act_checkpoint.pth",
    embedding_dim=768,
    freeze_action=True,  # 冻结 ACT，只训练融合层
)
```

### 批量编码

```python
# 批量处理多个 chunks
frames_list = [
    ["chunk1_frame0.jpg", "chunk1_frame1.jpg"],
    ["chunk2_frame0.jpg", "chunk2_frame1.jpg"],
]
texts = ["pick up the cup", "grasp the mug"]
actions_list = [torch.randn(100, 14), torch.randn(100, 14)]

embeddings = encoder.encode_batch(frames_list, texts, actions_list)
print(f"Embeddings shape: {embeddings.shape}")  # [2, 768]

# 计算相似度矩阵
similarity_matrix = embeddings @ embeddings.T
```

### 相似度匹配

```python
# 编码两个 chunks
emb1 = encoder(frames1, text1, actions1)
emb2 = encoder(frames2, text2, actions2)

# 计算相似度
similarity = encoder.compute_similarity(emb1, emb2, metric="cosine")
print(f"Similarity: {similarity:.4f}")
```

## 🎯 融合策略

### 1. **Concat** (默认)
```python
fusion_type="concat"
# VL [512] + Action [256] → Concat [768] → MLP → [768]
```
- **优点**: 保留所有信息
- **缺点**: 参数较多

### 2. **Add**
```python
fusion_type="add"
# VL [512] → Proj [768]
# Action [256] → Proj [768]
# 相加 → [768]
```
- **优点**: 参数少，简单
- **缺点**: 可能丢失信息

### 3. **Weighted** (推荐)
```python
fusion_type="weighted"
# 可学习的权重: α * VL + (1-α) * Action
```
- **优点**: 自动学习最优权重
- **缺点**: 需要训练数据

## 📂 项目结构

```
EgoSimMatch/
├── encoders/
│   ├── __init__.py
│   ├── vl_encoder.py           # Qwen-VL 编码器
│   ├── action_encoder.py       # ACT 动作编码器
│   └── multimodal_encoder.py   # 多模态融合
├── models/
│   └── act/                    # ACT 模型（从 mozbrain 移植）
│       ├── modeling_act.py
│       └── configuration_act.py
├── examples/
│   └── usage_example.py        # 使用示例
└── requirements.txt
```

## 🔧 配置说明

### VL Encoder 参数
- `vl_model_name`: Qwen-VL 模型路径
- `vl_output_dim`: VL embedding 维度（默认 512）
- `freeze_vl`: 是否冻结 VL backbone

### Action Encoder 参数
- `action_dim`: 动作维度（如 14 for bimanual）
- `action_chunk_len`: 动作序列长度
- `action_output_dim`: Action embedding 维度（默认 256）
- `freeze_action`: 是否冻结 Action encoder

### Fusion 参数
- `embedding_dim`: 最终 embedding 维度（默认 768）
- `fusion_type`: 融合方式（"concat", "add", "weighted"）

## 💡 使用建议

### 1. **数据去重任务**
```python
# 使用 concat 融合，保留所有信息
encoder = MultiModalEncoder(fusion_type="concat")
```

### 2. **语义匹配任务**
```python
# 使用 weighted 融合，自动学习权重
encoder = MultiModalEncoder(fusion_type="weighted")
```

### 3. **快速原型**
```python
# 冻结所有 backbone，只训练融合层
encoder = MultiModalEncoder(
    freeze_vl=True,
    freeze_action=True,
)
```

### 4. **端到端训练**
```python
# 解冻所有参数，端到端微调
encoder = MultiModalEncoder(
    freeze_vl=False,
    freeze_action=False,
)
```

## 🔍 下一步

1. **找到 ACT checkpoint**: 在 mozbrain 训练输出中找到 `.pth` 或 `.safetensors` 文件
2. **准备数据**: 确保数据格式符合要求（帧路径、文本、动作序列）
3. **运行示例**: `python examples/usage_example.py`
4. **训练/评估**: 根据你的任务编写训练脚本

## ❓ 常见问题

**Q: 找不到 ACT checkpoint 怎么办？**
A: 可以不加载预训练权重，从头训练 Action Encoder：
```python
encoder = MultiModalEncoder(
    act_checkpoint_path=None,  # 不加载
    freeze_action=False,       # 从头训练
)
```

**Q: Qwen-VL 显存不够怎么办？**
A: 可以换用更小的模型或使用 8-bit 量化：
```python
# 使用更小的模型
vl_model_name="Qwen/Qwen-VL-Chat"  # 或其他轻量版本
```

**Q: 如何调整 VL 和 Action 的权重？**
A: 使用 weighted 融合，权重会自动学习：
```python
encoder = MultiModalEncoder(fusion_type="weighted")
# 训练后查看权重
print(f"VL weight: {encoder.weight_vl}")
print(f"Action weight: {encoder.weight_action}")
```
