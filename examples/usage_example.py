"""多模态编码器使用示例。"""

import torch
from pathlib import Path
from encoders import MultiModalEncoder


def example_single_chunk():
    """单个 chunk 编码示例。"""
    print("=" * 60)
    print("示例 1: 单个 chunk 编码")
    print("=" * 60)

    # 初始化编码器
    encoder = MultiModalEncoder(
        vl_model_name="Qwen/Qwen-VL",  # 或本地路径
        vl_output_dim=512,
        action_dim=14,  # 双臂机器人动作维度
        action_chunk_len=100,
        action_output_dim=256,
        embedding_dim=768,
        fusion_type="concat",  # 可选: "concat", "add", "weighted"
        freeze_vl=True,
        freeze_action=True,
    )

    # 准备数据
    frames = [
        "path/to/frame_0.jpg",
        "path/to/frame_50.jpg",
        "path/to/frame_99.jpg",
    ]
    text = "pick up the cup and place it on the table"
    actions = torch.randn(100, 14)  # [chunk_len, action_dim]

    # 编码
    embedding = encoder(frames, text, actions, normalize=True)
    print(f"Embedding shape: {embedding.shape}")  # [768]
    print(f"Embedding norm: {embedding.norm().item():.4f}")  # 应该接近 1.0


def example_batch_encoding():
    """批量编码示例。"""
    print("\n" + "=" * 60)
    print("示例 2: 批量编码")
    print("=" * 60)

    encoder = MultiModalEncoder(
        vl_model_name="Qwen/Qwen-VL",
        embedding_dim=768,
        fusion_type="weighted",  # 可学习权重
    )

    # 准备批量数据
    batch_size = 4
    frames_list = [
        ["path/to/chunk1_frame0.jpg", "path/to/chunk1_frame1.jpg"],
        ["path/to/chunk2_frame0.jpg", "path/to/chunk2_frame1.jpg"],
        ["path/to/chunk3_frame0.jpg", "path/to/chunk3_frame1.jpg"],
        ["path/to/chunk4_frame0.jpg", "path/to/chunk4_frame1.jpg"],
    ]
    texts = [
        "pick up the cup",
        "grasp the mug",
        "place the cup on table",
        "put down the mug",
    ]
    actions_list = [torch.randn(100, 14) for _ in range(batch_size)]

    # 批量编码
    embeddings = encoder.encode_batch(frames_list, texts, actions_list)
    print(f"Embeddings shape: {embeddings.shape}")  # [4, 768]

    # 计算相似度矩阵
    similarity_matrix = embeddings @ embeddings.T
    print(f"\nSimilarity matrix:\n{similarity_matrix}")


def example_similarity_matching():
    """相似度匹配示例。"""
    print("\n" + "=" * 60)
    print("示例 3: 相似度匹配")
    print("=" * 60)

    encoder = MultiModalEncoder(
        vl_model_name="Qwen/Qwen-VL",
        embedding_dim=768,
    )

    # Chunk 1: "pick up the cup"
    frames1 = ["path/to/chunk1_frame0.jpg"]
    text1 = "pick up the cup"
    actions1 = torch.randn(100, 14)
    emb1 = encoder(frames1, text1, actions1)

    # Chunk 2: "grasp the mug" (语义相似)
    frames2 = ["path/to/chunk2_frame0.jpg"]
    text2 = "grasp the mug"
    actions2 = torch.randn(100, 14)
    emb2 = encoder(frames2, text2, actions2)

    # Chunk 3: "open the drawer" (语义不同)
    frames3 = ["path/to/chunk3_frame0.jpg"]
    text3 = "open the drawer"
    actions3 = torch.randn(100, 14)
    emb3 = encoder(frames3, text3, actions3)

    # 计算相似度
    sim_12 = encoder.compute_similarity(emb1, emb2, metric="cosine")
    sim_13 = encoder.compute_similarity(emb1, emb3, metric="cosine")

    print(f"Similarity (pick cup vs grasp mug): {sim_12:.4f}")  # 应该较高
    print(f"Similarity (pick cup vs open drawer): {sim_13:.4f}")  # 应该较低


def example_load_act_checkpoint():
    """从 ACT checkpoint 加载示例。"""
    print("\n" + "=" * 60)
    print("示例 4: 加载 ACT 预训练权重")
    print("=" * 60)

    # 从预训练模型加载
    encoder = MultiModalEncoder.from_pretrained(
        vl_model_name="Qwen/Qwen-VL",
        act_checkpoint_path="path/to/act_checkpoint.pth",  # ACT checkpoint 路径
        embedding_dim=768,
        freeze_action=True,  # 冻结 ACT encoder
    )

    print("Loaded encoder with pretrained ACT weights")

    # 查看可学习参数
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")


def example_fusion_types():
    """不同融合方式对比。"""
    print("\n" + "=" * 60)
    print("示例 5: 不同融合方式对比")
    print("=" * 60)

    frames = ["path/to/frame.jpg"]
    text = "pick up the cup"
    actions = torch.randn(100, 14)

    for fusion_type in ["concat", "add", "weighted"]:
        encoder = MultiModalEncoder(
            vl_model_name="Qwen/Qwen-VL",
            embedding_dim=768,
            fusion_type=fusion_type,
        )

        emb = encoder(frames, text, actions)
        print(f"{fusion_type:10s} | shape: {emb.shape} | norm: {emb.norm().item():.4f}")

        if fusion_type == "weighted":
            w_vl = torch.sigmoid(encoder.weight_vl).item()
            w_action = torch.sigmoid(encoder.weight_action).item()
            w_sum = w_vl + w_action
            print(f"           | VL weight: {w_vl/w_sum:.3f}, Action weight: {w_action/w_sum:.3f}")


if __name__ == "__main__":
    # 运行示例（需要根据实际情况修改路径）
    print("多模态编码器使用示例\n")

    # 注意：以下示例需要实际的数据路径和 checkpoint
    # 请根据你的数据修改路径

    try:
        example_single_chunk()
    except Exception as e:
        print(f"示例 1 失败: {e}")

    try:
        example_batch_encoding()
    except Exception as e:
        print(f"示例 2 失败: {e}")

    try:
        example_similarity_matching()
    except Exception as e:
        print(f"示例 3 失败: {e}")

    try:
        example_load_act_checkpoint()
    except Exception as e:
        print(f"示例 4 失败: {e}")

    try:
        example_fusion_types()
    except Exception as e:
        print(f"示例 5 失败: {e}")

    print("\n" + "=" * 60)
    print("示例完成")
    print("=" * 60)
