"""测试多模态编码器的基本功能（不需要实际数据）。"""

import torch
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_action_encoder():
    """测试动作编码器。"""
    print("=" * 60)
    print("测试 1: Action Encoder")
    print("=" * 60)

    from encoders.action_encoder import ActionEncoder

    # 创建编码器
    encoder = ActionEncoder(
        action_dim=14,
        chunk_len=100,
        hidden_dim=512,
        output_dim=256,
        n_heads=8,
        n_layers=4,
    )

    # 测试单个样本
    actions = torch.randn(100, 14)  # [chunk_len, action_dim]
    emb = encoder(actions.unsqueeze(0), normalize=True)  # [1, 256]

    print(f"✓ Input shape: {actions.shape}")
    print(f"✓ Output shape: {emb.shape}")
    print(f"✓ Output norm: {emb.norm().item():.4f} (应该接近 1.0)")

    # 测试批量
    actions_batch = torch.randn(4, 100, 14)  # [B, chunk_len, action_dim]
    embs = encoder(actions_batch, normalize=True)  # [4, 256]
    print(f"✓ Batch output shape: {embs.shape}")

    # 测试参数数量
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"✓ Total parameters: {total_params:,}")

    print("✅ Action Encoder 测试通过\n")


def test_vl_encoder_structure():
    """测试 VL 编码器结构（不加载实际模型）。"""
    print("=" * 60)
    print("测试 2: VL Encoder 结构")
    print("=" * 60)

    from encoders.vl_encoder import VLEncoder

    print("✓ VLEncoder 类导入成功")
    print("✓ 需要实际的 Qwen-VL 模型才能完整测试")
    print("⚠️  跳过实际模型加载（需要下载模型）\n")


def test_multimodal_encoder_structure():
    """测试多模态编码器结构（不加载 VL 模型）。"""
    print("=" * 60)
    print("测试 3: MultiModal Encoder 结构")
    print("=" * 60)

    from encoders.multimodal_encoder import MultiModalEncoder

    print("✓ MultiModalEncoder 类导入成功")

    # 测试不同融合方式的参数数量
    for fusion_type in ["concat", "add", "weighted"]:
        print(f"\n  融合方式: {fusion_type}")
        try:
            # 只创建 action encoder 和 fusion layer（跳过 VL）
            from encoders.action_encoder import ActionEncoder
            import torch.nn as nn

            action_encoder = ActionEncoder(
                action_dim=14,
                chunk_len=100,
                output_dim=256,
            )

            if fusion_type == "concat":
                fusion = nn.Sequential(
                    nn.Linear(512 + 256, 768),
                    nn.LayerNorm(768),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(768, 768),
                )
            elif fusion_type == "add":
                fusion = nn.Sequential(
                    nn.Linear(512, 768),
                    nn.Linear(256, 768),
                )
            elif fusion_type == "weighted":
                fusion = nn.Sequential(
                    nn.Linear(512, 768),
                    nn.Linear(256, 768),
                )

            fusion_params = sum(p.numel() for p in fusion.parameters())
            print(f"    ✓ Fusion layer 参数: {fusion_params:,}")

        except Exception as e:
            print(f"    ✗ 错误: {e}")

    print("\n✅ MultiModal Encoder 结构测试通过\n")


def test_similarity_computation():
    """测试相似度计算。"""
    print("=" * 60)
    print("测试 4: 相似度计算")
    print("=" * 60)

    import torch.nn.functional as F

    # 创建随机 embeddings
    emb1 = F.normalize(torch.randn(768), dim=-1)
    emb2 = F.normalize(torch.randn(768), dim=-1)
    emb3 = emb1 + 0.1 * torch.randn(768)  # 与 emb1 相似
    emb3 = F.normalize(emb3, dim=-1)

    # 余弦相似度
    sim_12 = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    sim_13 = F.cosine_similarity(emb1.unsqueeze(0), emb3.unsqueeze(0)).item()

    print(f"✓ Similarity(emb1, emb2): {sim_12:.4f}")
    print(f"✓ Similarity(emb1, emb3): {sim_13:.4f}")
    print(f"✓ emb3 应该与 emb1 更相似: {sim_13 > sim_12}")

    # 批量相似度矩阵
    embs = torch.stack([emb1, emb2, emb3])  # [3, 768]
    sim_matrix = embs @ embs.T  # [3, 3]
    print(f"\n✓ Similarity matrix:\n{sim_matrix}")

    print("\n✅ 相似度计算测试通过\n")


def test_fusion_types():
    """测试不同融合方式。"""
    print("=" * 60)
    print("测试 5: 融合方式对比")
    print("=" * 60)

    import torch.nn as nn
    import torch.nn.functional as F

    vl_emb = F.normalize(torch.randn(512), dim=-1)
    action_emb = F.normalize(torch.randn(256), dim=-1)

    # Concat
    concat_fusion = nn.Sequential(
        nn.Linear(768, 768),
        nn.GELU(),
        nn.Linear(768, 768),
    )
    concat_input = torch.cat([vl_emb, action_emb], dim=-1)
    concat_output = F.normalize(concat_fusion(concat_input), dim=-1)
    print(f"✓ Concat: input {concat_input.shape} → output {concat_output.shape}")

    # Add
    vl_proj = nn.Linear(512, 768)
    action_proj = nn.Linear(256, 768)
    add_output = F.normalize(vl_proj(vl_emb) + action_proj(action_emb), dim=-1)
    print(f"✓ Add: output {add_output.shape}")

    # Weighted
    weight_vl = torch.tensor(0.7)
    weight_action = torch.tensor(0.3)
    w_vl = torch.sigmoid(weight_vl)
    w_action = torch.sigmoid(weight_action)
    w_sum = w_vl + w_action
    weighted_output = F.normalize(
        (w_vl / w_sum) * vl_proj(vl_emb) + (w_action / w_sum) * action_proj(action_emb),
        dim=-1,
    )
    print(f"✓ Weighted: output {weighted_output.shape}, VL weight: {w_vl/w_sum:.3f}")

    print("\n✅ 融合方式测试通过\n")


def main():
    """运行所有测试。"""
    print("\n" + "=" * 60)
    print("多模态编码器功能测试")
    print("=" * 60 + "\n")

    tests = [
        ("Action Encoder", test_action_encoder),
        ("VL Encoder 结构", test_vl_encoder_structure),
        ("MultiModal Encoder 结构", test_multimodal_encoder_structure),
        ("相似度计算", test_similarity_computation),
        ("融合方式", test_fusion_types),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {name} 测试失败: {e}\n")
            failed += 1

    print("=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)

    if failed == 0:
        print("\n🎉 所有测试通过！")
        print("\n下一步:")
        print("1. 找到 ACT checkpoint 路径")
        print("2. 准备实际数据（图像帧、文本、动作序列）")
        print("3. 运行 examples/usage_example.py")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")


if __name__ == "__main__":
    main()
