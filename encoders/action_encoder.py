"""动作编码器 - 基于 ACT VAE Encoder。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from pathlib import Path


class ActionEncoder(nn.Module):
    """
    动作序列编码器，基于 ACT 的 VAE Encoder。

    将动作序列 [chunk_len, action_dim] 编码为固定维度的向量。
    """

    def __init__(
        self,
        action_dim: int = 14,
        chunk_len: int = 100,
        hidden_dim: int = 512,
        output_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            action_dim: 动作维度（如 14 for bimanual Aloha）
            chunk_len: 动作序列长度
            hidden_dim: Transformer 隐藏维度
            output_dim: 输出 embedding 维度
            n_heads: 注意力头数
            n_layers: Transformer 层数
            dropout: Dropout 率
        """
        super().__init__()
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # 输入投影：action_dim -> hidden_dim
        self.input_proj = nn.Linear(action_dim, hidden_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 输出投影：hidden_dim -> output_dim
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # 可学习的 [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(self, actions: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        编码动作序列。

        Args:
            actions: 动作序列 [batch_size, chunk_len, action_dim]
            normalize: 是否对输出做 L2 归一化

        Returns:
            action_emb: 动作 embedding [batch_size, output_dim]
        """
        batch_size = actions.shape[0]

        # 投影到隐藏空间
        x = self.input_proj(actions)  # [B, chunk_len, hidden_dim]

        # 添加 [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, hidden_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, chunk_len+1, hidden_dim]

        # Transformer 编码
        x = self.transformer(x)  # [B, chunk_len+1, hidden_dim]

        # 取 [CLS] token 的输出
        cls_output = x[:, 0]  # [B, hidden_dim]

        # 输出投影
        action_emb = self.output_proj(cls_output)  # [B, output_dim]

        # L2 归一化
        if normalize:
            action_emb = F.normalize(action_emb, dim=-1)

        return action_emb

    @classmethod
    def from_act_checkpoint(
        cls,
        checkpoint_path: str,
        output_dim: int = 256,
        freeze_backbone: bool = True,
    ) -> "ActionEncoder":
        """
        从 ACT checkpoint 加载预训练权重。

        Args:
            checkpoint_path: ACT 模型 checkpoint 路径
            output_dim: 输出维度
            freeze_backbone: 是否冻结 Transformer backbone

        Returns:
            加载了预训练权重的 ActionEncoder
        """
        # 加载 ACT checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # 从 checkpoint 中提取配置
        if "config" in checkpoint:
            config = checkpoint["config"]
            action_dim = config.get("action_dim", 14)
            chunk_len = config.get("chunk_size", 100)
            hidden_dim = config.get("dim_model", 512)
        else:
            # 默认配置
            action_dim = 14
            chunk_len = 100
            hidden_dim = 512

        # 创建编码器
        encoder = cls(
            action_dim=action_dim,
            chunk_len=chunk_len,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )

        # 加载预训练权重（如果存在）
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
            # 提取 VAE encoder 相关的权重
            encoder_state = {}
            for key, value in state_dict.items():
                if "vae_encoder" in key:
                    new_key = key.replace("vae_encoder.", "")
                    encoder_state[new_key] = value

            # 加载权重（部分加载，忽略不匹配的键）
            encoder.load_state_dict(encoder_state, strict=False)
            print(f"Loaded {len(encoder_state)} parameters from ACT checkpoint")

        # 冻结 backbone
        if freeze_backbone:
            for name, param in encoder.named_parameters():
                if "output_proj" not in name:
                    param.requires_grad = False
            print("Frozen Transformer backbone, only training output projection")

        return encoder

    def encode_batch(self, actions_list: list[torch.Tensor]) -> torch.Tensor:
        """
        批量编码动作序列（支持不同长度）。

        Args:
            actions_list: 动作序列列表，每个 [chunk_len_i, action_dim]

        Returns:
            action_embs: [batch_size, output_dim]
        """
        # Padding 到相同长度
        max_len = max(a.shape[0] for a in actions_list)
        padded_actions = []

        for actions in actions_list:
            if actions.shape[0] < max_len:
                # Zero padding
                pad_len = max_len - actions.shape[0]
                padded = F.pad(actions, (0, 0, 0, pad_len))
            else:
                padded = actions
            padded_actions.append(padded)

        # Stack 成 batch
        actions_batch = torch.stack(padded_actions)  # [B, max_len, action_dim]

        return self.forward(actions_batch)
