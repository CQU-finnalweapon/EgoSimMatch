"""多模态编码器 - 融合视觉-语言和动作。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
from PIL import Image

from .vl_encoder import VLEncoder
from .action_encoder import ActionEncoder


class MultiModalEncoder(nn.Module):
    """
    多模态编码器，融合视觉-语言（Qwen-VL）和动作（ACT）。

    架构：
        frames + text → VL Encoder → vl_emb [512]
        actions       → Action Encoder → action_emb [256]
                                ↓
                        concat + projection
                                ↓
                        final_emb [embedding_dim]
    """

    def __init__(
        self,
        vl_model_name: str = "Qwen/Qwen-VL",
        vl_output_dim: int = 512,
        action_dim: int = 14,
        action_chunk_len: int = 100,
        action_output_dim: int = 256,
        embedding_dim: int = 768,
        fusion_type: str = "concat",
        freeze_vl: bool = True,
        freeze_action: bool = True,
        device: Optional[str] = None,
    ):
        """
        Args:
            vl_model_name: Qwen-VL 模型名称
            vl_output_dim: VL 编码器输出维度
            action_dim: 动作维度
            action_chunk_len: 动作序列长度
            action_output_dim: 动作编码器输出维度
            embedding_dim: 最终 embedding 维度
            fusion_type: 融合方式 ("concat", "add", "weighted")
            freeze_vl: 是否冻结 VL encoder
            freeze_action: 是否冻结 Action encoder
            device: 设备
        """
        super().__init__()
        self.vl_output_dim = vl_output_dim
        self.action_output_dim = action_output_dim
        self.embedding_dim = embedding_dim
        self.fusion_type = fusion_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # VL 编码器
        self.vl_encoder = VLEncoder(
            model_name=vl_model_name,
            output_dim=vl_output_dim,
            freeze_backbone=freeze_vl,
            device=self.device,
        )

        # 动作编码器
        self.action_encoder = ActionEncoder(
            action_dim=action_dim,
            chunk_len=action_chunk_len,
            hidden_dim=512,
            output_dim=action_output_dim,
        ).to(self.device)

        if freeze_action:
            for param in self.action_encoder.parameters():
                param.requires_grad = False

        # 融合层
        self._build_fusion_layer()

    def _build_fusion_layer(self):
        """构建融合层。"""
        if self.fusion_type == "concat":
            # 拼接后投影
            fusion_input_dim = self.vl_output_dim + self.action_output_dim
            self.fusion = nn.Sequential(
                nn.Linear(fusion_input_dim, self.embedding_dim),
                nn.LayerNorm(self.embedding_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.embedding_dim, self.embedding_dim),
            ).to(self.device)

        elif self.fusion_type == "add":
            # 先投影到相同维度再相加
            self.vl_proj = nn.Linear(self.vl_output_dim, self.embedding_dim).to(
                self.device
            )
            self.action_proj = nn.Linear(
                self.action_output_dim, self.embedding_dim
            ).to(self.device)
            self.fusion = nn.Sequential(
                nn.LayerNorm(self.embedding_dim),
                nn.GELU(),
            ).to(self.device)

        elif self.fusion_type == "weighted":
            # 可学习的加权融合
            self.vl_proj = nn.Linear(self.vl_output_dim, self.embedding_dim).to(
                self.device
            )
            self.action_proj = nn.Linear(
                self.action_output_dim, self.embedding_dim
            ).to(self.device)
            self.weight_vl = nn.Parameter(torch.tensor(0.7))  # 初始权重 0.7
            self.weight_action = nn.Parameter(torch.tensor(0.3))  # 初始权重 0.3
            self.fusion = nn.Sequential(
                nn.LayerNorm(self.embedding_dim),
                nn.GELU(),
            ).to(self.device)

        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

    def forward(
        self,
        frames: List[Union[str, Image.Image]],
        text: str,
        actions: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        编码一个 chunk。

        Args:
            frames: 图像帧列表（路径或 PIL Image）
            text: 文本描述
            actions: 动作序列 [chunk_len, action_dim]
            normalize: 是否 L2 归一化

        Returns:
            emb: [embedding_dim]
        """
        # VL 编码
        vl_emb = self.vl_encoder.encode_frames_and_text(
            frames, text, normalize=False
        )  # [vl_output_dim]

        # 动作编码
        actions = actions.unsqueeze(0).to(self.device)  # [1, chunk_len, action_dim]
        action_emb = self.action_encoder(actions, normalize=False).squeeze(
            0
        )  # [action_output_dim]

        # 融合
        if self.fusion_type == "concat":
            combined = torch.cat([vl_emb, action_emb], dim=-1)
            emb = self.fusion(combined)

        elif self.fusion_type == "add":
            vl_proj = self.vl_proj(vl_emb)
            action_proj = self.action_proj(action_emb)
            emb = self.fusion(vl_proj + action_proj)

        elif self.fusion_type == "weighted":
            vl_proj = self.vl_proj(vl_emb)
            action_proj = self.action_proj(action_emb)
            # Softmax 归一化权重
            w_vl = torch.sigmoid(self.weight_vl)
            w_action = torch.sigmoid(self.weight_action)
            w_sum = w_vl + w_action
            emb = self.fusion((w_vl / w_sum) * vl_proj + (w_action / w_sum) * action_proj)

        # L2 归一化
        if normalize:
            emb = F.normalize(emb, dim=-1)

        return emb

    def encode_batch(
        self,
        frames_list: List[List],
        texts: List[str],
        actions_list: List[torch.Tensor],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        批量编码。

        Args:
            frames_list: 每个 chunk 的帧列表
            texts: 文本描述列表
            actions_list: 动作序列列表
            normalize: 是否 L2 归一化

        Returns:
            embs: [batch_size, embedding_dim]
        """
        embs = []
        for frames, text, actions in zip(frames_list, texts, actions_list):
            emb = self.forward(frames, text, actions, normalize=normalize)
            embs.append(emb)

        return torch.stack(embs)  # [B, embedding_dim]

    @classmethod
    def from_pretrained(
        cls,
        vl_model_name: str,
        act_checkpoint_path: Optional[str] = None,
        **kwargs,
    ) -> "MultiModalEncoder":
        """
        从预训练模型加载。

        Args:
            vl_model_name: Qwen-VL 模型名称或路径
            act_checkpoint_path: ACT checkpoint 路径（可选）
            **kwargs: 其他参数

        Returns:
            加载了预训练权重的 MultiModalEncoder
        """
        encoder = cls(vl_model_name=vl_model_name, **kwargs)

        # 如果提供了 ACT checkpoint，加载预训练权重
        if act_checkpoint_path:
            print(f"Loading ACT checkpoint from {act_checkpoint_path}")
            checkpoint = torch.load(act_checkpoint_path, map_location="cpu")

            if "model" in checkpoint:
                state_dict = checkpoint["model"]
                # 提取 VAE encoder 相关的权重
                encoder_state = {}
                for key, value in state_dict.items():
                    if "vae_encoder" in key:
                        new_key = key.replace("model.vae_encoder.", "")
                        encoder_state[new_key] = value

                # 加载到 action_encoder
                encoder.action_encoder.load_state_dict(encoder_state, strict=False)
                print(f"Loaded {len(encoder_state)} parameters from ACT checkpoint")

        return encoder

    def compute_similarity(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        metric: str = "cosine",
    ) -> float:
        """
        计算两个 embedding 的相似度。

        Args:
            emb1: [embedding_dim]
            emb2: [embedding_dim]
            metric: 相似度度量 ("cosine", "euclidean")

        Returns:
            similarity: 相似度分数
        """
        if metric == "cosine":
            return F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        elif metric == "euclidean":
            return -torch.dist(emb1, emb2, p=2).item()
        else:
            raise ValueError(f"Unknown metric: {metric}")
