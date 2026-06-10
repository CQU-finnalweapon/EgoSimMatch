"""视觉-语言编码器 - 基于 SigLIP2。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
from pathlib import Path
from PIL import Image

MODEL_PATH = str(Path(__file__).parent.parent / "models/siglip2/so400m-patch14-384")


class VLEncoder(nn.Module):
    """
    视觉-语言编码器，基于 SigLIP2-so400m-patch14-384。

    SigLIP2 是专为 embedding/检索训练的对比学习模型，图像和文本共享同一语义空间。
    图像和文本分别编码后可直接计算余弦相似度。

    支持两种编码模式：
    - encode_image: 编码单张或多帧图像（取平均）
    - encode_text: 编码文本描述
    - encode_frames_and_text: 融合多帧图像 + 文本（加权平均）
    """

    # SigLIP2-so400m 的原生输出维度
    NATIVE_DIM = 1152

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        output_dim: Optional[int] = None,
        image_weight: float = 0.7,
        freeze_backbone: bool = True,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ):
        """
        Args:
            model_path: SigLIP2 本地模型路径
            output_dim: 输出维度，None 表示使用原生 1152 维
            image_weight: 图像和文本融合时图像的权重（0~1）
            freeze_backbone: 是否冻结 SigLIP2 backbone
            device: 设备，None 自动选择
            dtype: 模型精度
        """
        super().__init__()
        self.model_path = model_path
        self.output_dim = output_dim or self.NATIVE_DIM
        self.image_weight = image_weight
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self._load_model(freeze_backbone)

        # 仅在需要降维时添加投影层
        if self.output_dim != self.NATIVE_DIM:
            self.proj = nn.Linear(self.NATIVE_DIM, self.output_dim).to(self.device)
        else:
            self.proj = None

    def _load_model(self, freeze_backbone: bool):
        from transformers import GemmaTokenizer, SiglipImageProcessor, AutoModel

        # SigLIP2 的 tokenizer_config 指向 GemmaTokenizer，需要分开加载
        self.tokenizer = GemmaTokenizer.from_pretrained(self.model_path)
        self.image_processor = SiglipImageProcessor.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
        ).to(self.device)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        print(f"Loaded SigLIP2 from {self.model_path} (dim={self.NATIVE_DIM})")

    def _project(self, emb: torch.Tensor) -> torch.Tensor:
        if self.proj is not None:
            emb = self.proj(emb.float())
        return emb

    @torch.no_grad()
    def encode_image(
        self,
        frames: List[Union[str, Image.Image]],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        编码图像帧序列，多帧取平均。

        Args:
            frames: 图像帧列表（路径字符串或 PIL Image）
            normalize: 是否 L2 归一化

        Returns:
            image_emb: [output_dim]
        """
        pil_frames = [
            Image.open(f).convert("RGB") if isinstance(f, str) else f.convert("RGB")
            for f in frames
        ]

        inputs = self.image_processor(images=pil_frames, return_tensors="pt")
        # image_processor 输出 float32，需要转成模型的 dtype
        inputs = {
            k: v.to(dtype=self.dtype, device=self.device) if v.is_floating_point() else v.to(self.device)
            for k, v in inputs.items()
        }

        image_features = self.model.get_image_features(**inputs)  # [N, 1152]
        emb = image_features.mean(dim=0)  # 多帧平均 [1152]

        emb = self._project(emb)

        if normalize:
            emb = F.normalize(emb, dim=-1)

        return emb

    @torch.no_grad()
    def encode_text(
        self,
        text: str,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        编码文本描述。

        Args:
            text: 文本描述
            normalize: 是否 L2 归一化

        Returns:
            text_emb: [output_dim]
        """
        inputs = self.tokenizer(
            [text], return_tensors="pt", padding=True, truncation=True, max_length=64
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        text_features = self.model.get_text_features(**inputs)  # [1, 1152]
        emb = text_features.squeeze(0)  # [1152]

        emb = self._project(emb)

        if normalize:
            emb = F.normalize(emb, dim=-1)

        return emb

    @torch.no_grad()
    def encode_frames_and_text(
        self,
        frames: List[Union[str, Image.Image]],
        text: str,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        融合编码图像帧序列 + 文本描述。

        图像和文本分别编码后按 image_weight 加权融合。

        Args:
            frames: 图像帧列表
            text: 文本描述
            normalize: 是否 L2 归一化

        Returns:
            vl_emb: [output_dim]
        """
        image_emb = self.encode_image(frames, normalize=False)
        text_emb = self.encode_text(text, normalize=False)

        emb = self.image_weight * image_emb + (1 - self.image_weight) * text_emb

        if normalize:
            emb = F.normalize(emb, dim=-1)

        return emb

    @torch.no_grad()
    def encode_batch(
        self,
        frames_list: List[List],
        texts: List[str],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        批量编码多个 chunk。

        Args:
            frames_list: 每个 chunk 的帧列表
            texts: 文本描述列表
            normalize: 是否 L2 归一化

        Returns:
            vl_embs: [batch_size, output_dim]
        """
        embs = [
            self.encode_frames_and_text(frames, text, normalize=normalize)
            for frames, text in zip(frames_list, texts)
        ]
        return torch.stack(embs)

    @torch.no_grad()
    def encode_image_batch(
        self,
        frames_list: List[List],
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        批量编码图像（不含文本），适合纯视觉相似度场景。

        Args:
            frames_list: 每个 chunk 的帧列表
            normalize: 是否 L2 归一化

        Returns:
            image_embs: [batch_size, output_dim]
        """
        embs = [
            self.encode_image(frames, normalize=normalize)
            for frames in frames_list
        ]
        return torch.stack(embs)

    def compute_similarity_matrix(
        self,
        embs1: torch.Tensor,
        embs2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算相似度矩阵。

        Args:
            embs1: [N, output_dim]，已 L2 归一化
            embs2: [M, output_dim]，已 L2 归一化，None 则计算 embs1 自身

        Returns:
            sim_matrix: [N, M] 或 [N, N]
        """
        if embs2 is None:
            return embs1 @ embs1.T
        return embs1 @ embs2.T
