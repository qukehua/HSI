import math
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    lengths = lengths.to(dtype=torch.long).clamp(min=1, max=max_len)
    frame_ids = torch.arange(max_len, device=lengths.device).reshape(1, max_len)
    return frame_ids < lengths.reshape(-1, 1)


def canonicalize_motion(
    motion: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
    root_joint: int = 0,
) -> torch.Tensor:
    """Shift every sequence by its first root joint and append root velocity."""
    if motion.ndim != 3:
        raise ValueError(f"Expected motion shape [B, T, D], got {tuple(motion.shape)}.")
    if motion.shape[-1] % 3 != 0:
        raise ValueError(f"Motion dim must be divisible by 3, got {motion.shape[-1]}.")

    batch_size, seq_len, dim = motion.shape
    joint_count = dim // 3
    if root_joint < 0 or root_joint >= joint_count:
        raise ValueError(f"root_joint={root_joint} is outside motion joint count {joint_count}.")

    points = motion.reshape(batch_size, seq_len, joint_count, 3)
    root0 = points[:, :1, root_joint:root_joint + 1].detach()
    shifted = points - root0
    root = shifted[:, :, root_joint]
    root_velocity = torch.zeros_like(root)
    root_velocity[:, 1:] = root[:, 1:] - root[:, :-1]
    features = torch.cat([shifted.reshape(batch_size, seq_len, dim), root_velocity], dim=-1)

    if lengths is not None:
        mask = lengths_to_mask(lengths.to(motion.device), seq_len).unsqueeze(-1)
        features = features.masked_fill(~mask, 0.0)
    return features


def masked_mean(sequence: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
    if lengths is None:
        return sequence.mean(dim=1)
    mask = lengths_to_mask(lengths.to(sequence.device), sequence.shape[1]).to(sequence.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (sequence * mask).sum(dim=1) / denom


class TextConditionEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(
        self,
        text: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        if text.ndim == 2:
            pooled = text
        elif text.ndim == 3:
            if text.shape[1] == 1:
                pooled = text[:, 0]
            else:
                pooled = masked_mean(text, lengths)
        else:
            raise ValueError(f"Expected text shape [B, D] or [B, T, D], got {tuple(text.shape)}.")
        embedding = self.net(pooled)
        return F.normalize(embedding, dim=-1) if normalize else embedding


class MotionSequenceEncoder(nn.Module):
    def __init__(
        self,
        motion_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        root_joint: int = 0,
    ):
        super().__init__()
        self.motion_dim = int(motion_dim)
        self.root_joint = int(root_joint)
        self.input = nn.Sequential(
            nn.Linear(self.motion_dim + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=gru_dropout,
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(
        self,
        motion: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        if motion.shape[-1] != self.motion_dim:
            raise ValueError(f"Expected motion dim {self.motion_dim}, got {motion.shape[-1]}.")
        features = canonicalize_motion(motion, lengths=lengths, root_joint=self.root_joint)
        features = self.input(features)
        if lengths is not None:
            lengths_cpu = lengths.detach().to("cpu", dtype=torch.long).clamp(min=1, max=motion.shape[1])
            packed = pack_padded_sequence(features, lengths_cpu, batch_first=True, enforce_sorted=False)
            _, hidden = self.gru(packed)
        else:
            _, hidden = self.gru(features)
        hidden = hidden.reshape(self.gru.num_layers, 2, motion.shape[0], self.gru.hidden_size)[-1]
        pooled = torch.cat([hidden[0], hidden[1]], dim=-1)
        embedding = self.output(pooled)
        return F.normalize(embedding, dim=-1) if normalize else embedding


class MotionEvaluator(nn.Module):
    def __init__(
        self,
        motion_dim: int,
        text_dim: int,
        hidden_dim: int = 512,
        embedding_dim: int = 512,
        text_hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        dropout: float = 0.1,
        root_joint: int = 0,
    ):
        super().__init__()
        text_hidden_dim = int(text_hidden_dim or hidden_dim)
        self.config = {
            "motion_dim": int(motion_dim),
            "text_dim": int(text_dim),
            "hidden_dim": int(hidden_dim),
            "embedding_dim": int(embedding_dim),
            "text_hidden_dim": int(text_hidden_dim),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "root_joint": int(root_joint),
        }
        self.motion_encoder = MotionSequenceEncoder(
            motion_dim=motion_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
            root_joint=root_joint,
        )
        self.text_encoder = TextConditionEncoder(
            input_dim=text_dim,
            hidden_dim=text_hidden_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07), dtype=torch.float32))

    def encode_motion(
        self,
        motion: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        return self.motion_encoder(motion, lengths, normalize=normalize)

    def encode_text(
        self,
        text: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        return self.text_encoder(text, lengths, normalize=normalize)

    def forward(
        self,
        motion: torch.Tensor,
        text: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        motion_embedding = self.encode_motion(motion, lengths)
        text_embedding = self.encode_text(text, lengths)
        return {
            "motion_embedding": motion_embedding,
            "text_embedding": text_embedding,
        }

    def contrastive_loss(
        self,
        motion: torch.Tensor,
        text: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.forward(motion, text, lengths)
        motion_embedding = outputs["motion_embedding"]
        text_embedding = outputs["text_embedding"]
        logits = self.logit_scale.exp().clamp(max=100.0) * text_embedding @ motion_embedding.t()
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_t2m = F.cross_entropy(logits, labels)
        loss_m2t = F.cross_entropy(logits.t(), labels)
        loss = 0.5 * (loss_t2m + loss_m2t)
        with torch.no_grad():
            top1_t2m = (logits.argmax(dim=1) == labels).float().mean()
            top1_m2t = (logits.argmax(dim=0) == labels).float().mean()
            pos_sim = (text_embedding * motion_embedding).sum(dim=-1).mean()
        return {
            "loss": loss,
            "loss_t2m": loss_t2m.detach(),
            "loss_m2t": loss_m2t.detach(),
            "top1_t2m": top1_t2m,
            "top1_m2t": top1_m2t,
            "pos_sim": pos_sim,
        }


def build_motion_evaluator_from_checkpoint(checkpoint: Dict, map_location=None) -> MotionEvaluator:
    config = dict(checkpoint["config"])
    model = MotionEvaluator(**config)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict)
    if map_location is not None:
        model.to(map_location)
    model.eval()
    return model
