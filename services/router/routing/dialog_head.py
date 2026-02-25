from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

ROLE_UNKNOWN = 0
ROLE_CALLER = 1
ROLE_AGENT = 2
ROLE_SYSTEM = 3
ROLE_VOCAB_SIZE = 4


@dataclass
class DialogTurn:
    role_id: int
    text: str


class DialogTransformerHead(nn.Module):
    def __init__(
        self,
        *,
        in_features: int,
        num_classes: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_turns: int = 64,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.max_turns = int(max_turns)

        self.input_proj = nn.Linear(self.in_features, self.d_model)
        self.role_emb = nn.Embedding(ROLE_VOCAB_SIZE, self.d_model)
        self.pos_emb = nn.Embedding(self.max_turns, self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.norm = nn.LayerNorm(self.d_model)
        self.classifier = nn.Linear(self.d_model, self.num_classes)

    def forward(
        self,
        turn_embeddings: torch.Tensor,
        role_ids: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = turn_embeddings.shape
        seq_len = min(seq_len, self.max_turns)
        x = turn_embeddings[:, :seq_len, :]
        roles = role_ids[:, :seq_len]
        mask = key_padding_mask[:, :seq_len]

        x = self.input_proj(x)
        pos_idx = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
        x = x + self.role_emb(roles) + self.pos_emb(pos_idx)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.norm(x)

        valid = (~mask).unsqueeze(-1).to(x.dtype)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.classifier(pooled)
