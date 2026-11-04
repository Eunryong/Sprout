from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from .codec import MAX_LEN, VOCAB, result_logits


@dataclass
class ModelCfg:
    d_model: int = 64
    n_layers: int = 3
    n_heads: int = 4
    d_ff: int = 128
    dropout: float = 0.0
    seed: int = 0
    vocab_size: int = 0  # 0 = the game's vocabulary (codec.VOCAB)
    max_len: int = 0  # 0 = codec.MAX_LEN

    def to_dict(self):
        return asdict(self)


class TinyLM(nn.Module):
    """Decoder-only transformer, randomly initialised. Right padding is safe because attention is causal
    and we read the logits at each row's last real position."""

    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        V, L = cfg.vocab_size or len(VOCAB), cfg.max_len or MAX_LEN
        self.tok = nn.Embedding(V, cfg.d_model)
        self.pos = nn.Embedding(L, cfg.d_model)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(cfg.n_layers)
        )
        self.ln = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, V)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        T = ids.shape[1]
        x = self.tok(ids) + self.pos(torch.arange(T, device=ids.device))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=ids.device)
        for layer in self.layers:
            x = layer(x, src_mask=mask, is_causal=True)
        return self.head(self.ln(x))


def build_model(cfg: ModelCfg) -> TinyLM:
    torch.manual_seed(cfg.seed)
    return TinyLM(cfg)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
