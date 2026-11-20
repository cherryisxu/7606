"""RMSNorm decoder-only GPT for the optional extension.

Same GPT family as src/model.py (depth, width, heads, tied embeddings, causal
attention, GELU MLP), except every LayerNorm is replaced by RMSNorm
(Zhang & Sennrich, 2019). Core checkpoints stay on LayerNorm.
"""
import math
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import CausalSelfAttention, GPT, GPTConfig, MLP  # noqa: E402


class RMSNorm(nn.Module):
    """Root-mean-square normalization. Scale only; no mean subtraction or bias."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Variance in fp32 so bf16 training does not underflow the rsqrt.
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_hat = x.float() * torch.rsqrt(variance + self.eps)
        return (x_hat * self.weight.float()).type_as(x)


class RMSBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class RMSNormGPT(GPT):
    """Baseline GPT with pre-norm residuals, using RMSNorm instead of LayerNorm."""

    def __init__(self, config):
        super().__init__(config)
        self.transformer.h = nn.ModuleList([RMSBlock(config) for _ in range(config.n_layer)])
        self.transformer.ln_f = RMSNorm(config.n_embd)
        self.transformer.h.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight") and "transformer.h." in name:
                nn.init.normal_(parameter, std=0.02 / math.sqrt(2 * config.n_layer))


def build_model(model_cfg):
    config = GPTConfig(**model_cfg)
    model = RMSNormGPT(config)
    if model.parameter_count() > 20_000_000:
        raise ValueError("model exceeds 20M parameters")
    return model


def load_rmsnorm_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "rmsnorm":
        raise ValueError("checkpoint is not an RMSNorm extension model")
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint
