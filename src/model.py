"""Small GPT, adapted from nanoGPT's MIT-licensed architecture (see LICENSE).

Student starter: complete the four TODO methods. See README.md.
"""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    attention_backend: str = "manual"

    def __post_init__(self):
        for name in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.attention_backend != "manual":
            raise ValueError("unknown attention backend")


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer("mask", torch.ones(config.block_size, config.block_size,
                                               dtype=torch.bool).tril(), persistent=False)

    def forward(self, x):
        """TODO 1: accept [B,T,C]; project Q/K/V; split heads; scale dot products; mask future keys; softmax over keys; attention dropout; merge heads; output projection and residual dropout. Use the provided layers/mask. Return [B,T,C]. Implement explicit attention, not SDPA or MultiheadAttention."""
        B, T, C = x.size()
        n_head = self.config.n_head
        head_dim = C // n_head

        # project to q, k, v
        qkv = self.c_attn(x)  # [B, T, 3C]
        q, k, v = qkv.split(C, dim=2)

        # split heads: [B, T, C] -> [B, n_head, T, head_dim]
        q = q.view(B, T, n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, head_dim).transpose(1, 2)

        # scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))  # [B, nh, T, T]
        # causal mask (True = keep, False = mask)
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # [B, nh, T, head_dim]
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # merge heads

        # output projection + residual dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """TODO 2: apply the provided expansion layer, GELU, projection layer and dropout in that order; preserve [B,T,C]."""
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        """TODO 3: implement a pre-normalized attention residual followed by a pre-normalized MLP residual, using the updated residual stream for the second normalization."""
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        if idx.ndim != 2 or not 0 < idx.shape[1] <= self.config.block_size:
            raise ValueError("input must be [batch, sequence] within block_size")
        pos = torch.arange(idx.shape[1], device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 eos_id=None, generator=None):
        """TODO 4: validate nonempty prompts, nonnegative length, finite positive temperature and positive optional top_k. Temporarily use eval mode and restore the prior mode even on failure. Autoregressively crop context, take last-position logits, apply temperature and optional top-k, sample with the supplied generator and append. Support optional per-row EOS: finished rows append EOS while other rows continue; stop when all finish. Return original prompt plus generated IDs. No gradients."""
        if idx.ndim != 2 or idx.shape[0] == 0 or idx.shape[1] == 0:
            raise ValueError("prompts must be a nonempty [batch, sequence] tensor")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        if not (isinstance(temperature, (int, float)) and math.isfinite(temperature) and temperature > 0):
            raise ValueError("temperature must be a finite positive number")
        if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
            raise ValueError("top_k must be a positive integer when provided")

        was_training = self.training
        self.eval()
        try:
            B = idx.shape[0]
            # track which sequences have already emitted EOS
            finished = torch.zeros(B, dtype=torch.bool, device=idx.device)
            for _ in range(max_new_tokens):
                if finished.all():
                    break
                # crop context if longer than block_size
                idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature  # [B, vocab]

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1, generator=generator)  # [B, 1]

                if eos_id is not None:
                    # already-finished rows keep appending EOS
                    next_token = torch.where(
                        finished.unsqueeze(1),
                        torch.full_like(next_token, eos_id),
                        next_token,
                    )
                    finished = finished | (next_token.squeeze(1) == eos_id)

                idx = torch.cat([idx, next_token], dim=1)
            return idx
        finally:
            self.train(was_training)
