"""Models for FFLM: GPT-2-style Transformer and a 1-layer LSTM baseline.

Matches the canonical baseline in Liu et al. 2023 Section 4
(6 layers, d=512, 8 heads -> ~19M params, dropout 0).
"""
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from .data import VOCAB_SIZE


class FFLMTransformer(nn.Module):
    def __init__(
        self,
        vocab_size=VOCAB_SIZE,
        n_positions=512,
        n_embd=512,
        n_layer=6,
        n_head=8,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    ):
        super().__init__()
        self.model = GPT2LMHeadModel(GPT2Config(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=resid_pdrop,
            embd_pdrop=embd_pdrop,
            attn_pdrop=attn_pdrop,
            use_cache=False,
        ))
        self.name = f"gpt2_L={n_layer}_d={n_embd}_H={n_head}"

    def forward(self, tokens):
        return self.model(input_ids=tokens).logits


class FFLMLSTM(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, hidden_size=128, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size)
        self.name = f"lstm_L={num_layers}_d={hidden_size}"

    def forward(self, tokens):
        h, _ = self.lstm(self.embed(tokens))
        return self.head(h)


def build_model(cfg):
    """Construct a model from a TrainConfig-like object."""
    if cfg.family == "gpt2":
        return FFLMTransformer(
            vocab_size=cfg.vocab_size,
            n_positions=cfg.n_positions,
            n_embd=cfg.n_embd,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            resid_pdrop=cfg.resid_pdrop,
            embd_pdrop=cfg.embd_pdrop,
            attn_pdrop=cfg.attn_pdrop,
        )
    if cfg.family == "lstm":
        return FFLMLSTM(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
        )
    raise ValueError(f"Unknown model family: {cfg.family!r}")
