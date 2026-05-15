"""Extract per-position LSTM cell and hidden states.

nn.LSTM does not expose c_t at every step, so we copy its weights into an
nn.LSTMCell and unroll manually. This is slow (Python loop over T_tokens) but
only used for analysis on a few hundred sequences.
"""
from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def extract_lstm_states(lstm_model, tokens, device):
    """Run the FFLMLSTM over `tokens` and return per-position (h, c).

    Args:
        lstm_model: an FFLMLSTM (single-layer LSTM, batch_first=True).
        tokens: LongTensor of shape (B, L).
        device: torch device.

    Returns:
        h_states: (B, L, hidden_size) — output of LSTM cell at each step
        c_states: (B, L, hidden_size) — cell state at each step
    """
    assert lstm_model.lstm.num_layers == 1, "single-layer LSTM only"
    tokens = tokens.to(device)
    embed = lstm_model.embed(tokens)  # (B, L, E)
    B, L, _ = embed.shape
    hidden = lstm_model.lstm.hidden_size

    cell = nn.LSTMCell(embed.size(-1), hidden).to(device)
    cell.weight_ih.data.copy_(lstm_model.lstm.weight_ih_l0.data)
    cell.weight_hh.data.copy_(lstm_model.lstm.weight_hh_l0.data)
    cell.bias_ih.data.copy_(lstm_model.lstm.bias_ih_l0.data)
    cell.bias_hh.data.copy_(lstm_model.lstm.bias_hh_l0.data)
    cell.eval()

    h = torch.zeros(B, hidden, device=device)
    c = torch.zeros(B, hidden, device=device)
    h_list, c_list = [], []
    for t in range(L):
        h, c = cell(embed[:, t, :], (h, c))
        h_list.append(h)
        c_list.append(c)
    return torch.stack(h_list, dim=1), torch.stack(c_list, dim=1)
