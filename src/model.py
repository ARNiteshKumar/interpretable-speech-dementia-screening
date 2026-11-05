"""Model: gated cross-attention fusion -> discourse encoder -> concept bottleneck.

The decision is forced through `n_concepts` clinically-meaningful markers and a
LINEAR head, so each concept's contribution to the logit is exactly w_k * c_k and
the contributions sum to the decision. This gives an exact additive scorecard
(ante-hoc interpretability) instead of a post-hoc approximation.
"""
import torch
import torch.nn as nn


class CrossAttnFusion(nn.Module):
    """Bidirectional cross-attention between linguistic (L) and acoustic (A) tokens,
    followed by a learned per-token gate:  H = g*L + (1-g)*A."""

    def __init__(self, d_text, d_aud, d=128, heads=4, dropout=0.3, maxlen=40):
        super().__init__()
        self.ln_l_in = nn.LayerNorm(d_text)
        self.ln_a_in = nn.LayerNorm(d_aud)
        self.pl = nn.Linear(d_text, d)
        self.pa = nn.Linear(d_aud, d)
        self.pos = nn.Embedding(maxlen, d)
        self.l2a = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.a2l = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln_l = nn.LayerNorm(d)
        self.ln_a = nn.LayerNorm(d)
        self.gate = nn.Linear(2 * d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, L, A, mask):
        T = L.size(1)
        pos = self.pos(torch.arange(T, device=L.device))[None]
        L = self.pl(self.ln_l_in(L)) + pos
        A = self.pa(self.ln_a_in(A)) + pos
        kpm = ~mask
        Lc, _ = self.l2a(L, A, A, key_padding_mask=kpm)   # text attends to audio
        Ac, _ = self.a2l(A, L, L, key_padding_mask=kpm)   # audio attends to text
        L = self.ln_l(L + Lc)
        A = self.ln_a(A + Ac)
        g = torch.sigmoid(self.gate(torch.cat([L, A], -1)))
        return self.drop(torch.relu(g * L + (1 - g) * A))


class DiscourseEncoder(nn.Module):
    """BiGRU + attention pooling over utterances (pauses, hesitation, topic drift)."""

    def __init__(self, d, hid=64, out=128, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(d, hid, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * hid, 1)
        self.proj = nn.Sequential(nn.Linear(2 * hid, out), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, x, lens):
        p = nn.utils.rnn.pack_padded_sequence(
            x, lens.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        o, _ = self.gru(p)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=x.size(1))
        m = torch.arange(o.size(1), device=x.device)[None, :] < lens[:, None].clamp(min=1)
        s = self.attn(o).squeeze(-1).masked_fill(~m, float("-inf"))
        w = torch.softmax(s, 1).unsqueeze(-1)
        return self.proj((o * w).sum(1))


class Net(nn.Module):
    """Full model. `bottleneck=False` gives the black-box control used to measure
    the *interpretability cost* (an MLP head with no concept layer)."""

    def __init__(self, text_dim, aud_dim, n_concepts, d=128, heads=4,
                 dropout=0.3, bottleneck=True, maxlen=40):
        super().__init__()
        self.bottleneck = bottleneck
        self.fusion = CrossAttnFusion(text_dim, aud_dim, d, heads, dropout, maxlen)
        self.disc = DiscourseEncoder(d, 64, 128, dropout)
        if bottleneck:
            self.to_concepts = nn.Linear(128, n_concepts)
            self.head = nn.Linear(n_concepts, 1)      # linear => exact additive scorecard
        else:
            self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(),
                                      nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, L, A, lens):
        mask = torch.arange(L.size(1), device=L.device)[None, :] < lens[:, None].clamp(min=1)
        H = self.fusion(L, A, mask)
        D = self.disc(H, lens)
        if self.bottleneck:
            c = self.to_concepts(D)
            return self.head(c).squeeze(-1), c
        return self.head(D).squeeze(-1), None

    @torch.no_grad()
    def scorecard(self, L, A, lens):
        """Per-sample explanation: contribution_k = w_k * c_k (these sum to the logit)."""
        assert self.bottleneck, "scorecard requires the concept bottleneck"
        logit, c = self.forward(L, A, lens)
        w = self.head.weight.detach().squeeze(0)
        return (c * w), logit                       # (B, n_concepts), (B,)
