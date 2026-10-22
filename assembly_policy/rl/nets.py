"""Networks (after Fig. 3 of the AlphaNIST paper): separate SimBa actor and critic with a point-cloud encoder.

  points (3 n_pts) -> Linear 256 -> ELU -> Linear 128 -> ELU          (point encoder)
  proprio (150)    -> running normalization (RN)
  concat (278) -> Linear 256 -> 2 x [x + Linear(Swish(Linear(LN(x), 1024)), 256)] -> LN -> head
Actor head: Linear 8 (Gaussian mean, state-independent log-std); critic head: Linear 1.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RunningNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, clip=10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.eps, self.clip = eps, clip

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, x.shape[-1]).float()
        n = torch.tensor(float(x.shape[0]), device=x.device)
        s1, s2 = x.sum(0), (x * x).sum(0)
        if float(n) == 0:   # no live environment this step (all workers lost): keep the statistics
            return
        bm = s1 / n
        bv = (s2 / n - bm * bm).clamp_min(0)
        tot = self.count + n
        delta = bm - self.mean
        self.mean += delta * n / tot
        self.var = (self.var * self.count + bv * n + delta * delta * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + self.eps)).clamp(-self.clip, self.clip)


class ResBlock(nn.Module):
    def __init__(self, dim=256, expansion=1024):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, expansion)
        self.fc2 = nn.Linear(expansion, dim)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.ln(x))))


class SimBa(nn.Module):
    def __init__(self, n_pts, proprio_dim, out_dim, hidden=256, expansion=1024, blocks=2, enc=(256, 128)):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(3 * n_pts, enc[0]), nn.ELU(), nn.Linear(enc[0], enc[1]), nn.ELU())
        self.inp = nn.Linear(enc[1] + proprio_dim, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden, expansion) for _ in range(blocks)])
        self.ln = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, pts, prop):
        return self.head(self.ln(self.blocks(self.inp(torch.cat([self.enc(pts), prop], -1)))))


class ActorCritic(nn.Module):
    def __init__(self, n_pts, proprio_dim=150, act_dim=8, init_std=1.0, max_std=2.0, min_std=1e-3):
        super().__init__()
        self.log_std_max = float(torch.log(torch.tensor(max_std)))
        self.log_std_min = float(torch.log(torch.tensor(min_std)))
        self.norm = RunningNorm(proprio_dim)
        self.actor = SimBa(n_pts, proprio_dim, act_dim)
        self.critic = SimBa(n_pts, proprio_dim, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(torch.log(torch.tensor(init_std)))))
        nn.init.orthogonal_(self.actor.head.weight, 0.01); nn.init.zeros_(self.actor.head.bias)
        nn.init.orthogonal_(self.critic.head.weight, 1.0); nn.init.zeros_(self.critic.head.bias)

    def std(self):
        return self.log_std.clamp(self.log_std_min, self.log_std_max).exp()

    @torch.no_grad()
    def project_(self):
        """Keep log_std inside its bounds in place (the clamp in std() alone has zero gradient outside the bounds,
        so a parameter that once stepped past a bound would freeze there)."""
        self.log_std.clamp_(self.log_std_min, self.log_std_max)

    def dist(self, pts, prop):
        mean = self.actor(pts, self.norm(prop))
        return torch.distributions.Normal(mean, self.std().expand_as(mean))

    def value(self, pts, prop):
        return self.critic(pts, self.norm(prop)).squeeze(-1)
