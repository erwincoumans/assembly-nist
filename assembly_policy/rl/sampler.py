"""Reset-entry samplers over the pooled multi-task bank.

beta (Success Guided Sampling, SGS; AlphaNIST paper Eq. 4): q_i ∝ eps_q + sqrt(p_i (1 - p_i)), p_i = mean of the last L outcomes of entry i,
normalized over the whole bank; an entry that has not been visited carries only the eps_q weight. Defaults:
eps_q = 1e-4, L = 50.
uniform: q_i = 1/M (paper ablation).
"""
from __future__ import annotations

import numpy as np


class ResetSampler:
    def __init__(self, n_entries, mode="beta", eps=1e-4, history=50, seed=0):
        assert mode in ("beta", "uniform")
        self.M, self.mode, self.eps, self.L = n_entries, mode, eps, history
        self.buf = np.zeros((n_entries, history), dtype=np.uint8)
        self.ptr = np.zeros(n_entries, dtype=np.int64)
        self.cnt = np.zeros(n_entries, dtype=np.int64)
        self.sum = np.zeros(n_entries, dtype=np.int64)
        self.rng = np.random.default_rng(seed)
        self._cdf = None
        self._group_cdf = None

    def p_hat(self):
        return np.where(self.cnt > 0, self.sum / np.maximum(np.minimum(self.cnt, self.L), 1), 0.0)

    def weights(self):
        if self.mode == "uniform":
            return np.ones(self.M)
        p = self.p_hat()
        return self.eps + np.sqrt(p * (1 - p))

    def sample(self, k):
        if k == 0:
            return np.zeros(0, dtype=np.int64)
        if self.mode == "uniform":
            return self.rng.integers(0, self.M, k)
        if self._cdf is None:
            w = self.weights()
            self._cdf = np.cumsum(w) / w.sum()
        return np.minimum(np.searchsorted(self._cdf, self.rng.random(k)), self.M - 1)

    def sample_within(self, groups, group_of):
        """One entry per element of groups, each drawn from the entries of its group (group_of: the group of every
        entry), by the same weights: for simulators whose environments keep their task (sim/gpu)."""
        groups = np.asarray(groups)
        out = np.empty(len(groups), dtype=np.int64)
        if self._group_cdf is None or self._group_cdf[0] is not group_of:
            self._group_cdf = (group_of, {})
        cache = self._group_cdf[1]
        for g in np.unique(groups):
            sel = np.nonzero(groups == g)[0]
            if g not in cache:
                idx = np.nonzero(group_of == g)[0]
                w = np.ones(len(idx)) if self.mode == "uniform" else self.weights()[idx]
                cache[g] = (idx, np.cumsum(w) / w.sum())
            idx, cdf = cache[g]
            out[sel] = idx[np.minimum(np.searchsorted(cdf, self.rng.random(len(sel))), len(idx) - 1)]
        return out

    def update(self, entries, outcomes):
        """Record episode outcomes (bool) for the given entries (may repeat)."""
        for e, y in zip(np.asarray(entries), np.asarray(outcomes, dtype=np.uint8)):
            old = self.buf[e, self.ptr[e]] if self.cnt[e] >= self.L else 0
            self.buf[e, self.ptr[e]] = y
            self.sum[e] += int(y) - int(old)
            self.ptr[e] = (self.ptr[e] + 1) % self.L
            self.cnt[e] += 1
        self._cdf = None
        self._group_cdf = None

    def state_dict(self):
        return dict(buf=self.buf, ptr=self.ptr, cnt=self.cnt, sum=self.sum, rng=self.rng.bit_generator.state)

    def load_state_dict(self, s):
        self.buf, self.ptr, self.cnt, self.sum = s["buf"], s["ptr"], s["cnt"], s["sum"]
        self.rng.bit_generator.state = s["rng"]
        self._cdf = None
        self._group_cdf = None

    def entropy_ratio(self):
        """exp(H(q)) / M: 1 for uniform, smaller when concentrated."""
        w = self.weights(); q = w / w.sum()
        return float(np.exp(-(q * np.log(q + 1e-30)).sum()) / self.M)
