"""Validated single-position off-policy estimates; all probabilities are marginal."""

import numpy as np


def validate(reward, action, pscore, pi, q):
    r = np.asarray(reward, float)
    a = np.asarray(action)
    p = np.asarray(pscore, float)
    pi = np.asarray(pi, float)
    q = np.asarray(q, float)
    n = len(r)
    if (
        n < 2
        or a.shape != (n,)
        or p.shape != (n,)
        or pi.ndim != 2
        or q.shape != pi.shape
        or len(pi) != n
    ):
        raise ValueError("shape")
    if (
        not np.issubdtype(a.dtype, np.integer)
        or np.any(a < 0)
        or np.any(a >= pi.shape[1])
    ):
        raise ValueError("actions")
    if not all(np.isfinite(x).all() for x in (r, p, pi, q)):
        raise ValueError("nonfinite")
    if np.any((r != 0) & (r != 1)) or np.any(p <= 0) or np.any(p > 1):
        raise ValueError("reward/propensity")
    if (
        np.any(pi < 0)
        or not np.allclose(pi.sum(1), 1, atol=1e-10)
        or np.any(q < 0)
        or np.any(q > 1)
    ):
        raise ValueError("distribution/reward model")
    return r, a, p, pi, q


def scores(reward, action, pscore, pi, q, clip=None):
    r, a, p, pi, q = validate(reward, action, pscore, pi, q)
    w = pi[np.arange(len(r)), a] / p
    if clip is not None:
        if not np.isfinite(clip) or clip <= 0:
            raise ValueError("clip")
        w = np.minimum(w, clip)
    dm = np.sum(pi * q, axis=1)
    ips = w * r
    dr = dm + w * (r - q[np.arange(len(r)), a])
    if w.sum() == 0:
        raise ValueError("no empirical overlap")
    sn = ips.mean() / w.mean()
    return {"DM": dm, "IPS": ips, "DR": dr, "SNIPS": sn + (ips - sn * w) / w.mean()}, w


def summary(reward, action, pscore, pi, q, clip=None):
    s, w = scores(reward, action, pscore, pi, q, clip)
    estimates = {
        k: {
            "value": float(v.mean()),
            "se_iid": float(v.std(ddof=1) / np.sqrt(len(v))),
            "ci95_iid": [
                float(v.mean() - 1.96 * v.std(ddof=1) / np.sqrt(len(v))),
                float(v.mean() + 1.96 * v.std(ddof=1) / np.sqrt(len(v))),
            ],
        }
        for k, v in s.items()
    }
    return {
        "n": len(w),
        "clicks": int(np.sum(reward)),
        "estimates": estimates,
        "ess": float(w.sum() ** 2 / np.sum(w * w)),
        "ess_fraction": float(w.sum() ** 2 / np.sum(w * w) / len(w)),
        "weight_mean": float(w.mean()),
        "weight_max": float(w.max()),
        "top_1pct_weight_share": float(
            np.sort(w)[-max(1, int(np.ceil(len(w) * 0.01))) :].sum() / w.sum()
        ),
        "clip": clip,
    }


def paired_difference(candidate, baseline, blocks=None, seed=2026):
    d = np.asarray(candidate) - np.asarray(baseline)
    se = d.std(ddof=1) / np.sqrt(len(d))
    out = {
        "value": float(d.mean()),
        "ci95_iid": [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)],
    }
    if blocks is not None:
        _, ix = np.unique(blocks, return_inverse=True)
        counts = np.bincount(ix)
        sums = np.bincount(ix, weights=d)
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(counts), (1000, len(counts)))
        boot = sums[draws].sum(1) / counts[draws].sum(1)
        out.update(
            hour_blocks=len(counts),
            ci95_hour_sensitivity=np.quantile(boot, [0.025, 0.975]).tolist(),
        )
    return out


def gate(result, days, paired, reference_ok):
    reasons = []
    if not reference_ok:
        reasons.append("Reference validation discrepancy")
    if days < 14:
        reasons.append("Fewer than 14 evaluation days; limited temporal evidence")
    if result["ess_fraction"] < 0.1:
        reasons.append("Effective sample size below 10%")
    if result["clicks"] < 100:
        reasons.append("Fewer than 100 observed clicks")
    if paired["ci95_iid"][0] <= 0:
        reasons.append("Paired DR lower bound does not exceed zero")
    vals = [v["value"] for v in result["estimates"].values()]
    if max(vals) - min(vals) > 0.002:
        reasons.append("Estimator disagreement exceeds 0.2 percentage points")
    return {
        "status": "HOLD" if reasons else "RESEARCH PASS",
        "reasons": reasons,
        "operational_authorization": False,
    }
