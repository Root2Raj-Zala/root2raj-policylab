"""Transparent empirical-Bayes reward model and supported stochastic policies."""

import numpy as np


class RewardModel:
    def __init__(self, k=34, strength=100):
        self.k = k
        self.strength = strength

    def fit(self, frame):
        if frame.empty:
            raise ValueError("empty train")
        self.cutoff = str(frame.event_time.max())
        self.global_mean = float(frame.click.mean())
        g = (
            frame.groupby("item_id")
            .click.agg(["sum", "count"])
            .reindex(range(self.k), fill_value=0)
        )
        self.action = (g["sum"].to_numpy() + self.strength * self.global_mean) / (
            g["count"].to_numpy() + self.strength
        )
        c = frame.groupby(["user_feature_0", "item_id"]).click.agg(["sum", "count"])
        self.context = {}
        for key, group in c.groupby(level=0):
            z = group.droplevel(0).reindex(range(self.k), fill_value=0)
            self.context[str(key)] = (
                z["sum"].to_numpy() + self.strength * self.action
            ) / (z["count"].to_numpy() + self.strength)
        return self

    def predict(self, frame, contextual=True):
        if not contextual:
            return np.tile(self.action, (len(frame), 1))
        return np.stack(
            [self.context.get(str(x), self.action) for x in frame.user_feature_0]
        )

    def artifact(self):
        return {
            "cutoff": self.cutoff,
            "global_mean": self.global_mean,
            "strength": self.strength,
            "action_rates": self.action.tolist(),
            "context_rates": {k: v.tolist() for k, v in self.context.items()},
            "features": ["item_id", "user_feature_0"],
        }


def policy(model, frame, name):
    if name == "uniform":
        return np.ones((len(frame), model.k)) / model.k
    kind, _, eps = name.split("_")
    eps = float(eps)
    q = model.predict(frame, kind == "contextual")
    p = np.full_like(q, eps / model.k)
    p[np.arange(len(frame)), np.argmax(q, axis=1)] += 1 - eps
    return p


def production_ts(prior, n_sim=1000000, seed=12345):
    # Vectorization preserves RandomState draw order of the official select_action loop.
    rng = np.random.RandomState(seed)
    a = np.asarray(prior["alpha"])
    b = np.asarray(prior["beta"])
    counts = np.zeros((len(a), 3), int)
    for start in range(0, n_sim, 10000):
        draws = rng.beta(a, b, size=(min(10000, n_sim - start), len(a)))
        ranks = np.argsort(draws, axis=1)[:, ::-1][:, :3]
        for pos in range(3):
            counts[:, pos] += np.bincount(ranks[:, pos], minlength=len(a))
    return counts / n_sim
