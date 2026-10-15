"""Explicitly synthetic stress study: known oracle, no real-data claims."""

from pathlib import Path
import numpy as np, pandas as pd
from policylab.ope import summary


def run(out, repetitions=250, n=4000):
    rng = np.random.default_rng(44)
    rows = []
    mu = np.array([0.01, 0.04, 0.10])
    pi0 = np.array([0.1, 0.2, 0.7])
    truth = float(pi0 @ mu)
    for behavior in [np.array([1 / 3] * 3), np.array([0.90, 0.09, 0.01])]:
        for rep in range(repetitions):
            a = rng.choice(3, n, p=behavior)
            r = (rng.random(n) < mu[a]).astype(int)
            pi = np.tile(pi0, (n, 1))
            q = np.ones((n, 3)) * 0.03
            s = summary(r, a, behavior[a], pi, q)
            for method, v in s["estimates"].items():
                rows.append(
                    {
                        "scenario": "balanced" if behavior[0] < 0.5 else "weak_overlap",
                        "rep": rep,
                        "method": method,
                        "oracle": truth,
                        "value": v["value"],
                        "covered_iid": v["ci95_iid"][0] <= truth <= v["ci95_iid"][1],
                        "ess": s["ess"],
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(Path(out) / "simulation-replicates.csv", index=False)
    df["error"] = df.value - df.oracle
    df["squared_error"] = df.error**2
    report = (
        df.groupby(["scenario", "method"])
        .agg(
            bias=("error", "mean"),
            mse=("squared_error", "mean"),
            coverage_iid=("covered_iid", "mean"),
            mean_ess=("ess", "mean"),
        )
        .reset_index()
    )
    report["rmse"] = np.sqrt(report.mse)
    report.to_csv(Path(out) / "simulation-summary.csv", index=False)
    return report
