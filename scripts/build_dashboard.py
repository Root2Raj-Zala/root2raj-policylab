from pathlib import Path
import json, sys
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from policylab.ope import paired_difference, gate

r = Path(__file__).resolve().parents[1]
out = r / "artifacts"
m = json.loads((out / "metrics.json").read_text())
a = pd.read_csv(out / "test-audit.csv.gz")
scenarios = []
for day in ["all"] + m["days"][5:]:
    z = a if day == "all" else a[a.day == day]
    for cap in [1, 2, 5, 10, 34]:
        w = np.minimum(z.weight.to_numpy(), cap)
        dm = z.dm.to_numpy()
        ips = w * z.click.to_numpy()
        dr = dm + w * (z.click.to_numpy() - z.q_logged.to_numpy())
        sn = ips.mean() / w.mean()
        s = {"DM": dm, "IPS": ips, "SNIPS": sn + (ips - sn * w) / w.mean(), "DR": dr}
        result = {
            "n": len(z),
            "clicks": int(z.click.sum()),
            "ess": float(w.sum() ** 2 / (w * w).sum()),
            "ess_fraction": float(w.sum() ** 2 / (w * w).sum() / len(z)),
            "weight_max": float(w.max()),
            "weight_mean": float(w.mean()),
            "estimates": {
                k: {
                    "value": float(v.mean()),
                    "ci95_iid": [
                        float(v.mean() - 1.96 * v.std(ddof=1) / np.sqrt(len(v))),
                        float(v.mean() + 1.96 * v.std(ddof=1) / np.sqrt(len(v))),
                    ],
                }
                for k, v in s.items()
            },
        }
        pair = paired_difference(dr, z.baseline_dr.to_numpy())
        result.update(
            scope=day,
            cap=cap,
            paired=pair,
            review=gate(result, 2 if day == "all" else 1, pair, False),
        )
        scenarios.append(result)
examples = (
    pd.concat([a[a.click == 1].head(8), a[a.click == 0].head(8)])
    .drop_duplicates(["source_sha", "source_row"])
    .to_dict(orient="records")
)
dashboard = {
    "version": "1.0.0",
    "source_rows": m["rows_all_positions"],
    "selected": m["selected"],
    "test_days": m["days"][5:],
    "scenarios": scenarios,
    "examples": examples,
    "reference_review": json.loads((out / "reference-review.json").read_text()),
    "source_manifest": json.loads((out / "source-manifest.json").read_text()),
    "protocol_sha256": m["protocol_sha256"],
    "scope": "Historical position-one Men campaign, 2019; conditional inference, no slate claim",
}
(out / "dashboard.json").write_text(
    json.dumps(dashboard, indent=2, allow_nan=False), encoding="utf-8"
)
print("dashboard", len(scenarios), "scenarios", len(examples), "real-row examples")
