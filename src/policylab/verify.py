"""Independent frozen audit reconciliation and artifact integrity checks."""

import json
from pathlib import Path
import numpy as np, pandas as pd
from policylab.warehouse import sha256


def verify(root):
    root = Path(root)
    out = root / "artifacts"
    m = json.loads((out / "metrics.json").read_text())
    a = pd.read_csv(out / "test-audit.csv.gz")
    checks = {}
    checks["actual_test_population"] = len(a) == m["test_rows"]
    checks["unique_source_row_identity"] = not a.duplicated(
        ["source_sha", "source_row"]
    ).any()
    checks["raw_hashes"] = (
        all(
            sha256(root / "data/raw" / x["policy"] / x["file"]) == x["source_sha256"]
            for x in json.loads((out / "source-manifest.json").read_text())["files"]
        )
        if (root / "data/raw/random/men.csv").exists()
        else None
    )
    checks["frozen_protocol"] = sha256(root / "protocol.json") == m["protocol_sha256"]
    checks["frozen_selection"] = sha256(out / "selection.json") == m["selection_sha256"]
    checks["strict_training_cutoff"] = bool(
        (
            pd.to_datetime(a.model_cutoff, utc=True)
            < pd.to_datetime(a.timestamp, utc=True)
        ).all()
    )
    checks["independent_ips"] = bool(
        np.isclose(
            (a.click * a.target_probability / a.propensity_score).mean(),
            m["test"]["estimates"]["IPS"]["value"],
        )
    )
    checks["independent_dr"] = bool(
        np.isclose(
            (a.dm + a.weight * (a.click - a.q_logged)).mean(),
            m["test"]["estimates"]["DR"]["value"],
        )
    )
    checks["independent_snips"] = bool(
        np.isclose(
            (a.weight * a.click).sum() / a.weight.sum(),
            m["test"]["estimates"]["SNIPS"]["value"],
        )
    )
    checks["paired_dr"] = bool(
        np.isclose(
            (a.dr - a.baseline_dr).mean(), m["test"]["paired_dr_vs_uniform"]["value"]
        )
    )
    checks["independent_ess"] = bool(
        np.isclose(a.weight.sum() ** 2 / (a.weight**2).sum(), m["test"]["ess"])
    )
    checks["gate_not_overridden"] = (
        m["review"]["status"] == "HOLD" and not m["review"]["operational_authorization"]
    )
    checks["seven_real_days"] = len(m["days"]) == 7
    checks["exploration_floor"] = bool(
        (a.target_probability >= float(m["selected"].split("_")[-1]) / 34 - 1e-12).all()
    )
    if not all(v is None or v for v in checks.values()):
        raise AssertionError(checks)
    return checks


if __name__ == "__main__":
    r = Path(__file__).resolve().parents[2]
    v = verify(r)
    (r / "artifacts/validation.json").write_text(
        json.dumps(v, indent=2), encoding="utf-8"
    )
    print(json.dumps(v, indent=2))
