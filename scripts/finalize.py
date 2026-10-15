from pathlib import Path
import sys, json, time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from policylab.verify import verify
from policylab.pipeline import save
from policylab.warehouse import Warehouse, sha256
from policylab.simulation import run as simulate
import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
out = root / "artifacts"
m = json.loads((out / "metrics.json").read_text())
save(out / "validation.json", verify(root))
w = Warehouse(root / "data/warehouse")
w.views()
before = w.reconcile()
start = time.perf_counter()
for policy in ["random", "bts"]:
    w.ingest(root / f"data/raw/{policy}/men.csv", policy)
w.views()
after = w.reconcile()
pd.testing.assert_frame_equal(before, after)
ledger = w.db.execute("SELECT * FROM shards ORDER BY policy,part").df()
assert all(sha256(p) == h for p, h in zip(ledger.path, ledger.parquet_sha))
assert ledger.rows.sum() == m["rows_all_positions"]
ledger.drop(columns="path").to_csv(out / "shard-ledger.csv", index=False)
save(
    out / "replay.json",
    {
        "status": "PASS",
        "real_rows": int(ledger.rows.sum()),
        "shards": len(ledger),
        "replayed_source_files": 2,
        "unchanged_rows_clicks_and_identities": True,
        "all_parquet_hashes_verified": True,
        "seconds": time.perf_counter() - start,
        "fault_injection": "separate six-row fixture, not synthetic production events",
    },
)
w.close()
print("full replay PASS", flush=True)
sim = simulate(out)
print("oracle simulations complete", flush=True)
plt.rcParams.update(
    {
        "figure.facecolor": "#091321",
        "axes.facecolor": "#091321",
        "text.color": "#e8f0ff",
        "axes.labelcolor": "#e8f0ff",
        "xtick.color": "#b9c8dd",
        "ytick.color": "#b9c8dd",
        "axes.edgecolor": "#557083",
        "font.size": 10,
        "svg.fonttype": "none",
    }
)
fig, ax = plt.subplots(figsize=(10, 5))
names = list(m["test"]["estimates"])
vals = [m["test"]["estimates"][k]["value"] * 100 for k in names]
err = (
    np.array(
        [
            [
                m["test"]["estimates"][k]["value"]
                - m["test"]["estimates"][k]["ci95_iid"][0]
                for k in names
            ],
            [
                m["test"]["estimates"][k]["ci95_iid"][1]
                - m["test"]["estimates"][k]["value"]
                for k in names
            ],
        ]
    )
    * 100
)
ax.errorbar(vals, range(4), xerr=err, fmt="o", color="#67e7e8", capsize=5)
ax.axvline(
    m["test"]["baseline"]["estimates"]["IPS"]["value"] * 100,
    color="#b796fa",
    ls="--",
    label="Observed uniform CTR",
)
ax.set_yticks(range(4), names)
ax.set_xlabel("Estimated click probability (%) · conditional iid 95% approximation")
ax.set_title(
    "PolicyLab | Frozen final policy · HOLD", loc="left", color="#e8f0ff", pad=20
)
ax.legend(facecolor="#15233a", labelcolor="#e8f0ff")
fig.tight_layout()
fig.savefig(out / "benchmark.svg")
fig.savefig(out / "benchmark.png", dpi=160)
plt.close(fig)
fig, ax = plt.subplots(figsize=(10, 4))
clips = m["test"]["clipping"]
ax.plot(
    [x["clip"] for x in clips],
    [x["estimates"]["IPS"]["value"] * 100 for x in clips],
    "-o",
    color="#67e7e8",
    label="IPS",
)
ax.plot(
    [x["clip"] for x in clips],
    [x["estimates"]["DR"]["value"] * 100 for x in clips],
    "-o",
    color="#b796fa",
    label="DR",
)
ax.set(
    xlabel="Weight cap (34 = effectively unclipped)",
    ylabel="Estimated CTR (%)",
    title="Clipping sensitivity · changes estimand through bias",
)
ax.legend(facecolor="#15233a", labelcolor="#e8f0ff")
fig.tight_layout()
fig.savefig(out / "clipping.svg")
plt.close(fig)
d = json.loads((out / "reference.json").read_text())
ad = pd.read_csv(out / "reference-action-diagnostics.csv")
tv = float(
    abs(
        ad[ad.position == 1].marginal_frequency.to_numpy()
        - ad[ad.position == 1].prior_mc_probability.to_numpy()
    ).sum()
    / 2
)
save(
    out / "reference-review.json",
    {
        "status": "MISMATCH",
        "position1_total_variation_prior_vs_logged_frequency": tv,
        "ips_minus_observed_bts": d["ips_minus_factual"],
        "combined_iid_standard_errors": d["ips_minus_factual"] / d["combined_iid_se"],
        "interpretation": "The pinned production-prior simulator does not reproduce observed position-one action frequencies. Algorithm parity and dataset counts pass, but the comparison cannot certify new-policy value. No single cause is established; prior/release alignment, policy dynamics and population/temporal differences require investigation. Do not treat observed BTS CTR as exact truth for an unmatched fixed-prior policy.",
        "publisher_table1_counts_match": True,
    },
)
files = {
    str(p.relative_to(out)).replace("\\", "/"): sha256(p)
    for p in out.iterdir()
    if p.is_file() and p.name not in ["integrity.json", "notebook-validation.json"]
}
save(out / "integrity.json", {"algorithm": "sha256", "files": files})
print(
    json.dumps(
        {
            "checks": len(verify(root)),
            "replay": int(ledger.rows.sum()),
            "simulation_replicates": 500,
            "reference_tv": tv,
        },
        indent=2,
    )
)
