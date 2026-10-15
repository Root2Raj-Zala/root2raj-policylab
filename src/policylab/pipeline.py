"""Run the complete frozen real-data protocol. No simulated rows enter real metrics."""

from pathlib import Path
import json, time, platform
import numpy as np
import pandas as pd
import yaml
from policylab.warehouse import Warehouse, sha256
from policylab.models import RewardModel, policy, production_ts
from policylab.ope import summary, scores, paired_difference, gate


def save(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8"
    )


def run(root):
    root = Path(root)
    out = root / "artifacts"
    out.mkdir(exist_ok=True)
    started = time.perf_counter()
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8-sig"))
    protocol_hash = sha256(root / "protocol.json")
    wh = Warehouse(root / "data/warehouse")
    manifest = []
    for name in ["random", "bts"]:
        manifest.append(wh.ingest(root / f"data/raw/{name}/men.csv", name))
        print("ingested", name, manifest[-1]["rows"], flush=True)
    wh.views()
    rec = wh.reconcile()
    rec.to_csv(out / "reconciliation.csv", index=False)
    assert (rec.rows == rec.identities).all()
    for m in manifest:
        row = rec[rec.policy == m["policy"]].iloc[0]
        assert int(row.rows) == m["rows"]
    source = {
        "dataset": "ZOZO Open Bandit Dataset",
        "campaign": "men",
        "license": "CC BY 4.0",
        "files": manifest,
        "revisions": json.loads((root / "data/upstream/revisions.json").read_text()),
    }
    save(out / "source-manifest.json", source)
    data = wh.position("random")
    bts = wh.position("bts")
    wh.close()
    days = sorted(data.day.unique())
    assert len(days) == 7
    assert set(data.item_id) == set(range(34))
    assert set(bts.item_id) == set(range(34))
    print("position 1", len(data), len(bts), days, flush=True)
    prior = yaml.safe_load((root / "data/upstream/prior_bts.yaml").read_text())["men"]
    ts = production_ts(prior)
    np.savetxt(
        out / "production-ts-action-distribution.csv",
        ts,
        delimiter=",",
        header="position_1,position_2,position_3",
        comments="",
    )
    # Cross-fitting is descriptive exchangeability-based reference evaluation, not temporal validation.
    rng = np.random.default_rng(12345)
    folds = np.empty(len(data), int)
    folds[rng.permutation(len(data))] = np.arange(len(data)) % 3
    q = np.empty((len(data), 34))
    for fold in range(3):
        q[folds == fold] = (
            RewardModel().fit(data[folds != fold]).predict(data[folds == fold])
        )
    target = np.tile(ts[:, 0], (len(data), 1))
    ref = summary(data.click, data.item_id, data.propensity_score, target, q)
    factual = float(bts.click.mean())
    bse = float(bts.click.std(ddof=1) / np.sqrt(len(bts)))
    diff = ref["estimates"]["IPS"]["value"] - factual
    se = np.hypot(ref["estimates"]["IPS"]["se_iid"], bse)
    # Compatibility diagnostic, not proof of population equivalence or paper table reproduction.
    reference_ok = abs(diff) <= 3 * se
    ref.update(
        observed_bts=factual,
        observed_bts_n=len(bts),
        observed_bts_se_iid=bse,
        ips_minus_factual=diff,
        combined_iid_se=float(se),
        within_3_iid_se=bool(reference_ok),
        mc_draws=1000000,
        seed=12345,
    )
    ref["daily"] = []
    for day in days:
        z = data[data.day == day]
        mask = data.day == day
        br = bts[bts.day == day]
        item = summary(z.click, z.item_id, z.propensity_score, target[mask], q[mask])
        item["day"] = day
        item["observed_bts"] = float(br.click.mean())
        item["observed_bts_n"] = len(br)
        ref["daily"].append(item)
    save(out / "reference.json", ref)
    train = data[data.day.isin(days[:4])]
    valid = data[data.day == days[4]]
    test = data[data.day.isin(days[5:])]
    model = RewardModel().fit(train)
    save(out / "reward-model.json", model.artifact())
    assert pd.Timestamp(model.cutoff) < valid.event_time.min() < test.event_time.min()
    candidates = protocol["candidates"]
    validation = []
    qv = model.predict(valid)
    uniform = policy(model, valid, "uniform")
    bs, _ = scores(valid.click, valid.item_id, valid.propensity_score, uniform, qv)
    for name in candidates:
        pi = policy(model, valid, name)
        s, _ = scores(valid.click, valid.item_id, valid.propensity_score, pi, qv)
        pair = paired_difference(s["DR"], bs["DR"])
        validation.append(
            {
                "name": name,
                "paired_dr": pair,
                "result": summary(
                    valid.click, valid.item_id, valid.propensity_score, pi, qv
                ),
            }
        )
    selected = sorted(
        validation, key=lambda z: (-z["paired_dr"]["ci95_iid"][0], z["name"])
    )[0]["name"]
    # Persist selection before first final outcome score.
    selection = {
        "selected": selected,
        "validation": validation,
        "train_days": days[:4],
        "validation_days": [days[4]],
        "test_days": days[5:],
        "protocol_sha256": protocol_hash,
        "model_sha256": sha256(out / "reward-model.json"),
    }
    save(out / "selection.json", selection)
    selection_hash = sha256(out / "selection.json")
    qt = model.predict(test)
    pi = policy(model, test, selected)
    unif = policy(model, test, "uniform")
    sc, w = scores(test.click, test.item_id, test.propensity_score, pi, qt)
    sb, _ = scores(test.click, test.item_id, test.propensity_score, unif, qt)
    pair = paired_difference(
        sc["DR"], sb["DR"], test.event_time.dt.strftime("%Y-%m-%dT%H").to_numpy()
    )
    result = summary(test.click, test.item_id, test.propensity_score, pi, qt)
    result["paired_dr_vs_uniform"] = pair
    result["baseline"] = summary(
        test.click, test.item_id, test.propensity_score, unif, qt
    )
    result["clipping"] = [
        summary(test.click, test.item_id, test.propensity_score, pi, qt, c)
        for c in protocol["clips"]
    ]
    result["daily"] = []
    for day in days[5:]:
        m = test.day == day
        z = test[m]
        r = summary(z.click, z.item_id, z.propensity_score, pi[m], qt[m])
        r["day"] = day
        result["daily"].append(r)
    # Counterfactual source-row audit, no fabricated unchosen outcomes.
    audit = test[
        [
            "source_sha",
            "source_row",
            "timestamp",
            "item_id",
            "position",
            "click",
            "propensity_score",
            "day",
        ]
    ].copy()
    audit["target_probability"] = pi[np.arange(len(test)), test.item_id]
    audit["weight"] = w
    audit["q_logged"] = qt[np.arange(len(test)), test.item_id]
    audit["dm"] = sc["DM"]
    audit["dr"] = sc["DR"]
    audit["baseline_dr"] = sb["DR"]
    audit["model_cutoff"] = model.cutoff
    audit.to_csv(
        out / "test-audit.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    review = gate(result, len(days[5:]), pair, reference_ok)
    metrics = {
        "version": "1.0.0",
        "rows_all_positions": sum(x["rows"] for x in manifest),
        "campaign": "men",
        "position": 1,
        "actions": 34,
        "days": days,
        "selected": selected,
        "train_rows": len(train),
        "validation_rows": len(valid),
        "test_rows": len(test),
        "reference": {k: v for k, v in ref.items() if k != "daily"},
        "test": result,
        "review": review,
        "protocol_sha256": protocol_hash,
        "selection_sha256": selection_hash,
        "runtime_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "zero_spend": True,
    }
    save(out / "metrics.json", metrics)
    print(
        json.dumps(
            {
                k: metrics[k]
                for k in [
                    "rows_all_positions",
                    "selected",
                    "test_rows",
                    "review",
                    "runtime_seconds",
                ]
            },
            indent=2,
        ),
        flush=True,
    )
    return metrics


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[2])
