"""Streaming, content-addressed ingestion with transactional shard visibility."""

from pathlib import Path
import hashlib, os
import duckdb
import numpy as np
import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(4 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def contract(df, k=34):
    if df.empty:
        raise ValueError("empty chunk")
    for col in [
        "item_id",
        "position",
        "click",
        "propensity_score",
        "timestamp",
        "user_feature_0",
    ]:
        if col not in df:
            raise ValueError("missing " + col)
    if (
        df[
            [
                "item_id",
                "position",
                "click",
                "propensity_score",
                "timestamp",
                "user_feature_0",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("null")
    a = df.item_id.to_numpy()
    p = df.propensity_score.to_numpy()
    if np.any(a != a.astype(int)) or np.any(a < 0) or np.any(a >= k):
        raise ValueError("action domain")
    if not df.position.isin([1, 2, 3]).all() or not df.click.isin([0, 1]).all():
        raise ValueError("reward/position")
    if not np.isfinite(p).all() or np.any(p <= 0) or np.any(p > 1):
        raise ValueError("propensity")
    pd.to_datetime(df.timestamp, utc=True, format="ISO8601", errors="raise")


class Warehouse:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = duckdb.connect(str(self.root / "warehouse.duckdb"))
        self.db.execute("SET threads=2")
        self.db.execute("SET TimeZone='UTC'")
        self.db.execute("SET memory_limit='1GB'")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS shards(source_sha VARCHAR, policy VARCHAR, part INTEGER, path VARCHAR, rows BIGINT, clicks BIGINT, parquet_sha VARCHAR, PRIMARY KEY(source_sha,part))"
        )

    def ingest(self, path, policy, chunk_size=100000, fail_part=None):
        path = Path(path)
        source = sha256(path)
        offset = 0
        with pd.read_csv(path, chunksize=chunk_size) as reader:
            for part, frame in enumerate(reader):
                n = len(frame)
                row = self.db.execute(
                    "SELECT rows FROM shards WHERE source_sha=? AND part=?",
                    [source, part],
                ).fetchone()
                if row:
                    if row[0] != n:
                        raise ValueError("changed chunk size")
                    offset += n
                    continue
                contract(frame)
                if policy == "random" and not np.allclose(
                    frame.propensity_score, 1 / 34
                ):
                    raise ValueError("random probability")
                frame = frame.drop(
                    columns=[c for c in frame if c.startswith("Unnamed:")]
                )
                frame.insert(
                    0, "source_row", np.arange(offset, offset + n, dtype=np.int64)
                )
                frame["source_sha"] = source
                frame["policy"] = policy
                frame["event_time"] = pd.to_datetime(
                    frame.timestamp, utc=True, format="ISO8601"
                )
                frame["day"] = frame.event_time.dt.strftime("%Y-%m-%d")
                folder = self.root / "parquet" / policy
                folder.mkdir(parents=True, exist_ok=True)
                target = folder / f"{source[:12]}-{part:05}.parquet"
                tmp = target.with_suffix(".pending")
                self.db.register("chunk", frame)
                self.db.execute(
                    "COPY chunk TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(tmp)]
                )
                self.db.unregister("chunk")
                digest = sha256(tmp)
                os.replace(tmp, target)
                self.db.execute("BEGIN")
                try:
                    self.db.execute(
                        "INSERT INTO shards VALUES (?,?,?,?,?,?,?)",
                        [
                            source,
                            policy,
                            part,
                            str(target),
                            n,
                            int(frame.click.sum()),
                            digest,
                        ],
                    )
                    if fail_part == part:
                        raise RuntimeError("injected interruption before commit")
                    self.db.execute("COMMIT")
                except BaseException:
                    self.db.execute("ROLLBACK")
                    raise
                offset += n
        return {
            "source_sha256": source,
            "bytes": path.stat().st_size,
            "rows": offset,
            "policy": policy,
            "file": path.name,
        }

    def views(self):
        paths = [
            r[0]
            for r in self.db.execute(
                "SELECT path FROM shards ORDER BY policy,part"
            ).fetchall()
        ]
        self.db.read_parquet(paths).create_view("exposures", replace=True)
        self.db.execute(
            "CREATE OR REPLACE TEMP VIEW rewards AS SELECT source_sha,source_row,click FROM exposures"
        )
        self.db.execute(
            "CREATE OR REPLACE TEMP VIEW logged_policies AS SELECT policy,position,item_id,count(*) AS exposures,avg(propensity_score) AS mean_propensity FROM exposures GROUP BY ALL"
        )

    def position(self, policy, position=1):
        return self.db.execute(
            "SELECT * FROM exposures WHERE policy=? AND position=? ORDER BY event_time,source_row",
            [policy, position],
        ).df()

    def reconcile(self):
        return self.db.execute(
            'SELECT policy, count(*) AS "rows", sum(click) clicks, count(DISTINCT (source_sha,source_row)) identities, min(event_time) first_time,max(event_time) last_time FROM exposures GROUP BY policy ORDER BY policy'
        ).df()

    def close(self):
        self.db.close()
