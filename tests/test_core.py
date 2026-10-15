import unittest, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from policylab.ope import scores, summary, paired_difference, gate
from policylab.models import RewardModel, policy, production_ts
from policylab.warehouse import Warehouse, contract, sha256


class EstimatorTests(unittest.TestCase):
    def setUp(self):
        self.r = np.array([0, 1, 1, 0])
        self.a = np.array([0, 1, 0, 1])
        self.p = np.ones(4) / 2
        self.pi = np.ones((4, 2)) / 2
        self.q = np.ones((4, 2)) * 0.3

    def args(self):
        return [self.r, self.a, self.p, self.pi, self.q]

    def test_identity_policy(self):
        s, w = scores(*self.args())
        self.assertTrue(np.all(w == 1))
        for k in ["IPS", "SNIPS", "DR"]:
            self.assertAlmostEqual(s[k].mean(), 0.5)

    def test_snips_scale_invariant(self):
        args = self.args()
        s, _ = scores(*args)
        args[2] = np.ones(4) / 4
        t, _ = scores(*args)
        self.assertAlmostEqual(s["SNIPS"].mean(), t["SNIPS"].mean())

    def test_dm_ignores_observed_reward(self):
        s, _ = scores(*self.args())
        args = self.args()
        args[0] = 1 - self.r
        t, _ = scores(*args)
        np.testing.assert_array_equal(s["DM"], t["DM"])

    def test_known_finite_population(self):
        # Enumerate balanced actions under known behavior and reward table.
        a = np.tile([0, 1], 100)
        r = (a == 1).astype(int)
        pi = np.tile([0.2, 0.8], (200, 1))
        q = np.tile([0.0, 1.0], (200, 1))
        s, _ = scores(r, a, np.ones(200) / 2, pi, q)
        for v in s.values():
            self.assertAlmostEqual(v.mean(), 0.8)

    def test_doubly_robust_with_exact_model_wrong_propensity(self):
        a = np.tile([0, 1], 100)
        r = a.copy()
        pi = np.tile([0.2, 0.8], (200, 1))
        q = np.tile([0.0, 1.0], (200, 1))
        s, _ = scores(r, a, np.ones(200) * 0.8, pi, q)
        self.assertAlmostEqual(s["DR"].mean(), 0.8)

    def test_exact_propensity_biased_model(self):
        a = np.tile([0, 1], 100)
        r = a.copy()
        pi = np.tile([0.2, 0.8], (200, 1))
        q = np.ones((200, 2)) * 0.1
        s, _ = scores(r, a, np.ones(200) / 2, pi, q)
        self.assertAlmostEqual(s["DR"].mean(), 0.8)

    def test_paired_uncertainty_cancels_identical_policy(self):
        d = paired_difference(np.arange(20), np.arange(20), np.repeat(range(4), 5))
        self.assertEqual(d["ci95_iid"], [0, 0])
        self.assertEqual(d["ci95_hour_sensitivity"], [0, 0])

    def test_ess_and_weight_tail(self):
        s = summary(*self.args())
        self.assertEqual(s["ess"], 4)
        self.assertEqual(s["weight_mean"], 1)

    def test_clipping_changes_ips_and_exposes_bias(self):
        args = self.args()
        args[3] = np.tile([0.0, 1.0], (4, 1))
        u, _ = scores(*args)
        c, _ = scores(*args, clip=1)
        self.assertNotEqual(u["IPS"].mean(), c["IPS"].mean())

    def test_zero_propensity(self):
        args = self.args()
        args[2][0] = 0
        with self.assertRaises(ValueError):
            scores(*args)

    def test_negative_probability(self):
        args = self.args()
        args[3][0] = [-1, 2]
        with self.assertRaises(ValueError):
            scores(*args)

    def test_unnormalized_policy(self):
        args = self.args()
        args[3] *= 2
        with self.assertRaises(ValueError):
            scores(*args)

    def test_nan(self):
        args = self.args()
        args[4][0, 0] = np.nan
        with self.assertRaises(ValueError):
            scores(*args)

    def test_out_of_range_action(self):
        args = self.args()
        args[1][0] = 2
        with self.assertRaises(ValueError):
            scores(*args)

    def test_missing_empirical_overlap(self):
        args = self.args()
        args[1][:] = 0
        args[3] = np.tile([0.0, 1.0], (4, 1))
        with self.assertRaises(ValueError):
            scores(*args)

    def test_invalid_reward(self):
        args = self.args()
        args[0][0] = 2
        with self.assertRaises(ValueError):
            scores(*args)

    def test_invalid_shape(self):
        args = self.args()
        args[4] = np.zeros((4, 3))
        with self.assertRaises(ValueError):
            scores(*args)

    def test_gate_holds_short_window(self):
        result = summary(*self.args())
        d = {"ci95_iid": [0.01, 0.02]}
        self.assertEqual(gate(result, 2, d, True)["status"], "HOLD")


class ArchitectureTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame(
            {
                "timestamp": ["2019-11-24T00:00:00Z"] * 6,
                "item_id": [0, 1, 2, 0, 1, 2],
                "position": [1, 2, 3, 1, 2, 3],
                "click": [0, 0, 1, 0, 0, 1],
                "propensity_score": [1 / 34] * 6,
                "user_feature_0": ["a"] * 6,
            }
        )

    def test_contract_valid(self):
        contract(self.frame())

    def test_mixed_iso_precision(self):
        z = self.frame()
        z.loc[0, "timestamp"] = "2019-11-24 00:00:00.123456+00:00"
        contract(z)

    def test_contract_rejects_bad_position(self):
        z = self.frame()
        z.loc[0, "position"] = 4
        with self.assertRaises(ValueError):
            contract(z)

    def test_contract_rejects_missing(self):
        with self.assertRaises(ValueError):
            contract(self.frame().drop(columns="click"))

    def test_contract_rejects_fractional_action(self):
        z = self.frame().astype({"item_id": float})
        z.loc[0, "item_id"] = 0.5
        with self.assertRaises(ValueError):
            contract(z)

    def test_crash_replay_idempotency_and_real_duplicate_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            self.frame().to_csv(p / "source.csv", index=False)
            w = Warehouse(p / "w")
            with self.assertRaisesRegex(RuntimeError, "injected"):
                w.ingest(p / "source.csv", "random", chunk_size=2, fail_part=1)
            self.assertEqual(
                w.db.execute("select sum(rows) from shards").fetchone()[0], 2
            )
            w.ingest(p / "source.csv", "random", chunk_size=2)
            w.views()
            before = w.reconcile()
            w.ingest(p / "source.csv", "random", chunk_size=2)
            w.views()
            after = w.reconcile()
            pd.testing.assert_frame_equal(before, after)
            self.assertEqual(int(after.iloc[0]["rows"]), 6)
            self.assertEqual(int(after.iloc[0].identities), 6)
            self.assertEqual(
                w.db.execute("select count(*) from exposures").fetchone()[0], 6
            )
            w.close()

    def test_changed_chunk_size_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            self.frame().to_csv(p / "source.csv", index=False)
            w = Warehouse(p / "w")
            w.ingest(p / "source.csv", "random", 2)
            with self.assertRaises(ValueError):
                w.ingest(p / "source.csv", "random", 3)
            w.close()

    def test_no_future_reward_dependency(self):
        z = self.frame()
        z["event_time"] = pd.to_datetime(z.timestamp)
        m = RewardModel().fit(z)
        before = m.predict(z)
        z.click = 1 - z.click
        np.testing.assert_array_equal(before, m.predict(z))

    def test_unknown_context_falls_back(self):
        z = self.frame()
        z["event_time"] = pd.to_datetime(z.timestamp)
        m = RewardModel().fit(z)
        z.user_feature_0 = "unseen"
        np.testing.assert_allclose(m.predict(z)[0], m.action)

    def test_policy_exploration_floor(self):
        z = self.frame()
        z["event_time"] = pd.to_datetime(z.timestamp)
        m = RewardModel().fit(z)
        p = policy(m, z, "contextual_epsilon_0.2")
        self.assertTrue(np.all(p >= 0.2 / 34))
        np.testing.assert_allclose(p.sum(1), 1)

    def test_ts_matches_official_loop(self):
        prior = {"alpha": [2, 4, 1, 3], "beta": [20, 40, 10, 30]}
        seed = 12345
        n = 201
        rng = np.random.RandomState(seed)
        counts = np.zeros((4, 3))
        for _ in range(n):
            ranks = rng.beta(prior["alpha"], prior["beta"]).argsort()[::-1][:3]
            for pos, a in enumerate(ranks):
                counts[a, pos] += 1
        np.testing.assert_array_equal(production_ts(prior, n, seed), counts / n)

    def test_hash_changes_on_edit(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "x"
            p.write_text("a")
            h = sha256(p)
            p.write_text("b")
            self.assertNotEqual(h, sha256(p))


if __name__ == "__main__":
    unittest.main()
