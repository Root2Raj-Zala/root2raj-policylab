# Model and decision card — PolicyLab 1.0

Owner: Ruturajsinh Zala (Root2Raj). AI-assisted personal research. Real historical ZOZOTOWN experiment logs, 24–30 November 2019. Data CC BY 4.0, original code MIT, upstream priors/reference Apache 2.0.

**Decision: HOLD.** This is a completed research and engineering release, with an intentionally withheld policy recommendation. It is not a production recommender.

## Population and protocol

4,530,676 real impressions across both Men policy logs and all three positions are ingested. Statistical work restricts to position one: 150,936 Random and 1,358,878 BTS records for the reference. Chronological train: 86,722; validation: 22,678; final: 41,536. The selected policy is `action_epsilon_0.5`. Simulated oracle experiments are separate.

## Final evidence

| Estimator | Estimated click probability |
|---|---:|
| DM | 0.8231% |
| IPS | 0.7199% |
| DR | 0.7892% |
| SNIPS | 0.7678% |

Paired DR difference vs uniform: 0.2512 percentage points; conditional row-iid 95% approximation [-0.0085, 0.5109]. This interval crosses zero. Hour-block sensitivity also crosses zero. Many impressions do not supply many independent days.

## Reference discrepancy

The official production-prior simulator gives IPS 0.5775% versus BTS observed 0.7245%. The frozen 3-combined-SE diagnostic fails. Actual action frequencies also disagree with the fixed-prior simulator (see reference-review.json and action diagnostics). The upstream algorithm is reproduced and tested against its literal loop, but matching the released logging policy is not certified. Observed BTS CTR cannot be called exact truth for a mismatched target. No single cause is established. Further alignment of the source release, priors, policy dynamics and exposure population is needed before any policy-value recommendation. The learned study therefore remains descriptive and conditional on Random logging assumptions.

## Appropriate and inappropriate use

Use this to inspect OPE equations, probability/support failures, selection discipline and auditable warehouse recovery. Do not use it to choose live fashion products, infer customer preferences/identities, claim a whole-slate causal effect, promise financial savings or assert current-market generalization. There are no protected-group fairness labels, reliable users, cost labels, long-term outcomes or original feature-vintage records. The reward model is deliberately transparent and small; sophistication lies in evaluation and evidence controls, not model size.

DM intervals condition on fitted predictions and omit model estimation error. Row-iid intervals generally omit repeated-user and temporal dependence. Hour resampling has too few days to warrant a coverage guarantee. Clipping changes bias, not just variance. A passing software check is not a validation of causal assumptions.
