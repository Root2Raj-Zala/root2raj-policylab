# Frozen experimental protocol

Read `protocol.json` and its SHA-256 in `artifacts/metrics.json`. The protocol was saved before the first analysis of complete outcomes. This is a timestamped development record, not registration with an independent registry.

## Estimand and identification

Expected binary click at the **first recommendation position**, for the historical Men campaign context population. Identification requires correct marginal logging probabilities, positive support, exchangeability under the randomized logging policy, consistency and a position-local reward without other-slate-item interference. Marginal item-position probabilities cannot identify a joint three-item slate value. Context hashes are not customer IDs. The experiment has no individual counterfactual ground truth.

## Two different experiments

**Reference validation:** implement the publisher's production-prior BernoulliTS policy with one million seeded Beta draws and the same descending top-three ranking. Compare Random-log OPE with the BTS-log observed reward mean at position one, using the complete matched period. Three-fold row cross-fitting supplies reward estimates. This is a modern implementation of the documented reference comparison; it does not reproduce a paper table, all estimators, bootstrap repetitions or the three-position pooled target. Both samples have sampling uncertainty. Monte Carlo policy probabilities and reported logged BTS propensities are approximate. A 3-combined-iid-SE diagnostic is a compatibility screen, not a scientific validation certificate.

**Chronological policy study:** first four UTC dates train a reward model and policies; date five selects the largest paired DR lower bound from four frozen candidates; the last two dates score only that selected specification. The reward model is not refitted on validation or final outcomes. Final alternative-candidate results are not used for selection. The empirical-Bayes model shrinks context-by-action click rates toward action rates, then global training CTR, with fixed strength 100 at both levels. The feature is anonymized user_feature_0 plus action; affinity scores are excluded because source vintage/availability is not reconstructed. No timestamp, row offset, future label or BTS outcome enters the learned policy. Unknown contexts use action rates. Candidate actions have epsilon/34 probability floors. Ties choose the lowest action index.

## Estimators

With w=pi(a|x)/p(a|x), DM=mean(sum_a pi*q), IPS=mean(w*r), SNIPS=sum(w*r)/sum(w), DR=mean(sum_a pi*q+w*(r-q_logged)). SNIPS uncertainty uses its ratio influence function. Paired DR differences subtract the uniform-policy score on the **same rows**, so baseline uncertainty and covariance enter the interval. Clipping truncates weights and can introduce bias; it is a sensitivity analysis and never retroactively selects the clipping threshold.

Row-iid normal intervals are conditional approximations. The separate hour-cluster percentile bootstrap resamples complete UTC-hour blocks (1,000 replicates); it is a dependence sensitivity analysis, not a guaranteed valid bound with only seven source days. No user-cluster claim is possible. Cross-fit reference folds are not future-safe training for an operational model.

## Decision

HOLD if reference mismatch, ESS fraction <10%, observed test clicks <100, paired DR lower bound <=0, estimator spread >0.002, or fewer than 14 evaluation days. This conservative temporal rule necessarily withholds release on this seven-day source. A HOLD is a useful research conclusion. No deployment, financial gain or long-run uplift is inferred.
