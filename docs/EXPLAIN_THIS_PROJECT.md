# Explain PolicyLab in an interview

A recommender changes which products people see. A click model alone cannot tell us whether a new recommendation rule is better, because the old rule determined what was observed. PolicyLab uses a real randomized fashion-recommendation experiment to evaluate that problem.

First I compare estimates against clicks observed under a separate real experiment policy. Then a separate chronological study trains simple interpretable policies on four earlier days, chooses one on another day and evaluates it on the remaining two. I compare four estimators and inspect how a few heavily weighted examples can dominate the answer. The warehouse preserves real source identities, detects bad probabilities and recovers safely from an interrupted ingest.

The result can be HOLD. There are only seven source days, user identities are unavailable and item-position probabilities cannot support arbitrary slate interactions. Statistical precision on many rows does not remove those limitations.

The implementation was developed with AI assistance. I should explain the estimand, assumptions, data contract and evidence before representing this as a skill. It is personal historical research, not client experience or an operational deployment.

Useful questions to practice: Why does IPS multiply by a probability ratio? Why is SNIPS biased in finite samples? What does DR protect against, and what does it not protect against? Why does clipping trade variance for bias? Why must selection happen before final testing? Why are two days not thousands of independent days? Which transaction makes a shard visible?
