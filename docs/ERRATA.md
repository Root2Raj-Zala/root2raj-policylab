# Documentation correction — 15 October 2026

The first model-card text incorrectly said that the UTC-hour bootstrap sensitivity interval crossed zero. The actual frozen `artifacts/metrics.json` has always reported `[0.0000274628, 0.0055988373]` in probability units, or **[+0.0027, +0.5599] percentage points**. It does not cross zero. The conditional row-iid interval **[-0.0085, +0.5109] percentage points** does cross zero.

The current model card corrects this sentence. No estimate, audit row, protocol, selection, evidence hash, warehouse record or HOLD decision changed. The sensitivity has only 48 hourly blocks across two final dates; reference mismatch and limited horizon still prevent certification.

The notebooks verify the immutable source-audit archive from GitHub commit `232f2e1d2d1db8f1ea7175077bc0f1363378cc0a`. That historical archive retains the earlier wording in its packaged model card. The notebook identifies this erratum explicitly; use the current repository model card and the actual numeric metrics for interpretation.
