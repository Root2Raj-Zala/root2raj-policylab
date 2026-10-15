# Reproduce without paid services

Python 3.12, CPU, about 3GB free storage plus a generous local margin. Downloaded logs total approximately 1.485GB; a complete run also writes Parquet, a DuckDB ledger and evidence. All local project files, virtual environments and caches should be kept in the project directory (the owner uses D:/Root2Rep/projects/policylab).

```powershell
$env:TEMP="$PWD/.cache/tmp"; $env:TMP=$env:TEMP
$env:PIP_CACHE_DIR="$PWD/.cache/pip"
$env:PYTHONPYCACHEPREFIX="$PWD/.cache/pycache"
New-Item -ItemType Directory -Force .cache/tmp | Out-Null
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install -e . --no-deps
.venv/Scripts/python -m unittest discover -s tests -v
.venv/Scripts/python scripts/download.py
.venv/Scripts/python scripts/run.py
.venv/Scripts/python scripts/finalize.py
```

Linux/Kaggle: replace the virtual-environment executable with `.venv/bin/python`, or use the installed notebook interpreter. `python scripts/download.py` uses pinned official data/source revisions in data/upstream/revisions.json. Raw data stays outside Git. The one-million-draw reference uses the original NumPy RandomState algorithm; the official-loop parity test guards the vectorization. Tiny floating-point differences may occur across environments. Compare estimates with numerical tolerances, never an expected positive gain.

The notebook offers an executed audit of the frozen held-out data and an explicit optional full-source rerun. GitHub CI runs tests, artifact checks and the notebook audit, without redownloading 1.5GB every push. Full benchmark results and full-source replay are separately recorded in artifacts. Neither the audit nor CI is falsely labeled a full raw-data rerun.

## Published Kaggle environment

The public notebook uses Python 3.13 on Kaggle CPU and explicitly installs/checks DuckDB 1.5.6 when the container has an older release. Runtime dependencies and caches use `/kaggle/temp`; source, reproducible outputs and evidence remain in `/kaggle/working/policylab-audit`. Version 1 exposed the older-container incompatibility; version 2 passed all 30 tests. The final release keeps caches out of saved outputs.

The size-compliant [Kaggle notebook](../notebooks/policylab-kaggle.ipynb) downloads the exact GitHub archive at commit `232f2e1d2d1db8f1ea7175077bc0f1363378cc0a` and verifies SHA-256 `56d684f36b692361718ef576451a494866cd11cc43294629e733a41828531921`. The [offline notebook](../notebooks/policylab.ipynb) embeds that same archive. Their default scientific cells are equivalent; only package transport differs.
