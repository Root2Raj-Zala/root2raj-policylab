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
