from pathlib import Path
import json, zipfile, io, base64, hashlib

r = Path(__file__).resolve().parents[1]
out = r / "artifacts"
# Seal all durable evidence after dashboard generation.
files = {
    str(p.relative_to(out)).replace("\\", "/"): hashlib.sha256(
        p.read_bytes()
    ).hexdigest()
    for p in out.iterdir()
    if p.is_file() and p.name not in ["integrity.json", "notebook-validation.json"]
}
(out / "integrity.json").write_text(
    json.dumps({"algorithm": "sha256", "files": files}, indent=2), encoding="utf-8"
)
include = []
for folder in ["src", "tests", "docs", "data/upstream", "artifacts"]:
    include += [
        p
        for p in (r / folder).rglob("*")
        if p.is_file() and "__pycache__" not in str(p) and p.suffix != ".pyc"
    ]
include += [
    r / p
    for p in [
        "protocol.json",
        "NOTICE",
        "LICENSE",
        "requirements.txt",
        "queries.sql",
        "README.md",
        "scripts/download.py",
        "scripts/run.py",
        "scripts/finalize.py",
        "scripts/build_dashboard.py",
    ]
]
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(include):
        info = zipfile.ZipInfo(
            str(p.relative_to(r)).replace("\\", "/"), (2026, 10, 15, 0, 0, 0)
        )
        info.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(info, p.read_bytes())
b = buf.getvalue()
package_hash = hashlib.sha256(b).hexdigest()
dest = r.parents[1] / "publish-kit/policylab-release"
dest.mkdir(exist_ok=True, parents=True)
(dest / "source-audit.zip").write_bytes(b)
cells = []


def md(x):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": x})


def code(x):
    cells.append(
        {
            "cell_type": "code",
            "metadata": {},
            "source": x,
            "execution_count": None,
            "outputs": [],
        }
    )


md("""# PolicyLab: Can logged clicks justify a recommendation-policy change?
**Ruturajsinh Zala Â· Root2Raj** | AI-assisted personal Data Science + Data Architecture research

4,530,676 real ZOZOTOWN impressions, two historical experiment policies, 34 Men-campaign actions, seven days (24â€“30 November 2019). This reproducible evidence audit deliberately concludes **HOLD**. It exposes a reference-policy mismatch and an inconclusive final improvement, rather than manufacturing a positive result.

[Interactive lab](https://root2raj.ruturaj1zala123.chatgpt.site/projects/policylab/) Â· [GitHub](https://github.com/Root2Raj-Zala/root2raj-policylab) Â· [Official data](https://research.zozo.com/data.html) Â· [Paper](https://arxiv.org/abs/2008.07146)

Data/derived audit: CC BY 4.0, Saito, Aihara, Matsutani & Narita. Original code MIT; upstream production priors Apache 2.0. Changes: position filtering, fitting, weighting and aggregation. No endorsement, client outcome or operational deployment is claimed.

The default run audits the **frozen real final-evaluation rows**, executes the estimator and recovery tests, verifies every packaged evidence hash and reruns the synthetic oracle study. Full 1.485GB raw-source processing is optional in the final section. These scopes are kept distinct.""")
code("""from pathlib import Path
import os,sys,json,hashlib,base64,io,zipfile,subprocess,importlib.util
ROOT=Path('/kaggle/working/policylab-audit') if Path('/kaggle').exists() else Path.cwd()/'.cache/policylab-notebook-audit'
ROOT.mkdir(parents=True,exist_ok=True)
CACHE_ROOT=Path('/kaggle/temp/policylab-cache') if Path('/kaggle').exists() else ROOT/'.cache'
for key,sub in [('TEMP','tmp'),('TMP','tmp'),('PYTHONPYCACHEPREFIX','pycache'),('MPLCONFIGDIR','matplotlib'),('PIP_CACHE_DIR','pip')]:
    p=CACHE_ROOT/sub;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
sys.pycache_prefix=os.environ['PYTHONPYCACHEPREFIX']
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
print('CPU-only, no paid API. Working directory:',ROOT)
""")
md(
    "## 1. Load the self-contained, hashed evidence package\nThe archive includes the exact research code, tests, protocol, final audit, reference diagnostics and source manifests. It excludes the large original logs and user-feature hashes from the public final audit."
)
code(
    "PACKAGE_SHA256="
    + repr(package_hash)
    + "\nPACKAGE_B64="
    + repr(base64.b64encode(b).decode())
    + "\npayload=base64.b64decode(PACKAGE_B64)\nassert hashlib.sha256(payload).hexdigest()==PACKAGE_SHA256\nwith zipfile.ZipFile(io.BytesIO(payload)) as z:\n    assert all(not Path(p).is_absolute() and '..' not in Path(p).parts for p in z.namelist())\n    z.extractall(ROOT)\nsys.path.insert(0,str(ROOT/'src'))\nimport importlib.metadata\ntry:\n    installed_duckdb=importlib.metadata.version('duckdb')\nexcept importlib.metadata.PackageNotFoundError:\n    installed_duckdb=None\nif installed_duckdb != '1.5.6':\n    DEPS=Path('/kaggle/temp/policylab-deps') if Path('/kaggle').exists() else ROOT/'.deps'\n    subprocess.check_call([sys.executable,'-m','pip','install','duckdb==1.5.6','--target',str(DEPS),'--upgrade','--disable-pip-version-check'])\n    sys.path.insert(0,str(DEPS))\nimport duckdb\nassert duckdb.__version__=='1.5.6',duckdb.__version__\nprint('Pinned DuckDB:',duckdb.__version__)\nprint('Verified archive:',PACKAGE_SHA256)\n"
)
md(
    "## 2. Execute meaningful scientific and recovery tests\nTests use separate fixtures with known answers. They cover DR misspecification identities, normalization and positivity failures, independent paired uncertainty, future outcomes, official-loop Monte Carlo parity, legitimate duplicate preservation and an injected pre-commit interruption. Fixtures never inflate the real-data count."
)
code(
    "import unittest\nsuite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))\nresult=unittest.TextTestRunner(verbosity=1).run(suite)\nassert result.wasSuccessful()\nprint('Tests passed:',result.testsRun)\n"
)
md(
    "## 3. Verify frozen real-data evidence independently\nRecompute IPS, SNIPS, DR, paired differences and ESS from the actual 41,536 held-out logged rows. Missing raw files are explicitly reported as not rechecked; their published hashes remain in the manifest."
)
code(
    "import numpy as np,pandas as pd\nfrom policylab.verify import verify\nchecks=verify(ROOT)\nassert all(v is None or v for v in checks.values())\ndisplay(pd.Series(checks,name='passed (None = raw files not downloaded)'))\nintegrity=json.loads((ROOT/'artifacts/integrity.json').read_text())\nfor name,h in integrity['files'].items():\n    assert hashlib.sha256((ROOT/'artifacts'/name).read_bytes()).hexdigest()==h,name\nprint('Evidence hashes verified:',len(integrity['files']))\naudit=pd.read_csv(ROOT/'artifacts/test-audit.csv.gz')\nassert not any(c.startswith('user_feature') for c in audit)\ndisplay(audit.drop(columns='source_sha').head(5))\n"
)
md(
    "## 4. Selection discipline and final evidence\nFirst four UTC days fit the fixed-strength empirical-Bayes reward model and candidate policies. Day five chooses one of four candidates by its paired DR lower bound. The final two days do not select the winner. The selected policy is an action-only rule with 50% uniform exploration. Contextual candidates were considered in validation; their final outcomes are not used for selection."
)
code(
    "metrics=json.loads((ROOT/'artifacts/metrics.json').read_text())\nselection=json.loads((ROOT/'artifacts/selection.json').read_text())\nprint(json.dumps({k:selection[k] for k in ['selected','train_days','validation_days','test_days']},indent=2))\ndisplay(pd.DataFrame(metrics['test']['estimates']).T)\nprint('Paired DR vs uniform:',metrics['test']['paired_dr_vs_uniform'])\nprint('Decision:',metrics['review'])\n"
)
code(
    "from IPython.display import SVG,display\ndisplay(SVG(filename=str(ROOT/'artifacts/benchmark.svg')))\ndisplay(SVG(filename=str(ROOT/'artifacts/clipping.svg')))\n"
)
md(
    "## 5. Challenge the reference\nThe documented production-prior BernoulliTS algorithm is reconstructed using one million Monte Carlo draws. A test compares our vectorization against the literal upstream loop. But algorithm parity is different from matching the historical logging policy: the reconstructed fixed-prior first-position distribution differs from observed exposure frequencies. Its estimate also misses the observed BTS CTR under the frozen diagnostic. The project therefore **does not certify the learned policy value**. No single cause is established; data/prior alignment, policy dynamics and population differences require further investigation."
)
code(
    "reference=json.loads((ROOT/'artifacts/reference.json').read_text())\nreview=json.loads((ROOT/'artifacts/reference-review.json').read_text())\nprint(json.dumps(review,indent=2))\ndisplay(pd.read_csv(ROOT/'artifacts/reference-action-diagnostics.csv').query('position == 1').sort_values('n',ascending=False).head(10))\nprint('Observed BTS CTR',reference['observed_bts'],'IPS reconstructed target',reference['estimates']['IPS']['value'])\n"
)
md(
    "## 6. Warehouse recovery evidence\nThe complete source replay and every committed Parquet hash were checked locally. The default notebook displays that receipt; the tests independently inject and recover a small interruption. A source SHA + original row offset identifies an ingestion record. Equal-valued rows are retained. This is a local single-writer visibility protocol, not a global streaming exactly-once claim."
)
code(
    "print(json.dumps(json.loads((ROOT/'artifacts/replay.json').read_text()),indent=2))\ndisplay(pd.read_csv(ROOT/'artifacts/reconciliation.csv'))\nprint('Protocol SHA:',metrics['protocol_sha256'])\n"
)
md(
    "## 7. Separate synthetic oracle experiment\nHere the true unobserved action rewards are known by construction. A deliberately biased reward model illustrates why DM can be wrong and how weak overlap raises IPS/DR uncertainty. These synthetic records are never called ZOZO impressions. The simulation tests behavior under its assumptions, not real-data validity."
)
code(
    "from policylab.simulation import run as simulate\nsimout=ROOT/'simulation-rerun';simout.mkdir(exist_ok=True)\nsimulation=simulate(simout,repetitions=250,n=4000)\ndisplay(simulation)\nexpected=pd.read_csv(ROOT/'artifacts/simulation-summary.csv')\nnp.testing.assert_allclose(simulation.select_dtypes('number'),expected.select_dtypes('number'),rtol=1e-9,atol=1e-10)\nprint('500 synthetic repetitions reproduced; separate from real counts')\n"
)
md(
    "## 8. Optional full-source rerun\nLeave the switch off for the evidence audit. Turn it on to download both complete Men logs from the pinned official release (about 1.485GB), ingest every row and rerun the protocol. Internet access and several GB of free disk are needed; CPU is sufficient. No paid API or GPU is used. This cell does not claim a raw-source rerun unless the switch is enabled."
)
code(
    "RUN_FULL_SOURCE=False\nif RUN_FULL_SOURCE:\n    subprocess.check_call([sys.executable,str(ROOT/'scripts/download.py')],cwd=ROOT)\n    from policylab.pipeline import run as full_run\n    rerun=full_run(ROOT)\n    assert rerun['rows_all_positions']==4530676\n    np.testing.assert_allclose(rerun['test']['estimates']['DR']['value'],metrics['test']['estimates']['DR']['value'],rtol=1e-9)\n    print('FULL RAW-SOURCE RERUN PASSED')\nelse:\n    print('Frozen real-audit scope completed. Full raw-source rerun was not requested.')\n"
)
md(
    "## What this result supports\nInspecting assumptions, estimation, failure cases and source lineage is useful portfolio evidence. Seven historical days, unavailable users, Monte Carlo propensities, an unresolved reference mismatch and conditional intervals cannot establish a production gain. Marginal item-position probabilities do not establish whole-slate effects. A large row count cannot repair those gaps. See the packaged protocol, architecture runbook, model card and plain-language walkthrough for the complete reasoning.\n\n**Attribution:** Saito, Y., Aihara, S., Matsutani, M., & Narita, Y. (2021), *Open Bandit Dataset and Pipeline*. Data: CC BY 4.0; transformed audit/aggregates retain attribution; upstream priors/reference Apache 2.0. No endorsement. Original implementation developed with AI assistance for Ruturajsinh Zala (Root2Raj), MIT."
)
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
    "cells": cells,
}
for i, c in enumerate(cells):
    c["id"] = f"policylab-{i:02}"
    c["source"] = c["source"].splitlines(keepends=True)
(r / "notebooks/policylab.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("notebook", len(cells), "cells", len(b), "package bytes", package_hash)
