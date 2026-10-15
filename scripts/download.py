import urllib.request, json, pathlib

root = pathlib.Path(__file__).resolve().parents[1]


def get(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Root2Raj PolicyLab https://github.com/Root2Raj-Zala"},
    )
    with (
        urllib.request.urlopen(req, timeout=180) as r,
        path.with_suffix(path.suffix + ".part").open("wb") as f,
    ):
        while b := r.read(4 * 1024 * 1024):
            f.write(b)
    path.with_suffix(path.suffix + ".part").replace(path)
    print(path.name, path.stat().st_size, flush=True)


rev = json.loads((root / "data/upstream/revisions.json").read_text())
sha = rev["huggingface"]
gh = rev["obp"]
for p in [
    "obp/policy/conf/prior_bts.yaml",
    "obp/policy/contextfree.py",
    "obp/dataset/real.py",
    "examples/obd/evaluate_off_policy_estimators.py",
    "LICENSE",
]:
    try:
        get(
            f"https://raw.githubusercontent.com/st-tech/zr-obp/{gh}/{p}",
            root / "data/upstream" / pathlib.Path(p).name,
        )
    except Exception as e:
        print(p, str(e), flush=True)
for policy in ["random", "bts"]:
    for name in ["men.csv", "item_context.csv"]:
        get(
            f"https://huggingface.co/datasets/zozonext/open-bandit/resolve/{sha}/{policy}/men/{name}",
            root / "data/raw" / policy / name,
        )
print("DOWNLOAD COMPLETE", flush=True)
