from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from policylab.pipeline import run

run(Path(__file__).resolve().parents[1])
