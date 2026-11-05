from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'

FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
HEAVY = RESULTS / "heavy"

# the tail is Testis ~ spermatocytes 5441
MIN_GENES = 7000
# neuropil is the mixup of neuronal and glial processes not a cell type
DROP_CONTEXTS = [("Cerebral cortex", "neuropil")]
DROP_RELIABILITY = ["Uncertain"]

POSITIVE = ["Low", "Medium", "High"]
NEGATIVE = ["Not detected"]
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}