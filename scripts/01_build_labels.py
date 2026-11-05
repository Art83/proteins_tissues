import numpy as np
import pandas as pd
from configs import (TABLES, ROOT, HEAVY, MIN_GENES, DROP_CONTEXTS,
                     DROP_RELIABILITY, POSITIVE, NEGATIVE, SPLIT_FRACTIONS)

# HPA dataset
HPA_ZIP = ROOT / "external" / "normal_ihc_data.tsv.zip"

SEED = 1

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)

key = ["Gene", "Tissue", "Cell type"]

hpa = pd.read_csv(HPA_ZIP, sep="\t")
print("rows read", len(hpa))

hpa = hpa.dropna(subset=["Level"])
print("rows with a level", len(hpa))

print("\n== Ascending/Descending rows vs gradient contexts")
print(pd.crosstab(
    hpa["Cell type"].str.contains("Gradient", regex=False).rename("gradient_context"),
    hpa["Level"].isin(["Ascending", "Descending"]).rename("ascending_or_descending"),
))

hpa = hpa[hpa["Level"].isin(POSITIVE + NEGATIVE)]
print("\nrows with a usable level", len(hpa))

shared = hpa.groupby("Gene name")["Gene"].nunique()
print("\n== symbols shared by more than one Ensembl gene")
print(shared[shared > 1].to_string())

# contexts
contexts = hpa.groupby(["Tissue", "Cell type"])["Gene"].nunique().rename("n_genes").reset_index()
drop_set = set(DROP_CONTEXTS)
contexts["dropped_by_rule"] = [(t, c) in drop_set for t, c in zip(contexts["Tissue"], contexts["Cell type"])]
contexts["kept"] = (contexts["n_genes"] >= MIN_GENES) & ~contexts["dropped_by_rule"]
contexts = contexts.sort_values("n_genes", ascending=False)
contexts.to_csv(TABLES / "01_contexts.csv", index=False)

kept = contexts[contexts["kept"]]
print("\n== contexts", len(contexts), "kept", len(kept))
print("dropped by rule")
print(contexts[contexts["dropped_by_rule"]].to_string(index=False))
print("shallowest kept")
print(kept.tail(5).to_string(index=False))
print("deepest excluded by depth")
print(contexts[~contexts["kept"] & ~contexts["dropped_by_rule"]].head(3).to_string(index=False))

hpa = hpa.merge(kept[["Tissue", "Cell type"]], on=["Tissue", "Cell type"])
print("\nrows in kept contexts", len(hpa))

# reliability
print("\n== reliability")
print("genes with more than one reliability value:", int((hpa.groupby("Gene")["Reliability"].nunique() > 1).sum()))
hpa["y"] = hpa["Level"].isin(POSITIVE).astype("int8")
print("positive rate, all reliabilities", round(hpa["y"].mean(), 4), "rows", len(hpa), "genes", hpa["Gene"].nunique())
hpa = hpa[~hpa["Reliability"].isin(DROP_RELIABILITY)]
print("positive rate, Uncertain removed ", round(hpa["y"].mean(), 4), "rows", len(hpa), "genes", hpa["Gene"].nunique())

# paired samples
dup = hpa[hpa.duplicated(key, keep=False)].sort_values(key + ["IHC tissue name"])
assert dup.groupby(key).size().max() == 2
paired = (
    dup.groupby(key)
    .agg(
        sample_a=("IHC tissue name", "first"), sample_b=("IHC tissue name", "last"),
        level_a=("Level", "first"), level_b=("Level", "last"),
        y_a=("y", "first"), y_b=("y", "last"),
    )
    .reset_index()
)
paired.to_csv(TABLES / "01_paired_samples.csv", index=False)
print("\n== paired samples, binary disagreement by context")
print(
    paired.assign(differs=paired["y_a"] != paired["y_b"])
    .groupby(["Tissue", "Cell type"])
    .agg(n_genes=("Gene", "size"), frac_y_differs=("differs", "mean"))
    .round(4)
    .to_string()
)

# collapse to one label per gene x context: present if either of two stained samples of the same tissue is positive
labels = (
    hpa.groupby(key, as_index=False)
    .agg(gene_name=("Gene name", "first"), y=("y", "max"))
    .rename(columns={"Gene": "gene", "Tissue": "tissue", "Cell type": "cell_type"})
)
labels["context"] = labels["tissue"] + " | " + labels["cell_type"]
labels.insert(0, "pair_id", np.arange(len(labels)))

n_genes = labels["gene"].nunique()
n_contexts = labels["context"].nunique()
print("\n== labels")
print("genes", n_genes, "contexts", n_contexts, "pairs", len(labels))
print("grid filled", round(len(labels) / (n_genes * n_contexts), 4))
print("positive rate", round(labels["y"].mean(), 4))

# splits
rng = np.random.default_rng(SEED)
names = list(SPLIT_FRACTIONS)
cum = np.cumsum(list(SPLIT_FRACTIONS.values()))[:-1]
units = {
    "split_pair": ("pair_id", labels["pair_id"].to_numpy()),
    "split_gene": ("gene", np.sort(labels["gene"].unique())),
    "split_context": ("context", np.sort(labels["context"].unique())),
}
for split_col, (unit_col, items) in units.items():
    shuffled = rng.permutation(items)
    assigned = np.repeat(names, np.diff(np.r_[0, np.round(cum * len(items)).astype(int), len(items)]))
    labels[split_col] = labels[unit_col].map(pd.Series(assigned, index=shuffled))

print("\n== splits")
for split_col in units:
    print(
        labels.groupby(split_col)
        .agg(pairs=("y", "size"), genes=("gene", "nunique"), contexts=("context", "nunique"), positive_rate=("y", "mean"))
        .round(4)
        .to_string()
    )
    print()

train = labels[labels["split_pair"] == "train"]
for part in ["val", "test"]:
    sub = labels[labels["split_pair"] == part]
    print(
        f"pair split {part}: gene unseen in train", int((~sub["gene"].isin(train["gene"])).sum()),
        "context unseen in train", int((~sub["context"].isin(train["context"])).sum()),
    )

labels.to_parquet(HEAVY / "01_labels.parquet", index=False)
