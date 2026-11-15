import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

from configs import HEAVY, TABLES

# Tabula Sapiens v2
# Brain: HBCA
SOURCES = {"ts": Path("D:/tabula/raw"), "brain": Path("D:/tabula/raw/brain")}
MAX_LEVELS = 400
SAMPLE_ROWS = 200
KEYWORDS = ["cell_type", "tissue", "donor", "compartment", "ontology", "free_annotation", "roi", "region",
            "dissection", "supercluster", "class", "cluster", "neuron"]

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 80)

# building one expression profile per HPA context.
# Contexts that share cells share a profile.
BUILD_PROFILES = True
TS_LAYER = "raw_counts"
MIN_CELLS = 50
CHUNK = 10000

# brain bridge
CORTEX_FILES = ["cerebral cortex A1C", "cerebral cortex M1C", "cerebral cortex MTG", "cerebral cortex S1C",
                "cerebral cortex STG", "cerebral cortex V1C"]
HIPPO_REGION_FILE = "Hippocampal CA4-DGC"
HIPPO_NEURON_FILES = ["Hippocampal CA1-3", "Hippocampal CA4-DG", "Hippocampal DG"]
HIPPO_REGION_ROI = "Human CA4C-DGC"
CEREBELLUM_CORTEX_FILES = ["Cerebellum CBL", "Cerebellum CBV"]
GLIA = ["astrocyte", "oligodendrocyte", "oligodendrocyte precursor cell", "central nervous system macrophage"]
GRANULAR_CLUSTERS = ["310", "311", "312", "308", "302"]
MOLECULAR_CLUSTERS = ["303", "304", "305", "306", "307"]


def cell_types(o, types):
    return o["cell_type"].astype(str).isin(types).to_numpy()


def clusters(o, ids):
    return o["cluster_id"].astype(str).isin(ids).to_numpy()


def hippocampal_neurons_elsewhere(o):
    return ((o["cell_type"].astype(str) == "neuron") & o["dissection"].astype(str).str.contains("hippocampus", case=False)
            & (o["roi"].astype(str) != HIPPO_REGION_ROI)).to_numpy()


BRAIN = []
for f in CORTEX_FILES:
    BRAIN += [("Cerebral cortex | neuronal cells", f, lambda o: cell_types(o, ["neuron"])),
              ("Cerebral cortex | glial cells", f, lambda o: cell_types(o, GLIA)),
              ("Cerebral cortex | endothelial cells", f, lambda o: cell_types(o, ["endothelial cell"]))]
BRAIN += [("Hippocampus | neuronal cells", HIPPO_REGION_FILE, lambda o: cell_types(o, ["neuron"])),
          ("Hippocampus | glial cells", HIPPO_REGION_FILE, lambda o: cell_types(o, GLIA))]
BRAIN += [("Hippocampus | neuronal cells", f, hippocampal_neurons_elsewhere) for f in HIPPO_NEURON_FILES]
BRAIN += [("Caudate | neuronal cells", "Caudate", lambda o: cell_types(o, ["neuron"])),
          ("Caudate | glial cells", "Caudate", lambda o: cell_types(o, GLIA))]
for f in CEREBELLUM_CORTEX_FILES:
    BRAIN += [("Cerebellum | cells in granular layer", f, lambda o: clusters(o, GRANULAR_CLUSTERS)),
              ("Cerebellum | cells in molecular layer", f, lambda o: clusters(o, MOLECULAR_CLUSTERS))]


def group_sums(path, key, codes, n_groups):
    with h5py.File(path, "r") as fh:
        node = fh[key]
        if isinstance(node, h5py.Dataset):
            n_rows, n_cols = node.shape
            out = np.zeros((n_groups, n_cols))
            for s in range(0, n_rows, CHUNK):
                c = codes[s:s + CHUNK]
                keep = np.flatnonzero(c >= 0)
                if len(keep):
                    ind = sp.csr_matrix((np.ones(len(keep)), (c[keep], keep)), shape=(n_groups, len(c)))
                    out += ind @ node[s:s + CHUNK].astype(np.float64)
            return out
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in node.attrs.items()}
        enc = attrs.get("encoding-type", attrs.get("h5sparse_format", "") + "_matrix")
        n_rows, n_cols = (int(v) for v in attrs.get("shape", attrs.get("h5sparse_shape")))
        indptr = node["indptr"][:]
        out = np.zeros((n_groups, n_cols))
        if enc == "csr_matrix":
            for s in range(0, n_rows, CHUNK):
                e = min(s + CHUNK, n_rows)
                keep = np.flatnonzero(codes[s:e] >= 0)
                if not len(keep):
                    continue
                a, b = indptr[s], indptr[e]
                block = sp.csr_matrix((node["data"][a:b].astype(np.float64), node["indices"][a:b], indptr[s:e + 1] - a),
                                      shape=(e - s, n_cols))
                ind = sp.csr_matrix((np.ones(len(keep)), (codes[s:e][keep], keep)), shape=(n_groups, e - s))
                out += (ind @ block).toarray()
        elif enc == "csc_matrix":
            for s in range(0, n_cols, CHUNK):
                e = min(s + CHUNK, n_cols)
                a, b = indptr[s], indptr[e]
                rows = node["indices"][a:b]
                col = np.repeat(np.arange(s, e), np.diff(indptr[s:e + 1]))
                c = codes[rows]
                keep = c >= 0
                out += np.bincount(c[keep] * n_cols + col[keep], weights=node["data"][a:b][keep],
                                   minlength=n_groups * n_cols).reshape(n_groups, n_cols)
        else:
            raise ValueError(f"{path} {key}: unknown encoding {enc}")
        return out


def sum_units(path, key, units, gene_ids, keep_genes):
    codes = np.full(len(next(iter(units.values()))), -1)
    for i, m in enumerate(units.values()):
        assert (codes[m] == -1).all(), f"{path}: overlapping cell selections"
        codes[m] = i
    sums = group_sums(path, key, codes, len(units))
    df = pd.DataFrame(sums.T[keep_genes], index=gene_ids[keep_genes], columns=list(units))
    return df.groupby(level=0).sum()


if BUILD_PROFILES:
    counts, meta = {}, []

    def add(pid, col, n_cells, donors, source, file, cells):
        counts[pid] = counts[pid].add(col, fill_value=0) if pid in counts else col
        meta.append({"profile_id": pid, "source": source, "file": file, "cells": cells, "n_cells": n_cells, "donors": set(donors)})

    draft = pd.read_csv("../external/context_bridge_draft.csv")
    draft["context"] = draft["hpa_tissue"] + " | " + draft["hpa_cell_type"]
    for ctx in ["Skin | keratinocytes", "Skin | epidermal cells"]:
        draft.loc[draft["context"] == ctx, ["ts_file", "ts_cell_types", "status", "note"]] = [
            "tongue.h5ad", "stratified squamous epithelial cell", "proxy", "no epidermis in the skin file"]
    ts = draft[draft["status"].isin(["match", "proxy"])].copy()
    ts["types"] = ts["ts_cell_types"].str.split("; ").map(tuple)
    ts["profile_id"] = "ts:" + ts["ts_file"].str.removesuffix(".h5ad") + ":" + ts["ts_cell_types"]
    for fname, rows in ts.groupby("ts_file"):
        path = SOURCES["ts"] / fname
        a = ad.read_h5ad(path, backed="r")
        obs, var = a.obs, a.var
        a.file.close()
        with h5py.File(path, "r") as fh:
            assert f"layers/{TS_LAYER}" in fh, f"{fname}: no layer {TS_LAYER}"
        is10x = (obs["method"].astype(str) == "10X").to_numpy()
        cls = obs["cell_ontology_class"].astype(str).to_numpy()
        wanted = sorted({t for types in rows["types"] for t in types})
        missing = [t for t in wanted if not (cls == t).any()]
        assert not missing, f"{fname}: {missing} not in cell_ontology_class"
        units = {t: is10x & (cls == t) for t in wanted}
        gene_ids = var["ensembl_id"].astype(str).str.split(".").str[0].to_numpy()
        keep_genes = ~var["ercc"].astype(bool).to_numpy()
        sums = sum_units(path, f"layers/{TS_LAYER}", units, gene_ids, keep_genes)
        donors = obs["donor"].astype(str).to_numpy()
        for pid, types in rows.drop_duplicates("profile_id")[["profile_id", "types"]].itertuples(index=False):
            m = np.logical_or.reduce([units[t] for t in types])
            add(pid, sums[list(types)].sum(axis=1), int(m.sum()), donors[m], "ts", fname, "; ".join(types))
        print(f"{fname}: {len(rows.drop_duplicates('profile_id'))} profiles, 10X cells {int(is10x.sum())} of {len(is10x)}")

    # brain atlas: raw counts in X
    brain_files = sorted({f for _, f, _ in BRAIN})
    for f in brain_files:
        path = SOURCES["brain"] / f"{f}.h5ad"
        a = ad.read_h5ad(path, backed="r")
        obs, var = a.obs, a.var
        a.file.close()
        rules = [(ctx, rule) for ctx, ff, rule in BRAIN if ff == f]
        units = {ctx: rule(obs) for ctx, rule in rules}
        gene_ids = pd.Index(var.index).astype(str).str.split(".").str[0].to_numpy()
        sums = sum_units(path, "X", units, gene_ids, np.ones(len(gene_ids), bool))
        donors = obs["donor_id"].astype(str).to_numpy()
        for ctx, m in units.items():
            add("brain:" + ctx, sums[ctx], int(m.sum()), donors[m], "brain", f, ctx)
        print(f"{f}.h5ad: " + ", ".join(f"{ctx.split(' | ')[1]} {int(m.sum())}" for ctx, m in units.items()))

    meta = pd.DataFrame(meta).groupby("profile_id").agg(
        source=("source", "first"), files=("file", lambda x: "; ".join(sorted(set(x)))), cells=("cells", "first"),
        n_cells=("n_cells", "sum"), donors=("donors", lambda x: len(set().union(*x)))).reset_index()
    raw = pd.DataFrame(counts)
    meta["library_size"] = meta["profile_id"].map(raw.sum(axis=0))  # over each atlas's own genes, before intersecting
    meta["kept"] = meta["n_cells"] >= MIN_CELLS

    ts_genes = raw.loc[:, meta.loc[meta["source"] == "ts", "profile_id"]].notna().all(axis=1)
    brain_genes = raw.loc[:, meta.loc[meta["source"] == "brain", "profile_id"]].notna().all(axis=1)
    shared = raw.index[ts_genes & brain_genes]
    kept = meta.loc[meta["kept"], "profile_id"].tolist()
    profiles = np.log1p(raw.loc[shared, kept] / meta.set_index("profile_id").loc[kept, "library_size"] * 1e6)
    profiles.astype(np.float32).rename_axis("gene").to_parquet(HEAVY / "04_profiles.parquet")
    print(f"\ngenes: Tabula Sapiens {int(ts_genes.sum())}, brain {int(brain_genes.sum())}, both {len(shared)}")

    contexts = pd.read_parquet(HEAVY / "01_labels.parquet", columns=["context", "tissue", "cell_type"]).drop_duplicates("context")
    brain_ctx = pd.DataFrame({"context": sorted({c for c, _, _ in BRAIN})})
    brain_ctx["profile_id"], brain_ctx["status"] = "brain:" + brain_ctx["context"], "match"
    bridge = contexts.merge(pd.concat([ts[["context", "profile_id", "status", "note"]], brain_ctx]), on="context", how="left")
    bridge = bridge.merge(meta[["profile_id", "n_cells", "donors"]], on="profile_id", how="left")
    bridge.loc[bridge["profile_id"].isin(meta.loc[~meta["kept"], "profile_id"]), "status"] = "too few cells"
    bridge["status"] = bridge["status"].fillna("none")
    bridge.loc[bridge["status"].isin(["none", "too few cells"]), "profile_id"] = None
    bridge["profile_shared_by"] = bridge.groupby("profile_id")["context"].transform("size")
    bridge.to_csv(TABLES / "04_context_bridge.csv", index=False)
    meta.to_csv(TABLES / "04_profile_meta.csv", index=False)
