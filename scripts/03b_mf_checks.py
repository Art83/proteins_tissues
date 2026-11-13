import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from utils import load_labels, within_auc
from configs import (TABLES, HEAVY, FIGURES)

SPLIT = "split_pair"
K_GRID = [1, 3, 5, 10, 20, 40, 60, 70, 80]
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_red", ["#2a78d6", "#f0efec", "#e34948"])

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)

labels, genes, contexts, parts = load_labels(SPLIT)
parts = {name: labels[labels[SPLIT] == name].reset_index(drop=True) for name in ["train", "val", "test"]}
n_genes, n_ctx = len(genes), len(contexts)

# 1. how similar two contexts are, from training labels with each gene's overall detection rate removed
Y = np.full((n_genes, n_ctx), np.nan)
Y[parts["train"]["g"], parts["train"]["c"]] = parts["train"]["y"]
gene_rate = np.nanmean(Y, axis=1)
R = Y - gene_rate[:, None]
varies = np.nanstd(Y, axis=1) > 0
S = pd.DataFrame(R[varies]).corr().to_numpy(copy=True)
np.fill_diagonal(S, np.nan)
print(f"== 1. context similarity from {varies.sum()} genes whose training labels vary")

nearest = np.nanargmax(S, axis=1)
neighbours = pd.DataFrame({
    "context": contexts,
    "nearest": contexts[nearest],
    "similarity": S[np.arange(n_ctx), nearest],
    "second": contexts[np.argsort(-np.nan_to_num(S, nan=-np.inf), axis=1)[:, 1]],
}).sort_values("similarity", ascending=False)
neighbours.to_csv(TABLES / "03b_context_neighbours.csv", index=False)
print("closest relative contexts")
print(neighbours.head(15).round(3).to_string(index=False))
print("least connected contexts")
print(neighbours.tail(5).round(3).to_string(index=False))


# 2. nearest-context baseline: predict a held-out pair from the gene's labels in its K most similar training contexts
def knn_scores(df, k):
    scores = np.empty(len(df))
    for c in range(n_ctx):
        rows = np.flatnonzero(df["c"].to_numpy() == c)
        if len(rows) == 0:
            continue
        order = np.argsort(-np.nan_to_num(S[c], nan=-np.inf))[: n_ctx - 1]
        yg = Y[df["g"].to_numpy()[rows]][:, order]
        observed = ~np.isnan(yg)
        take = observed & (np.cumsum(observed, axis=1) <= k)
        w = np.clip(S[c, order], 1e-6, None) * take
        num = (np.nan_to_num(yg) * w).sum(axis=1)
        den = w.sum(axis=1)
        fallback = gene_rate[df["g"].to_numpy()[rows]]
        scores[rows] = np.where(den > 0, num / np.where(den > 0, den, 1), fallback)
    return scores
print("\n== 2. nearest-context baseline, within-gene AUC")
grid = []
for k in K_GRID:
    auc, n = within_auc(parts["val"]["y"], knn_scores(parts["val"], k), parts["val"]["g"])
    grid.append({"k": k, "val_auc_within_gene": auc})
    print(f"k {k:>2}: val {auc:.5f}")
grid = pd.DataFrame(grid)
best_k = int(grid.loc[grid["val_auc_within_gene"].idxmax(), "k"])
print("selected k", best_k)

test = parts["test"].copy()
test["knn"] = knn_scores(test, best_k)
pred = pd.read_parquet(HEAVY / "03_mf_pred.parquet").query("part == 'test'")
wide = pred.pivot_table(index="pair_id", columns=["model", "seed"], values="p")
seeds = sorted(pred["seed"].unique())
for model in ["additive", "mf"]:
    for s in seeds:
        test[f"{model}_{s}"] = test["pair_id"].map(wide[(model, s)])


def aucs(df):
    out = {"knn": within_auc(df["y"], df["knn"], df["g"])[0]}
    for model in ["additive", "mf"]:
        out[model] = np.mean([within_auc(df["y"], df[f"{model}_{s}"], df["g"])[0] for s in seeds])
    return out


overall = aucs(test)
share = (overall["knn"] - overall["additive"]) / (overall["mf"] - overall["additive"])
print(f"test within-gene AUC: additive {overall['additive']:.5f}, nearest-context (k {best_k}) {overall['knn']:.5f}, "
      f"matrix factorisation {overall['mf']:.5f}")
print(f"share of the factorisation's gain over additive reached by the nearest-context baseline: {share:.3f}")
grid.assign(selected_k=best_k, test_additive=overall["additive"], test_knn=overall["knn"], test_mf=overall["mf"],
            share_of_gain=share).to_csv(TABLES / "03b_knn.csv", index=False)

# 3. does the gain depend on the held-out context having a close relative?
print("\n== 3. test within-gene AUC by how similar each context's nearest relative is (thirds of the 75 contexts)")
max_sim = np.nanmax(S, axis=1)
third = pd.qcut(max_sim, 3, labels=["low", "middle", "high"])
test["relative"] = np.asarray(third)[test["c"]]
rows = []
for level in ["low", "middle", "high"]:
    sub = test[test["relative"] == level]
    ctx = max_sim[np.asarray(third) == level]
    rows.append({"relative_similarity": level, "contexts": len(ctx), "similarity_range": f"{ctx.min():.2f} to {ctx.max():.2f}",
                 "test_pairs": len(sub), **aucs(sub)})
by_relative = pd.DataFrame(rows)
by_relative["mf_gain"] = by_relative["mf"] - by_relative["additive"]
by_relative.to_csv(TABLES / "03b_by_relative.csv", index=False)
print(by_relative.round(4).to_string(index=False))

# 4. map of the learned context vectors, ordered by average-linkage clustering for display only
print("\n== 4. context vectors")
vec = pd.read_parquet(HEAVY / "03_mf_vectors.parquet").query("level == 'context'").set_index("id").loc[contexts]
V = vec[[c for c in vec.columns if c.startswith("v")]].to_numpy()
Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
C = Vn @ Vn.T
iu = np.triu_indices(n_ctx, 1)
rho = spearmanr(C[iu], S[iu]).statistic
print(f"rank {V.shape[1]}; agreement between vector similarity and label similarity over {len(iu[0])} context pairs: Spearman {rho:.3f}")

Z = linkage(squareform(np.clip(1 - C, 0, None), checks=False), method="average")
order = leaves_list(Z)

fig, ax = plt.subplots(figsize=(13, 12))
im = ax.imshow(C[np.ix_(order, order)], cmap=DIVERGING, vmin=-1, vmax=1, interpolation="nearest")
ax.set_xticks(range(n_ctx))
ax.set_yticks(range(n_ctx))
ax.set_xticklabels(contexts[order], rotation=90, fontsize=5.5, color="#3d3d3a")
ax.set_yticklabels(contexts[order], fontsize=5.5, color="#3d3d3a")
ax.tick_params(length=0)
for spine in ax.spines.values():
    spine.set_visible(False)
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
cbar.set_label("cosine similarity of learned context vectors", fontsize=8, color="#3d3d3a")
cbar.ax.tick_params(labelsize=7, colors="#3d3d3a")
cbar.outline.set_visible(False)
ax.set_title(f"Context vectors from matrix factorisation (rank {V.shape[1]}), ordered by average-linkage clustering",
             fontsize=10, color="#1f1f1d", loc="left")
fig.tight_layout()
fig.savefig(FIGURES / "03b_context_map.png", dpi=300, facecolor="#fcfcfb")
