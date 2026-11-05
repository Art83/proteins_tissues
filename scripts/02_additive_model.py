import socket
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss, roc_auc_score

from utils import within_auc
from configs import (TABLES, HEAVY)

SPLIT = "split_pair"
SEED = 1
LAMBDA_GRID = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
LBFGS_MAX_ITER = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
machine = socket.gethostname()
torch.manual_seed(SEED)
print("device", device, "machine", machine, "torch", torch.__version__)

# 1. data as integer indices
labels = pd.read_parquet(HEAVY / "01_labels.parquet")
genes = np.sort(labels["gene"].unique())
contexts = np.sort(labels["context"].unique())
labels["g"] = np.searchsorted(genes, labels["gene"])
labels["c"] = np.searchsorted(contexts, labels["context"])
print("genes", len(genes), "contexts", len(contexts))

parts = {name: labels[labels[SPLIT] == name] for name in ["train", "val", "test"]}
for name, df in parts.items():
    print(f"{name:>5}: {len(df)} pairs, positive rate {df['y'].mean():.4f}")

train_rate = parts["train"].groupby("g")["y"].mean()
ctx_rate = parts["train"].groupby("c")["y"].mean()
print("\ngenes never detected in train:", int((train_rate == 0).sum()), " always detected:", int((train_rate == 1).sum()))
print("contexts never detected in train:", int((ctx_rate == 0).sum()), " always detected:", int((ctx_rate == 1).sum()))


def as_tensors(df, dtype):
    return (
        torch.tensor(df["g"].to_numpy(), device=device),
        torch.tensor(df["c"].to_numpy(), device=device),
        torch.tensor(df["y"].to_numpy(), dtype=dtype, device=device),
    )


# 2. the model
class Additive(nn.Module):
    def __init__(self, n_genes, n_contexts, use_gene=True, use_context=True, dtype=torch.float32):
        super().__init__()
        self.intercept = nn.Parameter(torch.zeros(1, dtype=dtype))
        self.gene = nn.Embedding(n_genes, 1, dtype=dtype) if use_gene else None
        self.context = nn.Embedding(n_contexts, 1, dtype=dtype) if use_context else None
        for emb in [self.gene, self.context]:
            if emb is not None:
                nn.init.zeros_(emb.weight)

    def forward(self, g, c):
        logit = self.intercept.expand(len(g))
        if self.gene is not None:
            logit = logit + self.gene(g).squeeze(1)
        if self.context is not None:
            logit = logit + self.context(c).squeeze(1)
        return logit

    def penalty(self):
        return sum((emb.weight ** 2).sum() for emb in [self.gene, self.context] if emb is not None)


def objective(model, lam, g, c, y):
    return F.binary_cross_entropy_with_logits(model(g, c), y) + 0.5 * lam * model.penalty()


def fit_lbfgs(model, lam, g, c, y):
    opt = torch.optim.LBFGS(
        model.parameters(), lr=1, max_iter=LBFGS_MAX_ITER, history_size=20,
        tolerance_grad=1e-10, tolerance_change=1e-14, line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        loss = objective(model, lam, g, c, y)
        loss.backward()
        return loss

    opt.step(closure)
    loss = closure()
    max_grad = max(p.grad.abs().max().item() for p in model.parameters())
    n_iter = opt.state[model.intercept]["n_iter"]
    return loss.item(), max_grad, n_iter


def evaluate(model, g, c, y):
    with torch.no_grad():
        p = torch.sigmoid(model(g, c)).cpu().numpy()
    y = y.cpu().numpy()
    return p, log_loss(y, p), roc_auc_score(y, p)


g_tr, c_tr, y_tr = as_tensors(parts["train"], torch.float64)

# 3. lambda chosen on validation, and the main-effect decomposition
print("\n== 4. lambda grid, float64 L-BFGS, selected on validation log loss")
g_va, c_va, y_va = as_tensors(parts["val"], torch.float64)
g_te, c_te, y_te = as_tensors(parts["test"], torch.float64)

grid = []
for lam in LAMBDA_GRID:
    m = Additive(len(genes), len(contexts), dtype=torch.float64).to(device)
    t0 = time.perf_counter()
    _, max_grad, n_iter = fit_lbfgs(m, lam, g_tr, c_tr, y_tr)
    _, ll, auc = evaluate(m, g_va, c_va, y_va)
    grid.append({"lambda": lam, "val_log_loss": ll, "val_auc": auc, "iterations": n_iter, "max_grad": max_grad})
    print(f"lambda {lam:.0e}: val log loss {ll:.5f}, val AUC {auc:.5f}, iterations {n_iter}, max |gradient| {max_grad:.1e}, {time.perf_counter() - t0:.1f} s")
grid = pd.DataFrame(grid)
best_lambda = float(grid.loc[grid["val_log_loss"].idxmin(), "lambda"])
print("selected lambda", best_lambda)
if best_lambda in (min(LAMBDA_GRID), max(LAMBDA_GRID)):
    print("WARNING selected lambda is at the edge of the grid")

rows_out = []
fitted = {}
for name, use_gene, use_context in [("intercept", False, False), ("context", False, True), ("gene", True, False), ("gene+context", True, True)]:
    m = Additive(len(genes), len(contexts), use_gene, use_context, dtype=torch.float64).to(device)
    fit_lbfgs(m, best_lambda, g_tr, c_tr, y_tr)
    fitted[name] = m
    row = {"model": name, "lambda": best_lambda, "machine": machine, "device": str(device)}
    for part, (g, c, y) in {"val": (g_va, c_va, y_va), "test": (g_te, c_te, y_te)}.items():
        p, row[f"{part}_log_loss"], row[f"{part}_auc"] = evaluate(m, g, c, y)
        row[f"{part}_auc_within_gene"], row[f"{part}_n_genes_scored"] = within_auc(y.cpu().numpy(), p, g.cpu().numpy())
        row[f"{part}_auc_within_context"], _ = within_auc(y.cpu().numpy(), p, c.cpu().numpy())
    rows_out.append(row)
summary = pd.DataFrame(rows_out)
print("\n" + summary.drop(columns=["lambda", "machine", "device"]).round(5).T.to_string(header=False))

best = fitted["gene+context"]
p_va, _, _ = evaluate(best, g_va, c_va, y_va)
p_te, _, _ = evaluate(best, g_te, c_te, y_te)
pd.concat([
    pd.DataFrame({"pair_id": parts["val"]["pair_id"].to_numpy(), "part": "val", "p": p_va}),
    pd.DataFrame({"pair_id": parts["test"]["pair_id"].to_numpy(), "part": "test", "p": p_te}),
]).to_parquet(HEAVY / "02_additive_pred.parquet", index=False)
pd.concat([
    pd.DataFrame({"level": "gene", "id": genes, "bias": best.gene.weight.detach().cpu().numpy()[:, 0]}),
    pd.DataFrame({"level": "context", "id": contexts, "bias": best.context.weight.detach().cpu().numpy()[:, 0]}),
    pd.DataFrame({"level": "intercept", "id": ["intercept"], "bias": best.intercept.detach().cpu().numpy()}),
]).to_parquet(HEAVY / "02_additive_coefs.parquet", index=False)
grid.to_csv(TABLES / "02_lambda_grid.csv", index=False)
summary.to_csv(TABLES / "02_additive_summary.csv", index=False)
