import socket
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import load_labels, metrics, within_auc, within_auc_parts
from configs import HEAVY, TABLES


SPLIT = "split_pair"
# rank and vector penalty are chosen on validation pairs from these grids
RANKS = [2, 4, 8, 16, 32]
LAMBDA_VEC_GRID = [1e-7, 1e-6, 1e-5, 1e-4]
SEEDS = [1, 2, 3]
# vectors start
INIT_SD = 0.05
# optimiser settings
LR = 0.05
BATCH_SIZE = 4096
EPOCHS = 60
PATIENCE = 8
N_BOOT = 2000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
machine = socket.gethostname()
torch.use_deterministic_algorithms(True, warn_only=True)
print("device", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "", "machine", machine, "torch", torch.__version__)

# 1. data, kept on the device for the whole script
labels, genes, contexts, parts = load_labels(SPLIT)
parts = {name: labels[labels[SPLIT] == name].reset_index(drop=True) for name in ["train", "val", "test"]}
data = {
    name: (
        torch.tensor(df["g"].to_numpy(), device=device),
        torch.tensor(df["c"].to_numpy(), device=device),
        torch.tensor(df["y"].to_numpy(), dtype=torch.float32, device=device),
    )
    for name, df in parts.items()
}
n_train = len(parts["train"])
print("genes", len(genes), "contexts", len(contexts), "train pairs", n_train)

LAMBDA_BIAS = float(pd.read_csv(TABLES / "02_additive_summary.csv")["lambda"].iloc[0])
print("bias lambda from script 02:", LAMBDA_BIAS)


# 2. the model
class MF(nn.Module):
    def __init__(self, n_genes, n_contexts, rank, init_sd):
        super().__init__()
        self.rank = rank
        self.intercept = nn.Parameter(torch.zeros(1))
        self.gene_bias = nn.Embedding(n_genes, 1)
        self.context_bias = nn.Embedding(n_contexts, 1)
        nn.init.zeros_(self.gene_bias.weight)
        nn.init.zeros_(self.context_bias.weight)
        if rank > 0:
            self.gene_vec = nn.Embedding(n_genes, rank)
            self.context_vec = nn.Embedding(n_contexts, rank)
            nn.init.normal_(self.gene_vec.weight, std=init_sd)
            nn.init.normal_(self.context_vec.weight, std=init_sd)

    def forward(self, g, c):
        logit = self.intercept + self.gene_bias(g).squeeze(1) + self.context_bias(c).squeeze(1)
        if self.rank > 0:
            logit = logit + (self.gene_vec(g) * self.context_vec(c)).sum(dim=1)
        return logit

    def penalties(self):
        bias = (self.gene_bias.weight ** 2).sum() + (self.context_bias.weight ** 2).sum()
        if self.rank == 0:
            return bias, torch.zeros((), device=bias.device)
        vec = (self.gene_vec.weight ** 2).sum() + (self.context_vec.weight ** 2).sum()
        return bias, vec


def predict(model, part):
    g, c, _ = data[part]
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(g, c)).cpu().numpy()


def score(model, part):
    return metrics(parts[part]["y"].to_numpy(), predict(model, part), parts[part]["g"].to_numpy(), parts[part]["c"].to_numpy())


def train(rank, lam_vec, seed, init_sd=INIT_SD, epochs=EPOCHS):
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    model = MF(len(genes), len(contexts), rank, init_sd).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g_tr, c_tr, y_tr = data["train"]

    best, best_epoch, best_state, history = np.inf, 0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train, device=device, generator=gen)
        batch_losses = []
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            bias_pen, vec_pen = model.penalties()
            loss = (
                F.binary_cross_entropy_with_logits(model(g_tr[idx], c_tr[idx]), y_tr[idx])
                + 0.5 * LAMBDA_BIAS * bias_pen + 0.5 * lam_vec * vec_pen
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_losses.append(loss.item())
        lr_now = sched.get_last_lr()[0]
        sched.step()

        val = score(model, "val")
        history.append({"epoch": epoch, "lr": lr_now, "train_objective": float(np.mean(batch_losses)), **{f"val_{k}": v for k, v in val.items()}})
        if val["log_loss"] < best:
            best, best_epoch = val["log_loss"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return model, best_epoch, epoch, history


# 3. grid over rank and vector penalty, three seeds each; rank 0 is the additive model trained the same way
print("\n== 3. grid")
settings = [(0, 0.0)] + [(r, lam) for r in RANKS for lam in LAMBDA_VEC_GRID]
runs, histories, states = [], [], {}
t_grid = time.perf_counter()
for rank, lam_vec in settings:
    for seed in SEEDS:
        t0 = time.perf_counter()
        model, best_epoch, last_epoch, history = train(rank, lam_vec, seed)
        val = score(model, "val")
        runs.append({"rank": rank, "lambda_vec": lam_vec, "seed": seed, "best_epoch": best_epoch, "epochs_run": last_epoch,
                     "seconds": time.perf_counter() - t0, **{f"val_{k}": v for k, v in val.items()}})
        histories += [{"rank": rank, "lambda_vec": lam_vec, "seed": seed, **h} for h in history]
        states[(rank, lam_vec, seed)] = {k: v.cpu() for k, v in model.state_dict().items()}
    r = pd.DataFrame(runs[-len(SEEDS):])
    print(f"rank {rank:>2}, lambda_vec {lam_vec:.0e}: val within-gene AUC {r['val_auc_within_gene'].mean():.5f} "
          f"(sd {r['val_auc_within_gene'].std():.5f}), val log loss {r['val_log_loss'].mean():.5f}, "
          f"best epochs {r['best_epoch'].tolist()}, {r['seconds'].sum():.0f} s")
print(f"grid total {time.perf_counter() - t_grid:.0f} s")

runs = pd.DataFrame(runs).assign(machine=machine, device=str(device))
runs.to_csv(TABLES / "03_mf_runs.csv", index=False)
pd.DataFrame(histories).to_csv(TABLES / "03_mf_history.csv", index=False)

by_setting = runs.groupby(["rank", "lambda_vec"])["val_auc_within_gene"].mean()
best_rank, best_lam = by_setting.drop(index=0, level="rank").idxmax()
print(f"\nselected rank {best_rank}, lambda_vec {best_lam:.0e} (mean val within-gene AUC {by_setting[(best_rank, best_lam)]:.5f})")
if best_rank in (min(RANKS), max(RANKS)):
    print("WARNING selected rank is at the edge of the grid")
if best_lam in (min(LAMBDA_VEC_GRID), max(LAMBDA_VEC_GRID)):
    print("WARNING selected lambda_vec is at the edge of the grid")

lbfgs = pd.read_parquet(HEAVY / "02_additive_pred.parquet").query("part == 'val'").merge(parts["val"], on="pair_id")
print(f"reference, script 02 L-BFGS additive: val within-gene AUC {within_auc(lbfgs['y'], lbfgs['p'], lbfgs['g'])[0]:.5f}")

# 4. test set, opened once, for rank 0 and the selected setting
print("\n== 4. test")
y_te, g_te, c_te = (parts["test"][k].to_numpy() for k in ["y", "g", "c"])
test_rows, preds, u_parts = [], [], {}
for rank, lam_vec in [(0, 0.0), (best_rank, best_lam)]:
    label = "additive" if rank == 0 else "mf"
    for seed in SEEDS:
        model = MF(len(genes), len(contexts), rank, INIT_SD).to(device)
        model.load_state_dict(states[(rank, lam_vec, seed)])
        for part in ["val", "test"]:
            preds.append(pd.DataFrame({"pair_id": parts[part]["pair_id"], "part": part, "model": label, "seed": seed, "p": predict(model, part)}))
        p_te = preds[-1]["p"].to_numpy()
        test_rows.append({"model": label, "rank": rank, "lambda_vec": lam_vec, "seed": seed, **{f"test_{k}": v for k, v in metrics(y_te, p_te, g_te, c_te).items()}})
        u_parts[(label, seed)] = within_auc_parts(y_te, p_te, g_te)

test = pd.DataFrame(test_rows)
test.to_csv(TABLES / "03_mf_test.csv", index=False)
pd.concat(preds).to_parquet(HEAVY / "03_mf_pred.parquet", index=False)
print(test.groupby("model")[["test_log_loss", "test_auc", "test_auc_within_gene", "test_auc_within_context"]].agg(["mean", "std"]).round(5).T.to_string())

# gain in within-gene AUC, averaged over seeds, with a bootstrap over genes
w = u_parts[("additive", SEEDS[0])]["w"].to_numpy()
u_add = np.stack([u_parts[("additive", s)]["u"].to_numpy() for s in SEEDS])
u_mf = np.stack([u_parts[("mf", s)]["u"].to_numpy() for s in SEEDS])
gain = ((u_mf - u_add).sum(axis=1) / w.sum()).mean()
rng = np.random.default_rng(1)
boot = np.empty(N_BOOT)
for b in range(N_BOOT):
    idx = rng.integers(0, len(w), len(w))
    boot[b] = ((u_mf[:, idx] - u_add[:, idx]).sum(axis=1) / w[idx].sum()).mean()
lo, hi = np.quantile(boot, [0.025, 0.975])
print(f"\ntest within-gene AUC gain over additive: {gain:+.5f}, 95% gene bootstrap interval {lo:+.5f} to {hi:+.5f}, "
      f"genes scored {len(w)}, bootstrap replicates <= 0: {(boot <= 0).mean():.4f}")
pd.DataFrame({"gain": [gain], "ci_lo": [lo], "ci_hi": [hi], "n_genes": [len(w)], "n_boot": [N_BOOT]}).to_csv(TABLES / "03_mf_gain.csv", index=False)

# learned vectors of the selected setting, seed 1, for scripts 03b, 05b, 06, 07 and 11
state = states[(best_rank, best_lam, SEEDS[0])]
cols = [f"v{i}" for i in range(best_rank)]
pd.concat([
    pd.DataFrame(state["gene_vec.weight"].numpy(), columns=cols).assign(level="gene", id=genes, bias=state["gene_bias.weight"].numpy()[:, 0]),
    pd.DataFrame(state["context_vec.weight"].numpy(), columns=cols).assign(level="context", id=contexts, bias=state["context_bias.weight"].numpy()[:, 0]),
]).to_parquet(HEAVY / "03_mf_vectors.parquet", index=False)


# the same vectors for every seed, so script 11 can test whether the programmes are stable across seeds
pd.concat([
    pd.concat([
        pd.DataFrame(st["gene_vec.weight"].numpy(), columns=cols).assign(level="gene", id=genes),
        pd.DataFrame(st["context_vec.weight"].numpy(), columns=cols).assign(level="context", id=contexts),
    ]).assign(seed=seed)
    for seed in SEEDS for st in [states[(best_rank, best_lam, seed)]]
]).to_parquet(HEAVY / "03_mf_vectors_seeds.parquet", index=False)

