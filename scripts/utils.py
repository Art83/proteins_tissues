import numpy as np
import pandas as pd


def within_auc_parts(y, p, groups):
    # Per group, u = number of positive-negative pairs ranked correctly (ties count half) and w = n_pos * n_neg.
    df = pd.DataFrame({"y": np.asarray(y).astype(np.int64), "p": np.asarray(p), "grp": np.asarray(groups)})
    df["rank"] = df.groupby("grp")["p"].rank()
    per = df.groupby("grp").agg(n=("y", "size"), n_pos=("y", "sum"))
    per["rank_pos"] = df[df["y"] == 1].groupby("grp")["rank"].sum()
    per = per[(per["n_pos"] > 0) & (per["n_pos"] < per["n"])]
    u = per["rank_pos"] - per["n_pos"] * (per["n_pos"] + 1) / 2
    w = per["n_pos"] * (per["n"] - per["n_pos"])
    return pd.DataFrame({"u": u, "w": w})


def within_auc(y, p, groups):
    # AUC over positive-negative pairs that share a group, i.e. the w-weighted mean of per-group AUCs
    parts = within_auc_parts(y, p, groups)
    auc = float(parts["u"].sum() / parts["w"].sum())
    assert 0 <= auc <= 1, auc
    return auc, len(parts)
