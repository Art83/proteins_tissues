import numpy as np
import pandas as pd

from configs import ROOT, TABLES, HEAVY

SC_ZIP = ROOT / "external" / "rna_single_cell_cluster.tsv.zip"
CHUNK = 2_000_000

pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 90)


BRAIN_NEURONS = ["brain excitatory neurons", "brain inhibitory neurons", "other brain neurons"]
BRAIN_GLIA = ["astrocytes", "oligodendrocytes", "oligodendrocyte progenitor cells", "microglia"]
SQUAMOUS = ["esophageal basal cells", "esophageal suprabasal cells", "esophageal apical cells"]
AIRWAY = ["respiratory basal cells", "respiratory ciliated cells", "respiratory secretory cells"]
KERATINOCYTES = ["basal keratinocytes", "suprabasal keratinocytes"]
BRIDGE = [
    ("Adipose tissue", "adipocytes", "adipose tissue", ["adipocytes"], "match", ""),
    ("Adrenal gland", "glandular cells", "adrenal gland", ["adrenal cortex cells"], "match", "medulla cells left out"),
    ("Appendix", "glandular cells", "colon", ["colonocytes", "goblet cells"], "proxy", "no appendix; same profile as Colon"),
    ("Appendix", "lymphoid tissue", "lymph node", ["b-cells", "t-cells"], "proxy", "no appendix; shares cells with the lymph node contexts"),
    ("Bone marrow", "hematopoietic cells", "bone marrow",
     ["hematopoietic stem cells", "multipotent progenitors", "megakaryocyte-erythroid progenitors", "erythrocyte progenitors",
      "megakaryocyte progenitors", "monocyte progenitors", "neutrophil progenitors"], "match", "progenitors, as in the Tabula Sapiens bridge"),
    ("Breast", "adipocytes", "adipose tissue", ["adipocytes"], "proxy", "no breast adipocytes; same profile as Adipose tissue"),
    ("Breast", "glandular cells", "breast", ["breast hormone-responsive cells", "breast secretory cells"], "match", "lactating cells left out"),
    ("Breast", "myoepithelial cells", "breast", ["breast myoepithelial cells"], "match", ""),
    ("Bronchus", "respiratory epithelial cells", "lung", AIRWAY, "match", "airway epithelium from the lung data"),
    ("Nasopharynx", "respiratory epithelial cells", "lung", AIRWAY, "proxy", "same profile as Bronchus"),
    ("Caudate", "neuronal cells", "brain neurons", BRAIN_NEURONS, "proxy", "no brain regions; one neuronal profile for all regions"),
    ("Hippocampus", "neuronal cells", "brain neurons", BRAIN_NEURONS, "proxy", "no brain regions; one neuronal profile for all regions"),
    ("Cerebral cortex", "neuronal cells", "brain neurons", BRAIN_NEURONS, "proxy", "no brain regions; one neuronal profile for all regions"),
    ("Caudate", "glial cells", "brain non-neurons", BRAIN_GLIA, "proxy", "no brain regions; one glial profile for all regions"),
    ("Hippocampus", "glial cells", "brain non-neurons", BRAIN_GLIA, "proxy", "no brain regions; one glial profile for all regions"),
    ("Cerebral cortex", "glial cells", "brain non-neurons", BRAIN_GLIA, "proxy", "no brain regions; one glial profile for all regions"),
    ("Cerebral cortex", "endothelial cells", "brain non-neurons", ["vascular endothelial cells"], "match", "brain endothelium, region not given"),
    ("Cervix", "squamous epithelial cells", "esophagus", SQUAMOUS, "proxy", "non-keratinised squamous epithelium"),
    ("Vagina", "squamous epithelial cells", "esophagus", SQUAMOUS, "proxy", "non-keratinised squamous epithelium"),
    ("Tonsil", "squamous epithelial cells", "esophagus", SQUAMOUS, "proxy", "non-keratinised squamous epithelium"),
    ("Esophagus", "squamous epithelial cells", "esophagus", SQUAMOUS, "match", ""),
    ("Oral mucosa", "squamous epithelial cells", "tongue", KERATINOCYTES, "match", "oral epithelium from tongue"),
    ("Colon", "glandular cells", "colon", ["colonocytes", "goblet cells"], "match", ""),
    ("Rectum", "glandular cells", "rectum", ["colonocytes", "goblet cells"], "match", ""),
    ("Duodenum", "glandular cells", "small intestine", ["enterocytes", "goblet cells"], "proxy", "segment not given; same profile as Small intestine"),
    ("Small intestine", "glandular cells", "small intestine", ["enterocytes", "goblet cells"], "match", ""),
    ("Endometrium", "cells in endometrial stroma", "endometrium", ["endometrial stromal cells"], "match", ""),
    ("Endometrium", "glandular cells", "endometrium", ["endometrial glandular cells"], "match", "luminal and secretory cells left out"),
    ("Epididymis", "glandular cells", "epididymis", ["epididymal principal cells", "epididymal basal cells", "epididymal clear cells"], "match",
     "duct epithelium; efferent duct cells left out"),
    ("Fallopian tube", "glandular cells", "fallopian tube", ["fallopian secretory cells", "fallopian tube ciliated cells"], "match", ""),
    ("Heart muscle", "cardiomyocytes", "heart muscle", ["cardiomyocytes"], "match", ""),
    ("Kidney", "cells in glomeruli", "kidney", ["podocytes", "mesangial cells", "parietal epithelial cells"], "match", ""),
    ("Kidney", "cells in tubules", "kidney",
     ["proximal tubule cells", "loop of henle epithelial cells", "distal convoluted tubule cells", "renal connecting tubule cells",
      "renal collecting duct principal cells", "renal collecting duct intercalated cells"], "match", ""),
    ("Liver", "cholangiocytes", "liver", ["cholangiocytes"], "match", ""),
    ("Liver", "hepatocytes", "liver", ["hepatocytes"], "match", ""),
    ("Lung", "alveolar cells", "lung", ["alveolar cells type 1", "alveolar cells type 2"], "match", ""),
    ("Lung", "macrophages", "lung", ["macrophages"], "match", ""),
    ("Lymph node", "germinal center cells", "lymph node", ["b-cells"], "match", ""),
    ("Lymph node", "non-germinal center cells", "lymph node", ["t-cells", "nk-cells"], "match", ""),
    ("Tonsil", "germinal center cells", "lymph node", ["b-cells"], "proxy", "no tonsil; same profile as Lymph node"),
    ("Tonsil", "non-germinal center cells", "lymph node", ["t-cells", "nk-cells"], "proxy", "no tonsil; same profile as Lymph node"),
    ("Ovary", "follicle cells", "ovary", ["granulosa cells"], "match", ""),
    ("Ovary", "ovarian stroma cells", "ovary", ["ovarian stromal cells"], "match", ""),
    ("Pancreas", "exocrine glandular cells", "pancreas", ["pancreatic acinar cells"], "match", "duct cells left out"),
    ("Pancreas", "pancreatic endocrine cells", "pancreas", ["pancreatic islet cells"], "match", ""),
    ("Placenta", "decidual cells", "placenta", ["decidual stromal cells"], "match", ""),
    ("Placenta", "trophoblastic cells", "placenta", ["cytotrophoblasts", "syncytiotrophoblasts", "extravillous trophoblasts", "migrating cytotrophoblasts"],
     "match", ""),
    ("Prostate", "glandular cells", "prostate", ["prostatic glandular cells", "basal prostatic cells"], "match", ""),
    ("Salivary gland", "glandular cells", "salivary gland", ["salivary acinar cells"], "match", ""),
    ("Skeletal muscle", "myocytes", "skeletal muscle", ["myonuclei"], "match", ""),
    ("Skin", "epidermal cells", "skin", KERATINOCYTES, "match", "same profile as Skin keratinocytes"),
    ("Skin", "keratinocytes", "skin", KERATINOCYTES, "match", ""),
    ("Skin", "fibroblasts", "skin", ["fibroblasts"], "match", ""),
    ("Skin", "melanocytes", "skin", ["melanocytes"], "match", ""),
    ("Smooth muscle", "smooth muscle cells", "urinary bladder", ["smooth muscle cells"], "proxy", "visceral smooth muscle from bladder"),
    ("Soft tissue", "fibroblasts", "adipose tissue", ["fibro-adipogenic progenitors"], "proxy", "no soft tissue; adipose stromal fibroblasts"),
    ("Spleen", "cells in red pulp", "spleen", ["macrophages", "erythrocytes"], "match", ""),
    ("Spleen", "cells in white pulp", "spleen", ["b-cells", "t-cells"], "match", ""),
    ("Stomach", "glandular cells", "stomach", ["foveolar cells", "mucous neck cells", "gastric chief cells", "parietal cells"], "match", ""),
    ("Testis", "Leydig cells", "testis", ["leydig cells"], "match", ""),
    ("Testis", "cells in seminiferous ducts", "testis",
     ["undifferentiated spermatogonia", "differentiating spermatogonia", "early primary spermatocytes", "late primary spermatocytes",
      "early spermatids", "late spermatids", "sertoli cells"], "match", "germ cells and Sertoli cells"),
    ("Urinary bladder", "urothelial cells", "urinary bladder", ["urothelial cells"], "match", ""),
]
NO_PROFILE = {  # contexts left without a profile
    ("Skin", "Langerhans"): "18,449 reads in total, too few for a stable profile",
    ("Soft tissue", "peripheral nerve"): "only Schwann cells from vasculature available, too weak a proxy",
}

# 1. read counts summed over clusters, for the tissue and cell type pairs the bridge uses
units = {(st, ct) for _, _, st, cts, _, _ in BRIDGE for ct in cts}
parts = []
for chunk in pd.read_csv(SC_ZIP, sep="\t", usecols=["Gene", "Tissue", "Cell type", "Read count"], chunksize=CHUNK):
    keep = pd.MultiIndex.from_frame(chunk[["Tissue", "Cell type"]]).isin(list(units))
    parts.append(chunk[keep].groupby(["Tissue", "Cell type", "Gene"])["Read count"].sum())
counts = pd.concat(parts).groupby(level=[0, 1, 2]).sum()
found = set(counts.index.droplevel(2).unique())
missing = units - found
assert not missing, f"not in the single-cell file: {sorted(missing)}"
print(f"units {len(units)}, genes {counts.index.get_level_values(2).nunique()}")

# 2. one profile per distinct (tissue, cell types) set: summed reads, counts per million over all genes, log1p
raw, meta = {}, []
for _, _, st, cts, _, _ in BRIDGE:
    pid = f"hpa:{st}:{'; '.join(cts)}"
    if pid in raw:
        continue
    col = sum(counts.loc[(st, ct)] for ct in cts)
    raw[pid] = col
    meta.append({"profile_id": pid, "source": "hpa", "files": st, "cells": "; ".join(cts), "reads": int(col.sum())})
raw = pd.DataFrame(raw).fillna(0.0)
meta = pd.DataFrame(meta)
meta["library_size"] = meta["profile_id"].map(raw.sum(axis=0))
meta["kept"] = True
profiles = np.log1p(raw / raw.sum(axis=0) * 1e6).astype(np.float32).rename_axis("gene")
profiles.to_parquet(HEAVY / "04b_profiles.parquet")
meta.to_csv(TABLES / "04b_profile_meta.csv", index=False)

# 3. context bridge in the same format as 08_context_bridge.csv
contexts = pd.read_parquet(HEAVY / "01_labels.parquet", columns=["context", "tissue", "cell_type"]).drop_duplicates("context")
rows = [{"context": f"{t} | {c}", "profile_id": f"hpa:{st}:{'; '.join(cts)}", "status": s, "note": n} for t, c, st, cts, s, n in BRIDGE]
rows += [{"context": f"{t} | {c}", "profile_id": None, "status": "none", "note": n} for (t, c), n in NO_PROFILE.items()]
bridge = contexts.merge(pd.DataFrame(rows), on="context", how="left")
bridge["status"] = bridge["status"].fillna("none")
bridge["note"] = bridge["note"].fillna("no counterpart in the single-cell data")
bridge = bridge.merge(meta[["profile_id", "reads"]], on="profile_id", how="left")
bridge["profile_shared_by"] = bridge.groupby("profile_id")["context"].transform("size")
bridge.to_csv(TABLES / "04b_context_bridge.csv", index=False)

print("\n== profiles (reads summed over clusters)")
print(meta[["profile_id", "reads"]].sort_values("reads").to_string(index=False))
print("\n== contexts by status")
print(bridge["status"].value_counts().to_string())
print(f"contexts with a profile {int(bridge['profile_id'].notna().sum())} of {len(bridge)}, distinct profiles {len(meta)}")
print("\n== contexts without a profile")
print(bridge.loc[bridge["profile_id"].isna(), ["context", "note"]].to_string(index=False))
print("\n== profiles shared by more than one context")
print(bridge.loc[bridge["profile_shared_by"] > 1].sort_values("profile_id")[["profile_id", "context"]].to_string(index=False))

