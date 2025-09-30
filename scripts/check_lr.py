import torch
import pandas as pd

# -----------------------------
# 路径配置
# -----------------------------
pt_path = "../data/E1_slide1.pt"
lr_pairs_path = "../data/lr_pairs.csv"

# -----------------------------
# 1. 加载数据
# -----------------------------
print(f"🔹 Loading graph: {pt_path}")
data = torch.load(pt_path, weights_only=False)

if not hasattr(data, "gene_names"):
    raise ValueError("⚠️ Data 对象缺少 gene_names，请检查 preprocess_visium.py 是否保存了基因名")

gene_names = [g.upper() for g in data.gene_names]

print("✅ gene_names 示例:", gene_names[:10])
print("✅ gene_names 总数:", len(gene_names))

# -----------------------------
# 2. 加载 L-R 配对
# -----------------------------
lr_pairs = pd.read_csv(lr_pairs_path)
lr_genes = set(lr_pairs["partner_a"].str.upper()) | set(lr_pairs["partner_b"].str.upper())

print("✅ lr_pairs 示例:")
print(lr_pairs.head())
print("✅ lr_pairs 基因总数:", len(lr_genes))

# -----------------------------
# 3. 计算交集
# -----------------------------
gene_set = set(gene_names)
overlap = gene_set & lr_genes

print(f"🔎 交集基因数: {len(overlap)}")
if len(overlap) > 0:
    print("🔎 示例交集基因:", list(overlap)[:20])
else:
    print("⚠️ 没有交集！请检查 gene_names 是否是 Ensembl 而 lr_pairs 是 Symbol")

