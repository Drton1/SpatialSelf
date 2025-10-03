import os
import re
import torch
import pandas as pd

# -----------------------------
# 路径配置
# -----------------------------
PT_DIR = "../data/processed_graphs"
SLIDES = [
    "Visium_Human_Lymph_Node.pt",
    "Human_Breast_Cancer.pt",
    "Human_Lung_Tissue.pt"
]
lr_pairs_path = "../data/lr_pairs_1.csv"
OUTPUT_DIR = "../data/lr_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 工具函数：清洗基因名
# -----------------------------
def clean_gene_name(g):
    g = g.upper()
    return re.sub(r"-\d+$", "", g)  # 去掉 -1/-2 后缀

# -----------------------------
# 1. 加载 L-R 配对
# -----------------------------
lr_pairs = pd.read_csv(lr_pairs_path)
lr_pairs["partner_a_clean"] = lr_pairs["partner_a"].str.upper().str.replace(r"-\d+$", "", regex=True)
lr_pairs["partner_b_clean"] = lr_pairs["partner_b"].str.upper().str.replace(r"-\d+$", "", regex=True)

# -----------------------------
# 2. 遍历多个切片
# -----------------------------
all_valid_pairs = []
for slide_file in SLIDES:
    pt_path = os.path.join(PT_DIR, slide_file)
    print(f"\n🔹 Loading graph: {pt_path}")
    data = torch.load(pt_path, weights_only=False)

    if not hasattr(data, "gene_names"):
        raise ValueError(f"{slide_file} 缺少 gene_names，请检查 preprocess_visium.py 是否保存了基因名")

    # 清洗基因名
    gene_names = [clean_gene_name(g) for g in data.gene_names]
    gene_set = set(gene_names)

    # 过滤出当前切片里可用的 L-R 配对
    valid_pairs = lr_pairs[
        lr_pairs["partner_a_clean"].isin(gene_set) &
        lr_pairs["partner_b_clean"].isin(gene_set)
    ].copy()

    valid_pairs["slide"] = slide_file
    print(f"✅ {slide_file}: 找到 {len(valid_pairs)} 个 L-R 配对")

    # 保存到单独 CSV
    out_csv = os.path.join(OUTPUT_DIR, f"{slide_file.replace('.pt','')}_lr_pairs.csv")
    valid_pairs.to_csv(out_csv, index=False)
    print(f"💾 已保存 {out_csv}")

    all_valid_pairs.append(valid_pairs)

# -----------------------------
# 3. 合并所有切片结果
# -----------------------------
if all_valid_pairs:
    merged = pd.concat(all_valid_pairs, ignore_index=True)
    merged_out = os.path.join(OUTPUT_DIR, "all_slides_lr_pairs.csv")
    merged.to_csv(merged_out, index=False)
    print(f"\n🌍 所有切片合并结果已保存到 {merged_out}")
