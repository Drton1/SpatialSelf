import torch
import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 工具函数：清洗基因名
# -----------------------------
def clean_gene_name(g):
    if pd.isna(g):
        return None
    g = str(g).upper()
    return re.sub(r"-\d+$", "", g)  # 去掉 -1/-2 后缀


# -----------------------------
# 基因表达稳定化：log1p + quantile_minmax
# -----------------------------
def per_gene_scale(X, q_low=0.01, q_high=0.99, eps=1e-8):
    X = np.log1p(X)  # log 平滑
    lo = np.quantile(X, q_low, axis=0, keepdims=True)
    hi = np.quantile(X, q_high, axis=0, keepdims=True)
    Xn = (X - lo) / (hi - lo + eps)
    return np.clip(Xn, 0.0, 1.0)


# -----------------------------
# 读取复合受体定义
# -----------------------------
def load_complex_dict(complex_csv):
    df = pd.read_csv(complex_csv)
    complex_dict = {}
    for _, row in df.iterrows():
        cname = row["complex_name"]
        rec_flag = row["receptor"] if "receptor" in row else False
        if rec_flag and isinstance(cname, str):
            # 收集所有 uniprot 列（转成 gene symbol 前最好映射；此处直接 clean）
            genes = [clean_gene_name(row[c]) for c in ["uniprot_1","uniprot_2","uniprot_3","uniprot_4"] if pd.notna(row[c])]
            if genes:
                complex_dict[cname.upper()] = [g for g in genes if g]
    return complex_dict


# -----------------------------
# 给 Data 添加 edge_attr (LR 分数 + 距离)
# -----------------------------
def add_edge_attributes(data, lr_pairs, complex_dict=None, debug=False):
    # 基因名字典
    gene2idx = {clean_gene_name(g): i for i, g in enumerate(data.gene_names)}

    # 基因表达矩阵稳定化
    expr = per_gene_scale(data.raw_expr.numpy())  # ndarray (spots, genes)

    lr_scores, dists = [], []
    debug_count = 0

    for src, dst in data.edge_index.t().tolist():
        edge_vals = []

        for _, row in lr_pairs.iterrows():
            lig, rec = clean_gene_name(row["partner_a"]), clean_gene_name(row["partner_b"])
            if lig in gene2idx:
                lig_idx = gene2idx[lig]
                lig_expr = expr[src, lig_idx]

                # receptor 可以是单基因或复合体
                rec_expr = None
                if rec in gene2idx:
                    rec_expr = expr[dst, gene2idx[rec]]
                elif complex_dict and rec in complex_dict:
                    rec_genes = [g for g in complex_dict[rec] if g in gene2idx]
                    if rec_genes:
                        rec_vals = [expr[dst, gene2idx[g]] for g in rec_genes]
                        rec_expr = np.exp(np.mean(np.log(np.array(rec_vals) + 1e-8)))  # 几何平均

                if rec_expr is not None:
                    edge_vals.append(lig_expr * rec_expr)

                    if debug and debug_count < 5:
                        print(f"[DEBUG] Edge ({src}->{dst}) "
                              f"{lig}={lig_expr:.3f}, {rec}={rec_expr:.3f}, "
                              f"乘积={lig_expr*rec_expr:.3f}")
                        debug_count += 1

        lr_score = np.mean(edge_vals) if edge_vals else 0.0
        lr_scores.append(lr_score)

        # 欧式距离
        p1, p2 = data.pos[src], data.pos[dst]
        dist = torch.norm(p1 - p2).item()
        dists.append(dist)

    edge_attr = np.vstack([lr_scores, dists]).T

    # 边特征归一化
    scaler = StandardScaler()
    edge_attr = scaler.fit_transform(edge_attr)

    data.edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    if debug:
        print(f"[DEBUG] lr_score 非零个数: {(np.array(lr_scores) > 0).sum()}")
        print(f"[DEBUG] lr_score 原始最大值: {np.max(lr_scores):.3f}, 最小值: {np.min(lr_scores):.3f}")
        print(f"[DEBUG] 标准化后均值: {data.edge_attr[:,0].mean():.3f}, 方差: {data.edge_attr[:,0].var():.3f}")

    return data


# -----------------------------
# 主程序：多切片处理
# -----------------------------
if __name__ == "__main__":
    PT_DIR = "../data/processed_graphs"
    SLIDES = [
        "Visium_Human_Lymph_Node.pt",
        "Human_Breast_Cancer.pt",
        "Human_Lung_Tissue.pt"
    ]
    lr_pairs_path = "../data/lr_results/clean_lr_pairs.csv"
    complex_path = "../data/L-Rgene/complex_input.csv"
    OUT_DIR = "../data/with_edge_attr"
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. 加载 L-R 配对和复合受体
    lr_pairs = pd.read_csv(lr_pairs_path)
    complex_dict = load_complex_dict(complex_path)
    print(f" L-R 配对总数: {len(lr_pairs)}")
    print(f" 复合受体数量: {len(complex_dict)}")

    # 2. 遍历切片
    for i, slide_file in enumerate(SLIDES):
        pt_path = os.path.join(PT_DIR, slide_file)
        print(f"\n🔹 Loading graph: {pt_path}")
        data = torch.load(pt_path, weights_only=False)

        if not hasattr(data, "raw_expr") or not hasattr(data, "gene_names"):
            raise ValueError(f"⚠️ {slide_file} 缺少 raw_expr 或 gene_names，请检查 preprocess_visium.py")

        # 添加 edge_attr (只对第一个切片打印 debug)
        debug_flag = (i == 0)
        print(" Calculating LR and distance with complex handling...")
        data = add_edge_attributes(data, lr_pairs, complex_dict=complex_dict, debug=debug_flag)

        # 保存
        safe_name = slide_file.replace(".pt", "_with_edge_attr.pt")
        out_path = os.path.join(OUT_DIR, safe_name)
        torch.save(data, out_path)
        print(f"✅ Saved with edge_attr to {out_path}")
        print(f"   edge_attr shape: {data.edge_attr.shape} (columns: [lr, dist])")
