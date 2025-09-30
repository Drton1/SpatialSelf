import torch
import pandas as pd
import numpy as np
import os

def add_edge_attributes(data, lr_pairs_path, debug=False):
    """
    给图数据 Data 添加 edge_attr，包含 'lr' 和 'dist'
    """
    lr_pairs = pd.read_csv(lr_pairs_path)
    gene2idx = {g.upper(): i for i, g in enumerate(data.gene_names)}

    lr_scores, dists = [], []
    debug_count = 0

    for src, dst in data.edge_index.t().tolist():
        edge_vals = []
        for _, row in lr_pairs.iterrows():
            lig, rec = row["partner_a"].upper(), row["partner_b"].upper()
            if lig in gene2idx and rec in gene2idx:
                lig_idx, rec_idx = gene2idx[lig], gene2idx[rec]
                lig_expr = data.raw_expr[src, lig_idx].item()
                rec_expr = data.raw_expr[dst, rec_idx].item()
                edge_vals.append(lig_expr * rec_expr)

                # Debug: 打印前 5 条匹配
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

    edge_attr = torch.tensor(np.vstack([lr_scores, dists]).T, dtype=torch.float)
    data.edge_attr = edge_attr

    if debug:
        print(f"[DEBUG] lr_score 非零个数: {(np.array(lr_scores) > 0).sum()}")
        print(f"[DEBUG] lr_score 最大值: {np.max(lr_scores):.3f}, 最小值: {np.min(lr_scores):.3f}")

    return data


if __name__ == "__main__":
    pt_path = "../data/E1_slide1.pt"
    lr_pairs_path = "../data/lr_pairs.csv"
    out_path = "../data/E1_slide1_with_edge_attr.pt"

    # 1. 加载图
    print(f" Loading graph: {pt_path}")
    data = torch.load(pt_path, weights_only=False)

    # 确保 raw_expr 和 gene_names 存在
    if not hasattr(data, "raw_expr") or not hasattr(data, "gene_names"):
        raise ValueError("⚠️ Data 对象缺少 raw_expr 或 gene_names，请先用 preprocess_visium.py 生成 .pt 文件")

    # 2. 添加 edge_attr (带 debug)
    print(" Calculating LR and distance...")
    data = add_edge_attributes(data, lr_pairs_path, debug=True)

    # 3. 保存
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(data, out_path)
    print(f" Saved with edge_attr to {out_path}")
    print(f"edge_attr shape: {data.edge_attr.shape} (columns: [lr, dist])")
