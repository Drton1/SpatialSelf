import os
import scanpy as sc
import pandas as pd
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, KDTree
from torch_geometric.data import Data
from scipy import sparse

# -----------------------------
# 路径配置
# -----------------------------
DATA_ROOT = "../data/"
OUTPUT_ROOT = "../data/processed_graphs"

SLIDES = [
    "Visium_Human_Lymph_Node",
    "Human Breast Cancer",
    "Human Lung Tissue"
]

LR_PAIRS_PATH = "../data/lr_pairs_1.csv"  # LR 配对表

# -----------------------------
# 工具函数
# -----------------------------
def _to_dense(x):
    if sparse.issparse(x):
        return x.toarray()
    return np.asarray(x)

def read_coords(slide_path: str):
    """读取空间坐标"""
    pos_file = None
    for candidate in ["tissue_positions_list.csv", "tissue_positions.csv"]:
        cand_path = os.path.join(slide_path, "spatial", candidate)
        if os.path.exists(cand_path):
            pos_file = cand_path
            break
    if pos_file is None:
        raise FileNotFoundError(f"No tissue_positions file under {slide_path}")
    positions = pd.read_csv(pos_file, header=None).set_index(0)
    return positions

def add_lr_topm_edges(expr, gene_names, coords, radius, m, lr_pairs):
    """在半径邻域内选 LR 值最高的 m 条边"""
    gene2idx = {g.upper(): i for i, g in enumerate(gene_names)}
    tree = KDTree(coords)
    neighbors = tree.query_radius(coords, r=radius)

    lr_edges = set()
    for i in range(len(coords)):
        cand = neighbors[i]
        cand = cand[cand != i]
        if cand.size == 0:
            continue

        scores = []
        for j in cand:
            score = 0.0
            for _, row in lr_pairs.iterrows():
                lig, rec = row["partner_a"].upper(), row["partner_b"].upper()
                if lig in gene2idx and rec in gene2idx:
                    score += float(expr[i, gene2idx[lig]] * expr[j, gene2idx[rec]])
                    score += float(expr[j, gene2idx[lig]] * expr[i, gene2idx[rec]])  # 双向
            scores.append((int(j), score))

        scores.sort(key=lambda x: -x[1])  # 按 LR 值排序
        for j, _ in scores[:m]:
            lr_edges.add((i, j))

    return lr_edges

def assign_best_lr_pair(expr, gene_names, edges, lr_pairs):
    """
    为每条边分配最强 LR 对
    expr: (n_spots, n_genes)
    gene_names: 基因名
    edges: [(i,j), ...]
    lr_pairs: DataFrame, 包含 partner_a, partner_b
    """
    gene2idx = {g.upper(): i for i, g in enumerate(gene_names)}
    edge_lr_pairs = []

    for (i, j) in edges:
        best_pair = "NA"
        best_score = 0.0
        for _, row in lr_pairs.iterrows():
            lig, rec = row["partner_a"].upper(), row["partner_b"].upper()
            if lig in gene2idx and rec in gene2idx:
                s1 = float(expr[i, gene2idx[lig]] * expr[j, gene2idx[rec]])
                s2 = float(expr[j, gene2idx[lig]] * expr[i, gene2idx[rec]])
                if s1 > best_score:
                    best_score = s1
                    best_pair = f"{lig}|{rec}"
                if s2 > best_score:
                    best_score = s2
                    best_pair = f"{lig}|{rec}"
        edge_lr_pairs.append(best_pair)
    return edge_lr_pairs

# -----------------------------
# 核心处理函数
# -----------------------------
def process_slide(
    slide_name,
    k=6,
    n_hvg=2500,
    pca_dim=256,
    radius=150.0,
    m=20
):
    slide_path = os.path.join(DATA_ROOT, slide_name)
    fmat_path = os.path.join(slide_path, "filtered_feature_bc_matrix")

    # 1) 表达矩阵
    if os.path.exists(fmat_path):
        h5_files = [f for f in os.listdir(fmat_path) if f.endswith(".h5")]
        if len(h5_files) > 0:
            print(f" Using H5 for {slide_name}")
            adata = sc.read_10x_h5(os.path.join(fmat_path, h5_files[0]))
        else:
            print(f" Using MTX for {slide_name}")
            adata = sc.read_10x_mtx(
                fmat_path,
                var_names="gene_symbols",
                make_unique=True
            )
    else:
        h5_files = [f for f in os.listdir(slide_path) if f.endswith(".h5")]
        if len(h5_files) == 0:
            raise FileNotFoundError(f"No expression matrix found for {slide_name}")
        print(f"🔹 Using H5 (root) for {slide_name}")
        adata = sc.read_10x_h5(os.path.join(slide_path, h5_files[0]))

    if "gene_symbols" in adata.var.columns:
        adata.var_names = adata.var["gene_symbols"]
    adata.var_names_make_unique()

    # 2) 空间坐标
    positions = read_coords(slide_path)
    barcodes = list(adata.obs_names)
    positions = positions.loc[barcodes]
    coords = positions.iloc[:, -2:].astype(float).values
    adata.obsm["spatial"] = coords
    assert adata.n_obs == coords.shape[0], f"Spot mismatch: {adata.n_obs} vs {coords.shape[0]}"
    print(f" Spot aligned: {adata.n_obs} spots")

    # 3) 标准化
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 4) HVG + PCA
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, subset=False)
    adata_hvg = adata[:, adata.var["highly_variable"]]
    X_pca = PCA(n_components=pca_dim).fit_transform(_to_dense(adata_hvg.X))

    # 5) 读取 LR 配对
    lr_pairs = pd.read_csv(LR_PAIRS_PATH)

    # 6) KNN 边
    nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
    knn_graph = nbrs.kneighbors_graph(coords).tocoo()
    edge_set = set(zip(knn_graph.row.astype(int), knn_graph.col.astype(int)))

    # 7) LR-topm 边
    raw_expr = _to_dense(adata.X)
    lr_edges = add_lr_topm_edges(raw_expr, adata.var_names, coords, radius, m, lr_pairs)

    # 8) 合并边
    all_edges = list(edge_set.union(lr_edges))
    edge_index = np.array(all_edges, dtype=np.int64).T

    # 9) 计算每条边最强的 LR 对名称
    edge_lr_pairs = assign_best_lr_pair(raw_expr, adata.var_names, all_edges, lr_pairs)

    # 10) PyG Data（包含 edge_lr_pairs）
    data = Data(
        x=torch.tensor(X_pca, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        pos=torch.tensor(coords, dtype=torch.float32),
        raw_expr=torch.tensor(raw_expr, dtype=torch.float32),
        gene_names=list(adata.var_names),
        edge_lr_pairs=edge_lr_pairs
    )

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    safe_name = slide_name.replace(" ", "_")
    out_path = os.path.join(OUTPUT_ROOT, f"{safe_name}.pt")
    torch.save(data, out_path)

    print(f"   Saved {slide_name} to {out_path}")
    print(f"   Nodes: {data.num_nodes}, PCA Features: {data.x.shape[1]}")
    print(f"   Raw expr: {tuple(data.raw_expr.shape)}, Genes: {len(data.gene_names)}")
    print(f"   Edges: {data.edge_index.shape[1]}")
    print(f"   KNN edges: {len(edge_set)}, LR-topm edges: {len(lr_edges)}, Union: {len(all_edges)}")
    print(f"   edge_lr_pairs saved: {len(edge_lr_pairs)}")
    print("-" * 50)
    return data

# -----------------------------
# 主程序
# -----------------------------
if __name__ == "__main__":
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    for slide in SLIDES:
        process_slide(slide)
