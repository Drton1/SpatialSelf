import os
import scanpy as sc
import pandas as pd
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

# -----------------------------
# 路径配置
# -----------------------------
DATA_DIR = "C:/Users/Drton1/PycharmProjects/SpatialSelf/data/Visium_Human_Lymph_Node"
OUTPUT_FILE = "../data/E1_slide1.pt"

# -----------------------------
# 1. 读取表达矩阵 (MTX)
# -----------------------------
adata = sc.read_10x_mtx(
    os.path.join(DATA_DIR, "filtered_feature_bc_matrix"),
    var_names="gene_symbols",   # 用 gene symbols 作为基因名
    make_unique=True
)

# -----------------------------
# 2. 读取空间坐标，并和 barcodes 对齐
# -----------------------------
barcodes = pd.read_csv(
    os.path.join(DATA_DIR, "filtered_feature_bc_matrix", "barcodes.tsv.gz"),
    header=None
)[0].tolist()

positions = pd.read_csv(
    os.path.join(DATA_DIR, "spatial", "tissue_positions_list.csv"),
    header=None
)

# 对齐 barcodes（只保留有表达的 spot）
positions = positions[positions[0].isin(barcodes)]
positions = positions.set_index(0).loc[barcodes]

# 取最后两列作为 (x, y) 坐标（兼容不同 SpaceRanger 版本）
adata.obsm["spatial"] = positions.iloc[:, -2:].values

# -----------------------------
# 3. 标准化
# -----------------------------
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# -----------------------------
# 4. HVG for PCA (不做 subset)
# -----------------------------
sc.pp.highly_variable_genes(adata, n_top_genes=2500, subset=False)

# PCA 只用 HVG
adata_hvg = adata[:, adata.var["highly_variable"]]
pca = PCA(n_components=256)
X_pca = pca.fit_transform(adata_hvg.X.toarray())

# -----------------------------
# 5. 构建图 (kNN, k=6)
# -----------------------------
coords = adata.obsm["spatial"]
nbrs = NearestNeighbors(n_neighbors=6).fit(coords)
edge_index = nbrs.kneighbors_graph(coords).tocoo()
edge_index = np.vstack((edge_index.row, edge_index.col))

# -----------------------------
# 6. 保存为 PyG Data
# -----------------------------
data = Data(
    x=torch.tensor(X_pca, dtype=torch.float),                     # HVG PCA 特征
    edge_index=torch.tensor(edge_index, dtype=torch.long),        # 邻接
    pos=torch.tensor(coords, dtype=torch.float),                  # 空间坐标
    raw_expr=torch.tensor(adata.X.toarray(), dtype=torch.float),  # 全量表达矩阵
    gene_names=list(adata.var_names)                              # 全部基因名
)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
torch.save(data, OUTPUT_FILE)

print(f"Graph data saved to {OUTPUT_FILE}")
print(f"Nodes: {data.x.shape[0]}, PCA Features: {data.x.shape[1]}")
print(f"Raw expr shape: {data.raw_expr.shape}, Genes: {len(data.gene_names)}")
print(f"Edges: {data.edge_index.shape[1]}")
