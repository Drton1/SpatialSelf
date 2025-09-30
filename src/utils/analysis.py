import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt


def plot_lr_activity(
    adata_path: str,
    lr_pairs_path: str,
    top_n: int = 20,
    heatmap: bool = True,
):
    """
    计算并可视化 L-R 配对在组织中的活跃度。

    参数
    ----
    adata_path : str
        Visium 表达矩阵路径 (10x filtered_feature_bc_matrix)
    lr_pairs_path : str
        经过过滤的 L-R 配对表 (CSV: partner_a, partner_b)
    top_n : int, default=20
        展示前多少个 L-R 配对
    heatmap : bool, default=True
        是否绘制热力图 (Top 10)
    """

    # 1. 读取数据
    adata = sc.read_10x_mtx(adata_path, var_names="gene_symbols", make_unique=True)
    lr_pairs = pd.read_csv(lr_pairs_path)

    # 2. 计算每个配对的平均活跃度
    scores = []
    for _, row in lr_pairs.iterrows():
        ligand, receptor = row["partner_a"], row["partner_b"]
        if ligand in adata.var_names and receptor in adata.var_names:
            lig_exp = adata[:, ligand].X.toarray().flatten()
            rec_exp = adata[:, receptor].X.toarray().flatten()
            lr_score = (lig_exp * rec_exp).mean()
            scores.append({"pair": f"{ligand}-{receptor}", "score": lr_score})

    df_scores = pd.DataFrame(scores).sort_values("score", ascending=False)

    if df_scores.empty:
        print("⚠️ 没有任何配对在表达矩阵中匹配到基因。")
        return None

    # 3. 直方图
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="score", y="pair", data=df_scores.head(top_n),
        hue="pair", dodge=False, legend=False, palette="viridis"
    )
    plt.title(f"Top {top_n} LR Pairs (by mean score)")
    plt.xlabel("Mean LR score across spots")
    plt.ylabel("L-R Pair")
    plt.tight_layout()
    plt.show()

    # 4. 热力图（Top 10 配对 × spot）
    if heatmap:
        top_pairs = df_scores.head(10)["pair"].tolist()
        heatmap_data = []
        for pair in top_pairs:
            lig, rec = pair.split("-")
            lig_exp = adata[:, lig].X.toarray().flatten()
            rec_exp = adata[:, rec].X.toarray().flatten()
            heatmap_data.append(lig_exp * rec_exp)

        heatmap_df = pd.DataFrame(
            heatmap_data, index=top_pairs, columns=adata.obs_names
        )

        plt.figure(figsize=(12, 6))
        sns.heatmap(heatmap_df, cmap="Reds", cbar_kws={"label": "LR score"})
        plt.title("Heatmap of Top 10 L-R pairs across spots")
        plt.xlabel("Spots")
        plt.ylabel("L-R Pair")
        plt.tight_layout()
        plt.show()

    return df_scores
