import pandas as pd
import os

# 路径
features_path = "../data/Visium_Human_Lymph_Node/filtered_feature_bc_matrix/features.tsv.gz"
interaction_path = "../data/L-Rgene/interaction_input.csv"
gene_input_path = "../data/L-Rgene/gene_input.csv"
complex_input_path = "../data/L-Rgene/complex_input.csv"
out_path = "../data/lr_pairs_1.csv"

# 1. Visium 基因
features = pd.read_csv(features_path, sep="\t", header=None)
visium_symbols = set(features[1].astype(str).str.upper())
print(f"✅ Visium 基因数: {len(visium_symbols)}")

# 2. gene_input: 建 uniprot → gene_name 映射
gene_input = pd.read_csv(gene_input_path)
uniprot2gene = dict(zip(gene_input["uniprot"], gene_input["gene_name"].str.upper()))

# 3. complex_input: 建 complex_name → gene_name 列表
complex_input = pd.read_csv(complex_input_path)

complex2genes = {}
for _, row in complex_input.iterrows():
    comp = str(row["complex_name"]).upper()
    genes = []
    for col in [c for c in complex_input.columns if c.startswith("uniprot_")]:
        uid = row[col]
        if pd.notna(uid) and uid in uniprot2gene:
            genes.append(uniprot2gene[uid])
    if genes:
        complex2genes[comp] = list(set(genes))  # 去重

print(f"✅ 复合物映射数: {len(complex2genes)}")

# 4. interaction_input: 展开
inter = pd.read_csv(interaction_path)
expanded_pairs = []
for _, row in inter.iterrows():
    a = str(row["partner_a"]).upper()
    b = str(row["partner_b"]).upper()

    a_list = complex2genes.get(a, [a])
    b_list = complex2genes.get(b, [b])

    for ga in a_list:
        for gb in b_list:
            expanded_pairs.append((ga.upper(), gb.upper()))

df_expanded = pd.DataFrame(expanded_pairs, columns=["partner_a", "partner_b"]).drop_duplicates()
print(f"✅ 展开后的基因–基因配对数: {len(df_expanded)}")

# 5. 和 Visium 交集
df_filtered = df_expanded[
    df_expanded["partner_a"].isin(visium_symbols) &
    df_expanded["partner_b"].isin(visium_symbols)
].drop_duplicates()

print(f"✅ 匹配 Visium 后 L-R 配对数: {len(df_filtered)}")

# 6. 保存
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df_filtered.to_csv(out_path, index=False)

print("✅ 已保存到", out_path)
print("示例输出:")
print(df_filtered.head(10))
