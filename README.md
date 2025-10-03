# SpatialSelf v1 Baseline

## 数据集
- **名称**: Visium_Human_Lymph_Node (人类淋巴结)  
- **下载地址**: [10x Genomics - Human Lymph Node](https://www.10xgenomics.com/datasets/human-lymph-node-1-standard-1-1-0)

## 特征构造
- **LR 计算方式**: `lig_expr * rec_expr`  
- **距离计算方式**: 欧式距离 

## 正负样本判别
- **正样本**: 每个 spot 的所有边中，`LR` 值排名前 20%  
- **负样本**: 其余 80%  
> 当前划分仅用于跑通 pipeline，非最终生物学标注。

## 编码器架构
- **模型**: 多层感知机 (MLP)  
- **结构**: `2 → 64 → 1`  
- **激活函数**: ReLU  
- **输入特征**: `[lr, dist]`

## 训练逻辑
- 将所有边一起输入 MLP  
- MLP 输出边的打分 (logit)  
- 与正负样本标签计算二分类交叉熵损失
- 反向传播更新参数

# SpatialSelf v1.5

## 数据集
- **名称**: Visium_Human_Lymph_Node (人类淋巴结)  
- **下载地址**: [10x Genomics - Human Lymph Node](https://www.10xgenomics.com/datasets/human-lymph-node-1-standard-1-1-0)
- **名称**: Human Lung Cancer (FFPE) (人类肺癌样本)  
- **下载地址**: [ human lung cancer (FFPE)](https://www.10xgenomics.com/datasets/human-lung-cancer-11-mm-capture-area-ffpe-2-standard)
- **名称**: Human breast cancer(formalin-fixed paraffin-embedded, FFPE) (人类乳腺癌样本)  
- **下载地址**: [ Human breast cancer(formalin-fixed paraffin-embedded, FFPE)]([https://www.10xgenomics.com/datasets/human-lung-cancer-11-mm-capture-area-ffpe-2-standard](https://www.10xgenomics.com/datasets/human-breast-cancer-visium-fresh-frozen-whole-transcriptome-1-standard)

## 改动
- 由单数据集，改为多数据集，train为Lymph_Node和breast cance构建的图数据集，val和test为Lung_Cancer图数据集
- 图的边的选取由保留每个点的 k 个最近邻近边，新增在半径邻域内计算 LR 分数（基于 lr_pairs.csv），为每个点挑出 LR 值最高的 m 条边，最好保留的边取两个边集合的交集。
- 在构造边特征的时，LR的计算方式增加了复合体的计算，通过读取complex_input.csv，找出复合体的LR基因，在 LR 计算时取几何平均。（后面可以根据生物意义需要改为选取max值），以及对[lr_score, dist]边特征进行标准化，保证数量级一致。




