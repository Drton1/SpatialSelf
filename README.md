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
- **名称**: uman breast cancer(formalin-fixed paraffin-embedded, FFPE) (人类乳腺癌样本)  
- **下载地址**: [ uman breast cancer(formalin-fixed paraffin-embedded, FFPE)](https://www.10xgenomics.com/datasets/human-lung-cancer-11-mm-capture-area-ffpe-2-standard)
