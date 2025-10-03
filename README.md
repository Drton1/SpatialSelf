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

##  数据集
- **Visium Human Lymph Node (fresh-frozen)**  
  [下载链接](https://www.10xgenomics.com/datasets/human-lymph-node-1-standard-1-1-0)

- **Human Lung Cancer (FFPE)**  
  [下载链接](https://www.10xgenomics.com/datasets/human-lung-cancer-11-mm-capture-area-ffpe-2-standard)

- **Human Breast Cancer ( Fresh-frozen)**  
  [下载链接](https://www.10xgenomics.com/datasets/human-breast-cancer-visium-fresh-frozen-whole-transcriptome-1-standard)

---

##  改动说明

### 1. 多数据集支持
- 由单数据集 (Lymph Node) → 扩展为多数据集：  
  - **Train**: Lymph Node + Breast Cancer  
  - **Val/Test**: Lung Cancer  

### 2. 图边构建优化
- **原始方法**: 每个点保留 k 近邻边 (k=6)  
- **改进方法**: 在半径邻域内计算 LR score，挑选 top-m 边  
- **最终边集**: `kNN 边 ∩ LR-top-m 边`，兼顾空间邻近和 LR 信号强度  

### 3. LR score 计算增强
- 支持 **复合受体 (complex receptor)**，基于 `complex_input.csv`  
- 多亚基复合体 → 取 **几何平均** (可选 max)  
- 基因表达 → **log1p + quantile scaling [0,1]** → 再计算 LR，更稳健  

### 4. 边特征归一化
- Edge features = `[LR score, distance]`  
- 在 **训练集拟合 StandardScaler**，对 val/test 仅做 transform，保证数值量级一致  

---






