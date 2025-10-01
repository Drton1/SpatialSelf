## v1
数据集：Visium_Human_Lymph_Node (人类淋巴结)  下载地址：https://www.10xgenomics.com/datasets/human-lymph-node-1-standard-1-1-0?utm_source=chatgpt.com
LR计算方式：lig_expr * rec_expr
距离计算方式：欧式距离
正负样本判别方式：正样本为该spot所有边LR值前百分之20，负样本为剩余百分之80 (该正负判别仅仅为了跑通pipline)
编码器架构：2→64→1 MLP  激活函数Relu，输入 [lr, dist]   
训练逻辑：所有边一起丢进去 → MLP 打分 → 计算损失 → 更新参数。


