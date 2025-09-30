import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib
matplotlib.use("Agg")  #  强制用非交互后端
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    # -----------------------------
    # 路径配置
    # -----------------------------
    pt_path = "../data/E1_slide1_with_edge_attr.pt"
    out_fig = "../figs/E1_baseline_roc_pr.png"
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)

    # -----------------------------
    # 1. 加载数据
    # -----------------------------
    data = torch.load(pt_path, weights_only=False)
    edge_attr = data.edge_attr  # [E, 2] → [lr, dist]
    lr = edge_attr[:, 0]
    dist = edge_attr[:, 1]

    print(f" Edge_attr shape: {edge_attr.shape}")
    print(f" lr min={lr.min().item():.4f}, max={lr.max().item():.4f}, 非零比例={(lr>0).float().mean().item():.4f}")

    # -----------------------------
    # 2. 标签构造 (分位数阈值)
    # -----------------------------
    threshold = torch.quantile(lr, 0.8)   # 取 top 10% 边作为正样本
    labels = (lr >= threshold).float()

    print(f" 正样本数: {int(labels.sum().item())}, 负样本数: {int((labels==0).sum().item())}")

    # -----------------------------
    # 3. 构建输入特征
    # -----------------------------
    x = torch.stack([lr, dist], dim=1)  # [E, 2]

    # -----------------------------
    # 4. 简单 MLP
    # -----------------------------
    mlp = nn.Sequential(
        nn.Linear(x.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

    # -----------------------------
    # 5. 前向 + 损失
    # -----------------------------
    pred = mlp(x).squeeze()
    loss = F.binary_cross_entropy_with_logits(pred, labels)

    # -----------------------------
    # 6. 计算 ROC / PR
    # -----------------------------
    probs = torch.sigmoid(pred).detach().numpy()
    labels_np = labels.numpy()

    roc_auc = roc_auc_score(labels_np, probs)
    aupr = average_precision_score(labels_np, probs)

    print(f" Loss: {loss.item():.4f}")
    print(f" ROC-AUC: {roc_auc:.3f}, AUPR: {aupr:.3f}")

    # -----------------------------
    # 7. 绘制曲线
    # -----------------------------
    fpr, tpr, _ = roc_curve(labels_np, probs)
    precision, recall, _ = precision_recall_curve(labels_np, probs)

    plt.figure(figsize=(10,4))

    # ROC
    plt.subplot(1,2,1)
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend()

    # PR
    plt.subplot(1,2,2)
    plt.plot(recall, precision, label=f"AUPR={aupr:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve"); plt.legend()

    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    print(f" ROC/PR 曲线已保存到 {out_fig}")
