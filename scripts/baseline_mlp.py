import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, roc_curve, precision_recall_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import numpy as np

# -----------------------------
# 指标函数
# -----------------------------
def evaluate_metrics(probs, labels, k_list=[100, 500, 1000]):
    labels_np = labels.numpy()
    probs_np = probs.detach().numpy()

    roc_auc = roc_auc_score(labels_np, probs_np)
    aupr = average_precision_score(labels_np, probs_np)

    # F1 / Precision (阈值=0.5)
    y_pred_bin = (probs_np >= 0.5).astype(int)
    f1 = f1_score(labels_np, y_pred_bin, zero_division=0)
    prec = precision_score(labels_np, y_pred_bin, zero_division=0)

    # Precision@k
    def precision_at_k(probs, labels, k):
        k = min(k, len(labels))  # 防止 k > 样本数
        idx = np.argsort(-probs)[:k]
        return labels[idx].sum() / k

    p_at_k = {f"P@{k}": precision_at_k(probs_np, labels_np, k) for k in k_list}

    return {
        "ROC-AUC": roc_auc,
        "AUPR": aupr,
        "F1": f1,
        "Precision": prec,
        **p_at_k
    }

# -----------------------------
# 主程序
# -----------------------------
if __name__ == "__main__":
    # 路径配置
    PT_DIR = "../data/with_edge_attr"
    OUT_FIG = "../figs/multislide_baseline.png"
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)

    slides = {
        "Lymph": torch.load(os.path.join(PT_DIR, "Visium_Human_Lymph_Node_with_edge_attr.pt"), weights_only=False),
        "Breast": torch.load(os.path.join(PT_DIR, "Human_Breast_Cancer_with_edge_attr.pt"), weights_only=False),
        "Lung": torch.load(os.path.join(PT_DIR, "Human_Lung_Tissue_with_edge_attr.pt"), weights_only=False),
    }

    # -----------------------------
    # Train / Val / Test split
    # -----------------------------
    train_data = [slides["Lymph"], slides["Breast"]]
    val_data = [slides["Lung"]]
    test_data = [slides["Lung"]]  # 这里简单设 val=test，你也可以换其他划分或交叉验证

    # 拼接训练集
    def concat_edges(datalist):
        X, y = [], []
        for data in datalist:
            edge_attr = data.edge_attr
            lr, dist = edge_attr[:,0], edge_attr[:,1]
            # 构造标签（top 20% 为正）
            threshold = torch.quantile(lr, 0.8)
            labels = (lr >= threshold).float()
            # 特征
            x = torch.stack([lr, dist], dim=1)
            X.append(x); y.append(labels)
        return torch.cat(X, dim=0), torch.cat(y, dim=0)

    X_train, y_train = concat_edges(train_data)
    X_val, y_val = concat_edges(val_data)
    X_test, y_test = concat_edges(test_data)

    print(f"Train edges: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

    # -----------------------------
    # MLP 模型
    # -----------------------------
    mlp = nn.Sequential(
        nn.Linear(2, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )

    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    best_val_auc, best_state = 0, None

    # -----------------------------
    # 训练循环
    # -----------------------------
    for epoch in range(30):
        mlp.train()
        pred = mlp(X_train).squeeze()
        loss = F.binary_cross_entropy_with_logits(pred, y_train)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        # 验证
        mlp.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(mlp(X_val).squeeze())
            metrics_val = evaluate_metrics(val_probs, y_val)
        if metrics_val["ROC-AUC"] > best_val_auc:
            best_val_auc = metrics_val["ROC-AUC"]
            best_state = mlp.state_dict()

        if epoch % 5 == 0:
            print(f"Epoch {epoch:02d}, Loss={loss.item():.4f}, Val AUC={metrics_val['ROC-AUC']:.3f}")

    # -----------------------------
    # 测试集评估
    # -----------------------------
    mlp.load_state_dict(best_state)
    mlp.eval()
    with torch.no_grad():
        test_probs = torch.sigmoid(mlp(X_test).squeeze())
        metrics_test = evaluate_metrics(test_probs, y_test)

    print("\n📊 Final Test Metrics:")
    for k,v in metrics_test.items():
        print(f" {k}: {v:.3f}")

    # -----------------------------
    # 绘制 ROC/PR 曲线
    # -----------------------------
    from sklearn.metrics import roc_curve, precision_recall_curve
    fpr, tpr, _ = roc_curve(y_test.numpy(), test_probs.numpy())
    precision, recall, _ = precision_recall_curve(y_test.numpy(), test_probs.numpy())

    plt.figure(figsize=(12,4))
    # ROC
    plt.subplot(1,2,1)
    plt.plot(fpr, tpr, label=f"AUC={metrics_test['ROC-AUC']:.3f}")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend()

    # PR
    plt.subplot(1,2,2)
    plt.plot(recall, precision, label=f"AUPR={metrics_test['AUPR']:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve"); plt.legend()

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300)
    print(f"✅ ROC/PR 曲线已保存到 {OUT_FIG}")
