import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, roc_curve, precision_recall_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import numpy as np

# ============================================================
# 指标函数
# ============================================================
def evaluate_metrics(probs, labels, k_list=[100, 500, 1000]):
    labels_np = labels.cpu().numpy()
    probs_np = probs.detach().cpu().numpy()
    results = {}

    if len(np.unique(labels_np)) > 1:
        results["ROC-AUC"] = roc_auc_score(labels_np, probs_np)
        results["AUPR"] = average_precision_score(labels_np, probs_np)
    else:
        results["ROC-AUC"] = float("nan")
        results["AUPR"] = float("nan")

    y_pred_bin = (probs_np >= 0.5).astype(int)
    results["F1"] = f1_score(labels_np, y_pred_bin, zero_division=0)
    results["Precision"] = precision_score(labels_np, y_pred_bin, zero_division=0)

    def precision_at_k(probs, labels, k):
        k = min(k, len(labels))
        idx = np.argsort(-probs)[:k]
        return labels[idx].sum() / k

    for k in k_list:
        results[f"P@{k}"] = precision_at_k(probs_np, labels_np, k)
    return results


# ============================================================
# 模型定义：GATv2 + Edge MLP
# ============================================================
class GATv2EdgePredictor(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64, heads=2):
        super().__init__()
        self.gnn1 = GATv2Conv(in_dim, hidden_dim, heads=heads, concat=True)
        self.gnn2 = GATv2Conv(hidden_dim * heads, hidden_dim, heads=1, concat=True)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_pairs, edge_attr):
        h = F.relu(self.gnn1(x, edge_index))
        h = F.relu(self.gnn2(h, edge_index))
        src, dst = edge_pairs
        h_src, h_dst = h[src], h[dst]
        edge_feat = torch.cat([h_src, h_dst, edge_attr], dim=-1)
        out = self.edge_mlp(edge_feat).squeeze()
        return out, h


# ============================================================
# InfoNCE 对比损失
# ============================================================
def info_nce_loss(h, edge_pairs, edge_attr, temperature=0.2, num_neg=5):
    src, dst = edge_pairs
    lr, dist = edge_attr[:, 0], edge_attr[:, 1]

    q_high = torch.quantile(lr, 0.9)
    q_low = torch.quantile(lr, 0.1)
    d_thr = torch.quantile(dist, 0.2)

    pos_mask = (dist <= d_thr) & (lr >= q_high)
    pos_src, pos_dst = src[pos_mask], dst[pos_mask]
    if len(pos_src) == 0:
        return torch.tensor(0.0, device=h.device, requires_grad=True)

    h_src, h_dst = h[pos_src], h[pos_dst]
    pos_score = F.cosine_similarity(h_src, h_dst)

    hard_mask = (dist <= d_thr) & (lr <= q_low)
    hard_src, hard_dst = src[hard_mask], dst[hard_mask]
    if len(hard_src) > 0:
        h_hard = h[hard_dst[:len(pos_src)]]
        hard_neg_score = F.cosine_similarity(h_src, h_hard).unsqueeze(1)
    else:
        hard_neg_score = None

    far_mask = (dist > d_thr)
    far_idx = torch.where(far_mask)[0]
    if len(far_idx) > 0:
        rand_idx = far_idx[torch.randint(0, len(far_idx), (len(pos_src)*num_neg,), device=h.device)]
        rand_dst = dst[rand_idx].view(len(pos_src), num_neg)
        h_neg = h[rand_dst]
        h_src_expanded = h_src.unsqueeze(1).expand_as(h_neg)
        rand_neg_score = F.cosine_similarity(h_src_expanded, h_neg, dim=-1)
    else:
        rand_neg_score = None

    neg_list = []
    if rand_neg_score is not None:
        neg_list.append(rand_neg_score)
    if hard_neg_score is not None:
        neg_list.append(hard_neg_score)
    neg_score = torch.cat(neg_list, dim=1)

    pos_score = pos_score / temperature
    neg_score = neg_score / temperature
    logits = torch.cat([pos_score.unsqueeze(1), neg_score], dim=1)
    labels = torch.zeros(len(pos_src), dtype=torch.long, device=h.device)
    return F.cross_entropy(logits, labels)


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    PT_DIR = "../data/with_edge_attr"
    OUT_FIG = "../figs/gatv2_infonce_bce_lr_earlystop.png"
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    slides = {
        "Lymph": torch.load(os.path.join(PT_DIR, "Visium_Human_Lymph_Node_with_edge_attr.pt"), weights_only=False),
        "Breast": torch.load(os.path.join(PT_DIR, "Human_Breast_Cancer_with_edge_attr.pt"), weights_only=False),
        "Lung": torch.load(os.path.join(PT_DIR, "Human_Lung_Tissue_with_edge_attr.pt"), weights_only=False),
    }

    train_data = slides["Lymph"]
    val_data = slides["Breast"]
    test_data = slides["Lung"]

    def build_dataset(data):
        edge_attr = data.edge_attr
        lr, dist = edge_attr[:, 0], edge_attr[:, 1]
        q_high = torch.quantile(lr, 0.8)
        q_low = torch.quantile(lr, 0.2)
        d_thr = torch.quantile(dist, 0.2)

        labels = torch.full_like(lr, -1)
        labels[(lr >= q_high) & (dist <= d_thr)] = 1
        labels[(lr <= q_low) & (dist <= d_thr)] = 0

        mask = labels >= 0
        edge_pairs = data.edge_index[:, mask]
        edge_attr = edge_attr[mask]

        edge_lr_pairs = None
        if hasattr(data, "edge_lr_pairs"):
            arr = np.array(data.edge_lr_pairs, dtype=object)
            edge_lr_pairs = arr[mask.cpu().numpy()]

        return data.x, data.edge_index, edge_pairs, labels[mask], edge_attr, edge_lr_pairs

    x_train, ei_train, ep_train, y_train, ea_train, _ = build_dataset(train_data)
    x_val, ei_val, ep_val, y_val, ea_val, val_edge_pairs = build_dataset(val_data)
    x_test, ei_test, ep_test, y_test, ea_test, test_edge_pairs = build_dataset(test_data)

    # GPU 迁移
    x_train, ei_train, ep_train, y_train, ea_train = (
        x_train.to(device), ei_train.to(device), ep_train.to(device), y_train.to(device), ea_train.to(device)
    )
    x_val, ei_val, ep_val, y_val, ea_val = (
        x_val.to(device), ei_val.to(device), ep_val.to(device), y_val.to(device), ea_val.to(device)
    )
    x_test, ei_test, ep_test, y_test, ea_test = (
        x_test.to(device), ei_test.to(device), ep_test.to(device), y_test.to(device), ea_test.to(device)
    )

    # 模型与优化器
    model = GATv2EdgePredictor(in_dim=x_train.shape[1], hidden_dim=64, heads=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    alpha = 5  # InfoNCE 权重
    patience = 15  # 早停耐心
    best_val_auc, best_state = -1, None
    wait = 0

    # ============================================================
    # 训练循环
    # ============================================================
    for epoch in range(300):
        model.train()
        pred, h = model(x_train, ei_train, ep_train, ea_train)
        loss_bce = F.binary_cross_entropy_with_logits(pred, y_train)
        loss_infonce = info_nce_loss(h, ep_train, ea_train, num_neg=10)
        loss = loss_bce + alpha * loss_infonce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(x_val, ei_val, ep_val, ea_val)
            val_probs = torch.sigmoid(val_logits)
            metrics_val = evaluate_metrics(val_probs, y_val)
        val_auc = metrics_val["ROC-AUC"]

        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1

        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | Loss={loss.item():.4f} | BCE={loss_bce.item():.4f} | "
                  f"InfoNCE={loss_infonce.item():.4f} | ValAUC={val_auc:.4f} | Best={best_val_auc:.4f} | Wait={wait}")

        # 早停条件
        if wait >= patience:
            print(f"⛔ Early stopping at epoch {epoch} (best AUC={best_val_auc:.4f})")
            break

    # 加载最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)

    # 测试评估
    model.eval()
    with torch.no_grad():
        test_logits, _ = model(x_test, ei_test, ep_test, ea_test)
        test_probs = torch.sigmoid(test_logits)
        metrics_test = evaluate_metrics(test_probs, y_test)

    print("\nFinal Test Metrics:")
    for k, v in metrics_test.items():
        print(f" {k}: {v:.3f}")

    # ============================================================
    # 验证集 Top-K 输出
    # ============================================================
    if val_edge_pairs is not None:
        with torch.no_grad():
            val_logits, _ = model(x_val, ei_val, ep_val, ea_val)
            val_probs = torch.sigmoid(val_logits)

        K = 50
        topk = torch.topk(val_probs, k=min(K, len(val_probs)))
        topk_idx = topk.indices.cpu().numpy()
        topk_scores = topk.values.cpu().numpy()

        print(f"\n验证集 Top-{K} 高分边对应的基因对:")
        for rank, (idx, score) in enumerate(zip(topk_idx, topk_scores), 1):
            pair = val_edge_pairs[idx]
            if isinstance(pair, str) and "|" in pair:
                lig, rec = pair.split("|", 1)
                print(f"{rank:03d}. {lig} → {rec} (score={score:.3f})")
            else:
                print(f"{rank:03d}. {pair} (score={score:.3f})")
    else:
        print("⚠️ val_data 中没有 edge_lr_pairs 字段，无法输出基因对。")
