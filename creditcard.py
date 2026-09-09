# =========================
# 0. 設定（避免中文亂碼）
# =========================
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# =========================
# 1. 匯入套件
# =========================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve, auc,
    precision_score, recall_score, f1_score,
    precision_recall_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import seaborn as sns

# =========================
# 2. 讀取資料
# =========================
df = pd.read_csv("C:\\Users\\user\\.vscode\\creditcard\\creditcard.csv")

# =========================
# 3. 前處理
# =========================
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])

X = df.drop("Class", axis=1)
y = df["Class"]

# =========================
# 4. 切分資料
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

# =========================
# 5. Logistic Regression
# =========================
lr = LogisticRegression(
    class_weight='balanced',
    max_iter=1000
)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# =========================
# 6. Random Forest
# =========================
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# =========================
# 7. XGBoost
# =========================
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])

xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

# =========================
# 8. 模型評估
# =========================
print("=== Logistic Regression ===")
print(classification_report(y_test, y_pred_lr))

print("=== Random Forest ===")
print(classification_report(y_test, y_pred_rf))

print("=== XGBoost ===")
print(classification_report(y_test, y_pred_xgb))

# =========================
# 9. 模型比較圖
# =========================
models_name = ["Logistic", "RandomForest", "XGBoost"]

precision_list = [
    precision_score(y_test, y_pred_lr),
    precision_score(y_test, y_pred_rf),
    precision_score(y_test, y_pred_xgb)
]

recall_list = [
    recall_score(y_test, y_pred_lr),
    recall_score(y_test, y_pred_rf),
    recall_score(y_test, y_pred_xgb)
]

f1_list = [
    f1_score(y_test, y_pred_lr),
    f1_score(y_test, y_pred_rf),
    f1_score(y_test, y_pred_xgb)
]

x = np.arange(len(models_name))
width = 0.25

plt.figure(figsize=(8,5))
plt.bar(x - width, precision_list, width, label="Precision")
plt.bar(x, recall_list, width, label="Recall")
plt.bar(x + width, f1_list, width, label="F1-score")

plt.xticks(x, models_name)
plt.title("模型效能比較")
plt.ylabel("分數")
plt.legend()

plt.savefig("model_comparison.png", dpi=300)
plt.show()

# =========================
# 10. ROC 曲線 + AUC
# =========================
plt.figure()

prob_lr = lr.predict_proba(X_test)[:,1]
prob_rf = rf.predict_proba(X_test)[:,1]
prob_xgb = xgb.predict_proba(X_test)[:,1]

fpr_lr, tpr_lr, _ = roc_curve(y_test, prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, prob_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, prob_xgb)

auc_lr = auc(fpr_lr, tpr_lr)
auc_rf = auc(fpr_rf, tpr_rf)
auc_xgb = auc(fpr_xgb, tpr_xgb)

plt.plot(fpr_lr, tpr_lr, label=f"Logistic (AUC={auc_lr:.3f})")
plt.plot(fpr_rf, tpr_rf, label=f"RandomForest (AUC={auc_rf:.3f})")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC={auc_xgb:.3f})")

plt.plot([0,1],[0,1],'--')

plt.xlabel("假陽性率 (FPR)")
plt.ylabel("真正率 (TPR)")
plt.title("ROC 曲線比較")
plt.legend()

plt.savefig("roc_compare.png", dpi=300)
plt.show()

# =========================
# 11. 混淆矩陣
# =========================
models = {
    "Logistic": y_pred_lr,
    "RandomForest": y_pred_rf,
    "XGBoost": y_pred_xgb
}

for name, pred in models.items():
    cm = confusion_matrix(y_test, pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title(f"{name} 混淆矩陣")
    plt.xlabel("預測")
    plt.ylabel("實際")

    plt.savefig(f"{name}_cm.png", dpi=300)
    plt.show()

# =========================
# 12. Threshold vs Precision / Recall
# =========================
thresholds = np.linspace(0,1,50)

recalls = []
precisions = []

for t in thresholds:
    pred = (prob_xgb > t).astype(int)
    recalls.append(recall_score(y_test, pred))
    precisions.append(precision_score(y_test, pred))

plt.figure()
plt.plot(thresholds, recalls, label="Recall")
plt.plot(thresholds, precisions, label="Precision")

plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold 對 Precision / Recall 影響")
plt.legend()

plt.savefig("threshold_curve.png", dpi=300)
plt.show()

# =========================
# 13. PR Curve（加分）
# =========================
precision_curve, recall_curve, _ = precision_recall_curve(y_test, prob_xgb)

plt.figure()
plt.plot(recall_curve, precision_curve)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

plt.savefig("pr_curve.png", dpi=300)
plt.show()