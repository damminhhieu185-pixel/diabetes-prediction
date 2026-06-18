import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

model  = joblib.load("backend/model/diabetes_model.pkl")
scaler = joblib.load("backend/model/scaler.pkl")

df = pd.read_csv("backend/diabetes_standardized.csv")
X  = df.drop(columns=["Outcome"])
y  = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_prob):.4f}")

# ── Confusion Matrix ──────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Không mắc (0)', 'Mắc bệnh (1)'],
            yticklabels=['Không mắc (0)', 'Mắc bệnh (1)'],
            annot_kws={"size": 14, "weight": "bold"})
plt.title("Confusion Matrix", fontsize=13, fontweight='bold')
plt.ylabel("Thực tế (Actual)")
plt.xlabel("Dự đoán (Predicted)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
print("Đã lưu: confusion_matrix.png")

# ── ROC Curve ─────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', linewidth=2, label=f"Random Forest (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label="Random (AUC = 0.500)")
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve", fontsize=13, fontweight='bold')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()
print("Đã lưu: roc_curve.png")



# ── Định nghĩa các mô hình ────────────────────────────────────
models = {
    "Random Forest":       model,  # dùng lại model đã load
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM":                 SVC(probability=True, random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=5),
    "XGBoost":             XGBClassifier(eval_metric='logloss', random_state=42),
}

# ── Train & đánh giá từng mô hình ────────────────────────────
results = {}
for name, m in models.items():
    if name != "Random Forest":
        m.fit(X_train, y_train)
    yp  = m.predict(X_test)
    ypr = m.predict_proba(X_test)[:, 1]
    results[name] = {
        "Accuracy":  round(accuracy_score(y_test, yp), 4),
        "Precision": round(precision_score(y_test, yp), 4),
        "Recall":    round(recall_score(y_test, yp), 4),
        "F1":        round(f1_score(y_test, yp), 4),
        "AUC-ROC":   round(roc_auc_score(y_test, ypr), 4),
    }


df_results = pd.DataFrame(results).T
print("\n=== BẢNG SO SÁNH CÁC MÔ HÌNH ===")
print(df_results.to_string())

# ── Vẽ biểu đồ so sánh ───────────────────────────────────────
df_results.plot(kind='bar', figsize=(12, 6), colormap='Set2', edgecolor='black')
plt.title("So Sánh Hiệu Suất Các Mô Hình", fontsize=14, fontweight='bold')
plt.xlabel("Mô hình")
plt.ylabel("Điểm số")
plt.xticks(rotation=15)
plt.ylim(0, 1.05)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()
print("Đã lưu: model_comparison.png")



# ── Feature Importance ────────────────────────────────────────
importances = model.feature_importances_
feat_df = pd.DataFrame({
    "Feature":    X.columns,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print("\n=== FEATURE IMPORTANCE ===")
print(feat_df.to_string(index=False))

# Bản dịch tiếng Việt
feature_translations = {
    'Pregnancies':              'Số lần mang thai',
    'Glucose':                  'Chỉ số Glucose',
    'BloodPressure':            'Huyết áp tâm trương',
    'SkinThickness':            'Độ dày nếp gấp da',
    'Insulin':                  'Chỉ số Insulin',
    'BMI':                      'Chỉ số BMI',
    'DiabetesPedigreeFunction': 'Phả hệ tiểu đường',
    'Age':                      'Tuổi',
    'Age_Group':                'Nhóm tuổi',
    'BMI_Category':             'Phân loại BMI',
    'Glucose_BMI':              'Tương tác Glucose * BMI',
}
feat_df["Feature_VI"] = feat_df["Feature"].map(lambda x: feature_translations.get(x, x))

# ── Vẽ biểu đồ ───────────────────────────────────────────────
plt.figure(figsize=(10, 6))
sns.barplot(data=feat_df, x="Importance", y="Feature_VI",
            palette="viridis", hue="Feature_VI", legend=False)
plt.title("Mức Độ Ảnh Hưởng Của Các Chỉ Số Đến Nguy Cơ Tiểu Đường",
          fontsize=13, fontweight='bold')
plt.xlabel("Feature Importance Score")
plt.ylabel("")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
print("Đã lưu: feature_importance.png")

from sklearn.model_selection import cross_val_score
from scipy import stats

# ── Cross-validation scores cho từng mô hình ─────────────────
print("\n=== KIỂM ĐỊNH THỐNG KÊ (t-test) ===")

rf_scores  = cross_val_score(model, X, y, cv=5, scoring='f1')
lr_scores  = cross_val_score(LogisticRegression(max_iter=1000, random_state=42), X, y, cv=5, scoring='f1')
svm_scores = cross_val_score(SVC(probability=True, random_state=42), X, y, cv=5, scoring='f1')
knn_scores = cross_val_score(KNeighborsClassifier(n_neighbors=5), X, y, cv=5, scoring='f1')
xgb_scores = cross_val_score(XGBClassifier(eval_metric='logloss', random_state=42), X, y, cv=5, scoring='f1')

print(f"\nRandom Forest  F1 trung bình: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
print(f"Logistic Reg.  F1 trung bình: {lr_scores.mean():.4f} ± {lr_scores.std():.4f}")
print(f"SVM            F1 trung bình: {svm_scores.mean():.4f} ± {svm_scores.std():.4f}")
print(f"KNN            F1 trung bình: {knn_scores.mean():.4f} ± {knn_scores.std():.4f}")
print(f"XGBoost        F1 trung bình: {xgb_scores.mean():.4f} ± {xgb_scores.std():.4f}")

# ── So sánh Random Forest với từng mô hình ───────────────────
comparisons = {
    "Logistic Regression": lr_scores,
    "SVM":                 svm_scores,
    "KNN":                 knn_scores,
    "XGBoost":             xgb_scores,
}

print("\n--- So sánh Random Forest với các mô hình khác ---")
for name, scores in comparisons.items():
    t_stat, p_value = stats.ttest_ind(rf_scores, scores)
    ket_luan = "Có ý nghĩa thống kê ✓" if p_value < 0.05 else "Không có ý nghĩa thống kê"
    print(f"RF vs {name:<22} t={t_stat:+.4f}  p={p_value:.4f}  → {ket_luan}")