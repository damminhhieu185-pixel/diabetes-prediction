import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, learning_curve, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ============================================================
# ĐỌC DỮ LIỆU
# ============================================================
df = pd.read_csv('backend/diabetes_standardized.csv')
X = df.drop(columns=['Outcome'])
y = df['Outcome']

model = joblib.load('backend/model/diabetes_model.pkl')

print("="*60)
print("PHAN 1: CROSS-VALIDATION CHI TIET")
print("="*60)

# ============================================================
# PHẦN 1: CROSS-VALIDATION 5-FOLD CHO TẤT CẢ MODEL
# ============================================================
models = {
    'Random Forest': model,
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {}
metrics = ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']

for name, m in models.items():
    cv_results[name] = {}
    for metric in metrics:
        scores = cross_val_score(m, X, y, cv=cv, scoring=metric)
        cv_results[name][metric] = scores
        print(f"{name} | {metric}: {scores.mean():.4f} +/- {scores.std():.4f}")
    print()

# ============================================================
# BIỂU ĐỒ 1: Boxplot Cross-Validation F1 Score 5 model
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Boxplot F1
f1_data = [cv_results[name]['f1'] for name in models.keys()]
bp = axes[0].boxplot(f1_data, patch_artist=True,
                     labels=list(models.keys()),
                     medianprops=dict(color='black', linewidth=2))
colors = ['#E25C5C', '#4A90D9', '#50C878', '#F4A460', '#9B59B6']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[0].set_title('So sanh F1-Score qua 5-Fold Cross Validation',
                  fontsize=13, fontweight='bold')
axes[0].set_ylabel('F1-Score', fontsize=12)
axes[0].set_xticklabels(list(models.keys()), rotation=15, ha='right')
axes[0].grid(axis='y', alpha=0.3)
axes[0].axhline(y=cv_results['Random Forest']['f1'].mean(),
                color='#E25C5C', linestyle='--', alpha=0.5,
                label=f"RF mean: {cv_results['Random Forest']['f1'].mean():.3f}")
axes[0].legend(fontsize=10)

# Boxplot AUC-ROC
auc_data = [cv_results[name]['roc_auc'] for name in models.keys()]
bp2 = axes[1].boxplot(auc_data, patch_artist=True,
                      labels=list(models.keys()),
                      medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1].set_title('So sanh AUC-ROC qua 5-Fold Cross Validation',
                  fontsize=13, fontweight='bold')
axes[1].set_ylabel('AUC-ROC', fontsize=12)
axes[1].set_xticklabels(list(models.keys()), rotation=15, ha='right')
axes[1].grid(axis='y', alpha=0.3)
axes[1].axhline(y=cv_results['Random Forest']['roc_auc'].mean(),
                color='#E25C5C', linestyle='--', alpha=0.5,
                label=f"RF mean: {cv_results['Random Forest']['roc_auc'].mean():.3f}")
axes[1].legend(fontsize=10)

plt.suptitle('Danh gia do on dinh cua cac mo hinh qua Cross Validation',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('cross_validation_boxplot.png', bbox_inches='tight', dpi=150)
plt.close()
print("Xong bieu do 1: cross_validation_boxplot.png")

# ============================================================
# BIỂU ĐỒ 2: Barplot tất cả metrics của Random Forest
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

rf_metrics = {
    'Accuracy': cv_results['Random Forest']['accuracy'].mean(),
    'F1-Score': cv_results['Random Forest']['f1'].mean(),
    'AUC-ROC': cv_results['Random Forest']['roc_auc'].mean(),
    'Precision': cv_results['Random Forest']['precision'].mean(),
    'Recall': cv_results['Random Forest']['recall'].mean()
}
rf_stds = {
    'Accuracy': cv_results['Random Forest']['accuracy'].std(),
    'F1-Score': cv_results['Random Forest']['f1'].std(),
    'AUC-ROC': cv_results['Random Forest']['roc_auc'].std(),
    'Precision': cv_results['Random Forest']['precision'].std(),
    'Recall': cv_results['Random Forest']['recall'].std()
}

bars = ax.bar(rf_metrics.keys(), rf_metrics.values(),
              yerr=rf_stds.values(), capsize=8,
              color=['#E25C5C','#4A90D9','#50C878','#F4A460','#9B59B6'],
              edgecolor='white', alpha=0.85, error_kw={'linewidth': 2})
for bar, val in zip(bars, rf_metrics.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.4f}', ha='center', va='bottom',
            fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.set_title('Tat ca chi so danh gia cua Random Forest (5-Fold CV)',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Diem so trung binh', fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('rf_metrics_barplot.png', bbox_inches='tight', dpi=150)
plt.close()
print("Xong bieu do 2: rf_metrics_barplot.png")

# ============================================================
# PHẦN 2: LEARNING CURVE
# ============================================================
print()
# ============================================================
# PHẦN 1.5: SO SÁNH TRƯỚC VÀ SAU KHI DÙNG SMOTE
# ============================================================
print()
print("="*60)
print("PHAN 1.5: SMOTE - XU LY MAT CAN BANG NHAN")
print("="*60)

# Trước SMOTE
print("Truoc SMOTE:")
print(f"  Khong benh (0): {(y==0).sum()} mau")
print(f"  Co benh    (1): {(y==1).sum()} mau")

# Áp dụng SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

print("\nSau SMOTE:")
print(f"  Khong benh (0): {(y_smote==0).sum()} mau")
print(f"  Co benh    (1): {(y_smote==1).sum()} mau")

# So sánh kết quả RF trước và sau SMOTE
rf_before = cross_val_score(model, X, y, cv=cv, scoring='f1')
rf_after = cross_val_score(
    RandomForestClassifier(n_estimators=300, max_depth=4,
                           min_samples_split=5, min_samples_leaf=2,
                           random_state=42),
    X_smote, y_smote, cv=cv, scoring='f1'
)

print(f"\nRandom Forest TRUOC SMOTE - F1: {rf_before.mean():.4f} +/- {rf_before.std():.4f}")
print(f"Random Forest SAU SMOTE   - F1: {rf_after.mean():.4f} +/- {rf_after.std():.4f}")

# Vẽ biểu đồ so sánh trước và sau SMOTE
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Biểu đồ tròn trước SMOTE
axes[0].pie([500, 268], labels=['Khong benh (500)', 'Co benh (268)'],
            colors=['#4A90D9', '#E25C5C'], autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11})
axes[0].set_title('Truoc SMOTE\n(Mat can bang)', fontsize=12, fontweight='bold')

# Biểu đồ tròn sau SMOTE
axes[1].pie([500, 500], labels=['Khong benh (500)', 'Co benh (500)'],
            colors=['#4A90D9', '#E25C5C'], autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11})
axes[1].set_title('Sau SMOTE\n(Can bang)', fontsize=12, fontweight='bold')

# Barplot so sánh F1 trước và sau SMOTE
means = [rf_before.mean(), rf_after.mean()]
stds = [rf_before.std(), rf_after.std()]
bars = axes[2].bar(['Truoc SMOTE', 'Sau SMOTE'], means,
                   yerr=stds, capsize=8,
                   color=['#4A90D9', '#E25C5C'],
                   edgecolor='white', alpha=0.85,
                   error_kw={'linewidth': 2})
for bar, val in zip(bars, means):
    axes[2].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f'{val:.4f}', ha='center', va='bottom',
                 fontsize=12, fontweight='bold')
axes[2].set_ylim(0, 1.0)
axes[2].set_title('F1-Score: Truoc vs Sau SMOTE',
                  fontsize=12, fontweight='bold')
axes[2].set_ylabel('F1-Score', fontsize=11)
axes[2].grid(axis='y', alpha=0.3)

plt.suptitle('Ket qua xu ly mat can bang nhan bang SMOTE',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('smote_comparison.png', bbox_inches='tight', dpi=150)
plt.close()
print("\nXong bieu do SMOTE: smote_comparison.png")
print("="*60)
print("PHAN 2: LEARNING CURVE")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Learning Curve cho Random Forest
train_sizes, train_scores, val_scores = learning_curve(
    model, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=cv,
    scoring='f1',
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

axes[0].plot(train_sizes, train_mean, 'o-', color='#E25C5C',
             linewidth=2, markersize=8, label='Train Score')
axes[0].fill_between(train_sizes,
                     train_mean - train_std,
                     train_mean + train_std,
                     alpha=0.15, color='#E25C5C')
axes[0].plot(train_sizes, val_mean, 's-', color='#4A90D9',
             linewidth=2, markersize=8, label='Validation Score')
axes[0].fill_between(train_sizes,
                     val_mean - val_std,
                     val_mean + val_std,
                     alpha=0.15, color='#4A90D9')
axes[0].set_title('Learning Curve - Random Forest', fontsize=13, fontweight='bold')
axes[0].set_xlabel('So mau huan luyen', fontsize=12)
axes[0].set_ylabel('F1-Score', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)
axes[0].set_ylim(0, 1.05)

# Learning Curve cho XGBoost (so sánh)
xgb = XGBClassifier(eval_metric='logloss', random_state=42)
train_sizes2, train_scores2, val_scores2 = learning_curve(
    xgb, X, y,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=cv,
    scoring='f1',
    n_jobs=-1
)

train_mean2 = train_scores2.mean(axis=1)
train_std2 = train_scores2.std(axis=1)
val_mean2 = val_scores2.mean(axis=1)
val_std2 = val_scores2.std(axis=1)

axes[1].plot(train_sizes2, train_mean2, 'o-', color='#E25C5C',
             linewidth=2, markersize=8, label='Train Score')
axes[1].fill_between(train_sizes2,
                     train_mean2 - train_std2,
                     train_mean2 + train_std2,
                     alpha=0.15, color='#E25C5C')
axes[1].plot(train_sizes2, val_mean2, 's-', color='#4A90D9',
             linewidth=2, markersize=8, label='Validation Score')
axes[1].fill_between(train_sizes2,
                     val_mean2 - val_std2,
                     val_mean2 + val_std2,
                     alpha=0.15, color='#4A90D9')
axes[1].set_title('Learning Curve - XGBoost', fontsize=13, fontweight='bold')
axes[1].set_xlabel('So mau huan luyen', fontsize=12)
axes[1].set_ylabel('F1-Score', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)
axes[1].set_ylim(0, 1.05)

plt.suptitle('Learning Curve: Random Forest vs XGBoost',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('learning_curve.png', bbox_inches='tight', dpi=150)
plt.close()
print("Xong bieu do 3: learning_curve.png")

print()
print("="*60)
print("HOAN THANH - 3 bieu do da duoc luu")
print("  1. cross_validation_boxplot.png")
print("  2. rf_metrics_barplot.png")
print("  3. learning_curve.png")
print("="*60)