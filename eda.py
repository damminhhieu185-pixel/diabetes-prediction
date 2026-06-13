import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('diabetes.csv')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

# ============================================================
# BIỂU ĐỒ 1: Phân phối Outcome
# ============================================================
fig, ax = plt.subplots(figsize=(6,6))
sizes = [500, 268]
labels = ['Không bệnh (500)', 'Có bệnh (268)']
colors = ['#4A90D9','#E25C5C']
wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
    autopct='%1.1f%%', startangle=90, textprops={'fontsize':13})
for at in autotexts:
    at.set_fontsize(13)
    at.set_color('white')
    at.set_fontweight('bold')
ax.set_title('Phan phoi nhan Outcome', fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('bieu_do_1_outcome.png', bbox_inches='tight')
plt.close()
print('Xong bieu do 1')

# ============================================================
# BIỂU ĐỒ 2: Histogram
# ============================================================
cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
        'Insulin','BMI','DiabetesPedigreeFunction','Age']
fig, axes = plt.subplots(2, 4, figsize=(18,9))
for i, col in enumerate(cols):
    ax = axes[i//4][i%4]
    ax.hist(df[df['Outcome']==0][col], bins=20, alpha=0.6,
            color='#4A90D9', label='Khong benh')
    ax.hist(df[df['Outcome']==1][col], bins=20, alpha=0.6,
            color='#E25C5C', label='Co benh')
    ax.set_title(col, fontsize=12, fontweight='bold')
    ax.set_xlabel('Gia tri', fontsize=10)
    ax.set_ylabel('So luong', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
plt.suptitle('Phan phoi cac dac trung theo nhom benh',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('bieu_do_2_histogram.png', bbox_inches='tight')
plt.close()
print('Xong bieu do 2')

# ============================================================
# BIỂU ĐỒ 3: Heatmap tương quan
# ============================================================
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='RdYlBu_r',
            ax=ax, linewidths=0.5, annot_kws={'size':11},
            vmin=-1, vmax=1)
ax.set_title('Ma tran tuong quan giua cac dac trung',
             fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('bieu_do_3_heatmap.png', bbox_inches='tight')
plt.close()
print('Xong bieu do 3')

# ============================================================
# BIỂU ĐỒ 4: Boxplot
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(18,9))
for i, col in enumerate(cols):
    ax = axes[i//4][i%4]
    data0 = df[df['Outcome']==0][col]
    data1 = df[df['Outcome']==1][col]
    bp = ax.boxplot([data0, data1], patch_artist=True,
                    labels=['Khong benh','Co benh'],
                    medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor('#4A90D9')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#E25C5C')
    bp['boxes'][1].set_alpha(0.7)
    ax.set_title(col, fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
plt.suptitle('Boxplot cac dac trung theo nhom benh',
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('bieu_do_4_boxplot.png', bbox_inches='tight')
plt.close()
print('Xong bieu do 4')

# ============================================================
# BIỂU ĐỒ 5: Tương quan với Outcome
# ============================================================
fig, ax = plt.subplots(figsize=(10,6))
corr_outcome = df.corr()['Outcome'].drop('Outcome').sort_values(ascending=True)
colors_bar = ['#E25C5C' if v > 0 else '#4A90D9' for v in corr_outcome]
bars = ax.barh(corr_outcome.index, corr_outcome.values,
               color=colors_bar, edgecolor='white', height=0.6)
for bar, val in zip(bars, corr_outcome.values):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=11, fontweight='bold')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('He so tuong quan voi Outcome', fontsize=12)
ax.set_title('Muc do tuong quan cua tung dac trung voi Outcome',
             fontsize=13, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('bieu_do_5_correlation.png', bbox_inches='tight')
plt.close()
print('Xong bieu do 5')

print()
print('HOAN THANH - 5 bieu do da duoc luu vao thu muc hien tai')