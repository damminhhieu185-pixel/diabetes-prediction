import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('diabetes.csv')

print("="*60)
print("PHAN TICH SAU TUNG COT")
print("="*60)

cols = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
        'Insulin','BMI','DiabetesPedigreeFunction','Age']

for col in cols:
    print(f"\n--- {col} ---")
    print(f"  Trung binh nguoi KHONG benh : {df[df['Outcome']==0][col].mean():.2f}")
    print(f"  Trung binh nguoi CO benh    : {df[df['Outcome']==1][col].mean():.2f}")
    diff = df[df['Outcome']==1][col].mean() - df[df['Outcome']==0][col].mean()
    print(f"  Chenh lech                  : {diff:+.2f}")

print()
print("="*60)
print("KIEM TRA OUTLIERS BANG IQR")
print("="*60)
for col in cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
    print(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")

print()
print("="*60)
print("TUONG QUAN VOI OUTCOME (xep hang)")
print("="*60)
corr = df.corr()['Outcome'].drop('Outcome').abs().sort_values(ascending=False)
for i, (col, val) in enumerate(corr.items(), 1):
    print(f"  #{i} {col}: {val:.4f}")