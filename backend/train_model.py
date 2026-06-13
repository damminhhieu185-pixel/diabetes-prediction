import os
import sys
import json
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Thiết lập UTF-8 cho stdout/stderr để tránh lỗi hiển thị tiếng Việt trên Windows console
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# Thiết lập thư mục lưu trữ mô hình
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

# Đường dẫn tệp dữ liệu chuẩn hóa sẵn
DATA_PATH = os.path.join(os.path.dirname(__file__), "diabetes_standardized.csv")


def download_dataset():
    """Kiểm tra sự tồn tại của tệp dữ liệu chuẩn hóa."""
    if not os.path.exists(DATA_PATH):
        print(f"Lỗi: Không tìm thấy tệp {DATA_PATH}!")
        raise FileNotFoundError(f"Vui lòng đặt tệp diabetes_standardized.csv tại: {DATA_PATH}")
    else:
        print("Tập dữ liệu đã chuẩn hóa tồn tại cục bộ.")

def load_and_preprocess_data():
    """Tải dữ liệu từ tệp chuẩn hóa sẵn, phân chia tập train/test."""
    df = pd.read_csv(DATA_PATH)
    print(f"Kích thước tập dữ liệu standardized: {df.shape}")
    
    # Phân chia đặc trưng (X) và nhãn mục tiêu (y)
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    # Chia tập dữ liệu thành Train (80%) và Test (20%) - Sử dụng stratify để giữ tỷ lệ nhãn ổn định
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Dữ liệu đã chuẩn hóa sẵn. Để giữ tương thích và tránh lỗi trong app.py,
    # ta lưu một dummy scaler được fit trực tiếp trên dữ liệu train đã chuẩn hóa này.
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Đã lưu StandardScaler vào: {scaler_path}")
    
    # Giá trị điền khuyết mặc định dùng cho lưu trữ metadata và tham chiếu
    imputation_values = {
        'Glucose': 117.0,
        'BloodPressure': 72.0,
        'SkinThickness': 29.0,
        'Insulin': 125.0,
        'BMI': 32.3
    }
    
    return X_train.values, X_test.values, y_train, y_test, X.columns.tolist(), imputation_values

def train_and_optimize_model(X_train, y_train):
    """Huấn luyện mô hình Random Forest sử dụng Grid Search tìm siêu tham số tốt nhất."""
    print("Bắt đầu huấn luyện mô hình Random Forest...")
    
    # Định nghĩa mô hình cơ sở
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    # Tập hợp các siêu tham số cần tìm kiếm để tránh overfitting
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [2, 4, 6]
    }
    
    # Sử dụng GridSearchCV với StratifiedKFold (cv=5) để tối ưu
    grid_search = GridSearchCV(
        estimator=rf, param_grid=param_grid, cv=5, 
        scoring='f1', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    print("Siêu tham số tối ưu nhất tìm được:")
    print(grid_search.best_params_)
    
    # Lưu mô hình tốt nhất
    model_path = os.path.join(MODEL_DIR, 'diabetes_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"Đã lưu mô hình tốt nhất vào: {model_path}")
    
    return best_model, grid_search.best_params_

def evaluate_and_plot(model, X_test, y_test, feature_names):
    """Đánh giá mô hình, tạo báo cáo và vẽ các đồ thị phân tích."""
    # Dự đoán trên tập kiểm thử
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Tính toán các chỉ số đánh giá
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "auc_roc": float(auc)
    }
    
    print("\n--- BÁO CÁO ĐÁNH GIÁ MÔ HÌNH ---")
    print(f"Độ chính xác (Accuracy): {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Độ nhạy): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC ROC: {auc:.4f}")
    print("\nChi tiết Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # --- Vẽ biểu đồ Độ quan trọng của các đặc trưng (Feature Importance) ---
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Bản dịch tiếng Việt cho các đặc trưng để trực quan hóa
    feature_translations = {
        'Pregnancies': 'Số lần mang thai',
        'Glucose': 'Chỉ số Glucose',
        'BloodPressure': 'Huyết áp tâm trương',
        'SkinThickness': 'Độ dày nếp gấp da',
        'Insulin': 'Chỉ số Insulin',
        'BMI': 'Chỉ số BMI',
        'DiabetesPedigreeFunction': 'Phả hệ tiểu đường',
        'Age': 'Tuổi',
        'Age_Group': 'Nhóm tuổi',
        'BMI_Category': 'Phân loại BMI',
        'Glucose_BMI': 'Tương tác Glucose * BMI'
    }
    
    translated_features = [feature_translations.get(f, f) for f in feature_names]
    sorted_features = [translated_features[i] for i in indices]
    sorted_importances = importances[indices]
    
    # Vẽ biểu đồ thanh ngang
    colors = sns.color_palette("viridis", len(sorted_importances))
    sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis", hue=sorted_features, legend=False)
    plt.title("Mức Độ Ảnh Hưởng Của Các Chỉ Số Y Tế Đến Nguy Cơ Tiểu Đường", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Mức độ quan trọng (Feature Importance Score)", fontsize=12)
    plt.tight_layout()
    
    feat_imp_path = os.path.join(MODEL_DIR, 'feature_importance.png')
    plt.savefig(feat_imp_path, dpi=300)
    plt.close()
    print(f"Đã lưu biểu đồ Feature Importance vào: {feat_imp_path}")
    
    # --- Vẽ Ma trận Nhầm lẫn (Confusion Matrix) ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', 
        xticklabels=['Không mắc bệnh (0)', 'Mắc bệnh (1)'],
        yticklabels=['Không mắc bệnh (0)', 'Mắc bệnh (1)'],
        cbar=False, annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title("Ma Trận Nhầm Lẫn (Confusion Matrix)", fontsize=13, fontweight='bold', pad=15)
    plt.ylabel("Thực tế (Actual)", fontsize=12)
    plt.xlabel("Dự đoán (Predicted)", fontsize=12)
    plt.tight_layout()
    
    cm_path = os.path.join(MODEL_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Đã lưu biểu đồ Confusion Matrix vào: {cm_path}")
    
    return metrics

def main():
    print("=== BẮT ĐẦU QUY TRÌNH XÂY DỰNG VÀ HUẤN LUYỆN MÔ HÌNH ===")
    download_dataset()
    
    # Load và xử lý dữ liệu
    X_train, X_test, y_train, y_test, feature_names, imputation_values = load_and_preprocess_data()
    
    # Huấn luyện mô hình tốt nhất
    best_model, best_params = train_and_optimize_model(X_train, y_train)
    
    # Đánh giá mô hình
    metrics = evaluate_and_plot(best_model, X_test, y_test, feature_names)
    
    # Lưu metadata tổng hợp (bao gồm cả giá trị điền khuyết)
    metadata = {
        "model_type": "Random Forest Classifier",
        "best_hyperparameters": best_params,
        "features": feature_names,
        "imputation_medians": imputation_values,
        "metrics": metrics
    }
    
    metadata_path = os.path.join(MODEL_DIR, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print(f"Đã lưu file metadata chi tiết vào: {metadata_path}")
    print("\n=== HOÀN THÀNH QUY TRÌNH HUẤN LUYỆN VÀ LƯU TRỮ MÔ HÌNH THÀNH CÔNG ===")

if __name__ == "__main__":
    main()
