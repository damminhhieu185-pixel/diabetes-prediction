import joblib
import json
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import os

print("⏳ Đang trích xuất Cây quyết định từ mô hình Random Forest...")

# 1. Khai báo đường dẫn đến file mô hình và metadata
model_path = os.path.join('model', 'diabetes_model.pkl')
metadata_path = os.path.join('model', 'metadata.json')

if not os.path.exists(model_path) or not os.path.exists(metadata_path):
    print("❌ Lỗi: Không tìm thấy mô hình. Hãy chắc chắn bạn đã chạy train_model.py trước.")
else:
    # 2. Tải mô hình và danh sách tên cột
    model = joblib.load(model_path)
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    feature_names = metadata['features']

    # 3. Rút ra MỘT cây quyết định đầu tiên (Estimator số 0) trong Rừng
    single_tree = model.estimators_[0]

    # 4. Thiết lập khung vẽ biểu đồ
    plt.figure(figsize=(20, 10)) # Kích thước ảnh to để chữ không bị mờ
    
    # Vẽ cây (Giới hạn max_depth=3 để cây không bị quá rậm rạp, dễ nhìn đưa vào báo cáo)
    plot_tree(
        single_tree, 
        feature_names=feature_names, 
        class_names=['An toàn (0)', 'Tiểu đường (1)'], 
        filled=True,      # Tô màu cho các hộp
        rounded=True,     # Bo góc các hộp
        max_depth=3,      # Cắt tỉa bớt các nhánh sâu để dễ đọc
        fontsize=10
    )
    
    plt.title("Trực quan hóa một Cây Quyết Định minh họa từ thuật toán Random Forest", fontsize=16, pad=20)
    
    # 5. Lưu ảnh
    output_file = os.path.join('model', 'decision_tree.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"🎉 Đã vẽ xong! Bạn hãy vào thư mục 'model' để lấy ảnh: {output_file}")