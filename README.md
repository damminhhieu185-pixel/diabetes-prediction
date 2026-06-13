# Xây dựng hệ thống dự đoán khả năng mắc bệnh tiểu đường dựa trên dữ liệu y tế bằng thuật toán Random Forest

## Giới thiệu
Đây là đồ án môn Học máy - xây dựng hệ thống dự đoán khả năng mắc bệnh tiểu đường
dựa trên dữ liệu y tế sử dụng thuật toán Random Forest.

## Nhóm thực hiện: Nhóm 10
| Thành viên | Vai trò | Nhiệm vụ |

| [Đàm Minh Hiếu | Nhóm trưởng | 
|   Thu thập dữ liệu
•	Tìm kiếm & tải dataset (Pima Indians, UCI)
•	Phân tích thống kê mô tả ban đầu (EDA)
•	Kiểm tra missing values, outliers, phân phối
•	Phối hợp tiến độ giữa các nhóm, họp daily 15 phút
•	Viết mục 1, 2 báo cáo: Giới thiệu & Dữ liệu
|
| [] | Thành viên | Tiền xử lý dữ liệu |
| [Tên P3] | Thành viên | Xây dựng mô hình Random Forest |
| [Tên P4] | Thành viên | Đánh giá và so sánh mô hình |
| [Tên P5] | Thành viên | Visualization và Demo app |
| [Tên P6] | Thành viên | Báo cáo và thuyết trình |

## Dataset
- **Tên:** Pima Indians Diabetes Database
- **Nguồn:** UCI Machine Learning Repository / Kaggle
- **Số mẫu:** 768 bệnh nhân
- **Số đặc trưng:** 8 chỉ số y tế
- **Nhãn:** Outcome (0 = không bệnh, 1 = có bệnh)

## Cấu trúc thư mục
diabetes-prediction/

│

├── diabetes.csv          # Dataset gốc

├── diabetes_clean.csv    # Dataset sau tiền xử lý (P2 tạo)

├── eda.py                # Code phân tích và vẽ biểu đồ EDA

├── phan_tich.py          # Code phân tích sâu dữ liệu

├── preprocessing.py      # Code tiền xử lý (P2)

├── model.py              # Code xây dựng mô hình (P3)

├── evaluation.py         # Code đánh giá mô hình (P4)

├── app.py                # Demo app Streamlit (P5)

├── model_rf.pkl          # Model đã huấn luyện (P3 tạo)

└── README.md             # Mô tả dự án

## Kết quả EDA nổi bật
- Glucose là yếu tố quan trọng nhất (tương quan 0.467 với Outcome)
- Người bệnh có Glucose trung bình 141.26 so với 109.98 của người không bệnh
- 48.7% giá trị Insulin bị ghi sai thành 0 — cần xử lý trước khi train
- Dataset mất cân bằng nhẹ: 65.1% không bệnh vs 34.9% có bệnh

## Cài đặt và chạy
bash

# Cài thư viện
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib

# Chạy EDA
python eda.py

# Chạy demo app (sau khi có model)
streamlit run app.py


## Môn học
Học máy — Trường Công Nghệ Thông Tin và Truyền Thông - Đại học Công Nghiệp Hà Nội