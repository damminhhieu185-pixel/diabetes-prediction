import streamlit as st
import requests

# 1. Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Dự đoán nguy cơ Tiểu Đường",
    page_icon="🩺",
    layout="wide"
)

# Đường dẫn gọi API tới file Flask (app.py) đang chạy
API_URL = "http://127.0.0.1:5000/predict"

# --- GIAO DIỆN CHÍNH ---
st.title("🩺 Hệ Thống Dự Đoán Nguy Cơ Tiểu Đường ")


# ================= KHU VỰC 1: NHẬP CHỈ SỐ SỨC KHỎE =================
st.header("📋 Chỉ số Sức Khỏe")

row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
with row1_col1:
    pregnancies = st.number_input("Số lần mang thai (Pregnancies)", min_value=0, max_value=20, value=1)
with row1_col2:
    glucose = st.number_input("Chỉ số Đường huyết (Glucose - mg/dL)", min_value=0, max_value=300, value=85)
with row1_col3:
    blood_pressure = st.number_input("Huyết áp tâm trương (BloodPressure - mmHg)", min_value=0, max_value=200, value=70)
with row1_col4:
    skin_thickness = st.number_input("Độ dày nếp gấp da (SkinThickness - mm)", min_value=0, max_value=100, value=20)

row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
with row2_col1:
    insulin = st.number_input("Chỉ số Insulin (mu U/ml)", min_value=0, max_value=900, value=75)
with row2_col2:
    bmi = st.number_input("Chỉ số Khối cơ thể (BMI)", min_value=0.0, max_value=70.0, value=21.5, step=0.1)
with row2_col3:
    dpf = st.number_input("Hệ số di truyền tiểu đường (Pedigree)", min_value=0.0, max_value=3.0, value=0.25, step=0.01, format="%.3f")
with row2_col4:
    age = st.number_input("Tuổi (Age)", min_value=1, max_value=120, value=23)
    
_, btn_col, _ = st.columns([1.5, 1, 1.5])
with btn_col:
    st.write("")
    submit_btn = st.button("Tính toán từ Mô Hình 📊", type="primary", use_container_width=True)

st.markdown("---")

# ================= KHU VỰC 2: HIỂN THỊ KẾT QUẢ TỪ API =================
st.header("📊 Kết Quả Phân Tích")

if submit_btn:
    # 1. Gom dữ liệu thô 8 đặc trưng gửi đi
    raw_input = {
        "Pregnancies": pregnancies, 
        "Glucose": glucose, 
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness, 
        "Insulin": insulin, 
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf, 
        "Age": age
    }
    
    try:
        # 2. Gọi thẳng đến file app.py qua API POST (đảm bảo logic mô hình luôn chuẩn 100%)
        response = requests.post(API_URL, json=raw_input)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "success":
                
                # Trích xuất dữ liệu trả về từ mô hình
                prediction = result["prediction"]
                probability = result["probability"]
                risk_status = result["risk_status"]
                recommendations = result["recommendations"]
                
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    if prediction == 1:
                        st.error("### ⚠️ CẢNH BÁO:\n**CÓ NGUY CƠ CAO MẮC BỆNH**")
                        st.metric(label="Xác suất mắc bệnh", value=f"{probability*100:.2f}%", delta="Nguy cơ cao", delta_color="inverse")
                    else:
                        st.success("### ✅ TRẠNG THÁI:\n**NGUY CƠ THẤP (BÌNH THƯỜNG)**")
                        st.metric(label="Xác suất mắc bệnh", value=f"{probability*100:.2f}%", delta="An toàn")
                        
                with res_col2:
                    st.subheader(f"Trạng thái hệ thống đánh giá: **{risk_status}**")
                    st.write("💡 **Lời khuyên & Khuyến nghị y tế:**")
                    for rec in recommendations:
                        st.write(f"- {rec}")
            else:
                st.error(f"Lỗi từ mô hình: {result.get('message')}")
        else:
            st.error("Lỗi: Không thể nhận kết quả từ app.py. Hãy chắc chắn bạn đã chạy `python app.py` ở một Terminal khác.")
            
    except requests.exceptions.ConnectionError:
        st.error("Không thể kết nối đến Máy chủ Mô hình! \n\n**Bạn cần mở thêm một Terminal mới và chạy lệnh:** `python app.py`")
else:
    st.info("Hãy nhập thông số và bấm nút phân tích. Ứng dụng sẽ lấy kết quả từ thuật toán trong file app.py để hiển thị.")