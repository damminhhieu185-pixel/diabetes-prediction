import os
import sys
import json
import joblib
import numpy as np

# Thiết lập UTF-8 cho stdout/stderr để tránh lỗi hiển thị tiếng Việt trên Windows console
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
# Kích hoạt CORS để frontend có thể gọi API từ cổng khác nếu cần
CORS(app)

# Đường dẫn thư mục mô hình
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

# Biến toàn cục chứa mô hình, bộ chuẩn hóa và cấu hình
model = None
scaler = None
metadata = None
imputation_medians = {}
feature_order = []

def load_model_assets():
    """Tải các file mô hình và metadata đã lưu."""
    global model, scaler, metadata, imputation_medians, feature_order
    try:
        model_path = os.path.join(MODEL_DIR, 'diabetes_model.pkl')
        scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
        metadata_path = os.path.join(MODEL_DIR, 'metadata.json')
        
        if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(metadata_path)):
            print("Cảnh báo: Không tìm thấy đầy đủ các file mô hình. Hãy chạy train_model.py trước!")
            return False
            
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        imputation_medians = metadata.get('imputation_medians', {})
        feature_order = metadata.get('features', [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ])
        
        print("Tải mô hình và tài sản thành công!")
        print("Đặc trưng đầu vào:", feature_order)
        print("Giá trị điền khuyết mặc định:", imputation_medians)
        return True
    except Exception as e:
        print(f"Lỗi khi tải tài sản mô hình: {e}")
        return False

# Cố gắng tải mô hình khi khởi động server
load_model_assets()

@app.route('/retrain', methods=['POST'])
def retrain():
    """Kích hoạt chạy lại quy trình tải dữ liệu, tiền xử lý và huấn luyện mô hình."""
    try:
        print("Kích hoạt tái huấn luyện mô hình theo yêu cầu...")
        import train_model
        # Chạy quy trình huấn luyện đầy đủ
        train_model.main()
        
        # Tải lại tài sản mô hình mới
        success = load_model_assets()
        if success:
            return jsonify({
                "status": "success",
                "message": "Đã tái huấn luyện mô hình Random Forest và cập nhật các chỉ số đánh giá mới thành công!"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Huấn luyện hoàn tất nhưng gặp lỗi khi tải lại các tài sản mô hình."
            }), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": f"Lỗi trong quá trình tái huấn luyện: {str(e)}"
        }), 500

def get_health_recommendation(probability, data):
    """Đưa ra lời khuyên y tế dựa trên mức độ rủi ro và chỉ số cụ thể."""
    risk_pct = probability * 100
    glucose = data.get('Glucose', 0)
    bmi = data.get('BMI', 0)
    
    # Nếu là dữ liệu đã chuẩn hóa, khôi phục lại giá trị thô để tính toán lời khuyên chính xác
    if 'Age_Group' in data or 'BMI_Category' in data or (glucose is not None and float(glucose) < 10):
        try:
            glucose = (float(glucose) + 3.999421) / 0.032875
            bmi = (float(bmi) + 4.861584) / 0.150080
        except Exception:
            pass
            
    recs = []
    
    if risk_pct < 30:
        status = "Nguy cơ thấp"
        recs.append("Chỉ số sức khỏe của bạn hiện tại nằm trong phạm vi an toàn. Hãy tiếp tục duy trì chế độ sinh hoạt lành mạnh.")
        recs.append("Duy trì chế độ ăn giàu chất xơ, hạn chế đồ ngọt và tập thể dục tối thiểu 150 phút/tuần.")
    elif risk_pct < 60:
        status = "Nguy cơ trung bình (Tiền tiểu đường)"
        recs.append("Mức nguy cơ của bạn ở mức cảnh báo. Có dấu hiệu chỉ số glucose hoặc BMI đang cao hơn bình thường.")
        if glucose > 120:
            recs.append("Chỉ số đường huyết (Glucose) của bạn đang hơi cao. Nên hạn chế ăn tinh bột tinh chế và đường sữa ngọt.")
        if bmi > 25:
            recs.append("Chỉ số BMI cho thấy bạn đang thừa cân. Hãy cố gắng giảm 5-7% trọng lượng cơ thể bằng cách tăng vận động.")
        recs.append("Nên lên kế hoạch đi kiểm tra sức khỏe và xét nghiệm đường huyết định kỳ mỗi 6 tháng.")
    else:
        status = "Nguy cơ cao"
        recs.append("Hệ thống phát hiện nguy cơ mắc tiểu đường của bạn rất cao. Cần tham khảo ý kiến bác sĩ chuyên khoa sớm.")
        recs.append("Thực hiện kiểm tra đường huyết lúc đói và xét nghiệm chỉ số HbA1c tại cơ sở y tế gần nhất.")
        if glucose > 140:
            recs.append("Đường huyết của bạn ở mức đáng báo động. Hãy kiểm tra chế độ ăn ngay lập tức và theo dõi sát sao.")
        recs.append("Hạn chế tối đa chất béo bão hòa, carbohydrate chuyển hóa nhanh và tuân thủ lối sống năng động.")
        
    return {
        "status": status,
        "recommendations": recs
    }

@app.route('/predict', methods=['POST'])
def predict():
    """Nhận dữ liệu thô hoặc chuẩn hóa, thực hiện tiền xử lý và dự đoán."""
    global model, scaler, imputation_medians, feature_order
    
    # Kiểm tra xem mô hình đã được tải chưa
    if model is None:
        # Thử tải lại lần nữa
        if not load_model_assets():
            return jsonify({
                "status": "error",
                "message": "Mô hình chưa được huấn luyện hoặc tải lên hệ thống. Vui lòng chạy train_model.py trước."
            }), 500
            
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "Yêu cầu rỗng hoặc định dạng JSON không hợp lệ"}), 400
            
        print("Dữ liệu nhận được:", data)
        
        # Kiểm tra xem dữ liệu đầu vào đã được chuẩn hóa sẵn chưa (chứa đặc trưng Age_Group hoặc BMI_Category)
        is_standardized = 'Age_Group' in data or 'BMI_Category' in data
        
        if is_standardized:
            processed_features = []
            for feature in feature_order:
                val = data.get(feature)
                if val is None:
                    return jsonify({"status": "error", "message": f"Dữ liệu chuẩn hóa thiếu đặc trưng: {feature}"}), 400
                try:
                    processed_features.append(float(val))
                except ValueError:
                    return jsonify({"status": "error", "message": f"Chỉ số {feature} phải là số thực."}), 400
            features_scaled = np.array(processed_features).reshape(1, -1)
            input_processed_info = {feature_order[i]: processed_features[i] for i in range(len(feature_order))}
        else:
            # Dữ liệu đầu vào dạng thô (8 đặc trưng ban đầu)
            try:
                pregnancies = float(data.get('Pregnancies', 0.0))
                glucose = float(data.get('Glucose', 0.0))
                bp = float(data.get('BloodPressure', 0.0))
                skin = float(data.get('SkinThickness', 0.0))
                insulin = float(data.get('Insulin', 0.0))
                bmi = float(data.get('BMI', 0.0))
                dpf = float(data.get('DiabetesPedigreeFunction', 0.0))
                age = float(data.get('Age', 0.0))
            except (ValueError, TypeError):
                return jsonify({"status": "error", "message": "Các chỉ số đầu vào phải là số thực."}), 400
            
            # 1. Điền khuyết các giá trị 0 bằng trung vị của tập dữ liệu
            glucose_imputed = 117.0 if glucose <= 0 else glucose
            bp_imputed = 72.0 if bp <= 0 else bp
            skin_imputed = 29.0 if skin <= 0 else skin
            insulin_imputed = 125.0 if insulin <= 0 else insulin
            bmi_imputed = 32.3 if bmi <= 0 else bmi
            
            # 2. Tạo đặc trưng nhóm tuổi Age_Group (chuẩn hóa trực tiếp)
            if age <= 30:
                age_group_std = -0.7851246923597033
            elif age <= 45:
                age_group_std = 0.4305522506488695
            elif age <= 60:
                age_group_std = 1.6462291936574422
            else:
                age_group_std = 2.861906136666015
                
            # 3. Tạo đặc trưng phân loại BMI BMI_Category (chuẩn hóa trực tiếp)
            if bmi_imputed < 18.5:
                bmi_cat_std = -3.2945277421017813
            elif bmi_imputed < 25.0:
                bmi_cat_std = -1.9600354921365029
            elif bmi_imputed < 30.0:
                bmi_cat_std = -0.6255432421712243
            else:
                bmi_cat_std = 0.7089490077940542
                
            # 4. Tính toán tương tác Glucose_BMI (chuẩn hóa trực tiếp với BMI được cap tại 51.3)
            bmi_capped = min(max(bmi_imputed, 18.2), 51.3)
            raw_gb = glucose_imputed * bmi_capped
            glucose_bmi_std = 0.000690 * raw_gb - 2.751843
            
            # 5. Chuẩn hóa và winsorize 8 đặc trưng cơ bản
            pregnancies_std = 0.296966 * pregnancies - 1.141852
            glucose_std = 0.032875 * glucose_imputed - 3.999421
            bp_std = 0.082721 * bp_imputed - 5.987934
            
            skin_std = 0.113824 * skin_imputed - 3.313196
            skin_std = max(skin_std, -2.516429)
            
            insulin_std = 0.126447 * insulin_imputed - 15.766834
            insulin_std = min(max(insulin_std, -1.494110), 1.414175)
            
            bmi_std = 0.150080 * bmi_imputed - 4.861584
            bmi_std = min(max(bmi_std, -2.130134), 2.836815)
            
            dpf_std = 3.503728 * dpf - 1.607910
            dpf_std = min(dpf_std, 2.596563)
            
            age_std = 0.085088 * age - 2.828392
            age_std = max(age_std, -1.041549)
            
            # 6. Sắp xếp các đặc trưng đã chuẩn hóa theo đúng thứ tự mà mô hình yêu cầu
            features_dict = {
                'Pregnancies': pregnancies_std,
                'Glucose': glucose_std,
                'BloodPressure': bp_std,
                'SkinThickness': skin_std,
                'Insulin': insulin_std,
                'BMI': bmi_std,
                'DiabetesPedigreeFunction': dpf_std,
                'Age': age_std,
                'Age_Group': age_group_std,
                'BMI_Category': bmi_cat_std,
                'Glucose_BMI': glucose_bmi_std
            }
            processed_features = [features_dict[feat] for feat in feature_order]
            features_scaled = np.array(processed_features).reshape(1, -1)
            
            # Ghi nhận các giá trị thô đã điền khuyết để hiển thị trong kết quả
            input_processed_info = {
                'Pregnancies': pregnancies,
                'Glucose': glucose_imputed,
                'BloodPressure': bp_imputed,
                'SkinThickness': skin_imputed,
                'Insulin': insulin_imputed,
                'BMI': bmi_imputed,
                'DiabetesPedigreeFunction': dpf,
                'Age': age
            }
            
        # Dự đoán phân lớp và xác suất
        prediction = int(model.predict(features_scaled)[0])
        probability = float(model.predict_proba(features_scaled)[0][1])
        
        # Tạo lời khuyên sức khỏe
        health_info = get_health_recommendation(probability, data)
        
        return jsonify({
            "status": "success",
            "prediction": prediction,
            "probability": probability,
            "risk_status": health_info["status"],
            "recommendations": health_info["recommendations"],
            "input_processed": input_processed_info
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Đã xảy ra lỗi hệ thống: {str(e)}"}), 500

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Trả về metadata và các chỉ số đo lường của mô hình."""
    global metadata
    if metadata is None:
        if not load_model_assets():
            return jsonify({"status": "error", "message": "Chưa có dữ liệu mô hình."}), 404
            
    return jsonify(metadata)

@app.route('/plots/<filename>', methods=['GET'])
def get_plot(filename):
    """Trả về các đồ thị dạng hình ảnh (Feature Importance hoặc Confusion Matrix)."""
    if filename not in ['feature_importance.png', 'confusion_matrix.png']:
        return jsonify({"status": "error", "message": "Tệp hình ảnh không hợp lệ."}), 400
    return send_from_directory(MODEL_DIR, filename)

@app.route('/health', methods=['GET'])
def health():
    """Đường dẫn kiểm tra tình trạng hoạt động của hệ thống."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    })

if __name__ == '__main__':
    # Chạy trên tất cả IP giao tiếp cổng 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
