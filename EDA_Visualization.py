import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

def main():
    print("⏳ Đang khởi động quá trình vẽ biểu đồ EDA...")

    # 1. Tạo thư mục 'charts' để chứa ảnh xuất ra (nếu chưa có)
    output_dir = 'charts'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Đã tạo thư mục: {output_dir}/")

    # 2. Đọc file dữ liệu sạch
    file_name = 'diabetes_clean.csv'
    try:
        df = pd.read_csv(file_name)
        print(f"✅ Đọc thành công dữ liệu từ {file_name} (Gồm {df.shape[0]} dòng, {df.shape[1]} cột)")
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file '{file_name}'. Vui lòng kiểm tra lại thư mục.")
        return

    # ---------------------------------------------------------
    # BIỂU ĐỒ 1: HEATMAP (MA TRẬN TƯƠNG QUAN)
    # ---------------------------------------------------------
    print("📊 Đang vẽ Heatmap...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Ma trận tương quan giữa các đặc trưng y tế", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/1_correlation_heatmap.png', dpi=300) # dpi=300 giúp ảnh nét hơn khi đưa vào Word
    plt.close() # Đóng biểu đồ trong bộ nhớ để tránh nặng máy

    # ---------------------------------------------------------
    # BIỂU ĐỒ 2: HISTOGRAM (PHÂN PHỐI DỮ LIỆU)
    # ---------------------------------------------------------
    print("📊 Đang vẽ Histogram...")
    df.hist(figsize=(16, 12), bins=20, color='skyblue', edgecolor='black')
    plt.suptitle("Phân phối của các đặc trưng", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/2_features_histogram.png', dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # BIỂU ĐỒ 3: BOXPLOT (KIỂM TRA OUTLIERS) - TỰ ĐỘNG CO GIÃN
    # ---------------------------------------------------------
    print("📊 Đang vẽ Boxplot...")
    
    # Lấy tất cả các cột trừ cột nhãn (thường là cột cuối cùng, ví dụ: 'Outcome')
    features = df.columns[:-1] 
    num_features = len(features)

    # Cấu hình lưới: 4 cột, số hàng tự động tính toán
    ncols = 4
    nrows = math.ceil(num_features / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 5 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(features):
        sns.boxplot(y=df[col], ax=axes[i], color='lightgreen')
        axes[i].set_title(f"Boxplot: {col}", fontsize=12)
        axes[i].set_ylabel("") # Ẩn nhãn trục Y cho gọn

    # Xóa các ô trống nếu số lượng biểu đồ bị lẻ
    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/3_features_boxplot.png', dpi=300)
    plt.close()

    print(f"🎉 Hoàn tất! Toàn bộ biểu đồ phân tích độ phân giải cao đã được lưu vào thư mục '{output_dir}'.")

if __name__ == "__main__":
    main()