import cv2
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

def predict_image(models, image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Khong the doc anh.")
        return

    img_resized = cv2.resize(img, (512, 512)) 
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_flattened = img_gray.flatten().reshape(1, -1)

    predictions = {}
    for model_name, model in models.items():
        pred = model.predict(img_flattened)
        species_name = "Lion" if pred[0] == 1 else "Horse" 
        predictions[model_name] = species_name  # Lưu tên loài vào từ điển

    # Vẽ các đặc trưng lên ảnh
    height, width = img_resized.shape[:2]
    features_text = f"Kich thuoc: {width}x{height}"
    cv2.putText(img_resized, features_text, (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Tạo không gian cho văn bản dự đoán
    img_with_text = np.zeros((height + 150, width, 3), dtype=np.uint8)
    img_with_text[:height, :] = img_resized  # Copy ảnh đã resize vào không gian mới

    # Vẽ văn bản dự đoán cho từng mô hình
    y_offset = height + 30
    for model_name, species_name in predictions.items():
        prediction_text = f"{model_name}: {species_name}"
        cv2.putText(img_with_text, prediction_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        y_offset += 30  # Tăng vị trí y để hiển thị dự đoán cho mô hình tiếp theo

    # Chuyển đổi BGR sang RGB
    img_rgb = cv2.cvtColor(img_with_text, cv2.COLOR_BGR2RGB)

    # Hiển thị ảnh sử dụng Matplotlib
    plt.imshow(img_rgb)
    plt.axis('off')  # Ẩn trục
    plt.title("Du doan anh")
    plt.show()

# Hàm main để chạy dự đoán
def main():
    # Load các mô hình đã được huấn luyện
    models = {}
    for model_name in ["SVM", "KNN", "Decision Tree"]:
        with open(f"{model_name}.pkl", "rb") as model_file:
            models[model_name] = pickle.load(model_file)

    # Nhập đường dẫn ảnh từ người dùng (hoặc gán sẵn đường dẫn)
    image_path = "./test9.jpg"  # Gán sẵn đường dẫn ảnh

    # Kiểm tra xem ảnh có tồn tại không
    if not os.path.exists(image_path):
        print("Duong dan anh khong hop le.")
        return

    # Dự đoán
    predict_image(models, image_path)

if __name__ == "__main__":
    main()
