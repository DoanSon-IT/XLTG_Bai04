import cv2
import os
import numpy as np

def augment_images(folder_path, output_folder, augment_count=100):
    os.makedirs(output_folder, exist_ok=True)
    images = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    for i in range(augment_count):
        img_name = np.random.choice(images)
        img_path = os.path.join(folder_path, img_name)
        image = cv2.imread(img_path)

        # Áp dụng biến đổi
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)  # Lật ngang
        if np.random.rand() > 0.5:
            angle = np.random.randint(-15, 15)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
            image = cv2.warpAffine(image, M, (w, h))
        
        # Lưu ảnh đã biến đổi vào thư mục output
        output_path = os.path.join(output_folder, f"augmented_{i}_{img_name}")
        cv2.imwrite(output_path, image)

# Sử dụng hàm để tăng ảnh cho Lion và Horse
augment_images("./Lion", "./Lion_augmented", augment_count=100)
augment_images("./Horse", "./Horse_augmented", augment_count=100)
