import cv2
import numpy as np
import os

def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (512, 512))  # Resize ảnh về cùng kích thước
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển ảnh sang grayscale
            images.append(img.flatten())  # Dùng ảnh phẳng cho các model
            labels.append(label)
    return images, labels

def prepare_data():
    lion_images, lion_labels = load_images_from_folder("./Lion", 0)

    horse_images, horse_labels = load_images_from_folder("./Horse", 1)

    lion_augmented_images, lion_augmented_labels = load_images_from_folder("./Lion_augmented", 0)
 
    horse_augmented_images, horse_augmented_labels = load_images_from_folder("./Horse_augmented", 1)

    images = lion_images + horse_images + lion_augmented_images + horse_augmented_images
    labels = lion_labels + horse_labels + lion_augmented_labels + horse_augmented_labels

    return np.array(images), np.array(labels)
