import os
import cv2
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor
import sqlite3
import pickle
from extract_features import extract_features
    
# ========== Lưu vào SQLite ==========
def insert_image(image_path, feature_vector):
    conn = sqlite3.connect("database/image_database.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT,
            feature_vector BLOB
        )
    """)
    
    feature_blob = pickle.dumps(feature_vector)
    cursor.execute("INSERT INTO images (path, feature_vector) VALUES (?, ?)", (image_path, feature_blob))
    conn.commit()
    conn.close()

# ========== Xử lý toàn bộ thư mục ảnh ==========
image_folder = "./Image_data/"
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

for img_file in image_files:
    img_path = os.path.join(image_folder, img_file)
    image = cv2.imread(img_path)
    features = extract_features(image)
    insert_image(img_path, features)
    print(f"✅ Đã lưu đặc trưng của ảnh {img_file} vào database.")

print("🎉 Hoàn tất lưu trữ đặc trưng tất cả ảnh.")
