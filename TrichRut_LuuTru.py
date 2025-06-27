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
def insert_image(image_path, hsv, spatial_hsv, lbp, spatial_lbp):
    if not os.path.exists("database"):
        os.makedirs("database")
    conn = sqlite3.connect("database/image_database.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT,
            hsv_feature BLOB,
            spatial_hsv_feature BLOB,
            lbp_feature BLOB,
            spatial_lbp_feature BLOB
        )
    """)
    
    cursor.execute("""
        INSERT INTO images (path, hsv_feature, spatial_hsv_feature, lbp_feature, spatial_lbp_feature)
        VALUES (?, ?, ?, ?, ?)
    """, (
        image_path,
        pickle.dumps(hsv),
        pickle.dumps(spatial_hsv),
        pickle.dumps(lbp),
        pickle.dumps(spatial_lbp)
    ))
    conn.commit()
    conn.close()

# ========== Xử lý toàn bộ thư mục ảnh ==========
image_folder = "./Image_data/"
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

for img_file in image_files:
    img_path = os.path.join(image_folder, img_file)
    image = cv2.imread(img_path)
    hsv, spatial_hsv, lbp, spatial_lbp = extract_features(image)
    insert_image(img_path, hsv, spatial_hsv, lbp, spatial_lbp)
    print(f"✅ Đã lưu đặc trưng của ảnh {img_file} vào database.")

print("🎉 Hoàn tất lưu trữ đặc trưng tất cả ảnh.")
