import cv2
import numpy as np
from matplotlib import pyplot as plt
import sqlite3
import os
import pickle
from PIL import Image
from HamCanThiet import normalize_image, extract_color_histogram






def insert_image(image_path, feature_vector):
    conn = sqlite3.connect("database2/image_database.db")
    cursor = conn.cursor()
    
    # Chuyển vector thành dạng BLOB (bytes)
    feature_blob = pickle.dumps(feature_vector)
    cursor.execute("INSERT INTO images (path, feature_vector) VALUES (?, ?)", (image_path, feature_blob))
    conn.commit()
    conn.close()



# Lưu toàn bộ ảnh trong thư mục vào database
image_folder = "./Image_data/"
image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]

for img_file in image_files:
    img_path = os.path.join(image_folder, img_file)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Không thể đọc ảnh từ đường dẫn: {img_path}")
        continue
    features = extract_color_histogram(normalize_image(image))
    insert_image(img_path, features)
    print(f"Đã lưu xong ảnh {img_file} vào database!")

print("!!! Đã lưu xong vector vào database !!!")

