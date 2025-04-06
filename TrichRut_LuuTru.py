import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import sqlite3
import pickle
import os

# Load ResNet50
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Bỏ lớp cuối
model.eval()

# Tiền xử lý ảnh
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    return image

# Trích xuất đặc trưng
def extract_features(image_path):
    image = preprocess_image(image_path)
    with torch.no_grad():
        feature = model(image)
    return feature.view(-1).numpy()

# Thêm ảnh vào database
def insert_image(image_path, feature_vector):
    conn = sqlite3.connect("database/image_database.db")
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
    features = extract_features(img_path)
    insert_image(img_path, features)
    print(f"Đã lưu xong ảnh {img_file} vào database!")

print("!!! Đã lưu xong vector vào database !!!")
