import torch
import numpy as np
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from torchvision import models
import torchvision.transforms as transforms
import pickle
import matplotlib.pyplot as plt

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

# Lấy toàn bộ dữ liệu từ database
def load_database():
    conn = sqlite3.connect("database/image_database.db")
    cursor = conn.cursor()
    
    cursor.execute("SELECT path, feature_vector FROM images")
    data = cursor.fetchall()
    
    conn.close()
    
    image_paths = []
    feature_vectors = []
    
    for path, blob in data:
        image_paths.append(path)
        feature_vectors.append(pickle.loads(blob))
    
    return image_paths, np.array(feature_vectors)

# Tìm kiếm ảnh tương tự
def search_similar_images(query_img_path, top_k=3):
    query_vector = extract_features(query_img_path).reshape(1, -1)
    
    # Load toàn bộ database
    image_paths, feature_vectors = load_database()
    
    # Tính toán Cosine Similarity
    similarities = cosine_similarity(query_vector, feature_vectors)
    
    # Sắp xếp theo độ tương đồng giảm dần
    indices = np.argsort(similarities[0])[::-1][:top_k]
    
    return [image_paths[i] for i in indices]

# Test tìm kiếm
query_image = "./Image_test/qua-nho-2.jpg"
result_images = search_similar_images(query_image)

print("Ảnh giống nhất:", result_images)

# Hiển thị 3 ảnh kết quả trong 1 hàng
plt.figure(figsize=(6, 3))
for i, img_path in enumerate(result_images):
    img = Image.open(img_path)
    plt.subplot(1, len(result_images), i+1)
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Ảnh {i+1}")
plt.show()


