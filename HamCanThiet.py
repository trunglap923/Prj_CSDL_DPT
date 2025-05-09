import cv2
from PIL import Image
import numpy as np
import sqlite3
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image


def extract_color_histogram(image, bins=256, regions=4):
    """
    Trích xuất histogram theo vùng từ ảnh.
    - image: Ảnh đầu vào (NumPy array, RGB).
    - bins: Số lượng bins cho mỗi histogram.
    - regions: Số vùng chia theo chiều ngang và dọc (regions x regions).
    """
    height, width, _ = image.shape
    region_height = height // regions
    region_width = width // regions

    feature_vector = []

    # Duyệt qua từng vùng
    for i in range(regions):
        for j in range(regions):
            # Xác định tọa độ vùng
            start_y = i * region_height
            end_y = (i + 1) * region_height
            start_x = j * region_width
            end_x = (j + 1) * region_width

            # Cắt vùng từ ảnh
            region = image[start_y:end_y, start_x:end_x]

            # Tính histogram cho từng kênh màu (R, G, B)
            hist_r = cv2.calcHist([region], [0], None, [bins], [0, 256])
            hist_g = cv2.calcHist([region], [1], None, [bins], [0, 256])
            hist_b = cv2.calcHist([region], [2], None, [bins], [0, 256])

            # Chuẩn hóa histogram
            hist_r = cv2.normalize(hist_r, hist_r).flatten()
            hist_g = cv2.normalize(hist_g, hist_g).flatten()
            hist_b = cv2.normalize(hist_b, hist_b).flatten()

            # Kết hợp histogram của vùng vào vector đặc trưng
            feature_vector.extend(hist_r)
            feature_vector.extend(hist_g)
            feature_vector.extend(hist_b)

    return np.array(feature_vector)

#Chuẩn hóa đầu vào thành NumPy array ở không gian màu RGB.    
def normalize_image(image):
    if isinstance(image, Image.Image):
        # Nếu là ảnh PIL, chuyển đổi sang NumPy array
        image = np.array(image)
    elif isinstance(image, np.ndarray):
        # Nếu là ảnh từ cv2 (BGR), chuyển sang RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Đầu vào phải là ảnh từ PIL hoặc cv2.")
    
    return image


# Load dữ liệu từ database
def load_database():
    conn = sqlite3.connect("database2/image_database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT path, feature_vector FROM images")
    data = cursor.fetchall()
    conn.close()
    
    image_paths, feature_vectors = [], []
    for path, blob in data:
        image_paths.append(path)
        feature_vectors.append(pickle.loads(blob))
    
    return image_paths, np.array(feature_vectors)

# Tìm ảnh tương tự
def find_similar_images(image: Image.Image, top_k=3):
    query_vector = extract_color_histogram(image).reshape(1, -1)
    image_paths, feature_vectors = load_database()
    
    # Tính toán độ tương đồng Cosine Similarity
    similarities = cosine_similarity(query_vector, feature_vectors)
    indices = np.argsort(similarities[0])[::-1][:top_k]
    
    results = [{"path": image_paths[i], "similarity": float(similarities[0][i])} for i in indices]
    return results
