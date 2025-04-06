import sqlite3
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from extract_features import extract_features
from PIL import Image

# Load dữ liệu từ database
def load_database():
    conn = sqlite3.connect("database/image_database.db")
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
    query_vector = extract_features(image).reshape(1, -1)
    image_paths, feature_vectors = load_database()
    
    # Tính toán độ tương đồng Cosine Similarity
    similarities = cosine_similarity(query_vector, feature_vectors)
    indices = np.argsort(similarities[0])[::-1][:top_k]
    
    results = [{"path": image_paths[i], "similarity": float(similarities[0][i])} for i in indices]
    return results
