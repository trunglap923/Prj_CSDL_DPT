import sqlite3
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
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

def chi_square_distance(p, q, eps=1e-10):
    return 0.5 * np.sum(((p - q) ** 2) / (p + q + eps))

# Tìm ảnh tương tự
def find_similar_images(image, metric='cosine', top_k=3):
    query_vector = extract_features(image).reshape(1, -1)
    image_paths, feature_vectors = load_database()
    
    if (metric == 'cosine'):
        # Tính toán độ tương đồng Cosine Similarity
        similarities = cosine_similarity(query_vector, feature_vectors)
        indices = np.argsort(similarities[0])[::-1][:top_k]
    
        # Lấy ra vector query và vector top_k ảnh trong database
        query_vector = query_vector.flatten()
        top_k_vectors = feature_vectors[indices]
        
        results = [
            {
                "path": image_paths[i],
                "similarity": float(similarities[0][i]),
                "vector": top_k_vectors[idx].tolist()
            }
            for idx, i in enumerate(indices)
        ]
    elif (metric == 'euclidean'):
        # Tính toán khoảng cách Euclidean
        distances = euclidean_distances(query_vector, feature_vectors)
        indices = np.argsort(distances[0])[:top_k]
        
        # Lấy ra vector query và vector top_k ảnh trong database
        query_vector = query_vector.flatten()
        top_k_vectors = feature_vectors[indices]
        
        results = [
            {
                "path": image_paths[i],
                "similarity": float(distances[0][i]),
                "vector": top_k_vectors[idx].tolist()
            }
            for idx, i in enumerate(indices)
        ]
    elif (metric == 'chi_square'):
        # Tính toán khoảng cách Chi-Square
        distances = [chi_square_distance(query_vector.flatten(), vec.flatten()) for vec in feature_vectors]
        indices = np.argsort(distances)[:top_k]
        
        # Lấy ra vector query và vector top_k ảnh trong database
        query_vector = query_vector.flatten()
        top_k_vectors = feature_vectors[indices]
        
        results = [
            {
                "path": image_paths[i],
                "similarity": float(distances[i]),
                "vector": top_k_vectors[idx].tolist()
            }
            for idx, i in enumerate(indices)
        ]
    
    
    final_result = {
        "vector_query": query_vector.tolist(),
        "results": results
    }
    return final_result
