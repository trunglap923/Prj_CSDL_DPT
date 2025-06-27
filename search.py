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
    cursor.execute("SELECT path, hsv_feature, spatial_hsv_feature, lbp_feature, spatial_lbp_feature FROM images")
    data = cursor.fetchall()
    conn.close()
    
    image_paths = []
    hsv_vectors = []
    spatial_hsv_vectors = []
    lbp_vectors = []
    spatial_lbp_vectors = []

    for row in data:
        path, hsv_blob, spatial_hsv_blob, lbp_blob, spatial_lbp_blob = row
        image_paths.append(path)
        hsv_vectors.append(pickle.loads(hsv_blob))
        spatial_hsv_vectors.append(pickle.loads(spatial_hsv_blob))
        lbp_vectors.append(pickle.loads(lbp_blob))
        spatial_lbp_vectors.append(pickle.loads(spatial_lbp_blob))

    return image_paths, np.array(hsv_vectors), np.array(spatial_hsv_vectors), np.array(lbp_vectors), np.array(spatial_lbp_vectors)

# Tìm ảnh tương tự
def find_similar_images(image, metric='euclidean', top_k=3):
    # Trích xuất đặc trưng truy vấn
    hsv, spatial_hsv, lbp, spatial_lbp = extract_features(image)

    image_paths, db_hsv, db_spatial_hsv, db_lbp, db_spatial_lbp = load_database()

    # Chọn hàm đo khoảng cách
    def compute_distance(query, db_vectors):
        if metric == 'cosine':
            sims = cosine_similarity(query.reshape(1, -1), db_vectors)[0]
            return -sims 
        elif metric == 'euclidean':
            dists = euclidean_distances(query.reshape(1, -1), db_vectors)[0]
            return dists
        else:
            raise ValueError("Unsupported metric: " + metric)

    # Tính khoảng cách từng phần
    dist_hsv = compute_distance(hsv, db_hsv)
    dist_spatial_hsv = compute_distance(spatial_hsv, db_spatial_hsv)
    dist_lbp = compute_distance(lbp, db_lbp)
    dist_spatial_lbp = compute_distance(spatial_lbp, db_spatial_lbp)

    # Tổng hợp điểm với trọng số
    total_distance = (
        0.2 * dist_hsv +
        0.2 * dist_spatial_hsv +
        0.3 * dist_lbp +
        0.3 * dist_spatial_lbp
    )

    indices = np.argsort(total_distance)[:top_k]

    results = []
    for idx in indices:
        results.append({
            "path": image_paths[idx],
            "similarity": float(total_distance[idx]),
            "hsv": db_hsv[idx].tolist(),
            "spatial_hsv": db_spatial_hsv[idx].tolist(),
            "lbp": db_lbp[idx].tolist(),
            "spatial_lbp": db_spatial_lbp[idx].tolist()
        })

    final_result = {
        "query_vector": {
            "hsv": hsv.tolist(),
            "spatial_hsv": spatial_hsv.tolist(),
            "lbp": lbp.tolist(),
            "spatial_lbp": spatial_lbp.tolist()
        },
        "results": results
    }

    return final_result
