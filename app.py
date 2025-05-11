from flask import Flask, request, render_template, jsonify, send_from_directory
from search import find_similar_images
import cv2
import os
import shutil

app = Flask(__name__)

# Đường dẫn thư mục upload ảnh
UPLOAD_FOLDER = "static/uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/Image_data/<path:filename>')
def serve_image(filename):
    return send_from_directory("Image_data", filename)

@app.route("/search", methods=["POST"])
def search_image():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})
    
    # Lưu ảnh tải lên
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Tìm ảnh tương tự
    image = cv2.imread(file_path)
    final_results = find_similar_images(image, metric='euclidean', top_k=3)
    query_vector = final_results["vector_query"]
    results = final_results["results"]
    
    # In ra vector query và vector top_k ảnh trong database
    top_k_vectors = [result["vector"] for result in results]
    
    # print("Query Vector:", query_vector)
    print(len(query_vector))
    # print("Top K Vectors:")
    # for i, vector in enumerate(top_k_vectors):
    #     print(f"Vector {i+1}:", vector)

    return jsonify({"results": results, "uploaded_image": file.filename})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)