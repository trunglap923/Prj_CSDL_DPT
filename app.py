from flask import Flask, request, render_template, jsonify, send_from_directory
from PIL import Image
import os
from HamCanThiet import normalize_image,find_similar_images

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
    image = Image.open(file_path)
    normalized_image=normalize_image(image)
    results = find_similar_images(normalized_image)

    return jsonify({"results": results, "uploaded_image": file.filename})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
