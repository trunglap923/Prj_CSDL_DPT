import sqlite3

# Thêm thư mục database nếu chưa có
import os
if not os.path.exists("database2"):
    os.makedirs("database2")

# Kết nối SQLite
conn = sqlite3.connect("database2/image_database.db")
cursor = conn.cursor()

# Tạo bảng lưu trữ ảnh và vector đặc trưng
cursor.execute("""
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    feature_vector BLOB NOT NULL
)
""")

conn.commit()
conn.close()
