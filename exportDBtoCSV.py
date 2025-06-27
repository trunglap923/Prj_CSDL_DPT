import csv
import sqlite3
import pickle
import numpy as np
np.set_printoptions(threshold=np.inf)

conn = sqlite3.connect("database/image_database.db")
cursor = conn.cursor()
cursor.execute("""
    SELECT path, hsv_feature, spatial_hsv_feature, lbp_feature, spatial_lbp_feature
    FROM images
""")

with open("export_db_format.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["path", "hsv_feature", "spatial_hsv_feature", "lbp_feature", "spatial_lbp_feature"])

    for row in cursor.fetchall():
        path = row[0]
        hsv = pickle.loads(row[1])
        shsv = pickle.loads(row[2])
        lbp = pickle.loads(row[3])
        slbp = pickle.loads(row[4])

        # Ghi mỗi vector đặc trưng thành chuỗi (dễ đọc lại từ CSV sau này)
        writer.writerow([
            path,
            hsv,
            shsv,
            lbp,
            slbp
        ])

