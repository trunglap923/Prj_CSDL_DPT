import csv
import sqlite3
import pickle

conn = sqlite3.connect("database/image_database.db")
cursor = conn.cursor()
cursor.execute("SELECT path, feature_vector FROM images")

with open("export_vectors.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    for row in cursor.fetchall():
        path = row[0]
        vector = pickle.loads(row[1])
        writer.writerow([path] + list(vector))
