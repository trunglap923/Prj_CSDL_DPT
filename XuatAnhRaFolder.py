import os
import shutil

source_folder = r'./Ảnh trái cây'
destination_folder = r'./Image_data'

os.makedirs(destination_folder, exist_ok=True)

for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file.endswith('.jpg'):
            source_file = os.path.join(root, file)
            destination_file = os.path.join(destination_folder, file)
            shutil.copy(source_file, destination_file)
            print(f'Đã xuất ảnh {file} ra folder Image_data')

print('Xuất ảnh hoàn tất!')