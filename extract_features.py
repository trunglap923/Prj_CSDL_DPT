import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor
import cv2

# ========== Tham số cho LBP và GLCM ==========
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
# GLCM_DISTANCES = [1]
# GLCM_ANGLES = [0]  # 0 độ

# ========== Đặc trưng HSV ==========
def extract_hsv_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
    hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
    return cv2.normalize(hist, hist).flatten()

# ========== Đặc trưng Histogram theo vùng ==========
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

# ========== Đặc trưng LBP ==========
def extract_lbp_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

# ========== Đặc trưng Gabor ==========
def extract_gabor_features(image, frequencies=[0.1, 0.2, 0.3], thetas=[0, np.pi/4, np.pi/2]):
    image = image.astype(np.float32) / 255.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = []
    for freq in frequencies:
        for theta in thetas:
            real, imag = gabor(gray, frequency=freq, theta=theta)
            mean_val = real.mean()
            std_val = real.std()
            features.extend([mean_val, std_val])
    return np.array(features)

# ========== Đặc trưng GLCM ==========
# def extract_glcm_features(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     glcm = graycomatrix(gray, distances=GLCM_DISTANCES, angles=GLCM_ANGLES, levels=256, symmetric=True, normed=True)
#     features = [
#         # graycoprops(glcm, 'contrast')[0, 0],
#         graycoprops(glcm, 'correlation')[0, 0],
#         graycoprops(glcm, 'energy')[0, 0],
#         graycoprops(glcm, 'homogeneity')[0, 0],
#     ]
#     return np.array(features)

# ========== Tổng hợp đặc trưng ==========
def extract_features(image):
    image = cv2.resize(image, (100, 100))
    hsv_feat = extract_hsv_features(image)
    region_hist_feat = extract_color_histogram(image)
    lbp_feat = extract_lbp_features(image)
    gabor_feat = extract_gabor_features(image)
    
    combined = np.concatenate((hsv_feat * 1.0, region_hist_feat * 1.0, lbp_feat * 3.0, gabor_feat * 5.0))
    return combined

