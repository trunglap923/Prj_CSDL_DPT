import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
from skimage.filters import gabor
from sklearn.preprocessing import normalize
import cv2
import math


# ========== Đặc trưng HSV ==========
def extract_hsv_features(image):
    # Chuyển đổi ảnh từ BGR (OpenCV) sang RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Chuyển RGB sang HSV dùng hàm thủ công
    def rgb_to_hsv(pixel):
        r, g, b = pixel
        r, g, b = r / 255.0, g / 255.0, b / 255.0
        v = max(r, g, b)
        delta = v - min(r, g, b)

        if delta == 0:
            h = 0
            s = 0
        else:
            s = delta / v
            if r == v:
                h = (g - b) / delta
            elif g == v:
                h = 2 + (b - r) / delta
            else:
                h = 4 + (r - g) / delta
            h = (h / 6) % 1.0

        return [int(h * 180), int(s * 255), int(v * 255)]

    def covert_image_rgb_to_hsv(img):
        hsv_image = []
        for i in img:
            hsv_row = []
            for j in i:
                hsv_row.append(rgb_to_hsv(j))
            hsv_image.append(hsv_row)
        return np.array(hsv_image)

    def my_calcHist(image, channels, histSize, ranges):
        hist = np.zeros(histSize, dtype=np.int64)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                bin_vals = [image[i, j, c] for c in channels]
                bin_idxs = [
                    (bin_vals[c] - ranges[c][0]) * histSize[c] //
                    (ranges[c][1] - ranges[c][0])
                    for c in range(len(channels))
                ]
                # Đảm bảo không vượt quá chỉ số
                bin_idxs = [min(idx, histSize[i] - 1) for i, idx in enumerate(bin_idxs)]
                hist[tuple(bin_idxs)] += 1
        return hist

    hsv_image = covert_image_rgb_to_hsv(image_rgb)
    channels = [0, 1, 2]
    histSize = [12, 12, 3]
    ranges = [(0, 180), (0, 256), (0, 256)]

    histogram = my_calcHist(hsv_image, channels, histSize, ranges)
    hist_flat = histogram.flatten().astype("float")
    hist_flat /= (hist_flat.sum() + 1e-7)  # Chuẩn hóa
    return hist_flat

# ========== Đặc trưng Histogram theo vùng ==========
def extract_color_histogram(image, bins=256, regions=4):
    """
    Trích xuất histogram theo vùng từ ảnh HSV.
    - image: Ảnh đầu vào (NumPy array, RGB).
    - bins: Số lượng bins cho mỗi histogram.
    - regions: Số vùng chia theo chiều ngang và dọc (regions x regions).
    """
    # Chuyển ảnh từ RGB sang HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    height, width, _ = hsv_image.shape
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

            # Cắt vùng từ ảnh HSV
            region = hsv_image[start_y:end_y, start_x:end_x]

            # Tính histogram cho từng kênh màu HSV (H, S, V)
            hist_h = cv2.calcHist([region], [0], None, [bins], [0, 180])  # Hue: 0-179
            hist_s = cv2.calcHist([region], [1], None, [bins], [0, 256])  # Saturation: 0-255
            hist_v = cv2.calcHist([region], [2], None, [bins], [0, 256])  # Value: 0-255

            # Chuẩn hóa histogram
            hist_h = cv2.normalize(hist_h, hist_h).flatten()
            hist_s = cv2.normalize(hist_s, hist_s).flatten()
            hist_v = cv2.normalize(hist_v, hist_v).flatten()

            # Kết hợp histogram của vùng vào vector đặc trưng
            feature_vector.extend(hist_h)
            feature_vector.extend(hist_s)
            feature_vector.extend(hist_v)

    return np.array(feature_vector)

# ========== Đặc trưng LBP ==========
# ========== Tham số và hàm LBP ==========
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
class LBP(object):
    def __init__(self, radius=1, npoints=8, counter_clockwise=True, interpolation="bilinear"):
        self.radius = radius
        self.npoints = npoints
        self.interpolation = interpolation
        self.counter_clockwise= counter_clockwise
        assert self.radius > 0 and self.npoints > 0
        assert interpolation in ("bilinear", "nearest")
        self.get_pixel_func = self._get_pixel_nearest if self.interpolation == "nearest" else self._get_pixel_bilinear
        
        start_angle_radian = 0
        angle_radian = 2*math.pi/npoints
        circle_direction = 1 if counter_clockwise else -1
        neighbor_positions = []
        for pos in range(self.npoints):
            delta_x = math.cos(start_angle_radian+circle_direction*pos*angle_radian) * self.radius
            delta_y = -(math.sin(start_angle_radian+circle_direction*pos*angle_radian) * self.radius)
            neighbor_positions.append((delta_x, delta_y))
        neighbor_positions.reverse()
        self.neighbor_positions = neighbor_positions
        assert len(self.neighbor_positions) == npoints
        pass
    
    def _get_pixel_nearest(self, image, x, y, w, h):
        xx = round(x)
        yy = round(y)
        if xx < 0 or yy < 0 or xx >= w or yy >= h:
            return 0
        else:
            return image[yy, xx]
    
    def _get_pixel_bilinear(self, image, x, y, w, h):
        xmin, xmax = math.floor(x), math.ceil(x)
        ymin, ymax = math.floor(y), math.ceil(y)
        
        intensity_top_left = 0 if xmin<0 or ymin<0 or xmin>=w or ymin>=h else image[ymin, xmin]
        intensity_top_right = 0 if xmax<0 or ymin<0 or xmax>=w or ymin>=h else image[ymin, xmax]
        intensity_bottom_left = 0 if xmin<0 or ymax<0 or xmin>=w or ymax>=h else image[ymax, xmin]
        intensity_bottom_right = 0 if xmax<0 or ymax<0 or xmax>=w or ymax>=h else image[ymax, xmax]
        
        weight_x = x - xmin
        weight_y = y - ymin
        
        intensity_at_top = (1-weight_x) * intensity_top_left + weight_x * intensity_top_right
        intensity_at_bottom= (1-weight_x) * intensity_bottom_left + weight_x * intensity_bottom_right
        
        final_intensity = (1-weight_y) * intensity_at_top + weight_y * intensity_at_bottom        
        return final_intensity
    
    def __call__(self, image):
        assert len(image.shape) == 2
        h, w = image.shape
        result = np.zeros([h, w])
        for y in range(h):
            for x in range(w):
                center_intensity = image[y, x]
                binary_vector = [0] * self.npoints
                for npos in range(self.npoints):
                    new_x = x + self.neighbor_positions[npos][0]
                    new_y = y + self.neighbor_positions[npos][1]              
                    
                    neighbor_intensity = self.get_pixel_func(image, new_x, new_y, w, h)
                    
                    if center_intensity <= neighbor_intensity:
                        binary_vector[npos] = 1
                binary_str = "".join([str(e) for e in binary_vector]) # '00001001'
                decimal_value = int(binary_str, 2) # convert binary string to decimal
                result[y, x] = decimal_value
        return result
    
# spatial_hist 
def lbp_spatial_histogram(lbp_image, grid_x=4, grid_y=4, npoints=8):
    h, w = lbp_image.shape
    num_bins = 2 ** npoints
    hist_vector = []
    step_x = w // grid_x
    step_y = h // grid_y
    for i in range(grid_y):
        for j in range(grid_x):
            x_start = j * step_x
            y_start = i * step_y
            x_end = x_start + step_x
            y_end = y_start + step_y
            cell = lbp_image[y_start:y_end, x_start:x_end]
            hist, _ = np.histogram(cell.ravel(), bins=num_bins, range=(0, num_bins))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6)
            hist_vector.extend(hist)
    return np.array(hist_vector)

# global_hist 
def lbp_to_feature_vector(lbp_image, npoints=8):
    num_bins = 2 ** npoints
    hist, _ = np.histogram(lbp_image.ravel(), bins=num_bins, range=(0, num_bins))
    
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_lbp_features(image, grid_x=4, grid_y=4, npoints=8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Dùng thư viện
    # lbp_result = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="default") 
    
    # Code chay
    lbp = LBP()
    lbp_result = lbp(gray)
    
    if grid_x == 1 and grid_x == 1:
        feature_vector = lbp_to_feature_vector(lbp_result, npoints=npoints)
    else:
        feature_vector = lbp_spatial_histogram(lbp_result, grid_x=grid_x, grid_y=grid_y, npoints=npoints)
    return feature_vector

# # ========== Tổng hợp đặc trưng ==========

def extract_features(image):
    image = cv2.resize(image, (100, 100))

    hsv = normalize(extract_hsv_features(image).reshape(1, -1))[0]
    spatial_hsv = normalize(extract_color_histogram(image).reshape(1, -1))[0]
    lbp = normalize(extract_lbp_features(image, grid_x=1, grid_y=1, npoints=8).reshape(1, -1))[0]
    spatial_lbp = normalize(extract_lbp_features(image, grid_x=4, grid_y=4, npoints=8).reshape(1, -1))[0]
    
    return np.concatenate([hsv, spatial_hsv, lbp, spatial_lbp])



def main():
    image_path = 'Image_test/0effc19dfa42fbad48ec445fa846f408.jpg'

    image = cv2.imread(image_path)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = extract_features(image_rgb)
    
if __name__ == '__main__':
    main()
