import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms

# Load mô hình ResNet50
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Bỏ lớp cuối
model.eval()

# Tiền xử lý ảnh
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Trích xuất đặc trưng ảnh
def extract_features(image: Image.Image):
    image = preprocess_image(image)
    with torch.no_grad():
        feature = model(image)
    return feature.view(-1).numpy()
