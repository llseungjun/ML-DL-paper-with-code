from PIL import Image
import torchvision.transforms as transforms
import torch

# 이미지 전처리 변환 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_path, device):
    """이미지 파일 경로를 입력받아 모델에 입력 가능한 텐서로 변환"""
    image = Image.open(image_path).convert("RGB")  # RGB 변환
    image = transform(image).unsqueeze(0)  # (1, 3, 224, 224) 배치 차원 추가
    return image.to(device)
