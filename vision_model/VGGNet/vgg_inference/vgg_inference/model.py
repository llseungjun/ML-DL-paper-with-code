import torch
import torchvision.models as models

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """Pretrained VGG 모델 로드"""
    model = models.vgg16(pretrained=True)
    model = model.to(device)
    model.eval()  # Inference 모드
    return model

def predict(image_tensor, model):
    """이미지 텐서를 입력받아 예측 결과 반환"""
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    return predicted_class
