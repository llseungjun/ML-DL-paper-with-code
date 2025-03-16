import torch
from .custom_model import VGGNet, VGG_with_BN, VGG11

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name="VGG11",num_classes=10,model_path="./VGG11_best.pth"):
    # VGG16, VGG batchnorm 중에서 선택
    if model_name == "VGGNet":
        model = VGGNet(num_classes = num_classes).to(device)
    elif model_name == "VGG_with_BN":
        model = VGG_with_BN(num_classes = num_classes).to(device)
    else:
        model = VGG11(num_classes = num_classes).to(device)
    # load pretrain model
    if model_path:
        import requests

        url = "https://raw.githubusercontent.com/llseungjun/ML-DL-paper-with-code/main/vision_model/VGGNet/vgg_inference/VGG11_best.pth"  # GitHub URL
        model_path = "VGG11_best.pth"

        # 모델 다운로드
        response = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(response.content)
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model weights loaded from {model_path}")

    model.eval()  # Inference 모드
    return model

def predict(image_tensor, model):
    """이미지 텐서를 입력받아 예측 결과 반환"""
    with torch.no_grad():
        output = model(image_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    return predicted_class
