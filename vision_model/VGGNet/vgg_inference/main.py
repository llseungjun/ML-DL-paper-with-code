from vgg_inference import load_model, predict, preprocess_image, get_label_name
import torch

# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드
model = load_model()

# 예측할 이미지 경로
image_path = "test_image.jpg"

# 이미지 전처리 및 예측 수행
image_tensor = preprocess_image(image_path, device)
predicted_class = predict(image_tensor, model)
predicted_label = get_label_name(predicted_class)

print(f"Predicted class index: {predicted_class}")
print(f"Predicted class name: {predicted_label}")
