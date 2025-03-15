import streamlit as st
from PIL import Image
from vgg_inference import load_model, predict, preprocess_image, get_label_name
import torch

# Streamlit 페이지 설정
st.set_page_config(page_title="VGG Image Classifier", layout="centered")

# 모델 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model()

# 제목
st.title("🔍 VGG Image Classifier")
st.write("Upload an image to classify using VGG-16.")

# 이미지 업로더
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 이미지 표시
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # 이미지 추론 버튼
    if st.button("Classify Image"):
        # 이미지 전처리 및 예측
        image_tensor = preprocess_image(uploaded_file, device)
        predicted_class = predict(image_tensor, model)
        predicted_label = get_label_name(predicted_class)

        # 결과 출력
        st.success(f"**Predicted Class:** {predicted_label} (Class Index: {predicted_class})")