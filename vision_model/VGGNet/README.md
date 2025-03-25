1. **프로젝트 개요**  
   Streamlit을 활용하여 VGGNet 기반 이미지 분류를 수행하는 Proof of Concept(POC) 프로젝트입니다.  
사용자가 이미지를 업로드하면 사전 학습된 VGGNet 모델을 통해 분류 결과를 반환합니다.   

2. **프로젝트 구조**  
   ```bash
    📂 vggnet_agent/
    │── 📂 src/                 # 모델 및 유틸리티 코드
    │   ├── __init__.py        # 패키지 관리 파일
    │   ├── custom_model.py   # VGGNet 모델 구조
    │   ├── model.py          # VGGNet 모델 로드 및 예측
    │   ├── preprocess.py     # 이미지 전처리
    │   ├── utils.py          # 기타 유틸리티 함수
    │── 📂 data/               # 샘플 이미지 데이터 (테스트용)
    │── 📂 models/             # 사전 학습된 모델 저장 폴더
    │── 📂 requirements.txt    # 필요한 패키지 목록
    │── app.py              # Streamlit 메인 실행 파일

   ``` 

3. **기능 소개**  
   - 10가지 클래스 ['plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck'] 중 하나에 해당하는 이미지 파일을 드래그 앤 드롭을 통해 업로드
   - 알맞는 클래스로 분류 수행

4. **설치 및 실행 방법**  
   ```bash
    cd vgg_agent # 1. vgg_agent 폴더로 이동 

    conda create -name YOUR_ENV_NAME python=3.9 # 2. conda 가상환경 생성
   
    conda activate YOUR_ENV_NAME # 3. conda 가상환경 생성

    pip install -r requirements.txt # 4. requirements 설치 

    streamlit run app.py # 5. streamlit 실행
   ```