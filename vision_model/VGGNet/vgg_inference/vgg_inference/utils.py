import requests

LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

def get_labels():
    """ImageNet 클래스 라벨 리스트 반환"""
    labels = requests.get(LABELS_URL).text.split("\n")
    return labels

def get_label_name(class_index):
    """클래스 인덱스를 실제 라벨명으로 변환"""
    labels = get_labels()
    return labels[class_index] if class_index < len(labels) else "Unknown"
