def get_labels():
    """cifar10 클래스 라벨 리스트 반환"""
    labels = ['plane', 'car', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']
    return labels

def get_label_name(class_index):
    """클래스 인덱스를 실제 라벨명으로 변환"""
    labels = get_labels()
    return labels[class_index] if class_index < len(labels) else "Unknown"
