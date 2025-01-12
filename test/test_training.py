import pytest
from ultralytics import YOLO
from PIL import Image

@pytest.fixture
def model():
    """Fixture để tải mô hình YOLOv11n"""
    model = YOLO('yolov11n.pt')  # Tải mô hình từ tệp yolov11n.pt
    return model

def test_model_inference(model):
    """Kiểm tra việc dự đoán của mô hình"""
    img_path = 'test_image.jpg'  # Đảm bảo bạn có một ảnh test
    results = model(img_path)
    
    assert results is not None
    assert len(results.pandas().xywh) > 0  # Kiểm tra xem mô hình có phát hiện đối tượng trong ảnh không

def test_class_names(model):
    """Kiểm tra tên lớp của các đối tượng được nhận diện"""
    img_path = 'test_image.jpg'  # Đảm bảo bạn có một ảnh test
    results = model(img_path)
    
    detected_classes = results.pandas().xywh['class'].values
    assert all(isinstance(c, int) for c in detected_classes)  # Kiểm tra các lớp là kiểu số nguyên (ID lớp)
    assert len(detected_classes) > 0  # Kiểm tra rằng có ít nhất một lớp được nhận diện
