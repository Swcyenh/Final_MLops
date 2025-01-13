import cv2
import torch
import numpy as np

class YOLOv11nDetector:
    def __init__(self, model_path):
        """
        Khởi tạo YOLOv11n Detector với mô hình được cung cấp.
        """
        if not model_path:
            raise ValueError("Đường dẫn tới mô hình YOLOv11n là bắt buộc.")
        
        # Load custom YOLOv11n model
        self.model = torch.load(model_path)  # Load YOLOv11n model
        self.model.eval()  # Set model to evaluation mode

    def detect(self, image_path):
        """
        Phát hiện đối tượng trong hình ảnh.
        Args:
            image_path (str): Đường dẫn tới hình ảnh cần xử lý.
        Returns:
            pandas.DataFrame: Kết quả dự đoán với các cột [x_center, y_center, width, height].
        """
        # Đọc và chuyển đổi hình ảnh
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Không tìm thấy file hình ảnh tại: {image_path}")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Perform detection
        results = self.model(img_rgb)  # Model trả về kết quả dự đoán
        
        # Chuyển đổi kết quả về định dạng dễ sử dụng
        return results.pandas().xywh[0]  # Trả về tọa độ trung tâm, chiều rộng, chiều cao

def detect_with_yolov11n(image_path, model_path):
    """
    Hàm tiện ích để phát hiện đối tượng trong hình ảnh sử dụng YOLOv11n.
    Args:
        image_path (str): Đường dẫn tới hình ảnh cần xử lý.
        model_path (str): Đường dẫn tới tệp mô hình YOLOv11n (.pt).
    Returns:
        pandas.DataFrame: Kết quả phát hiện đối tượng.
    """
    # Tạo YOLOv11nDetector
    detector = YOLOv11nDetector(model_path)

    # Thực hiện phát hiện và trả về kết quả
    return detector.detect(image_path)
