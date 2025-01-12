import cv2
import torch
import numpy as np

class YOLOv11nDetector:
    def __init__(self, model_path=None):
        # Tải mô hình YOLOv5 nhỏ gọn từ Ultralytics (thay YOLOv11n nếu bạn có mô hình này)
        self.model = torch.hub.load('ultralytics/yolov5:v5.0', 'yolov5n')  # YOLOv5n là mô hình nhỏ gọn, thay bằng YOLOv11n nếu có
        if model_path:
            self.model.load_state_dict(torch.load(model_path))  # Nếu bạn có mô hình YOLOv11n cụ thể

    def detect(self, image_path):
        # Đọc ảnh đầu vào
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Phát hiện đối tượng trong ảnh
        results = self.model(img_rgb)  # Kết quả trả về dưới dạng một đối tượng chứa các dự đoán
        
        # Trả về kết quả dạng pandas dataframe
        return results.pandas().xywh[0]  # Dự đoán trong hệ tọa độ [x_center, y_center, width, height]

    def display_results(self, image_path):
        # Hiển thị ảnh với các hộp chứa đối tượng
        results = self.detect(image_path)
        img = cv2.imread(image_path)
        
        for _, row in results.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vẽ hình chữ nhật quanh đối tượng

        # Hiển thị ảnh kết quả
        cv2.imshow("Detection Results", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
