import torch
import os
from ultralytics import YOLO
from pathlib import Path

def save_model(model, save_path, epoch, loss):
    """
    Lưu mô hình YOLOv11n sau mỗi epoch.
    
    Parameters:
    - model: Mô hình YOLOv11n đã huấn luyện.
    - save_path: Đường dẫn lưu mô hình.
    - epoch: Số epoch hiện tại.
    - loss: Mức độ mất mát (loss).
    """
    model_path = os.path.join(save_path, f'yolov11n_epoch_{epoch}_loss_{loss:.4f}.pt')
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

def track_performance(metrics, epoch, loss, accuracy):
    """
    Theo dõi hiệu suất mô hình YOLOv11n trong quá trình huấn luyện.
    
    Parameters:
    - metrics: Danh sách lưu trữ các thông số (mất mát, độ chính xác) của mô hình.
    - epoch: Số epoch hiện tại.
    - loss: Mức độ mất mát (loss).
    - accuracy: Độ chính xác của mô hình.
    """
    metrics.append({'epoch': epoch, 'loss': loss, 'accuracy': accuracy})
    print(f'Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}')

def train_yolov11n(data_config, model_config, epochs=50, batch_size=16, img_size=640):
    """
    Hàm huấn luyện mô hình YOLOv11n.
    
    Parameters:
    - data_config: Đường dẫn tới tệp cấu hình dữ liệu (data.yaml).
    - model_config: Đường dẫn tới tệp cấu hình mô hình (model.yaml).
    - epochs: Số epoch huấn luyện.
    - batch_size: Kích thước batch.
    - img_size: Kích thước ảnh đầu vào.
    """
    metrics = []

    # Tải mô hình YOLOv11n
    model = YOLO('yolov11n.pt')  # Tải mô hình YOLOv11n từ tệp yolov11n.pt

    # Huấn luyện mô hình với các tham số
    for epoch in range(epochs):
        # Huấn luyện và lấy kết quả
        results = model.train(data=data_config, epochs=1, batch_size=batch_size, img_size=img_size)
        loss = results.box_loss  # Mất mát (box_loss) từ huấn luyện
        accuracy = results.metrics["mAP_0.5"]  # Đo độ chính xác (mAP) tại IoU=0.5
        
        track_performance(metrics, epoch, loss, accuracy)  # Theo dõi hiệu suất
        save_model(model, save_path='./models', epoch=epoch, loss=loss)  # Lưu mô hình

if __name__ == '__main__':
    data_path = 'data.yaml'  # Đảm bảo bạn đã có tệp cấu hình dữ liệu
    model_path = 'yolov11n.yaml'  # Đảm bảo bạn đã có tệp cấu hình mô hình YOLOv11n
    train_yolov11n(data_path, model_path, epochs=100, batch_size=16, img_size=640)
