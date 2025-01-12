import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        Khởi tạo bộ dữ liệu tùy chỉnh.
        
        Parameters:
        - images: Danh sách đường dẫn tới các ảnh.
        - labels: Danh sách nhãn tương ứng với các ảnh.
        - transform: Các phép biến đổi cần áp dụng cho ảnh (ví dụ: resize, normalize).
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])  # Đọc ảnh
        label = self.labels[idx]  # Lấy nhãn

        # Áp dụng các phép biến đổi nếu có
        if self.transform:
            image = self.transform(image)

        return image, label

def load_data(image_dir, label_dir, test_size=0.2, batch_size=16):
    """
    Hàm tải và xử lý dữ liệu huấn luyện.
    
    Parameters:
    - image_dir: Thư mục chứa các ảnh huấn luyện.
    - label_dir: Thư mục chứa các tệp nhãn.
    - test_size: Tỷ lệ dữ liệu dùng để kiểm tra.
    - batch_size: Kích thước batch.
    
    Returns:
    - train_loader: Bộ dữ liệu huấn luyện.
    - test_loader: Bộ dữ liệu kiểm tra.
    """
    # Lấy danh sách đường dẫn tới ảnh và nhãn
    images = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith('.jpg')]
    labels = [os.path.join(label_dir, fname.replace('.jpg', '.txt')) for fname in os.listdir(image_dir) if fname.endswith('.jpg')]

    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=test_size)

    # Áp dụng các phép biến đổi ảnh (resize, normalize, etc.)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Tạo DataLoader cho tập huấn luyện và kiểm tra
    train_dataset = CustomDataset(train_images, train_labels, transform=transform)
    test_dataset = CustomDataset(test_images, test_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == '__main__':
    image_dir = 'data/images'
    label_dir = 'data/labels'
    train_loader, test_loader = load_data(image_dir, label_dir)
    print(f'Train loader: {len(train_loader)} batches')
    print(f'Test loader: {len(test_loader)} batches')
