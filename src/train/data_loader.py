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
        # Đọc ảnh
        image = cv2.imread(self.images[idx])
        if image is None:
            raise FileNotFoundError(f"Không tìm thấy ảnh: {self.images[idx]}")
        
        # Đọc nhãn từ file .txt
        label_path = self.labels[idx]
        with open(label_path, 'r') as f:
            label = np.array([list(map(float, line.split())) for line in f])

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

    if len(images) == 0 or len(labels) == 0:
        raise FileNotFoundError("Thư mục ảnh hoặc nhãn rỗng hoặc không tìm thấy tệp phù hợp.")

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
    # Cập nhật đường dẫn tới dataset mới
    image_dir = 'src/train/data/images'  # Thư mục chứa ảnh
    label_dir = 'src/train/data/labels'  # Thư mục chứa nhãn

    try:
        train_loader, test_loader = load_data(image_dir, label_dir)
        print(f'Train loader: {len(train_loader)} batches')
        print(f'Test loader: {len(test_loader)} batches')
    except FileNotFoundError as e:
        print(e)
