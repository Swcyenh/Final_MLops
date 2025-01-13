import cv2
import os

def load_image(image_path):
    """Tải ảnh từ đường dẫn và kiểm tra sự tồn tại của file."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image at {image_path} not found.")
    return cv2.imread(image_path)

def save_results(results, output_path):
    """Lưu kết quả phát hiện vào tệp văn bản."""
    with open(output_path, 'w') as file:
        for result in results:
            file.write(f"{result}\n")

def show_image(img):
    """Hiển thị ảnh bằng OpenCV."""
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
