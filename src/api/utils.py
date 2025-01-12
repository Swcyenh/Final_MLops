import os

def save_results(results, output_path):
    """Lưu kết quả phát hiện đối tượng vào tệp văn bản."""
    with open(output_path, 'w') as f:
        for result in results.itertuples():
            f.write(f"{result}\n")

def ensure_upload_folder_exists(folder='uploads'):
    """Đảm bảo thư mục uploads tồn tại"""
    if not os.path.exists(folder):
        os.makedirs(folder)
