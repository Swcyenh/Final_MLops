from flask import Flask, request, jsonify
from pathlib import Path
from PIL import Image
import io
import sys

# Append project directory to sys.path
sys.path.append('D:/MLOPS_FINAL/Final_MLops')

# Import các hàm xử lý từ detection
from src.detection.yolov11n_detector import detect_with_yolov11n
from src.detection.easyocr_detector import detect_text_with_easyocr

app = Flask(__name__)

# Đường dẫn lưu trữ tạm thời (nếu cần)
TEMP_IMAGE_PATH = Path("temp_images")
TEMP_IMAGE_PATH.mkdir(exist_ok=True)

@app.route("/detect/objects", methods=["POST"])
def detect_objects():
    """
    API endpoint để phát hiện đối tượng trong hình ảnh sử dụng YOLOv11n.
    """
    if 'file' not in request.files:
        return jsonify({"error": "Không tìm thấy file trong request."}), 400

    file = request.files['file']
    if file.content_type not in ["image/jpeg", "image/png"]:
        return jsonify({"error": "File không phải là định dạng hình ảnh (JPEG/PNG)."}), 400

    try:
        # Đọc nội dung file và lưu tạm thời
        temp_file = TEMP_IMAGE_PATH / file.filename
        file.save(temp_file)

        # Gọi YOLOv11n để phát hiện đối tượng
        results = detect_with_yolov11n(temp_file)

        # Xóa file tạm thời sau khi xử lý
        temp_file.unlink()

        return jsonify({"status": "success", "results": results})

    except Exception as e:
        return jsonify({"error": f"Lỗi xử lý: {str(e)}"}), 500

@app.route("/detect/text", methods=["POST"])
def detect_text():
    """
    API endpoint để nhận diện văn bản trong hình ảnh sử dụng EasyOCR.
    """
    if 'file' not in request.files:
        return jsonify({"error": "Không tìm thấy file trong request."}), 400

    file = request.files['file']
    if file.content_type not in ["image/jpeg", "image/png"]:
        return jsonify({"error": "File không phải là định dạng hình ảnh (JPEG/PNG)."}), 400

    try:
        # Đọc nội dung file
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))

        # Gọi EasyOCR để nhận diện văn bản
        results = detect_text_with_easyocr(image)

        return jsonify({"status": "success", "text": results})

    except Exception as e:
        return jsonify({"error": f"Lỗi xử lý: {str(e)}"}), 500

@app.route("/", methods=["GET"])
def root():
    """
    Endpoint mặc định.
    """
    return jsonify({"message": "Chào mừng đến với Image Detection API. Hãy thử POST lên /detect/objects hoặc /detect/text."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
