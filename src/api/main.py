from flask import Flask, request, jsonify
from detection.yolov11n_detector import YOLOv11nDetector
from detection.easyocr_detector import EasyOCRDetector
import os

# Khởi tạo Flask app
app = Flask(__name__)

# Tạo đối tượng YOLO và EasyOCR
yolo_detector = YOLOv11nDetector()
ocr_detector = EasyOCRDetector()

def save_results(results, filename):
    """Lưu kết quả vào tệp."""
    with open(filename, 'w') as f:
        for result in results:
            f.write(f"{result}\n")

@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    """API nhận ảnh và trả về kết quả phát hiện đối tượng."""
    file = request.files.get('image')
    
    if file:
        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)
        
        # Phát hiện đối tượng
        result = yolo_detector.detect(image_path)
        
        # Lưu kết quả vào tệp (nếu cần)
        save_results(result, 'results.txt')

        return jsonify(result.to_dict(orient='records'))
    else:
        return jsonify({"error": "No image file provided"}), 400


@app.route('/ocr_text', methods=['POST'])
def ocr_text():
    """API nhận ảnh và trả về văn bản nhận diện."""
    file = request.files.get('image')
    
    if file:
        image_path = os.path.join('uploads', file.filename)
        file.save(image_path)
        
        # Phát hiện văn bản
        texts = ocr_detector.detect_text(image_path)
        
        return jsonify({"texts": texts})
    else:
        return jsonify({"error": "No image file provided"}), 400


if __name__ == '__main__':
    app.run(debug=True)
