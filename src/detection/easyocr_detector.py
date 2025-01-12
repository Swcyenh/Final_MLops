import easyocr
import cv2

class EasyOCRDetector:
    def __init__(self, lang='en'):
        # Khởi tạo EasyOCR Reader cho ngôn ngữ được chỉ định
        self.reader = easyocr.Reader([lang])

    def detect_text(self, image_path):
        # Phát hiện văn bản trong ảnh
        results = self.reader.readtext(image_path)
        texts = [result[1] for result in results]  # Lấy văn bản từ kết quả
        return texts

    def display_texts(self, image_path):
        # Vẽ văn bản nhận diện được lên ảnh
        img = cv2.imread(image_path)
        results = self.reader.readtext(image_path)

        for result in results:
            (x, y), (w, h) = result[0][0], result[0][2]
            text = result[1]
            # Vẽ hộp chứa văn bản và ghi chú
            cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Hiển thị ảnh với văn bản nhận diện
        cv2.imshow("OCR Results", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
