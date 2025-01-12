import easyocr
import cv2

class EasyOCRDetector:
    def __init__(self, lang='en'):
        # Initialize EasyOCR Reader with the specified language
        self.reader = easyocr.Reader([lang])

    def detect_text(self, image_path):
        # Detect text in the image
        results = self.reader.readtext(image_path)
        texts = [result[1] for result in results]  # Extract the detected text
        return texts

    def display_texts(self, image_path):
        # Draw the detected text on the image
        img = cv2.imread(image_path)
        results = self.reader.readtext(image_path)

        for result in results:
            (x, y), (w, h) = result[0][0], result[0][2]
            text = result[1]
            # Draw bounding box and text
            cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Show the image with detected text
        cv2.imshow("OCR Results", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Function to call the EasyOCR detector for text detection
def detect_text_with_easyocr(image_path, lang='en'):
    detector = EasyOCRDetector(lang)
    return detector.detect_text(image_path)
