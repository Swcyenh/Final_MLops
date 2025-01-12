import cv2
import torch
import numpy as np

class YOLOv11nDetector:
    def __init__(self, model_path=None):
        # Load your custom YOLOv11n model (.pt file)
        if model_path:
            self.model = torch.load(model_path)  # Load custom YOLOv11n model
            self.model.eval()  # Set model to evaluation mode
        else:
            # If no model path is provided, load a default YOLOv5n model
            self.model = torch.hub.load('ultralytics/yolov5:v5.0', 'yolov5n')  # Example for YOLOv5n

    def detect(self, image_path):
        # Read input image
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Perform detection
        results = self.model(img_rgb)  # Model will return results as a pandas dataframe

        # Return predictions as a pandas dataframe
        return results.pandas().xywh[0]  # Predictions in [x_center, y_center, width, height]

def detect_with_yolov11n(image_path, model_path=None):
    # Instantiate the detector with the provided model path (if any)
    detector = YOLOv11nDetector(model_path)

    # Perform detection and return results
    return detector.detect(image_path)
