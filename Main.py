from ultralytics import YOLO
from Run import Run
from Manual import Manuel
import numpy, math
import cv2

class Main:
    def __init__(self, fp, mod) -> None:
        self.file_path = fp
        self.model = mod
        self.frame_status = []

    print(cv2.__version__)







# Import the model to be used
model = YOLO("yolov8n.pt", "v8")




