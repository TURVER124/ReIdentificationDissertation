from ultralytics import YOLO
from Run import Run
import numpy, math
import cv2

# Import the model to be used
model = YOLO("yolov8n.pt", "v8")

file_path = 'videos/Video3/Video3_Clip2.mp4'

run = Run(1, model, file_path)
run.main()