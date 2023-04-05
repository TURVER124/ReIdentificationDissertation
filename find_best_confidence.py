from ultralytics import YOLO
import numpy, Player, math, os, time
import cv2

model = YOLO("yolov8n.pt", "v8")


image = "conf_test10.png"
input_path = "confidence_level/"
output_path = "runs/detect/predict10/"

conf = 0.30

for i in range (10):
    conf = float(round(conf, 2))
    detect_params = model.predict(source=input_path+image, conf=conf, save=True)
    time.sleep(1)
    os.rename(output_path+image, output_path+"conf_"+str(conf))
    time.sleep(1)
    conf += 0.05