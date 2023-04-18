from ultralytics import YOLO
from Run import Run
import numpy, math
import cv2

print(cv2.__version__)

# Import the model to be used
model = YOLO("yolov8n.pt", "v8")

file_path = 'videos/Video3/Clip4.mp4'

run = Run(1, model, file_path, "TRACKING")
run.main()

tot_frames = len(run.frames)
maintained_frames = 0

for frame in run.frames:
    if frame.maintained:
        maintained_frames += 1

percent_consistent = round((maintained_frames/tot_frames) * 100, 2)
print(f"ID's remained consistent in: {percent_consistent} percent of the frames")
