from ultralytics import YOLO
import numpy
import cv2

model = YOLO("yolov8n.pt", "v8")

# detection_output = model.predict(source = "images/test_image_close.PNG", conf=0.25, save=True)

# print(detection_output)

# print(detection_output[0].numpy())

frame_width = 640
frame_height = 480

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('videos/dataset_test_vid.mp4')

width  = cap.get(3)
height  = cap.get(4)

print(width)
print(height)

if not cap.isOpened():
    print("Unable to open camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame not recieved. Exiting...")
        break

    frame = cv2.resize(frame, (frame_width, frame_height))

    detect_params = model.predict(source=[frame], conf=0.40, save=False)

    params = detect_params[0].numpy()
    if len(params) != 0:
        for count in range(len(detect_params[0])):
            # print(count)

            detections = detect_params[0].boxes
            single_detect = detections[count]
            class_id = single_detect.cls.numpy()[0]
            conf = single_detect.conf.numpy()[0]
            bb = single_detect.xyxy.numpy()[0]

            cv2.rectangle(frame,
                (int(bb[0]), int(bb[1])),
                (int(bb[2]), int(bb[3])),3)


    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()