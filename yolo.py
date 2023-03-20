from ultralytics import YOLO
import numpy
import cv2

model = YOLO("yolov8x.pt", "v8")
# model.train (epochs=5)

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('videos/Video3/Video3_Clip2.mp4')

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

    #Resize the video so that the detection algorithm runs slower
    scale_fact = 0.5
    # frame = cv2.resize(frame, (int(width*scale_fact), int(height*scale_fact)))

    detect_params = model.predict(source=[frame], conf=0.75, save=False)

    params = detect_params[0].numpy()
    player_id = 0
    if len(params) != 0:
        for count in range(len(detect_params[0])):

            detections = detect_params[0].boxes
            single_detect = detections[count]
            class_id = single_detect.cls.numpy()[0]
            conf = single_detect.conf.numpy()[0]
            bb = single_detect.xyxy.numpy()[0]

            if class_id == 0.0:
                #Draw the bounding box onto each person class
                cv2.rectangle(frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])), (0,0,255), 2)
                #Add the confidence rate and player id number to the bounding box
                cv2.putText(frame, (str(player_id) + " - " + str(conf)),
                            (int(bb[0]), int(bb[1])-10), 3, 0.6, (255,255,255))
                
                player_id += 1
                


    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()