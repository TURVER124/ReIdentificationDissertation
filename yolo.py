from ultralytics import YOLO
import numpy, Player
import cv2

model = YOLO("yolov8n.pt", "v8")
# model.train (epochs=5)

#Open video file, getting the height and width of the frames
cap = cv2.VideoCapture('videos/Video3/Video3_Clip2.mp4')
width  = cap.get(3)
height  = cap.get(4)

#If the video is not found, print error message and exit
if not cap.isOpened():
    print("Unable to open specified video")
    exit()

first_frame = True

while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame not recieved. Exiting...")
        break

    # Set an array of player objects to keep track of each player
    players = []
    num_players = []

    #Resize the video so that the detection algorithm runs slower
    scale_fact = 0.5
    # frame = cv2.resize(frame, (int(width*scale_fact), int(height*scale_fact)))

    #For each frame run the detection algorithm (tracking by detection) and convert
    #parameter to numpy array
    detect_params = model.predict(source=[frame], conf=0.60, save=False)
    params = detect_params[0].numpy()
    player_id = 0
    if len(params) != 0:
        for count in range(len(detect_params[0])):

            # Split into different parameters from the detection
            detections = detect_params[0].boxes
            single_detect = detections[count]
            class_id = single_detect.cls.numpy()[0]
            conf = single_detect.conf.numpy()[0]
            bb = single_detect.xyxy.numpy()[0]

            # Check if the detection is a person
            if class_id == 0.0:
                if first_frame:
                    new_player = Player.Player(player_id, bb, conf, "Blue")
                    players.append(new_player)
                #Draw the bounding box onto each person class
                cv2.rectangle(frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])), (0,0,255), 2)
                #Add the confidence rate and player id number to the bounding box
                cv2.putText(frame, (str(player_id) + " - " + str(conf)),
                            (int(bb[0]), int(bb[1])-10), 3, 0.6, (255,255,255))
                
                player_id += 1
                first_frame = False

                print(players)
                

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()