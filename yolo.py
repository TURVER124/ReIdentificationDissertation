from ultralytics import YOLO
import numpy, Player, math
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

# Set an array of player objects to keep track of each player
players = []
num_players = 0
first_frame = True

# Check detection against each player and find the smallest Euclidean distance
def comp_detect_to_player_bb(detect_bb):
    closest = 10000000000000
    id = -1

    for player in players:
        start_diff = bb_diff(player.bound_box[0], player.bound_box[1], detect_bb[0], detect_bb[1])
        end_diff = bb_diff(player.bound_box[2], player.bound_box[3], detect_bb[2], detect_bb[3])
        tot_diff = start_diff + end_diff
        if tot_diff < closest:
            closest = tot_diff
            id = player.id

    return closest, id


# Find the Euclidean distance between the start of two bounding boxes
def bb_diff(player_x, player_y, detect_x, detect_y):
    euclid = math.sqrt(((player_x - detect_x) * (player_x - detect_x)) +
                        ((player_y - detect_y) * (player_y - detect_y)))
    
    return abs(euclid)

def get_shirt_colour():
    print()


# for x in range(20):
while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame not recieved. Exiting...")
        break

    #Resize the video so that the detection algorithm runs slower
    scale_fact = 0.5
    # frame = cv2.resize(frame, (int(width*scale_fact), int(height*scale_fact)))

    #For each frame run the detection algorithm (tracking by detection) and convert
    #parameter to numpy array
    detect_params = model.predict(source=[frame], conf=0.45, save=False)
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
                # Get detections on the first frame to get a basic list of players
                if first_frame:
                    new_player = Player.Player(player_id, bb, conf, "Blue")
                    players.append(new_player)
                    display_id = "Player_" + str(players[player_id].id)
                    player_id += 1
                else:
                    dist_diff, player_comp_id = comp_detect_to_player_bb(bb)
                    # print("This detection is " + str(players[player_comp_id].id))
                    if (dist_diff < 20):
                        players[player_comp_id].bound_box = bb
                        display_id = "Player_" + str(players[player_comp_id].id)
                    else:
                        new_player = Player.Player(player_id, bb, conf, "Blue")
                        players.append(new_player)
                        display_id = "Player_" + str(players[player_id].id)
                        player_id += 1
                #Draw the bounding box onto each person class
                cv2.rectangle(frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])), (0,0,255), 2)
                #Add the confidence rate and player id number to the bounding box
                cv2.putText(frame, (display_id),
                            (int(bb[0]), int(bb[1])-10), 4, 0.8, (255,255,255))
                
        first_frame = False
                

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()