from ultralytics import YOLO
from Player import Player
import numpy, math
import cv2

model = YOLO("yolov8n.pt", "v8")
# model.train (epochs=5)

#Open video file, getting the height and width of the frames
cap = cv2.VideoCapture('videos/Video3/Video3_Clip1.mp4')
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
player_id = 0

# Check detection against each player and find the smallest Euclidean distance
def comp_detect_to_player_bb(detect_bb):
    players_by_dist = []
    dist = []

    for player in players:
        start_diff = bb_diff(player.bound_box[0], player.bound_box[1], detect_bb[0], detect_bb[1])
        end_diff = bb_diff(player.bound_box[2], player.bound_box[3], detect_bb[2], detect_bb[3])
        tot_diff = start_diff + end_diff
        players_by_dist.append(player)
        dist.append(tot_diff)

    players_by_dist = numpy.array(players_by_dist)
    dist = numpy.array(dist)
    index = dist.argsort()
    sorted_player_dist = players_by_dist[index]
    dist.sort()

    return sorted_player_dist, dist


# Find the Euclidean distance between the start of two bounding boxes
def bb_diff(player_x, player_y, detect_x, detect_y):
    euclid = math.sqrt(((player_x - detect_x) * (player_x - detect_x)) +
                        ((player_y - detect_y) * (player_y - detect_y)))
    
    return abs(euclid)

def get_shirt_colour():
    print()

# Main loop taking each frame at a time until the video is finished
# for x in range(3):
while True:
    ret, frame = cap.read()

    if not ret:
        print("Frame not recieved. Exiting...")
        break

    # Resize the video so that the detection algorithm runs faster
    # scale_fact = 0.5
    # frame = cv2.resize(frame, (int(width*scale_fact), int(height*scale_fact)))

    # For each frame run the detection algorithm (tracking by detection) and convert
    # parameter to numpy array
    detect_params = model.predict(source=[frame], conf=0.45, save=False)
    params = detect_params[0].numpy()
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
                    new_player = Player(player_id, bb, conf, "Blue")
                    players.append(new_player)
                    display_id = "Player_" + str(players[player_id].id)
                    player_id += 1
                
                # If the detection is not on the first frame
                else:
                    # Get the distances from the current bouning box to the already detected ones
                    players_by_distance, distances = comp_detect_to_player_bb(bb)
                    print(str(players_by_distance[0]) + "\t&&\t" + str(players_by_distance[1]))
                    print(distances[0])
                    if (distances[0] < 20): # If the distance is below the threshold then this is likely to be the same player
                        players_by_distance[0].bound_box = bb
                        display_id = "Player_" + str(players_by_distance[0].id)
                    else: # If the distance is above the threshold then this is likely to be a player not already in the list
                        new_player = Player(player_id, bb, conf, "")
                        players.append(new_player)
                        display_id = "Player_" + str(players[player_id].id)
                        player_id += 1
                # Draw the bounding box onto each person class
                cv2.rectangle(frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])), (0,0,255), 2)
                # Add the confidence rate and player id number to the bounding box
                cv2.putText(frame, (display_id),
                            (int(bb[0]), int(bb[1])-10), 4, 0.8, (255,255,255))
                
        first_frame = False
        print(player_id)
                

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()