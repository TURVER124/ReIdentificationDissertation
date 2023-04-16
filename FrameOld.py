from Player import Player
import cv2, math, numpy
from itertools import chain

class Frame:
    def __init__(self, index, frame_img) -> None:
        self.index = index # Index of the frame in the video sqauence
        self.frame_image = frame_img # The actual image of the frame as an array
        self.frame_anot = None
        self.player_list = [] # List of players contained in this frame
        self.maintained = False # Whether the player identities have been correctly maintained

    def num_players(self):
        num_players = len(self.player_list)
        return num_players

    def __str__(self):
        return f'Frame Index: {self.index} - {self.maintained}'
    
    # Run the Yolo detection on the single frame populate players
    def run_detection(self, model):
        # Run the Yolo detection on this single frame
        detect_params = model.predict(source=[self.frame_image], conf=0.45, save=False)
        params = detect_params[0].numpy()
        player_id = 0 # ID number of the player to be used on the 

        if len(params) != 0: # If objects have been detected
            detected_object_list = detect_params[0].boxes
            # Loop through each detected object
            for count in range(len(detect_params[0])): 
                single_object = detected_object_list[count]
                class_id = single_object.cls.numpy()[0]
                conf = single_object.conf.numpy()[0]
                bb = single_object.xyxy.numpy()[0]

                if class_id == 0.0: # Check the class of the detected object is a person
                    next_player = Player(bb, conf) # Create new player object
                    if self.index == 0:
                        next_player.id = player_id
                        player_id += 1
                    self.player_list.append(next_player) # Add the new player to the list

    # Annotate the frame with the player bounding boxes and their IDs
    def annotate(self):
        ano_frame = self.frame_image.copy()
        for player in self.player_list:
            bb = player.bound_box
            p_id = player.id
            cv2.rectangle(ano_frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])), (0,0,255), 2)
            # Add the confidence rate and player id number to the bounding box
            output_id = f'PlayerID {p_id}'
            cv2.putText(ano_frame, (output_id),
                        (int(bb[0]), int(bb[1])-10), 4, 0.8, (255,255,255))
        self.frame_anot = ano_frame.copy()

    # Label players in order of detection
    def label_detec_order(self):
        index = 0
        for cur_player in self.player_list:
            cur_player.id = index
            index += 1

    # Simple tracking method, keep id using Euclidean dist of bouding boxes from one frame to another 
    def tracking(self, prev_frame, top_n):
        closests = []
        distances = []
        for cur_player in self.player_list:
            closest = []
            dist = []
            for prev_player in prev_frame.player_list:
                start_diff = self.bb_diff(prev_player.bound_box[0], prev_player.bound_box[1],
                                          cur_player.bound_box[0], cur_player.bound_box[1])
                end_diff = self.bb_diff(prev_player.bound_box[2], prev_player.bound_box[3],
                                        cur_player.bound_box[2], cur_player.bound_box[3])
                tot_diff = start_diff + end_diff
                closest.append(prev_player)
                dist.append(tot_diff)
            
            closest = numpy.array(closest)
            dist = numpy.array(dist)
            index = dist.argsort()
            closest_player_detections = closest[index]
            dist.sort()
            closests.append(closest_player_detections[:top_n])
            distances.append(dist[:top_n])

        self.matchup(closests, distances, top_n)

        return closests, distances

    def matchup(self, closests, distances, num_posib):
        for i in range(len(closests)):
            for j in range(len(closests)):
                if i != j and closests[i][0] == closests[j][0]:
                    print(f"{self.player_list[i]} : {self.player_list[j]}")


    # Find the Euclidean distance between the start of two bounding boxes
    def bb_diff(self, player_x, player_y, detect_x, detect_y):
        euclid = math.sqrt(((player_x - detect_x) * (player_x - detect_x)) +
                            ((player_y - detect_y) * (player_y - detect_y)))
        
        return abs(euclid)


    def find_large_dist(self, dist, amount):
        largest = 0
        index = []
        for count in range(len(dist)):
            if dist[count][0] > largest:
                largest = dist[count][0]                
                index = count
        return index
    
    
    def get_highest_id(self, frames):
        largest_id = 0
        for frame in frames:
            for player in frame.player_list:
                if player.id > largest_id:
                    largest_id = player.id
        return largest_id


    # Determine the ids of the players in the current frame using specified heuristic
    # NONE - No re-identification used just labeled in the order players are detected
    # TRACKING - Just the tracking by detection using the closest player from the last frame
    #            which is calculated using the Euclidean distance between current and previous bb
    # SPEED - 
    # DIR - 
    # SPEED_DIR - 
    def determine_ids(self, frames, heuristic):
        top_n = 2

        if heuristic == "NONE":
            self.label_detec_order()

        elif heuristic == "TRACKING":
            most_rec_frame = (frames[-1:])[0]
            closest_match, distance_diff = self.tracking(most_rec_frame, top_n)

            # Check if a new player has been detected
            player_num_dif = len(self.player_list) - len(most_rec_frame.player_list)
            if player_num_dif > 0:
                fur_dist_ind = self.find_large_dist(distance_diff, player_num_dif)
            else:
                fur_dist_ind = -1

            index = 0
            for player in self.player_list:
                if index != fur_dist_ind:
                    player.id = (closest_match[index][0]).id
                    index += 1
                else:
                    print("Entered this")
                    player.id = self.get_highest_id(frames)
                    index += 1