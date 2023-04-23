from Player import Player
import cv2, copy

class Frame:
    def __init__(self, index, frame_img=None) -> None:
        self.index = index # Index of the frame in the video sqauence
        self.frame_image = frame_img # The actual image of the frame as an array
        self.frame_anot = None
        self.player_list = [] # List of players contained in this frame
        self.maintained = False # Whether the player identities have been correctly maintained

    def num_players(self):
        num_players = len(self.player_list)
        return num_players

    def __str__(self):
        string = f'Frame index: {self.index}\n'
        string += f'Frame maintained: {self.maintained}\n'
        for player in self.player_list:
            string += f'Player: {player.id} - {player.bound_box}\n'
        
        return string
    
    # Run the Yolo detection on the single frame populate players
    def run_detection(self, model):
        # Run the Yolo detection on this single frame
        detect_params = model.predict(source=[self.frame_image], conf=0.45, save=False, show=False)
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
                    next_player.get_colour(self.frame_image)
                    if self.index == 0:
                        next_player.id = player_id
                        player_id += 1
                    self.player_list.append(next_player) # Add the new player to the list

    # Annotate the frame with the player bounding boxes and their IDs
    def annotate(self):
        ano_frame = self.frame_image.copy()
        for player in self.player_list:
            if player.id == 0:
                player.get_colour(self.frame_image)
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
    def tracking(self, frames):
        prev_frame = (frames[-1:])[0]
        poss_play_ids = copy.deepcopy(self.player_list)
        taken_ids = []

        # Iterate through each player in the previous frame
        for prev_player in prev_frame.player_list:
            lowest_diff = 100000000
            lowest_ind = 0
            index = 0
            for cur_player in self.player_list:
                start_diff, end_diff = prev_player.bb_diff(cur_player)
                tot_diff = start_diff + end_diff
                if tot_diff < lowest_diff:
                    lowest_diff = tot_diff
                    lowest_ind = index
                index += 1
                
            if poss_play_ids[lowest_ind].dist_diff > lowest_diff:
                poss_play_ids[lowest_ind].dist_diff = lowest_diff
                poss_play_ids[lowest_ind].id = prev_player.id
                taken_ids.append(prev_player.id)


        next_id = self.get_next_avil_id(frames)
        for player in poss_play_ids:
            if player.id == -1:
                pos_matches = []
                if len(frames) < 5:
                    for s_frame in frames:
                        for pos_player in s_frame.player_list:
                            if pos_player.id not in taken_ids:
                                pos_matches.append(copy.deepcopy(pos_player))
                else:
                    for s_frame in frames[-5:]:
                        for pos_player in s_frame.player_list:
                            if pos_player.id not in taken_ids:
                                pos_matches.append(copy.deepcopy(pos_player))
                if len(pos_matches) > 0:
                    closest = None
                    diff = 1000000
                    for match in pos_matches:
                        start_diff, end_diff = match.bb_diff(player)
                        tot_diff = start_diff + end_diff
                        if tot_diff < diff:
                            diff = tot_diff
                            closest = match
                    if diff < closest.get_dist_thrshold(player):
                        player.id = closest.id
                    else:
                        player.id = next_id
                        next_id += 1
                else:
                    player.id = next_id
                    next_id += 1

        return poss_play_ids

    # Find the next free player id number
    def get_next_avil_id(self, frames):
        largest_id = 0
        for frame in frames:
            for player in frame.player_list:
                if player.id > largest_id:
                    largest_id = player.id

        return largest_id + 1

    # Find the Euclidean distance between the start of two bounding boxes


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
    # COLOUR - 
    # SPEED - 
    # DIR - 
    # SPEED_DIR - 
    def determine_ids(self, frames, heuristic):
        if heuristic == "NONE":
            self.label_detec_order()

        elif heuristic == "TRACKING":
            pos_ids = self.tracking(frames)
            for i in range(len(self.player_list)):
                self.player_list[i].id = pos_ids[i].id
