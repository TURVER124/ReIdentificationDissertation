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
        if len(self.player_list) > 0:
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
        else:
            cv2.putText(ano_frame, '*Yolo Detection Failed*', 
                            (20, 50), 4, 1, (255,255,255))
        self.frame_anot = ano_frame.copy()


    # Label players in order of detection
    def label_detec_order(self):
        index = 0
        for cur_player in self.player_list:
            cur_player.id = index
            index += 1

    # Find the Euclidean distance between the start of two bounding boxes
    def find_large_dist(self, dist):
        largest = 0
        index = []
        for count in range(len(dist)):
            if dist[count][0] > largest:
                largest = dist[count][0]                
                index = count
        return index
    
    
    # Find the largest id being used thoughout all frames
    def get_highest_id(self, frames):
        largest_id = 0
        for frame in frames:
            for player in frame.player_list:
                if player.id > largest_id:
                    largest_id = player.id
        return largest_id
    

    # Compare two frames so check if they have the same index and player list
    def compare(self, sec_frame):
        if self.index == sec_frame.index:
            equal_players = 0

            for f1_player in self.player_list:
                for f2_player in sec_frame.player_list:
                    same_player = f1_player.equal(f2_player)
                    if same_player:
                        equal_players += 1
            if equal_players == len(self.player_list):
                return True
            else:
                return False
        else:
            return False
        
    
    # Simple tracking method, keep id using Euclidean dist of bouding boxes from one frame to another 
    def bound_box_diff(self, frames):
        poss_play_ids = copy.deepcopy(self.player_list)
        highest_id = self.get_highest_id(frames)

        most_rec_players = []
        found = False

        for id in range(highest_id+1):
            found = False
            for frame in frames[::-1]:
                if not found:
                    for player in frame.player_list:
                        if player.id == id:
                            most_rec_players.append(player)
                            found = True
                            break
                else:
                    break

        for cur_player in poss_play_ids:
            closest = 0
            smallest_diff = 100000
            for past_player in most_rec_players:
                start_diff, end_diff = cur_player.bb_diff(past_player)
                tot_diff = start_diff + end_diff
                if tot_diff < smallest_diff:
                    smallest_diff = tot_diff
                    closest = past_player

            cur_player.id = closest.id
            most_rec_players.remove(closest)

        return poss_play_ids
        
    # Identify players by the colour of their shirt
    def colour(self, frames):
        poss_play_ids = copy.deepcopy(self.player_list)
        highest_id = self.get_highest_id(frames)

        most_rec_players = []

        for id in range(highest_id+1):
            found = False
            for frame in frames:
                if not found:
                    for player in frame.player_list:
                        if player.id == id:
                            most_rec_players.append(player)
                            found = True
                            break
                else:
                    break
        
        # print("\nNew Frame")
        for cur_player in poss_play_ids:
            closest = 0
            smallest_diff = 100000
            for past_player in most_rec_players:
                diff = cur_player.comp_colour(past_player)
                if diff < smallest_diff:
                    smallest_diff = diff
                    closest = past_player
                    # print(f'{cur_player.colour} : {closest.colour}')

            cur_player.id = closest.id
            most_rec_players.remove(closest)

        return poss_play_ids                       



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

        elif heuristic == "BB_DIFF":
            pos_ids = self.bound_box_diff(frames)
            for i in range(len(self.player_list)):
                self.player_list[i].id = pos_ids[i].id

        elif heuristic == "COLOUR":
            pos_ids = self.colour(frames)
            for i in range(len(self.player_list)):
                self.player_list[i].id = pos_ids[i].id