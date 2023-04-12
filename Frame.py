from Player import Player
import cv2

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

    # Determine the ids of the players in the current frame using specified heuristic
    # TRACKING - Just the tracking by detection using the closest player from the last frame
    #            which is calculated using the Euclidean distance and a threshold
    # SPEED - 
    # DIR - 
    # SPEED_DIR - 
    def determine_ids(self, frames, heuristic):
        print("Determining the IDs")