import cv2, copy
import numpy as np
from Frame import Frame
from Player import Player
from ultralytics import YOLO

class Manuel:
    def __init__(self, model, fp, mod_let) -> None:
        self.model = model # Yolo model to be used
        self.model_letter = mod_let
        self.file_path = fp+'.mp4' # Path to the video this run is using
        self.frames = [] # List of frames
    
    def main(self):
        # Open the video into OpenCV
        cap = cv2.VideoCapture(self.file_path)
        width  = cap.get(3)
        height  = cap.get(4)

        # If the video is not found, print error message and exit
        if not cap.isOpened():
            print("Unable to open specified video")
            exit()

        # Variables to be used
        frame_index = 0
        quit = False

        # Setup through each frame of the video until finished
        while not quit:
            ret, frame = cap.read()

            if not ret: # If frame not found then exit
                # print("Frame not recieved. Exiting...")
                break

            sf = 720 / width 
            frame = cv2.resize(frame, dsize=(720, int(height*sf)))

            # Create a new frame object and run the detector on it
            current_frame = Frame(frame_index, frame)

            # Output the frame to screen
            if (frame_index == 0):
                current_frame.run_detection(self.model)
                current_frame.annotate()
                cv2.imshow('First Frame', current_frame.frame_anot)
            else:
                detect_params = self.model.predict(source=[current_frame.frame_image], conf=0.45, save=False, show=False)
                params = detect_params[0].numpy()
                num_people = self.get_num_people(detect_params)
                if len(params) != 0: # If objects have been detected
                    detected_object_list = detect_params[0].boxes
                    # Loop through each detected object
                    for count in range(len(detect_params[0])): 
                        single_object = detected_object_list[count]
                        class_id = single_object.cls.numpy()[0]
                        bb = single_object.xyxy.numpy()[0]

                        if class_id == 0.0: # Check the class of the detected object is a person
                            next_player = Player(bb) # Create new player object
                        
                            temp_frame = copy.copy(current_frame.frame_image)
                            cv2.rectangle(temp_frame,
                                    (int(bb[0]), int(bb[1])),
                                    (int(bb[2]), int(bb[3])), (255,0,0), 2)
                            cv2.putText(temp_frame, str(num_people), 
                                        (100, 100), 4, 3, (0,0,0))
                            cv2.imshow('Player', temp_frame)

                            if num_people != 2:
                                break

                            key = cv2.waitKey(0)
                            while key != ord('0') and key != ord('1') and key != ord('/') and key != ord('q'):
                                key = cv2.waitKey(0)
                            if  key == ord('0'):
                                next_player.id = 0
                                current_frame.player_list.append(next_player)
                            elif key == ord('1'):
                                next_player.id = 1
                                current_frame.player_list.append(next_player)
                            elif key == ord('/'):
                                break
                            elif key == ord('q'):
                                quit = True
                    # cv2.destroyAllWindows()

            current_frame.annotate()
            cv2.imshow('Frame', current_frame.frame_anot)

            self.frames.append(current_frame)
            frame_index += 1

        cap.release()
        cv2.destroyAllWindows()


    def get_num_people(self, detect_params):
        people = 0
        params = detect_params[0].numpy()
        if len(params) != 0: # If objects have been detected
            detected_object_list = detect_params[0].boxes
            # Loop through each detected object
            for count in range(len(detect_params[0])): 
                single_object = detected_object_list[count]
                class_id = single_object.cls.numpy()[0]
                if class_id == 0.0:
                    people += 1
        return people

    def export(self):
        file = open(self.file_path + f'_man_label_{self.model_letter}.txt', 'w')

        for frame in self.frames:
            file.write('*\n')
            file.write(f'{frame.index}\n')

            for player in frame.player_list:
                file.write(f'{player.id}\n')
                file.write(f"{(player.bound_box).tostring().hex()}\n")

        file.write('/')
        file.close()

    def import_run(self):
        file = open(self.file_path + f'_man_label_{self.model_letter}.txt', 'r')

        lines = file.readlines()
        index = 0
        finish = False
        start = True

        while not finish:
            if lines[index] == '*\n':
                if not start:
                    self.frames.append(cur_frame)
                cur_frame = Frame(int(lines[index+1]))
                start = False
                index += 2
            elif lines[index] == '/':
                finish = True
            else:
                bb = np.frombuffer(bytes.fromhex(lines[index+1]), dtype=np.float32)
                new_player = Player(bb)
                new_player.id = int(lines[index])
                cur_frame.player_list.append(new_player)
                index += 2