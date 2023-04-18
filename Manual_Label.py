import cv2, time, copy
from Frame import Frame
from Player import Player
from ultralytics import YOLO

class Manuel:
    def __init__(self, model, vid_path) -> None:
        self.model = model # Yolo model to be used
        self.vid_path = vid_path # Path to the video this run is using
        self.frames = [] # List of frames
    
    def main(self):
        # Open the video into OpenCV
        cap = cv2.VideoCapture(self.vid_path)
        width  = cap.get(3)
        height  = cap.get(4)

        # If the video is not found, print error message and exit
        if not cap.isOpened():
            print("Unable to open specified video")
            exit()

        # Variables to be used
        frame_index = 0

        # Setup through each frame of the video until finished
        while True:
            ret, frame = cap.read()
            if not ret: # If frame not found then exit
                print("Frame not recieved. Exiting...")
                break

            # Create a new frame object and run the detector on it
            current_frame = Frame(frame_index, frame)

            # Output the frame to screen
            if (frame_index == 0):
                current_frame.run_detection(self.model)
                current_frame.annotate()
                cv2.imshow('First Frame', current_frame.frame_anot)
            else:
                detect_params = model.predict(source=[self.frame_image], conf=0.45, save=False, show=False)
                params = detect_params[0].numpy()
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
                        
                        temp_frame = copy.copy(current_frame.frame_image)
                        cv2.rectangle(temp_frame,
                                (int(bb[0]), int(bb[1])),
                                (int(bb[2]), int(bb[3])), (255,0,0), 2)
                        cv2.imshow('Player', temp_frame)

            current_frame.annotate()
            # Manual input as to whether the system has been able to maintain consistant player identity
            # Enter 0 if the identities are maintained from first frame
            # Enter 1 if identities have been switched

            resize = cv2.resize(current_frame.frame_anot,dsize=None,fx=1.3,fy=1.3)
            cv2.imshow('window_name', resize)
            cv2.setWindowTitle('window_name', f'Frame {current_frame.index}')
            key = cv2.waitKey(0)
            while key != ord('0') and key != ord('1') and key != ord('q'):
                key = cv2.waitKey(0)
            if  key == ord('0'):
                current_frame.maintained = False
            elif key == ord('1'):
                current_frame.maintained = True
            elif key == ord('q'):
                break

            self.frames.append(current_frame)
            frame_index += 1

        cap.release()
        cv2.destroyAllWindows()


model = YOLO("yolov8n.pt", "v8")
file_path = 'videos/Video3/Clip4.mp4'

new_man = Manuel(model, file_path)

new_man.main()