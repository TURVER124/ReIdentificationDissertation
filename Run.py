import cv2, os
from Frame import Frame

class Run:
    def __init__(self, model, vid_path, meth="NONE", show=False) -> None:
        self.model = model # Yolo model to be used
        self.vid_path = vid_path+'.mp4' # Path to the video this run is using
        self.frames = [] # List of frames
        self.method = meth
        self.show = show

    def __str__(self):
        return f'Run {self.id}'
    
    def frame_by_frame(self):
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

            sf = 720 / width 
            frame = cv2.resize(frame, dsize=(720, int(height*sf)))

            # Create a new frame object and run the detector on it
            current_frame = Frame(frame_index, frame)
            current_frame.run_detection(self.model)

            # Output the frame to screen
            if (frame_index == 0):
                current_frame.annotate()
                cv2.imshow('First Frame', current_frame.frame_anot)
            else:
                current_frame.determine_ids(self.frames, self.method)

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



    def auto_run(self):
        # Open the video into OpenCV
        cap = cv2.VideoCapture(self.vid_path)
        width  = cap.get(3)
        height  = cap.get(4)

        if not os.path.exists(f'{self.vid_path[:-4]}_{self.method}'):
            os.mkdir(f'{self.vid_path[:-4]}_{self.method}')

        # If the video is not found, print error message and exit
        if not cap.isOpened():
            print("Unable to open specified video")
            exit()

        # Variables to be used
        frame_index = 0

        # Setup through each frame of the video until finished
        # for x in range(5):
        while True:
            ret, frame = cap.read()
            if not ret: # If frame not found then exit
                # print("Frame not recieved. Exiting...")
                break

            sf = 720 / width 
            frame = cv2.resize(frame, dsize=(720, int(height*sf)))

            # Create a new frame object and run the detector on it
            current_frame = Frame(frame_index, frame)
            current_frame.run_detection(self.model)

            # Output the frame to screen
            if (frame_index == 0):
                current_frame.annotate()
                if self.show:
                    cv2.imshow('First Frame', current_frame.frame_anot)
            else:
                # Check if both players have been correctly detected
                if len(current_frame.player_list) == 2:
                    current_frame.determine_ids(self.frames, self.method)
                else:
                    current_frame.player_list = []

                current_frame.annotate()

            resize = cv2.resize(current_frame.frame_anot,dsize=None,fx=1.3,fy=1.3)
            if self.show:
                # cv2.putText(resize, str(current_frame.index), 
                #             (100, 100), 4, 3, (0,0,0))
                cv2.imshow('window_name', resize)
                cv2.setWindowTitle('window_name', f'Frame {current_frame.index}')

                cv2.imwrite(f'{self.vid_path[:-4]}_{self.method}/{current_frame.index}.jpeg', resize)

            self.frames.append(current_frame)
            frame_index += 1

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()