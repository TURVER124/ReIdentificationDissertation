import cv2, time
from Frame import Frame

class Run:
    def __init__(self, id, model, vid_path, heur="NONE") -> None:
        self.id = id # ID number of the current run
        self.model = model # Yolo model to be used
        self.vid_path = vid_path # Path to the video this run is using
        self.frames = [] # List of frames
        self.percent_corr = 0 # Percentage of fames with consistent identities
        self.heuristic = heur

    def __str__(self):
        return f'Run {self.id}'
    
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
            current_frame.run_detection(self.model)

            # Output the frame to screen
            if (frame_index == 0):
                current_frame.annotate()
                cv2.imshow('First Frame', current_frame.frame_anot)
            else:
                if len(self.frames) < 5:
                    current_frame.determine_ids(self.frames, self.heuristic)
                else:
                    current_frame.determine_ids(self.frames[-5:], self.heuristic)

            current_frame.annotate()
            # Manual input as to whether the system has been able to maintain consistant player identity
            # Enter 0 if the identities are maintained from first frame
            # Enter 1 if identities have been switched
            cv2.imshow('window_name', current_frame.frame_anot)
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
