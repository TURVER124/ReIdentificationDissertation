from ultralytics import YOLO
from Run import Run
from Manual import Manuel

class Main:
    def __init__(self, fp, mod, meth='NONE', show=False) -> None:
        self.file_path = fp
        self.method = meth
        self.model = mod
        self.frame_status = []
        self.show = show


    def run_comparison(self):

        new_run = Run(self.model, self.file_path, self.method, self.show)
        ground_truth = Manuel(self.model, self.file_path)

        new_run.auto_run()
        ground_truth.import_run()

        for index in range(len(new_run.frames)-1):
            if len(new_run.frames[index].player_list) > 0 and len(ground_truth.frames[index].player_list) > 0:            
                equal_frames = new_run.frames[index].compare(ground_truth.frames[index])
                if equal_frames:
                    self.frame_status.append('T')
                else:
                    self.frame_status.append('F')
            else:
                self.frame_status.append('X')

    def perc_maintined(self):
        num_correct = 0
        for res in self.frame_status:
            if res == 'T':
                num_correct += 1

        percent = round((num_correct/len(self.frame_status)*100), 2)

        return str(percent)
    
    def perc_yolo(self):
        num_detect = 0
        for res in self.frame_status:
            if res != 'X':
                num_detect += 1

        percent = round((num_detect/len(self.frame_status)*100), 2)

        return str(percent)

# Import the model to be used
model = YOLO("yolov8n.pt", "v8")

path = 'videos/Video3/Clip4'

method = ['NONE', 'BB_DIFF']


for i in range(len(method)):
    print(f"Running with tracking method: {method[i]}")
    run = Main(path, model, method[i], show=True)
    run.run_comparison()
    print(run.frame_status)
    print(f'Percentage Of Yolo Complete Detections: {run.perc_yolo()}%')
    print(f'Percentage Maintained Consistent Identity: {run.perc_maintined()}%')
    print("\n")
