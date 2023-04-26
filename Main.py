from ultralytics import YOLO
from Run import Run
from Manual import Manuel
import numpy as np
from colorama import Fore

class Main:
    def __init__(self, fp, mod, ml, meth='NONE', show=False) -> None:
        self.file_path = fp
        self.method = meth
        self.model = mod
        self.model_letter = ml
        self.frame_status = []
        self.show = show


    def run_comparison(self):

        new_run = Run(self.model, self.file_path, self.method, self.show)
        ground_truth = Manuel(self.model, self.file_path, self.model_letter)

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
        yolo_misses = 0
        for res in self.frame_status:
            if res == 'T':
                num_correct += 1
            if res == 'X':
                yolo_misses += 1

        percent = round((num_correct/len(self.frame_status)*100), 2)
        percent_of_yolo = round((num_correct/(len(self.frame_status)-yolo_misses)*100), 2)

        return str(percent), str(percent_of_yolo)
    
    def perc_yolo(self):
        num_detect = 0
        for res in self.frame_status:
            if res != 'X':
                num_detect += 1

        percent = round((num_detect/len(self.frame_status)*100), 2)

        return str(percent)

# Import the model to be used
model = YOLO("yolov8x.pt", "v8")
model_letter = 'x'

# method = ['NONE', 'BB_DIFF', 'COLOUR']
method = ['NONE']

total_perc = [0.0, 0.0, 0.0]

for vid in range(6):
    for clip in range(3):
        # path = f'videos/Video{vid+1}/Clip{clip+1}'
        path = 'videos/Video7/Clip1'

        # Keep percentage for each video
        video_perc_main_yolo = []

        print(f'Video {vid+1} - Clip {clip+1}')
        save_file = open(f'{path}_results_{model_letter}.txt', 'w')
        for i in range(len(method)):
            run = Main(path, model, model_letter, method[i], show=True)
            run.run_comparison()
            perc_main, perc_main_yolo = run.perc_maintined()
            perc_yolo = run.perc_yolo()

            print(f'{method[i]}: \n{run.frame_status}')

            video_perc_main_yolo.append(perc_main_yolo)

            total_perc[i] += float(perc_main_yolo)

            save_file.write(f"Running with tracking method: {method[i]}\n")
            save_file.write(f'Each frame status: \n--\n {run.frame_status}\n--\n')
            save_file.write(f'Percentage Of Yolo Complete Detections: {perc_yolo}%\n')
            save_file.write(f'Percentage Maintained Consistent Identity: {run.perc_maintined()}%\n')
            save_file.write(f'Percentage Of Yolo Detections Where ID Maintained: {perc_main_yolo}%\n\n\n')

        if float(run.perc_yolo()) < 80.0:
                print(Fore.RED + "**********NOT USEFUL VIDEO**********")
        print(Fore.GREEN + f'Percentage of frames with complete detections: {perc_yolo}%')
        print(f'Method:                                        {method}')
        print(f'Percentage of frames with maintained IDs:      {video_perc_main_yolo}')
        print(Fore.WHITE + "\n")
        save_file.close()
    print("\n")


print("\n\n")
total_perc = np.array(total_perc) / float((vid+1)*(clip+1))
print(f'Method:                          {method}')
print(f'Total percentage over all tests: {total_perc}')




# path = 'videos/Video7/Clip3'

# new_man = Manuel(model, path, model_letter)

# new_man.main()
# new_man.export()