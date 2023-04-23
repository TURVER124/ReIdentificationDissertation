import math, cv2
from sklearn.cluster import KMeans
from collections import Counter
import numpy as np

class Player:
    def __init__(self, bb, conf=0) -> None:
        self.id = -1
        self.bound_box = bb
        self.confidence = conf
        self.colour = []
        self.dist_diff = 100000
        self.speed = 0
        self.direction = None

    def get_bb_size(self):
        width = (self.bound_box[2]) - (self.bound_box[0])
        height = (self.bound_box[3]) - (self.bound_box[1])

        return (width, height)
    
    def bb_diff(self, sec_player):
        s_euclid = math.sqrt(((sec_player.bound_box[0] - self.bound_box[0]) * (sec_player.bound_box[0] - self.bound_box[0])) +
                            ((sec_player.bound_box[1] - self.bound_box[1]) * (sec_player.bound_box[1] - self.bound_box[1])))
        
        e_euclid = math.sqrt(((sec_player.bound_box[2] - self.bound_box[2]) * (sec_player.bound_box[2] - self.bound_box[2])) +
                            ((sec_player.bound_box[3] - self.bound_box[3]) * (sec_player.bound_box[3] - self.bound_box[3])))
        
        return (abs(s_euclid)), (abs(e_euclid))
    
    def get_dist_thrshold(self, preceding):
        avg_w = ((self.bound_box[2] - self.bound_box[0]) + (preceding.bound_box[2] - preceding.bound_box[0]) / 2)
        avg_h = ((self.bound_box[3] - self.bound_box[1]) + (preceding.bound_box[2] - preceding.bound_box[0]) / 2)
        thresh = (avg_w + avg_h) * 0.6
        return thresh
    
    # Helper function to view the colour pallet of the returned most common colour clusters
    # taken from: https://towardsdatascience.com/finding-most-common-colors-in-python-47ea0767a06a
    def palette(self, clusters):
        width = 300
        palette = np.zeros((50, width, 3), np.uint8)
        
        n_pixels = len(clusters.labels_)
        counter = Counter(clusters.labels_) # count how many pixels per cluster
        perc = {}
        for i in counter:
            perc[i] = np.round(counter[i]/n_pixels, 2)
        perc = dict(sorted(perc.items()))
        
        clust_cent = clusters.cluster_centers_

        step = 0
        
        for idx, centers in enumerate(clusters.cluster_centers_): 
            palette[:, step:int(step + perc[idx]*width+1), :] = centers
            step += int(perc[idx]*width+1)

        list_perc = []
        for i in perc:
            list_perc.append(perc[i])
            
        return list_perc, clust_cent, palette
    
    def get_colour(self, frame):
        roi = frame[int(self.bound_box[1]):int(self.bound_box[3]), int(self.bound_box[0]):int(self.bound_box[2])]

        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Green threshold and mask
        lower_green = np.array([30, 25, 25])
        upper_green = np.array([80, 255,255])
        mask = cv2.inRange(roi_hsv, lower_green, upper_green)

        inv_mask = cv2.bitwise_not(mask)
        result = cv2.bitwise_and(roi, roi, mask = inv_mask)

        clt = KMeans(n_clusters = 2, n_init='auto')
        perc, centers, palette = self.palette(clt.fit(result.reshape(-1, 3)))

        centers = np.array(centers)
        perc = np.array(perc)
        inds = (-perc).argsort()
        sorted_centers = centers[inds]
        sorted_perc = perc[inds]

        colour = [int(value) for value in sorted_centers[1]]        
        self.colour = colour

    def is_same_colour(self, comp_player):
        threshold = 10
        if comp_player.colour[0] in range (self.colour[0]-threshold, self.colour[0]+threshold):
            if comp_player.colour[1] in range (self.colour[1]-threshold, self.colour[1]+threshold):
                if comp_player.colour[2] in range (self.colour[2]-threshold, self.colour[2]+threshold):
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False
        
    
    def equal(self, sec_player):
        if self.id == sec_player.id:
            if (self.bound_box == sec_player.bound_box).all():
                return True
            else:
                return False
        else:
            return False
        

    def __str__(self):
        return f'Player ID: {self.id}'
    