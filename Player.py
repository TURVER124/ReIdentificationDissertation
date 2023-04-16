import math

class Player:
    def __init__(self, bb, conf) -> None:
        self.id = -1
        self.bound_box = bb
        self.confidence = conf
        self.colour = (0,0,0)
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

    def __str__(self):
        return f'Player ID: {self.id}'
    