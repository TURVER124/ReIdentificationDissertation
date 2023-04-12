class Player:
    def __init__(self, bb, conf) -> None:
        self.id = -1
        self.bound_box = bb
        self.confidence = conf
        self.colour = (0,0,0)

    def __str__(self):
        return f'Player ID: {self.id}'