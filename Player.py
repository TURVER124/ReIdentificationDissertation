class Player:
    def __init__(self, id, bb, conf, colour) -> None:
        self.id = id
        self.bound_box = bb
        self.confidence = conf
        self.colour = colour