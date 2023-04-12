class Frame:
    def __init__(self, index, play_lst, maint) -> None:
        self.index = index # Index of the frame in the video sqauence
        self.player_list = play_lst # List of players contained in this frame
        self.maintained = maint # Whether the correct identities have been maintained

    def num_players(self):
        num_players = len(self.player_list)
        return num_players

    def __str__(self):
        return f'Frame Index: {self.index} - {self.maintained}'
    
    