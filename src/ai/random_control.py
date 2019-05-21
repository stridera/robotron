import random


class RandomControl:

    def __init__(self):
        pass

    def play(self, _):
        move = random.randint(1, 8)
        shoot = random.randint(1, 8)
        return move, shoot, []
