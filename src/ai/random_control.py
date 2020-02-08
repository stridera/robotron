import random


class RandomControl:

    def __init__(self):
        pass

    @staticmethod
    def desc():
        return "Move and shoot completely randomly!"

    def reset(self):
        pass

    def play(self, _):
        move = random.randint(1, 8)
        shoot = random.randint(1, 8)
        return move, shoot, []
