from enum import Enum
from .noise_filter import NoiseFilter
import math


class SpriteTypes(Enum):
    UNKNOWN = 0
    PLAYER = 1
    CIVILIAN = 2
    GRUNT = 3
    HULK = 4
    SPHEROID = 5
    ENFORCER = 6
    BRAIN = 7
    TANK = 8
    QUARK = 9
    ELECTRODE = 10
    BULLET = 11

    def __int__(self):
        return self.value

    def __str__(self):
        reps = ['u', 'p', 'c', 'g', 'h', 's', 'e', 'b', 't', 'q', 'x', 'z']
        return reps[self.value]


class Sprite:

    def __init__(self, x, y, w, h, type=SpriteTypes.UNKNOWN):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.vx = 0
        self.vy = 0
        self.bearing = 0
        self.certainty = 0
        self.lastSeen = 0
        self.spriteClass = NoiseFilter(1, type)

    def getRepresentation(self):
        spriteClass = SpriteTypes(self.spriteClass.get())
        return str(spriteClass).upper() if self.certainty else str(spriteClass)

    def getCenterXY(self):
        return (int(self.x + (self.w / 2.0)), int(self.y + (self.h / 2.0)))

    def update(self, x, y, w, h, c=SpriteTypes.UNKNOWN, certainty=0):
        self.bearing = 90 - (180 / math.pi) * math.atan2(y - self.y, x - self.x)
        self.vx = x - self.x
        self.vy = y - self.y

        self.x = x
        self.y = y
        self.w = w
        self.h = h

        if c != 0:
            self.spriteClass.set(c)
        self.certainty = (self.certainty + certainty) / 2

    def isMatch(self, x, y, w, h):
        '''
        Lets only match closely shaped objects
        '''
        percentThreshold = 0.10
        wperc = abs(1 - (self.w / w))
        hperc = abs(1 - (self.h / h))
        return wperc < percentThreshold and hperc < percentThreshold
