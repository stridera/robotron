from enum import Enum
import cv2
import imagehash
from PIL import Image

from .image_hashes import ROBOTRON_SPRITE_HASHES


class Sprite:
    SCORE_THRESHOLD = 10

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

        IGNORE = 99

        def __int__(self):
            return self.value

        def __str__(self):
            return [

            ][self.value]

        def __repr__(self):
            reps = ['u', 'p', 'c', 'g', 'h', 's', 'e', 'b', 't', 'q', 'x', 'z']
            return reps[self.value]

    def __init__(self, x, y, w, h, type=SpriteTypes.UNKNOWN):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.t = type

    def getRepresentation(self):
        spriteClass = self.SpriteTypes(self.t)
        return repr(spriteClass).upper()# if self.certainty else str(spriteClass)

    def getCenterXY(self):
        return (int(self.x + (self.w / 2.0)), int(self.y + (self.h / 2.0)))

    def update(self, x, y, w, h, guess, cert = 0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.t = guess

    # @classmethod
    @staticmethod
    def match(image):
        image = cv2.resize(image, (20, 20))
        sHash = imagehash.average_hash(Image.fromarray(image))

        bestScore = 999
        bestClass = 0
        for i, classHashes in enumerate(ROBOTRON_SPRITE_HASHES):
            for spriteHash in classHashes:
                score = sHash - imagehash.hex_to_hash(spriteHash)
                if score < Sprite.SCORE_THRESHOLD and score < bestScore:
                    bestScore = score
                    bestClass = i

        return bestClass, bestScore, sHash
