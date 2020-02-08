import numpy as np
import hashlib
import base64
from .image_hashes import NUMBERS_HASH


class LevelProcessor():
    LEVEL_BOX = np.array([586, 615])
    LEVEL_SIZE = np.array([7, 12])
    LEVEL_BUFFER = 2

    def __init__(self):
        ''' constructor '''

    def h6(self, w):
        h = hashlib.md5(w).digest()
        return base64.b64encode(h)[:6].lower().decode("utf-8")

    def numberCleanup(self, image):
        binary_output = np.zeros_like(image)
        binary_output[image >= 30] = 1
        return binary_output

    def getDigitFromImageBinary(self, LEVEL_bin):
        digest = self.h6(LEVEL_bin)

        if digest in NUMBERS_HASH:
            return NUMBERS_HASH[digest]

        # print(type(digest), type(list(NUMBERS_HASH.keys())[0]))
        # print("Missing Hash", digest, LEVEL_bin)
        return -1

    def getLevel(self, gray):
        level = -1
        tl = np.copy(self.LEVEL_BOX)
        br = tl + self.LEVEL_SIZE
        level_img = gray[tl[1]:br[1], tl[0]:br[0]]
        level_bin = self.numberCleanup(level_img)

        place = 0
        while (np.count_nonzero(level_bin)):
            digit = self.getDigitFromImageBinary(level_bin)
            if digit == -1:
                return -1

            if place == 0:
                level = digit
            else:
                level += digit * (10 ** place)

            tl[0] = tl[0] - self.LEVEL_SIZE[0] - self.LEVEL_BUFFER
            br[0] = tl[0] + self.LEVEL_SIZE[0]
            level_img = gray[tl[1]:br[1], tl[0]:br[0]]
            level_bin = self.numberCleanup(level_img)
            place += 1

        return level
