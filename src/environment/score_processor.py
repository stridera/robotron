import numpy as np
import hashlib
import base64
from .image_hashes import NUMBERS_HASH


class ScoreProcessor():
    SCORE_BOX = np.array([479, 91])
    SCORE_SIZE = np.array([12, 14])
    SCORE_BUFFER = 2

    def __init__(self):
        ''' constructor '''

    def h6(self, w):
        h = hashlib.md5(w).digest()
        return base64.b64encode(h)[:6].lower().decode("utf-8")

    def numberCleanup(self, image):
        binary_output = np.zeros_like(image)
        binary_output[image >= 30] = 1
        return binary_output

    def getDigitFromImageBinary(self, score_bin):
        digest = self.h6(score_bin)

        if digest in NUMBERS_HASH:
            return NUMBERS_HASH[digest]

        # print(type(digest), type(list(NUMBERS_HASH.keys())[0]))
        # print("Missing Hash", digest, score_bin)
        return -1

    def getScore(self, gray):
        score = -1
        tl = np.copy(self.SCORE_BOX)
        br = tl + self.SCORE_SIZE
        score_img = gray[tl[1]:br[1], tl[0]:br[0]]
        score_bin = self.numberCleanup(score_img)

        place = 0
        while (np.count_nonzero(score_bin)):
            digit = self.getDigitFromImageBinary(score_bin)
            if digit == -1:
                return -1

            if place == 0:
                score = digit
            else:
                score += digit * (10 ** place)

            tl[0] = tl[0] - self.SCORE_SIZE[0] - self.SCORE_BUFFER
            br[0] = tl[0] + self.SCORE_SIZE[0]
            score_img = gray[tl[1]:br[1], tl[0]:br[0]]
            score_bin = self.numberCleanup(score_img)
            place += 1

        return score
