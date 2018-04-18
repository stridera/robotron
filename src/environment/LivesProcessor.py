import numpy as np

class LivesProcessor():
    LIFE_BOX = np.array([501, 91])
    LIFE_SIZE = np.array([14, 14])
    LIFE_BUFFER = 2

    def __init__(self):
        ''' constructor '''

    def imgCleanup(self, image):
        binary_output = np.zeros_like(image)
        binary_output[image >= 30] = 1
        return binary_output

    def getLives(self, gray):
        lives = 0

        tl = np.copy(self.LIFE_BOX)
        br = tl + self.LIFE_SIZE
        lives_img = gray[tl[1]:br[1], tl[0]:br[0]]
        lives_bin = self.imgCleanup(lives_img)
        while (np.count_nonzero(lives_bin)):
            lives += 1
            tl[0] = tl[0] + self.LIFE_SIZE[0] + self.LIFE_BUFFER
            br[0] = tl[0] + self.LIFE_SIZE[0]
            lives_img = gray[tl[1]:br[1], tl[0]:br[0]]
            lives_bin = self.imgCleanup(lives_img)
        return lives