import cv2
# import numpy as np

from .ScoreProcessor import ScoreProcessor
from .LivesProcessor import LivesProcessor


class Environment():
    GAMEBOX = [114, 309, 608, 975]

    def __init__(self, show=True):
        ''' constructor '''
        self.ScoreProcessor = ScoreProcessor()
        self.LivesProcessor = LivesProcessor()

        self.frame = 0
        self.score = 0
        self.lives = 0
        self.showScreen = show

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

    def reset(self):
        self.frame = 0
        self.score = 0
        self.lives = 0

    def hideScreen(self):
        self.showScreen = False

    def showScreen(self):
        self.showScreen = True

    def getGamebox(self, image):
        (x, y, x1, y1) = self.GAMEBOX
        return image[x:x1, y:y1]

    def overlayText(
        self,
        image,
        text,
        location,
        size=3,
        weight=8,
        color=(255, 255, 255)
    ):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (x, y0) = location
        dy = 40
        for i, line in enumerate(text.split('\n')):
            y = y0 + i * dy
            cv2.putText(image, line, (x, y), font, size, color, weight)
        return image

    def process(self):
        active = False
        done = False

        ret, image = self.cap.read()

        if not ret:
            print("Nothing captured.")
            return (False, None, 0, False)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        score = self.ScoreProcessor.getScore(gray)

        if (not done and score != -1):
            lives = self.LivesProcessor.getLives(gray)

            if (lives < self.lives):
                done = True

            active = True
            self.frame += 1
            self.score = score
            self.lives = lives

        gamebox = self.getGamebox(gray)

        reward = (self.frame / 10) + (self.score / 10)

        if self.showScreen:
            msg = ("Frame: {}\nScore: {}\n"
                   "Lives: {}\nReward: {}\nGame Over: {}").format(
                self.frame, score, self.lives, reward, done)
            image = self.overlayText(image, msg, (5, 40), weight=4, size=1)
            # print(msg)
            cv2.imshow('frame', image)
            cv2.waitKey(1)

        return (active, gamebox, int(reward), done)

    def close(self):
        self.cap.release()
        if self.showScreen:
            cv2.destroyAllWindows()
