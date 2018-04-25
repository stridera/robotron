import cv2
import sys
# import numpy as np

from .ScoreProcessor import ScoreProcessor
from .LivesProcessor import LivesProcessor
from .NoiseFilter import NoiseFilter


class Environment():
    GAMEBOX = [114, 309, 608, 975]
    FILTERSIZE = 15

    def __init__(self, show=False):
        ''' constructor '''
        self.ScoreProcessor = ScoreProcessor()
        self.LivesProcessor = LivesProcessor()

        self.score = NoiseFilter(self.FILTERSIZE)
        self.lives = NoiseFilter(self.FILTERSIZE)

        self.frame = 0
        self.reward = 0
        self.showScreen = show

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

    def reset(self):
        self.frame = 0
        self.reward = 0
        self.score = NoiseFilter(self.FILTERSIZE)
        self.lives = NoiseFilter(self.FILTERSIZE)

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

    def process(self, image_only=False):
        active = False
        done = False

        ret, image = self.cap.read()

        if not ret:
            print("Nothing captured.")
            return (False, None, 0, False)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gamebox = self.getGamebox(gray)

        if (image_only):
            return gamebox

        score = self.ScoreProcessor.getScore(gray)

        if (not done and score != -1):
            lives = self.LivesProcessor.getLives(gray)

            if (self.frame > self.FILTERSIZE and lives < self.lives.get()):
                done = True

            active = True
            self.frame += 1
            self.score.set(score)
            self.lives.set(lives)

        # TODO: Make this return only reward from last process()
        reward = (self.frame / 10) + (self.score.get() / 10)
        reward_delta = reward - self.reward
        self.reward = reward

        if self.showScreen:
            msg = ("Frame: {}\nScore: {}\n"
                   "Lives: {}\nReward: {}\nGame Over: {}").format(
                self.frame, self.score.get(), self.lives.get(), reward, done)
            image = self.overlayText(image, msg, (5, 40), weight=4, size=1)
            # print(msg)
            cv2.imshow('frame', image)
            cv2.waitKey(1)
        else:
            msg = ("Frame: {}\tScore: {}\t"
                   "Lives: {}\tReward: {}\tGame Over: {}           \r").format(
                self.frame, self.score.get(), self.lives.get(), reward, done)
            sys.stdout.write(msg)
            sys.stdout.flush()

        return (active, gamebox, int(reward_delta), done)

    def close(self):
        self.cap.release()
        if self.showScreen:
            cv2.destroyAllWindows()
