import cv2
import sys
import time
import numpy as np

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
        self.maxLives = 0
        self.rewards = np.array([0, 0])

        self.frame = 0
        self.showScreen = show

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

    def reset(self):
        self.rewards = np.array([0, 0])
        self.frame = 0
        self.score.zero()
        self.lives.zero()

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
        # Lets slow it down so it's not doing 60fps
        time.sleep(0.1)

        active = False
        done = False

        ret, image = self.cap.read()

        if not ret:
            print("Nothing captured.")
            return (False, None, 0, 0, False)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gamebox = self.getGamebox(gray)

        if (image_only):
            return gamebox

        score = self.ScoreProcessor.getScore(gray)
        score_delta = 0
        if (score != -1):
            self.lives.set(self.LivesProcessor.getLives(gray))
            if (self.frame > self.FILTERSIZE and self.lives.get() < self.maxLives):
                done = True

            if self.lives.get() > self.maxLives:
                self.maxLives = self.lives.get()

            active = True
            self.frame += 1
            score_before = self.score.get()
            self.score.set(score)
            score_after = self.score.get()
            score_delta = score_after - score_before

        # Calculate Movement Reward
        movement_reward = 1
        while score_delta >= 1000:
            # Assume we collected a civilian - humans are worth an increasing point value.
            # The first human scores 1000 points, the second is worth 2000 points and so
            # on until the point value reaches 5000. The point value will remain at 5000
            # for all the remaining humans in the same wave. When the wave is completed or
            # you have been killed, the points awarded for saving another human will be
            # reset to 1000. For our purpose, we'll just give it a boost of 10.
            movement_reward += 100
            while score_delta >= 1000:
                score_delta -= 1000

        score_delta = int(score_delta / 10)
        self.rewards += [movement_reward, score_delta]

        if self.showScreen:
            msg = ("Frame: {}\nScore: {}\nLives: {}\n"
                   "Movement Reward: {}\nShooting Reward: {}\nGame Over: {}").format(
                self.frame, self.score.get(), self.lives.get(),
                self.rewards[0], self.rewards[1], done)
            image = self.overlayText(image, msg, (5, 40), weight=4, size=1)
            # print(msg)
            cv2.imshow('frame', image)
            cv2.waitKey(1)
        else:
            msg = ("Frame: {}    Score: {}    Active: {}    Lives: {}    "
                   "Movement Reward: {}    Shooting Reward: {}    Game Over: {}             \r").format(
                self.frame, self.score.get(), active, self.lives.get(),
                self.rewards[0], self.rewards[1], done)
            sys.stdout.write(msg)
            sys.stdout.flush()

        return (active, gamebox, movement_reward, score_delta, done)

    def close(self):
        self.cap.release()
        if self.showScreen:
            cv2.destroyAllWindows()
