import cv2
import sys
import time
import numpy as np

from .tracker import Tracker
from .score_processor import ScoreProcessor
from .lives_processor import LivesProcessor
from .noise_filter import NoiseFilter


class Environment():
    GAMEBOX = [114, 309, 608, 975]
    IMAGE_SIZE = (720, 1280)
    BOARD_SIZE = (493, 666)
    FILTERSIZE = 7

    def __init__(self):
        ''' constructor '''
        self.tracker = Tracker(self.IMAGE_SIZE, self.BOARD_SIZE)
        self.scoreProcessor = ScoreProcessor()
        self.livesProcessor = LivesProcessor()

        self.score = NoiseFilter(self.FILTERSIZE)
        self.lives = NoiseFilter(self.FILTERSIZE)
        self.maxLives = 0
        self.rewards = np.array([0, 0])

        self.frame = 0

    def reset(self):
        self.tracker.reset()
        self.rewards = np.array([0, 0])
        self.frame = 0
        self.score.zero()
        self.lives.zero()

    def getGamebox(self, image):
        (x, y, x1, y1) = self.GAMEBOX
        return image[x:x1, y:y1]

    def overlayText(self, image, text, location, size=3, weight=8, color=(255, 255, 255)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        (x, y0) = location
        dy = 40
        for i, line in enumerate(text.split('\n')):
            y = y0 + i * dy
            cv2.putText(image, line, (x, y), font, size, color, weight)
        return image

    def process(self, image):
        active = False
        done = False

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gamebox = self.getGamebox(gray)
        score = self.scoreProcessor.getScore(gray)
        score_delta = 0
        sprites = []
        sprite_map_image = None

        if (score != -1):
            self.lives.set(self.livesProcessor.getLives(gray))
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

            sprites, sprite_map_image = self.tracker.update(gamebox)

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

        data = {
            'frame': self.frame,
            'score': self.score.get(),
            'lives': self.lives.get(),
            # 'level': self.level,
            'movement_reward': self.rewards[0],
            'shooting_reward': self.rewards[1],
            'active': active,
            'game_over': done,
            'sprites': sprites,
        }

        # msg = ("Frame: {}    Score: {}    Active: {}    Lives: {}    "
        #        "Movement Reward: {}    Shooting Reward: {}    Game Over: {}             \r").format(
        #            self.frame, self.score.get(), active, self.lives.get(), self.rewards[0], self.rewards[1], done)
        # sys.stdout.write(msg)
        # sys.stdout.flush()

        return data, sprite_map_image
