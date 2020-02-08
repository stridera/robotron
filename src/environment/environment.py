"""
Process and manage the environment
"""

import cv2
import numpy as np

# from .tracker import Tracker
from .sprite_grid import SpriteGrid
from .score_processor import ScoreProcessor
from .lives_processor import LivesProcessor
from .level_processor import LevelProcessor
from .noise_filter import NoiseFilter
from .utils import crop

class Environment():
    """
    Setup and manage the game environment

    """
    IMAGE_SIZE = (720, 1280) # Expected image size
    GAMEBOX = [116, 309, 608, 974] # Area to crop

    # Size to filter for noise.  Higher number means means we skip more noise, but creates a delay before we accept data
    FILTERSIZE = 7

    def __init__(self):
        ''' constructor '''
        left, top, right, bottom = self.GAMEBOX
        self.tracker = SpriteGrid(self.IMAGE_SIZE, (bottom - top, right - left))
        # self.tracker = Tracker(self.IMAGE_SIZE, (bottom - top, right - left))

        self.score_processor = ScoreProcessor()
        self.lives_processor = LivesProcessor()
        self.level_processor = LevelProcessor()

        self.score = NoiseFilter(self.FILTERSIZE)
        self.lives = NoiseFilter(self.FILTERSIZE)

        self.max_lives = 0 # Number of lines to allow the player to get down to before resetting.
        self.rewards = np.array([0, 0])

        self.player_location = ((bottom - top + 1) // 2, (right - left + 1) // 2)

        self.frame = 0

    def reset(self):
        """
        Resets the environment.
        """

        self.tracker.reset()
        self.rewards = np.array([0, 0])
        self.frame = 0
        self.score.zero()
        self.lives.zero()

        left, top, right, bottom = self.GAMEBOX
        self.player_location = ((bottom - top + 1) // 2, (right - left + 1) // 2)


    def add_character_movement_hint(self, direction):
        """
        Add a hint to where we think the character moved to.

        Arguments:
            direction {int} -- Direction from 0-8.
        """
        px = 1
        py = 1
        self.player_location = np.add(self.player_location, (
            # x   y
            (0, 0),  # No movement
            (0, -py),  # Up
            (px, -py),  # Up/Right
            (px, py),  # Right
            (px, py),  # Down Right
            (0, py),  # Down
            (-px, py),  # Down Left
            (-px, 0),  # Left
            (-px, -py),  # Left Up
        )[direction])

    def process(self, image):
        """
        Process a single frame

        Arguments:
            image {ndarray} -- Input image

        Returns:
            list(list, ndarray) -- Returns a list of data processed and an annotated image for debugging.
        """
        active = False
        game_over = False
        score_delta = 0
        sprites = []
        sprite_map_image = None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gamebox = crop(image, Environment.GAMEBOX)
        score = self.score_processor.getScore(gray)
        level = self.level_processor.getLevel(gray)
        print(level)

        if score != -1:
            self.lives.set(self.lives_processor.getLives(gray))
            if (self.frame > self.FILTERSIZE and self.lives.get() < self.max_lives):
                game_over = True

            if self.lives.get() > self.max_lives:
                self.max_lives = self.lives.get()

            active = True
            self.frame += 1
            score_before = self.score.get()
            self.score.set(score)
            score_after = self.score.get()
            score_delta = score_after - score_before

            # sprites, sprite_map_image = self.tracker.update(gamebox, image)
            sprite_map_image = gamebox.copy()
            x, y = self.player_location
            cv2.rectangle(sprite_map_image, (x-5, y-5), (x+5, y+5), (0, 0, 255), 1)

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

        score_delta = score_delta // 10
        self.rewards += [movement_reward, score_delta]

        data = {
            'frame': self.frame,
            'score': self.score.get(),
            'lives': self.lives.get(),
            # 'level': self.level,
            'player_location': self.player_location,
            'movement_reward': self.rewards[0],
            'shooting_reward': self.rewards[1],
            'active': active,
            'game_over': game_over,
            'sprites': sprites,
        }

        # msg = ("Frame: {}    Score: {}    Active: {}    Lives: {}    "
        #        "Movement Reward: {}    Shooting Reward: {}    Game Over: {}             \r").format(
        #            self.frame, self.score.get(), active, self.lives.get(), self.rewards[0], self.rewards[1], done)
        # sys.stdout.write(msg)
        # sys.stdout.flush()

        return data, sprite_map_image
