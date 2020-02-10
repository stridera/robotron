"""
Process and manage the environment
"""

import cv2

from .score_processor import ScoreProcessor
from .lives_processor import LivesProcessor

from .noise_filter import NoiseFilter
from .utils import crop


class Environment():
    """ Setup and manage the game environment """

    IMAGE_SIZE = (720, 1280)  # Expected image size
    GAMEBOX = [116, 309, 608, 974]  # Area to crop

    # Size to filter for noise.  Higher number means means we skip more noise, but creates a delay before we recognize
    # a change
    FILTERSIZE = 5
    # Number of inactive frames before we believe the game is over.  Should be enough to handle level transitions.
    MAX_INACTIVE = 0

    CIVILIAN_REWARD = 100
    DEATH_REWARD = -100

    def __init__(self):
        ''' constructor '''

        self.score_processor = ScoreProcessor()
        self.lives_processor = LivesProcessor()

        self.score = NoiseFilter(self.FILTERSIZE)
        self.lives = NoiseFilter(self.FILTERSIZE, 3)

        self.last_score = 0
        self.last_lives = 0

        self.frame = 0

        # Number of frames since we've had a valid score collection.  Used to determine end of game.
        self.inactive_frames = -1

    def reset(self):
        """ Resets the environment. """

        self.frame = 0
        self.inactive_frames = -1
        self.score.zero()
        self.lives.set_all(3)
        self.last_score = 0
        self.last_lives = 0

    def process(self, image):
        """
        Process a single frame

        Arguments:
            image {ndarray} -- Input image

        Returns:
            list(list, ndarray) -- Returns a list of data processed and an annotated image for debugging.
        """
        movement_reward = 0
        active = False
        game_over = False
        score_delta = 0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gamebox = crop(image, Environment.GAMEBOX)
        score = self.score_processor.get_score(gray)

        if score == -1:
            if self.inactive_frames != -1:
                self.inactive_frames += 1
                if self.inactive_frames > self.MAX_INACTIVE:
                    game_over = True
        else:
            self.inactive_frames = 0
            active = True
            self.frame += 1

            movement_reward += 1

            lives_after = self.lives.set(self.lives_processor.getLives(gray))
            if lives_after < self.last_lives:
                movement_reward -= self.DEATH_REWARD
            self.last_lives = lives_after

            score_after = self.score.set(score)
            score_delta = score_after - self.last_score
            self.last_score = score_after

            # Calculate Movement Reward

            while score_delta >= 1000:
                # Assume we collected a civilian - humans are worth an increasing point value.
                # The first human scores 1000 points, the second is worth 2000 points and so
                # on until the point value reaches 5000. The point value will remain at 5000
                # for all the remaining humans in the same wave. When the wave is completed or
                # you have been killed, the points awarded for saving another human will be
                # reset to 1000.
                #
                # For our purpose, we want to remove all of these from the score delta used for
                # the shooting reward and add a bump to the movement reward.
                movement_reward += self.CIVILIAN_REWARD
                while score_delta >= 1000:
                    score_delta -= 1000

            if score_delta > 0:
                score_delta = score_delta // 10
            else:
                score_delta = 0

        data = {
            'frame': self.frame,
            'score': self.score.get(),
            'lives': self.lives.get(),
            'movement_reward': movement_reward,
            'shooting_reward': score_delta,
            'active': active,
            'game_over': game_over,
            'inactive_frame_count': self.inactive_frames,
        }

        return gamebox, data
