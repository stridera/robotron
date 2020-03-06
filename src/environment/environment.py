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

    def __init__(self, output_image_size):
        ''' constructor '''
        self.output_image_size = output_image_size

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
        self.lives.set_all(2)
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
        active = False
        score_delta = 0
        reward = 0.
        dead = False
        reset_required = False

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        gamebox = crop(thresh, Environment.GAMEBOX)
        score = self.score_processor.get_score(gray)

        if score == -1:
            if self.inactive_frames != -1:
                self.inactive_frames += 1
        else:
            self.inactive_frames = 0
            active = True
            self.frame += 1

            lives_after = self.lives.set(self.lives_processor.getLives(gray))
            if lives_after < self.last_lives:
                dead = True

            self.last_lives = lives_after

            if self.last_lives == 0:
                reset_required = True

            score_after = self.score.set(score)
            score_delta = score_after - self.last_score
            self.last_score = score_after

            # Calculate Movement Reward

            # while score_delta >= 1000:
            #     # Assume we collected a civilian - humans are worth an increasing point value.
            #     # The first human scores 1000 points, the second is worth 2000 points and so
            #     # on until the point value reaches 5000. The point value will remain at 5000
            #     # for all the remaining humans in the same wave. When the wave is completed or
            #     # you have been killed, the points awarded for saving another human will be
            #     # reset to 1000.
            #     #
            #     # For our purpose, we want to remove all of these from the score delta used for
            #     # the shooting reward and add a bump to the movement reward.
            #     reward = 1
            #     while score_delta >= 1000:
            #         score_delta -= 1000

            # A positive score (civilian grabbed, enemy shot) gives a 1
            if score_delta > 0:
                reward = 1.

            # If they died, no matter what else happened, we give a negative reward.
            if dead:
                reward = -1.

        data = {
            'frame': self.frame,
            'score': self.score.get(),
            'lives': self.lives.get(),
            'reward': reward,
            'active': active,
            'dead': dead,
            'reset_required': reset_required,
            'inactive_frame_count': self.inactive_frames,
        }

        # default size: 492, 665
        return cv2.resize(gamebox / 255., self.output_image_size, interpolation=cv2.INTER_LINEAR), data
