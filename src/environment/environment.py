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
    GAMEBOX = [116, 310, 608, 974]  # Area to crop

    # Size to filter for noise.  Higher number means means we skip more noise, but creates a delay before we recognize
    # a change
    FILTERSIZE = 3
    # Number of inactive frames before we believe the game is over.  Should be enough to handle level transitions.
    MAX_INACTIVE = 0

    CIVILIAN_REWARD = 50
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
        self.lives.set_all(2)
        self.last_score = 0
        self.last_lives = 0
        self.game_over = False

    def process(self, image):
        """
        Process a single frame

        Arguments:
            image {ndarray} -- Input image

        Returns:
            list(list, ndarray) -- Returns a list of data processed and an annotated image for debugging.
        """
        active = False

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gamebox = crop(gray, Environment.GAMEBOX)
        score = self.score_processor.get_score(gray)

        if score != -1:
            active = True
            self.lives.set(self.lives_processor.getLives(gray))

        return gamebox, active, score
