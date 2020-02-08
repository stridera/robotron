# -*- coding: utf-8 -*-
"""
Process the grid
"""

import os
import time
import random
from collections import defaultdict

import cv2
import numpy as np

from .sprites import Sprite


class SpriteGrid():
    """ Class Docstring

    """

    def __init__(self, imageSize, boardSize=None):
        self.imageSize = imageSize
        self.boardSize = boardSize
        self.seenHashes = defaultdict()
        self.sprites = []

    def reset(self):
        """Reset the class
        """
        pass

    def getGridImage(self, img=None):
        if img is None:
            rep = np.zeros(np.concatenate([self.boardSize, [3]]), np.uint8)
        else:
            rep = img.copy()

        for sprite in self.sprites:
            r = sprite.getRepresentation()
            x = int(sprite.x + (sprite.w // 2))
            y = int(sprite.y + (sprite.h // 2))
            # cv2.rectangle(rep, (x-5, y-10), (x+5, y+10), (0, 0, 255), cv2.FILLED)
            cv2.putText(rep, r, (x-5, y+3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return rep

    def getLocations(self):
        locs = []
        # px, py = self.sprites[0].getCenterXY()
        for sprite in self.sprites:
            x, y = sprite.getCenterXY()
            r = sprite.getRepresentation()
            d = 0#dist.euclidean((px, py), (x, y))
            locs.append((d, r, x, y))
        return locs

    @staticmethod
    def crop(img, x, y, w, h):
        return img[y:y+h, x:x+w]

    @staticmethod
    def saveImage(image, imageType, name):
        filename = '/home/strider/Code/robotron/scratch/{}/{}.jpg'.format(imageType, name)
        path = os.path.dirname(filename)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        if not os.path.isfile(filename):
            cv2.imwrite(filename, image)

    def update(self, image, color):
        # Lets save 1% of all screens for tagging and testing
        if random.random() < 0.01:
            t = time.time()
            self.saveImage(color, "screens", t)
            print("Saving Screen. ", t)
        
        tagged = image.copy()
        self.sprites = []
        _, thresh = cv2.threshold(image, 16, 255, 0)
        blur = cv2.GaussianBlur(thresh, (3, 3), 0)
        contours, _ = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in enumerate(contours):
            rect = cv2.boundingRect(contour[1])
            (x, y, w, h) = rect
            if h > 10 and w > 10 and h < 40 and w < 40:
                cv2.rectangle(tagged, (x, y), (x+w, y+h), (255, 255, 255), 2)
                sprite = self.crop(thresh, x, y, w, h)
                # hsh = self.imghash(sprite)
                guess, score, hsh = Sprite.match(sprite)
                self.sprites.append(Sprite(x, y, w, h, guess))

                if (hsh not in self.seenHashes):
                    self.seenHashes[hsh] = 1
                    self.saveImage(sprite, 'gray', hsh)
                    sprite = self.crop(color, x+309, y+115, w, h)
                    self.saveImage(sprite, 'color', hsh)

        return self.getLocations(), self.getGridImage()
