import numpy as np
import cv2
from collections import defaultdict


class SpriteGrid():
    GAMEBOX = [115, 309, 608, 974]

    def __init__(self):

        self.seenHashes = defaultdict()
        (x, y, x1, y1) = self.GAMEBOX

    def imghash(self, img):
        img = cv2.resize(img, (20, 20))
        return imagehash.average_hash(Image.fromarray(img))

    def match(self, img):
        img = cv2.resize(img, (20, 20))
        sHash = imagehash.average_hash(Image.fromarray(img))

        bestScore = 999
        bestClass = 0
        for i, classHashes in enumerate(self.spriteHashMap):
            for spriteHash in classHashes:
                score = sHash - spriteHash
                if score < self.SCORE_THRESHOLD and score < bestScore:
                    bestScore = score
                    bestClass = i

        return bestClass, bestScore, sHash

    @staticmethod
    def crop(img, x, y, w, h):
        return img[y:y+h, x:x+w]

    @staticmethod
    def saveImage(imageType, classStr, image):
        sHash = imagehash.average_hash(Image.fromarray(image))
        filename = '/home/strider/Code/robotron/scratch/{}/{}/{}.jpg'.format(imageType, classStr, sHash)
        path = os.path.dirname(filename)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        if not os.path.isfile(filename):
            cv2.imwrite(filename, image)

    def update(self, image):
        tagged = image.copy()
        _, thresh = cv2.threshold(image, 16, 255, 0)
        blur = cv2.GaussianBlur(thresh, (3, 3), 0)
        contours, _ = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in enumerate(contours):
            rect = cv2.boundingRect(contour[1])
            (x, y, w, h) = rect
            if h > 10 and w > 10:
                hsh = self.imghash(self.crop(image, x, y, w, h))
                cv2.rectangle(tagged, (x,y), (x+w,y+h), (255, 255, 255), 2)

        return tagged
