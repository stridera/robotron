import numpy as np
import cv2
from scipy.spatial import distance as dist
import imagehash
from PIL import Image

from .sprites import Sprite


class Tracker:
    '''
        Based on https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
        Plan:
            - Initialize with the player in the middle of the field
            - Mark all other objects on the field
            - Give each object a guess based on size/hash
            - Update will try to keep track of bearing and velocity
    '''

    SCORE_THRESHOLD = 10

    def __init__(self, imageSize, boardSize=None):
        self.maxMissing = 1
        self.nextID = 1
        self.sprites = {}
        self.imageSize = imageSize
        self.boardSize = boardSize
        self.spriteHashMap = []
        self.playerDistances = []

        self.reset()

        # for sprite_classes in self.HASHES:
        #     hashes = []
        #     for hashStr in sprite_classes:
        #         hashes.append(imagehash.hex_to_hash(hashStr))
        #     self.spriteHashMap.append(hashes)

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

    def getSprites(self):
        return self.sprites

    def getLocations(self):
        locs = []
        px, py = self.sprites[0].getCenterXY()
        for key in self.sprites:
            sprite = self.sprites[key]
            x, y = sprite.getCenterXY()
            r = sprite.getRepresentation()
            d = dist.euclidean((px, py), (x, y))
            locs.append((d, r, x, y))
        return locs

    def getGridImage(self, img=None):
        if img is None:
            rep = np.zeros(np.concatenate([self.boardSize, [3]]), np.uint8)
        else:
            rep = img.copy()

        for key in self.sprites:
            sprite = self.sprites[key]
            r = sprite.getRepresentation()
            x = int(sprite.x + (sprite.w // 2))
            y = int(sprite.y + (sprite.h // 2))
            # cv2.rectangle(rep, (x-5, y-10), (x+5, y+10), (0, 0, 255), cv2.FILLED)
            cv2.putText(rep, r, (x-5, y+3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return rep

    def reset(self):
        self.sprites = {}
        self.sprites[0] = Sprite(666 / 2, 493 / 2, 15, 25, 1)
        self.nextID = 1

    def addSprite(self, x, y, w, h, c=0):
        if (c == 1):
            print("Can not add a player entry... only update it.")
            return
        # print("Adding sprite", x, y, w, h, c)
        self.sprites[self.nextID] = Sprite(x, y, w, h, c)
        self.nextID += 1

    def delSprite(self, spriteID):
        # s = self.sprites[spriteID]
        # print("Deleting sprite", s.x, s.y, s.getRepresentation())
        del self.sprites[spriteID]

    def handleMissing(self, spriteID):
        if spriteID == 0:
            ''' Return, since the player can't go missing '''
            return
        self.sprites[spriteID].lastSeen += 1

        if self.sprites[spriteID].lastSeen > self.maxMissing:
            self.delSprite(spriteID)

    def midpoint(self, x1, y1, x2, y2):
        return (x1+x2)/2, (y1+y2)/2

    def crop(self, img, x, y, w, h):
        return img[y:y+h, x:x+w]

    def withinPercentage(self, newValue, oldValue, percentage=0.05):
        perc = 1 - abs(oldValue - newValue)
        return perc < percentage

    def update(self, image, _):
        _, thresh = cv2.threshold(image, 16, 255, 0)
        blur = cv2.GaussianBlur(thresh, (3, 3), 0)
        contours, _ = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        ''' If we don't find anything, lets just increase the lastseen marker for stuff '''
        if len(contours) == 0:
            for spriteID in list(self.sprites.keys()):
                self.handleMissing(spriteID)

            return self.getLocations(), self.getGridImage()

        ''' Step through each contour and either add it as new, or update existing '''
        newSpriteRects = []
        for contour in enumerate(contours):
            rect = cv2.boundingRect(contour[1])
            (x, y, w, h) = rect

            # Since enemies explode into horizontal/vertical bands, ignore small bands
            if h > 10 and w > 10:
                newSpriteRects.append(rect)

        '''
        If our current object list of sprites is empty just add everything.  Pretty much what happens on first run.
        Enemies come in first, so they'll be added and we'll pretend we know where the player is despite them not appearing for a bit
        '''
        if len(self.sprites) <= 1:
            for rect in newSpriteRects:
                sprite_class, _, _ = self.match(self.crop(thresh, *rect))
                self.addSprite(*rect, sprite_class)
            return self.getLocations(), self.getGridImage()

        '''
        If we're here, we need to look at all existing objects and see if we can link them with new detectBions
        New Plan:
         - Add everything if empty
         - Step through each existing sprite and see if there is a corrisponding sprite in the new
         - Step through remaining new sprites and add them
         - Step through remaining old sprites and delete them
        '''
        spriteIDs = list(self.sprites.keys())
        existingSpriteCenters = np.zeros((len(spriteIDs), 2), dtype="int")
        for i, sID in enumerate(spriteIDs):
            existingSpriteCenters[i] = self.sprites[sID].getCenterXY()

        newSpriteCenters = np.zeros((len(newSpriteRects), 2), dtype="int")
        for i, rect in enumerate(newSpriteRects):
            (x, y, w, h) = rect
            newSpriteCenters[i] = (x + (w / 2.0), y + (h / 2.0))

        distances = dist.cdist(existingSpriteCenters, newSpriteCenters)
        if distances.size == 0:
            return self.getLocations(), self.getGridImage()

        self.playerDistances = distances[0]

        # sort the distances
        oldSpriteArgs = distances.min(axis=1).argsort()
        newSortedSpriteArgs = distances.argsort(axis=1)

        usedOldSprites = set()
        usedNewSprites = set()

        '''
        Step through all the sprites from the last frame and try to find a matching sprite on the new frame
        '''
        for oldSpriteArg in oldSpriteArgs:
            if oldSpriteArg in usedOldSprites:
                print("It actually happened.  Twas seen again.")
                continue

            spriteID = spriteIDs[oldSpriteArg]
            oldSprite = self.sprites[spriteID]

            """
            Now we should go through each new sprite and compare them:
            - If the current new sprite is too far away to be a likely candidate,
                break out of the loop and we'll handle the sprite disappearing afterward.
            - First check the size.  If the new sprite is within ... lets say 5% of the
                old sprite, and it hasn't been seen before, we'll say it's the same as the
                old one and mark it used.
            - If the size doesn't match, check to see if it's like 150% bigger.  If so,
                we'll mark it as moving and assume that it's just two sprites hitting each other.
            - If the old sprite type is unknown, lets try to match it and use it for the update.
            """
            for newSpriteArg in newSortedSpriteArgs[oldSpriteArg]:
                sdist = distances[oldSpriteArg][newSpriteArg]

                if sdist > 50:
                    ''' The closest sprite is too far away to be a likely match. '''
                    break

                doUpdate = False
                certainty = 0
                (x, y, w, h) = newSpriteRects[newSpriteArg]
                spriteClass = oldSprite.getRepresentation()
                spriteGuess, guessScore, sHash = self.match(self.crop(thresh, *newSpriteRects[newSpriteArg]))

                if True:  # oldSprite.isMatch(*newSpriteRects[newSpriteArg]):
                    if newSpriteArg in usedNewSprites:
                        ''' We've already seen this sprite.  '''
                        continue

                    if spriteClass == spriteGuess:
                        certainty = 1

                    doUpdate = True
                else:
                    ''' Sprite dims don't match... check to see if it's much larger and '''
                    ''' we'll assume it's part of a group of objects  '''
                    if (w > oldSprite.w * 1.5 or h > oldSprite.h * 1.5):
                        x, y = self.midpoint(oldSprite.x, oldSprite.y, x, y)
                        doUpdate = True

                if doUpdate:
                    usedOldSprites.add(oldSpriteArg)
                    usedNewSprites.add(newSpriteArg)

                    if spriteGuess == 0:
                        c = self.crop(image, *newSpriteRects[newSpriteArg])
                        hsh = self.imghash(c)
                        cv2.imwrite('/home/strider/Code/robotron/resources/test/{}-{}.jpg'.format(hsh, guessScore), c)

                    if spriteID == 0 and spriteClass != 1:
                        certainty = 0
                        spriteClass = 1

                    if spriteClass == 1:
                        if spriteID != 0:
                            spriteID = 0

                    self.sprites[spriteID].update(*newSpriteRects[newSpriteArg], spriteGuess, certainty)

        oldUnusedSprites = set(range(0, distances.shape[0])).difference(usedOldSprites)
        newUnusedSprites = set(range(0, distances.shape[1])).difference(usedNewSprites)

        ''' Step through any sprite not matched and mark it as missing '''
        for oldSprite in oldUnusedSprites:
            objectID = spriteIDs[oldSprite]
            # self.handleMissing(objectID)

        for newSprite in newUnusedSprites:
            rect = newSpriteRects[newSprite]
            self.addSprite(*rect)

        # return the set of trackable objects
        return self.getLocations(), self.getGridImage()
