import cv2
import numpy as np

import imagehash
from PIL import Image
from scipy.spatial import distance as dist

from .sprites import SpriteTypes, Sprite

IMAGE_SIZE = (720, 1280)
BOARD_SIZE = (493, 666)


class Tracker:
    '''
        Based on https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
        Plan:
            - Initialize with the player in the middle of the field
            - Mark all other objects on the field
            - Give each object a guess based on size/hash
            - Update will try to keep track of bearing and velocity
    '''
    HASHES = [
        # Unknown
        [],
        # Player
        [
            'ffc34242426666c3', '7c7e3c183c3c1838', '787e3c187c3e3c7c', '78fc783078783038', '78fc783078f8787c',
            '18ff3c3c7edb2424', '18ff3c3c7efc6c0c', '18ff3c3c7eff3630', '00ff3c187edb2424', '7eff3c187eff6c0c',
            '7eff3c187e3f3630'
        ],
        # Civilian
        [
            '383c3e3c3c7e1838', '383c1e1e7efc6cec', '383c1e3e7e3f6c6c', '1c3c7c3c3c7e181c', '1c3c78787efe3636',
            '1c3c783c7e3f3636', '3c7e7edbdafc2424', '3c7e7edbfe7c2430', '3c7e7edbfcfc240c', '3c7e7edb5b3f2424',
            '3c7e7edb3f3f2430', '3c7e7edb7b3e240c', '1818181818383818', '1818181c3c3e3e68', '1818181c3cfcf878',
            '3838383838383818', '30303870783e3e3c', '3038707878f8fc2c', '3030fcfcb4362e68', '3030fcfcb4362e20',
            '3030fcfcb6366a18', '0c0c3f3f2d6c7416', '0c0c3f3f6d6c5618', '0c0c3f3f2d6c7404', '3c7c3c3c3c3c1838',
            '3c3c3c3c7e6626e6', '3c3c3c3c7f6626e7', '3c3e3c3c3c3c181c', '3c3c3c3cfe6664e6', '3c3c3c3c7e4e64e6',
            '3c3c3c7eff3c2466', '3c3c3c7e7f3e2720', '3c3c3c7efe7ce404'
        ],
        # Grunt
        ['183c3cffbd183c66', '183c3cffbd383c0c', '183c3cffbd1c3c30'],
        # Hulk
        [
            '183c3c3c3c181800', '18183c7e7c3c6600', '18183c3efe3c6600', '1818ffffffbd2400', '1818ffffff7f0c0c',
            '1818fffffffe3030', '10387c7c7c303000', '10307c7c7e78c600', '10307c7efc78c400'
        ],
        # Sphereoid
        [
            '0000001818000000', '0000001818180000', '0000183c3c180000', '0000183c3c3c0000', '00003c66667e1800',
            '00187e6666663c00', '183c4242c3421818', '181800c3c3000018'
        ],
        # Enforcer
        [
            '00187e187eff3c3c', '000000183c181800', '0000183c3c3c1800', '0000181c1c1c1808', '00183c3c7e7e1818',
            '001c3e3e3e3e3c1c'
        ],
        # Brain
        [
            '3c7e7e7e7c301030', '3c7e7e7e7c307008', '3c7e7e7e7c307028', '387cfefc3c1c0808', '387cfefc3c1c1c04',
            '107effff7e183c18', '107effff7e183c10', '107effff7e183c08', '187effff7e183c10', '187effff7e183c08'
        ],
        # Tank
        ['183c18667e7e7e42', '183c187e7e7e7e42', '003c187a7e7e7e42', '003c187e7e7e7e42'],
        # Electrode
        [
            '005a3c3cff3c7e5a', '00183c3c7e3c3c18', '000018183c180000', '005a7e7ee77e7e5a', '00183c3c7e3c1818',
            '0000001c3c180000', '007e7e7e7e7e7e7e', '0000183c3c3c0000', '0003070f1f3f7fff', '000002060e1e3e3e',
            '000000040e1c1c00', '003c3c3c3c3c3c3c', '0018181818181818', '00183c7eff7e3c18', '00183c3c7e3c1800',
            '00ff8199915b82ff', '0000ffdbd1dfff7e', '0000007e7e3c0000', '007e7e66c3667e7e', '00183c3c663c3c00',
            '0000183c3c3c1800', '7ec1bda5b5857d03', '00003c24fcf4fc00', '00007e2e3c300000'
        ],
        # Quark
        [
            '0000001818180000', '0000183c3c3c0000', '00003c3c3c3c0000', '003c7e7e7e7e7e00', '007e667c5a247600',
            '005a2400812466db', '99426600812466db', '1800008181000018', '0000000000000000'
        ],
        # Bullet
        ['18181818ff181818', '0002663c183c6666', '0040663c183c6666'],
    ]

    SCORE_THRESHOLD = 10

    def __init__(self, imageSize, boardSize=None):
        self.maxMissing = 1
        self.nextID = 1
        self.sprites = {}
        self.imageSize = imageSize
        self.boardSize = boardSize
        self.spriteHashMap = []
        self.zzzz = 0

        self.reset()

        for sprite_classes in self.HASHES:
            hashes = []
            for hashStr in sprite_classes:
                hashes.append(imagehash.hex_to_hash(hashStr))
            self.spriteHashMap.append(hashes)

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

        return bestClass, bestScore

    def getSprites(self):
        return self.sprites

    def getLocations(self):
        locs = []
        for key in self.sprites:
            sprite = self.sprites[key]
            x = int((float(sprite.x) / float(self.imageSize[0])) * self.boardSize[0])
            y = int((float(sprite.y) / float(self.imageSize[1])) * self.boardSize[1])
            r = sprite.getRepresentation()
            locs.append((r, x, y))
        return locs

    def getGridImage(self, img=None):
        if img is None:
            rep = np.zeros(np.concatenate([self.imageSize, [3]]), np.uint8)
        else:
            rep = img.copy()

        for key in self.sprites:
            sprite = self.sprites[key]
            r = sprite.getRepresentation()
            x, y = sprite.getCenterXY()
            cv2.rectangle(rep, (x - 5, y - 10), (x + 5, y + 10), (0, 0, 255), cv2.FILLED)
            cv2.putText(rep, r, (x - 5, y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        return rep

    def reset(self):
        self.sprites = {}
        self.sprites[0] = Sprite(666 / 2, 493 / 2, 15, 25, SpriteTypes.PLAYER)
        self.nextID = 1

    def addSprite(self, x, y, w, h, c=SpriteTypes.UNKNOWN):
        if (c == SpriteTypes.PLAYER):
            print("Can not add a player entry... only update it.")
            return


#         print("Adding sprite", x, y, w, h, c)
        self.sprites[self.nextID] = Sprite(x, y, w, h, c)
        self.nextID += 1

    def delSprite(self, spriteID):
        s = self.sprites[spriteID]
        #         print("Deleting sprite", s.x, s.y, s.getRepresentation())
        del self.sprites[spriteID]

    def handleMissing(self, spriteID):
        if spriteID == 0:
            ''' Return, since the player can't go missing '''
            return
        self.sprites[spriteID].lastSeen += 1

        if self.sprites[spriteID].lastSeen > self.maxMissing:
            self.delSprite(spriteID)

    def midpoint(self, x1, y1, x2, y2):
        return (x1 + x2) / 2, (y1 + y2) / 2

    def crop(self, img, x, y, w, h):
        return img[y:y + h, x:x + w]

    def update(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 16, 255, 0)
        blur = cv2.GaussianBlur(thresh, (3, 3), 0)
        contours, _ = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ''' If we don't find anything, lets just increase the lastseen marker for stuff '''
        if len(contours) == 0:
            for spriteID in list(self.sprites.keys()):
                self.handleMissing(spriteID)
            return self.sprites, self.getGridImage()

        ''' Step through each contour and either add it as new, or update existing '''
        newSpriteRects = []
        for contour in enumerate(contours):
            rect = cv2.boundingRect(contour[1])
            (x, y, w, h) = rect
            print(np.count_nonzero(self.crop(thresh, *rect)))

            # Since enemies explode into horizontal/vertical bands, ignore small bands
            if h > 10 and w > 10 and np.count_nonzero(self.crop(thresh, *rect)) > 20:
                newSpriteRects.append(rect)

        '''
        If our current object list of sprites is empty just add everything.  Pretty much what happens on first run.
        Enemies come in first, so they'll be added and we'll pretend we know where the player is despite them not appearing for a bit
        '''
        if len(self.sprites) == 1:
            for rect in newSpriteRects:
                sprite_class, _ = self.match(self.crop(thresh, *rect))
                self.addSprite(*rect, sprite_class)
            return self.sprites, self.getGridImage()
        '''
        If we're here, we need to look at all existing objects and see if we can link them with new detections
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
            - If the current new sprite is too far away to be a likely candidate, break out of the loop and we'll handle the sprite disappearing afterward.
            - First check the size.  If the new sprite is within ... lets say 5% of the old sprite, and it hasn't been seen before, we'll say it's the same as the old one and mark it used.
            - If the size doesn't match, check to see if it's like 150% bigger.  If so, we'll mark it as moving and assume that it's just two sprites hitting each other.
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
                spriteClass = oldSprite.spriteClass
                spriteGuess, guessScore = self.match(self.crop(thresh, *newSpriteRects[newSpriteArg]))

                if oldSprite.isMatch(*newSpriteRects[newSpriteArg]):
                    if newSpriteArg in usedNewSprites:
                        ''' We've already seen this sprite.  '''
                        continue

                    if spriteClass == spriteGuess:
                        certainty = 1

                    doUpdate = True
                else:
                    ''' Sprite dims don't match... check to see if it's much larger and we'll assume it's part of a group of objects  '''
                    if (w > oldSprite.w * 1.5 or h > oldSprite.h * 1.5):
                        x, y = self.midpoint(oldSprite.x, oldSprite.y, x, y)
                        doUpdate = True

                if doUpdate:
                    usedOldSprites.add(oldSpriteArg)
                    usedNewSprites.add(newSpriteArg)

                    if spriteGuess == SpriteTypes.UNKNOWN:
                        c = self.crop(thresh, *newSpriteRects[newSpriteArg])
                        hsh = self.imghash(c)
                        cv2.imwrite('/home/strider/Code/robotron/resources/test/{}-{}.jpg'.format(hsh, guessScore), c)

                    if spriteClass == SpriteTypes.PLAYER:
                        if spriteID != 0:
                            spriteID = 0

                    self.sprites[spriteID].update(*newSpriteRects[newSpriteArg], spriteGuess, certainty)

        oldUnusedSprites = set(range(0, distances.shape[0])).difference(usedOldSprites)
        newUnusedSprites = set(range(0, distances.shape[1])).difference(usedNewSprites)
        ''' Step through any sprite not matched and mark it as missing '''
        for oldSprite in oldUnusedSprites:
            objectID = spriteIDs[oldSprite]
            self.handleMissing(objectID)

        for newSprite in newUnusedSprites:
            rect = newSpriteRects[newSprite]
            self.addSprite(*rect)

        # return the set of trackable objects
        return self.sprites, self.getGridImage()
