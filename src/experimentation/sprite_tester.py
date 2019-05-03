import cv2
import numpy as np
import os


class SpriteTester:

    def __init__(self, hasher, scorer):
        self.path = '/home/strider/Code/robotron/resources/images/'

        if not hasher or not scorer:
            raise ValueError("Hasher and Scorer required")
        if not callable(hasher) or not callable(scorer):
            raise ValueError("Hasher and Scorer need to be functions")

        self.hasher = hasher
        self.scorer = scorer

        self.classes = self.loadClasses()
        self.spriteClasses, self.spriteNames, self.sprites = self.loadSprites()

        self.hashes = self.hashSprites()

    def loadClasses(self):
        spriteClasses = open(self.path+'robotronclasses.txt', 'r')
        classes = []
        for line in spriteClasses:
            (name, _) = line.split()
            classes.append(name)

        return classes

    def loadSprites(self):
        spriteDefFile = open(self.path+'robotronsprites.txt', 'r')
        spriteClasses = open(self.path+'robotronclasses.txt', 'r')
        spriteSheet = cv2.imread(self.path+"robotronsprites.jpg")

        ssh, ssw, _ = spriteSheet.shape
        x = 0
        y = 0

        sprites = {}
        spriteClasses = {}
        rowheight = 0
        i = 0
        for line in spriteDefFile:
            i += 1
            (name, c, sid, w, h, _) = line.split()
            w = int(w) * 4
            h = int(h) * 2
            if x + w > ssw:
                x = 0
                y += rowheight + 10
                rowheight = 0
            sprite = spriteSheet[y:y+h, x:x+w]
            sprites[name] = sprite[:, :, ::-1]
            spriteClasses[name] = c
            x += w + 10
            if h > rowheight:
                rowheight = h

        images = []
        names = []
        for name in sprites.keys():
            sprite = sprites[name]
            spriteClass = spriteClasses[name]
            if spriteClass == '0':
                continue
            images.append(sprite)
            names.append(name)
        return spriteClasses, names, images

    def hashSprites(self):
        hashes = []
        for name, ref in zip(self.spriteNames, self.sprites):
            hash = self.hasher(ref)
            hashes.append(hash)
        return hashes

    def getHashes(self):
        return self.spriteNames, self.hashes

    def getClass(self, name):
        return self.spriteClasses[name]

    def testSprite(self, img):
        sprite = self.hasher(img)
        results = []
        for name, ref_hash in zip(self.spriteNames, self.hashes):
            classID = self.spriteClasses[name]
            score = self.scorer(sprite, ref_hash)
            results.append((name, classID, round(score, 3)))
        results = sorted(results, key=lambda score: score[2])
        return results

    def runTests(self):
        results = []
        typeResults = {}
        fullResults = []
        for dirClassName in os.listdir(self.path + 'test/'):
            typeResults[dirClassName] = []
            dirClass = self.classes.index(dirClassName)
            if not dirClass:
                print("Class {} not found!".format(type))
                continue
            subpath = "{}/test/{}/".format(self.path, dirClassName)
            for testImageName in os.listdir(subpath):
                img = cv2.imread(subpath + testImageName)
                guess = self.testSprite(img)
                fullResults.append((testImageName, dirClassName, guess))
                success = 1 if int(guess[0][1]) == int(dirClass) else 0
                results.append(success)
                typeResults[dirClassName].append(success)

        if len(results) == 0:
            return 0

        resultPercentage = 100 * (np.count_nonzero(results) / len(results))
        typeResultsPercentage = {}
        for k in typeResults.keys():
            if len(typeResults[k]) == 0:
                typeResultsPercentage[k]
            else:
                typeResultsPercentage[k] = 100 * (np.count_nonzero(typeResults[k]) / len(typeResults[k]))

        return (resultPercentage, typeResultsPercentage, fullResults)
