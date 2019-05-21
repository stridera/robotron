import math
from enum import Enum


class AlgorithmicControl:
    EPSILON = 10.0

    class States(Enum):
        NONE = 0
        CIVILIAN = 1
        ENEMIES = 2

    def __init__(self):
        self.frame = 0
        self.center = (334, 247)

    @staticmethod
    def desc():
        return "Collect Civs, Shoot baddies, stay in center."

    def reset(self):
        self.frame = 0

    def isBetween(self, a, b, c):
        (ax, ay), (bx, by), (cx, cy) = a, b, c
        crossproduct = (cy - ay) * (bx - ax) - (cx - ax) * (by - ay)

        # compare versus epsilon for floating point values, or != 0 if using integers
        if abs(crossproduct) > self.EPSILON:
            return False

        dotproduct = (cx - ax) * (bx - ax) + (cy - ay)*(by - ay)
        if dotproduct < 0:
            return False

        squaredlengthba = (bx - ax)*(bx - ax) + (by - ay)*(by - ay)
        if dotproduct > squaredlengthba:
            return False

        return True

    def getDirection(self, x1, y1, x2, y2):
        deltaX = x2 - x1
        deltaY = y2 - y1

        degrees_temp = (math.atan2(deltaY, deltaX)/math.pi*180)+22
        if degrees_temp < 0:
            degrees_final = 360 + degrees_temp
        else:
            degrees_final = degrees_temp

        point = round((degrees_final + 20) / 45) + 2
        if point > 8:
            point -= 8

        while abs(degrees_final) > 45:
            degrees_final -= 45

        return point, degrees_final

    def getDistance(self, x1, y1, x2, y2):
        return math.sqrt(((x1-x2)**2)+((y1-y2)**2))

    def toDir(self, d):
        return d % 8 + 1

    def midPoint(self, x1, y1, x2, y2):
        return ((x1+x2)//2, (y1+y2)//2)

    def play(self, sprites):
        """
        - Check for Civilian.  If found, focus on closest and:
        - - Check to see if there is anything between the player and the civ if so:
        - - - Shoot at whatever it is and try to move around it
        """
        self.frame += 1
        closest_enemy = None
        closest_civ = None
        goal = self.center

        if len(sprites) == 0:
            return 0, 0, []

        _, _, px, py = sprites[0]

        close_enemies = []
        sorted_sprites = sorted(sprites)
        for d, r, x, y in sorted_sprites:
            if r == 'p':
                continue

            if r == 'c':
                if closest_civ is None:
                    closest_civ = (x, y)
                continue

            if d < 150:
                close_enemies.append((x, y))

            if r != 'h' and closest_enemy is None:
                closest_enemy = (x, y)

        # If we have a civilian, lets grab the closest.
        if closest_civ is not None:
            goal = closest_civ
        else:
            if closest_enemy is None:
                goal = self.center
            else:
                goal = self.midPoint(*self.center, *closest_enemy)

        if goal is None:
            return 0, self.toDir(self.frame % 8), []

        initial, _ = self.getDirection(px, py, *goal)

        # If anything is close, focus fire on it and move away
        goalDir = 0
        if len(close_enemies) > 0:
            # Lets find an escape path:
            quadrants = set()
            for x, y in close_enemies:
                monster_dir, ang = self.getDirection(px, py, x, y)
                quadrants.add(monster_dir)

            for i in range(8):
                check = self.toDir(initial + i)
                left = self.toDir(check - 1)
                right = self.toDir(check + 1)
                if len(quadrants.intersection([check, left, right])) == 0:
                    goalDir = check

            if goalDir is None:
                # Fine... any direction, just get out of here!
                for i in range(8):
                    check = self.toDir(initial + i)
                    if check not in quadrants:
                        goalDir = check
                        break

            # Now we have a direction, lets start shooting... one per enemy per frame
            move = goalDir
            enemy = list(close_enemies)[self.frame % len(close_enemies)]
            shoot, _ = self.getDirection(px, py, *enemy)
            arrows = []
        else:
            # TODO: Move around enemies instead of waiting for them to get close
            move = initial
            if closest_enemy is not None:
                shoot, _ = self.getDirection(px, py, *closest_enemy)
            else:
                shoot = self.toDir(self.frame % 8)

            arrows = []
            arrows.append(((px, py), goal, "{}/{:.2f}".format(move, self.getDistance(px, py, *goal))))
            arrows.append(((px, py), closest_enemy, shoot))

        if self.frame < 150:
            move = 0

        return move, shoot, arrows
