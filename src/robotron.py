#!/usr/bin/env python
# -*- coding: utf-8 -*-

import environment
import control
import cv2
import numpy as np
import time
from statistics import mean, median
import math


class Robotron:
    WINDOW_NAME = "Robotron"
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, device=2):
        self.controller = control.Controller()
        self.output = control.Output()
        self.env = environment.Environment()
        self.cap = cv2.VideoCapture(device)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

        self.running = False
        self.profData = {
            'all': [0]
        }
        self.maxProf = 100

        self.arrows = []

        cv2.namedWindow(self.WINDOW_NAME)

    def __del__(self):
        if self.output:
            self.output.close()

        if self.cap:
            self.cap.release()

        print("Killing all windows.")
        cv2.destroyAllWindows()

    def addProfData(self, name, delta):
        self.profData[name].append(delta)
        if len(self.profData[name]) > self.maxProf:
            self.profData[name] = self.profData[name][-self.maxProf:]

    def reset(self):
        """ Reset the game from an already running game. """
        # self.output.reset()
        self.env.reset()
        self.running = False

    def handleInput(self, key):
        if not self.running:
            self.output.none()

        if key == -1:
            return
        elif key == ord('`'):
            self.running = not self.running
        elif key == ord('Q'):
            exit(1)
        elif key == ord('e'):
            self.output.start()
        elif key == ord('q'):
            self.output.back()
        elif self.controller.attached() and key == ord('c'):
            self.controller.run()
        else:
            move_key = {
                ord('w'): 1,
                ord('d'): 3,
                ord('s'): 5,
                ord('a'): 7
            }

            shoot_key = {
                ord('i'): 1,
                ord('l'): 3,
                ord('k'): 5,
                ord('j'): 7
            }

            move = move_key.get(key, 0)
            shoot = shoot_key.get(key, 0)
            if move or shoot:
                self.output.move_and_shoot(move, shoot)

    def midpoint(self, p1, p2):
        (x1, y1) = p1
        (x2, y2) = p2
        return ((x1+x2)//2, (y1+y2)//2)

    def showScreen(self, image, spriteGrid=None, data=None):
        """ Show the screen and data """
        if spriteGrid is not None and data is not None:

            for i, (p1, p2, name) in enumerate(self.arrows):
                if p2 is not None:
                    color = (0, 255, 0) if i == 0 else (0, 0, 255)
                    cv2.arrowedLine(spriteGrid, p1, p2, color, 1)
                    t = "{}".format(name)
                    cv2.putText(spriteGrid, t, self.midpoint(p1, p2), self.FONT, 0.6, color, 1, cv2.LINE_AA)

            ih, _, _ = image.shape
            sgh, w, _ = spriteGrid.shape

            dataPanel = np.zeros((ih - sgh, w, 3), dtype=np.uint8)
            datastr = [
                "Score: {}".format(data['score']),
                "Lives: {}".format(data['lives']),
                "Active: {}".format(data['active']),
                "Playing: {}".format(self.running),
                "Times: Max: {:.4f}  Average: {:.4f}".format(max(self.profData['all']), mean(self.profData['all']))
            ]

            for i, line in enumerate(datastr):
                cv2.putText(dataPanel, line, (15, (30 * i) + 30), self.FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            sidePanel = np.vstack((spriteGrid, dataPanel))
            image = np.hstack((sidePanel, image))

        cv2.imshow(self.WINDOW_NAME, image)
        return cv2.waitKey(1)

    def getDirection(self, x1, y1, x2, y2):
        deltaX = x2 - x1
        deltaY = y2 - y1

        degrees_temp = (math.atan2(deltaY, deltaX)/math.pi*180)+22
        if degrees_temp < 0:
            degrees_final = 360 + degrees_temp
        else:
            degrees_final = degrees_temp

        point = round(degrees_final / 45) + 2
        if point > 8:
            point -= 8

        return point

    def play(self, data):
        active = data['active']
        lives = data['lives']

        if active:
            sprites = data['sprites']

            if sprites is None:
                return

            # Shoot at closest enemy, move toward closest civilian
            closest_enemy = None
            closest_civ = None

            _, _, px, py = sprites[0]

            sorted_sprites = sorted(sprites)
            for d, r, x, y in sorted_sprites:
                if d == 0:
                    continue
                if r == 'c' and closest_civ is None:
                    closest_civ = (x, y)
                elif closest_enemy is None:
                    closest_enemy = (x, y)

                if closest_civ is not None and closest_enemy is not None:
                    break

            if closest_enemy is None:
                shoot = 0
            else:
                shoot = self.getDirection(px, py, *closest_enemy)

            if closest_civ is None:
                move = shoot + 4
                if move > 8:
                    move -= 8
            else:
                move = self.getDirection(px, py, *closest_civ)

            self.arrows = []
            self.arrows.append(((px, py), closest_civ, move))
            self.arrows.append(((px, py), closest_enemy, shoot))

            print('blah', (px, py), closest_enemy, closest_civ, move, shoot)
            self.output.move_and_shoot(move, shoot)

    def run(self):
        try:
            while self.cap and self.cap.isOpened():
                start = time.time()
                status, image = self.cap.read()

                if not status:
                    print("No image received.")
                    continue

                data, spriteGrid = self.env.process(image)

                if self.running:
                    self.play(data)
                else:
                    self.reset()

                try:
                    window = cv2.getWindowProperty('Robotron', 0)
                    if window < 0:
                        exit(1)
                except Exception:
                    exit(1)

                resp = self.showScreen(image, spriteGrid, data)
                self.handleInput(resp)
                end = time.time()
                self.addProfData('all', end - start)

        except KeyboardInterrupt:
            print("Interrupt detected.  Exiting...")


if __name__ == '__main__':
    r = Robotron()#'/home/strider/Code/robotron/resources/video/robotron-1.mp4')
    r.run()
