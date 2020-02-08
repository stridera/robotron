#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
from statistics import mean

import ai
import capture
import control
import environment


class Robotron:
    WINDOW_NAME = "Robotron"
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, capDevice=2, arduinoPort='/dev/ttyACM0'):
        self.ai = ai.AlgorithmicControl()
        # self.ai = ai.RandomControl()

        self.cap = capture.VideoCapture(capDevice)
        # self.cap = capture.VideoCapture('/home/strider/Code/robotron/resources/video/robotron-1.mp4')

        self.controller = control.Controller()
        self.output = control.Output(arduinoPort)
        self.env = environment.Environment()

        self.lives = 0
        self.running = False
        self.profData = {
            'all': [0]
        }
        self.maxProf = 100
        self.lastActive = 0
        self.using_controller = False

        self.arrows = []

        cv2.namedWindow(self.WINDOW_NAME)

    def __del__(self):
        if self.output:
            self.output.close()

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
        self.ai.reset()

    def handleInput(self, key):
        if not self.running and not self.using_controller:
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
        elif self.controller is not None and self.controller.attached() and key == ord('c'):
            self.using_controller = not self.using_controller
            print("Using controller: ",
                  "True" if self.using_controller else "False")
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
                    cv2.putText(spriteGrid, t, self.midpoint(p1, p2),
                                self.FONT, 0.6, color, 1, cv2.LINE_AA)

            ih, _, _ = image.shape
            sgh, w, _ = spriteGrid.shape

            dataPanel = np.zeros((ih - sgh, w, 3), dtype=np.uint8)
            if self.using_controller:
                playing = "Using Controller"
            else:
                playing = self.running

            datastr = [
                "Method: {}".format(self.ai.desc()),
                "Score: {}".format(data['score']),
                "Lives: {}".format(data['lives']),
                "Active: {}".format(data['active']),
                "Playing: {}".format(playing),
                "Times: Max: {:.4f}  Average: {:.4f}".format(
                    max(self.profData['all']), mean(self.profData['all']))
            ]

            for i, line in enumerate(datastr):
                cv2.putText(dataPanel, line, (15, (30 * i) + 30),
                            self.FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            side_panel = np.vstack((spriteGrid, dataPanel))
            image = np.hstack((side_panel, image))

        cv2.imshow(self.WINDOW_NAME, image)
        return cv2.waitKey(1)

    def run(self):
        try:
            frame = 0
            for image in self.cap:
                if image is None:
                    print("No image received.")
                    continue

                frame += 1
                start = time.time()

                data, sprite_grid = self.env.process(image)

                active = data['active']

                if self.using_controller:
                    (left, right, back, start, xbox) = self.controller.read()
                    if start:
                        self.output.start()
                    else:
                        self.env.add_character_movement_hint(left)
                        self.output.move_and_shoot(
                            left, frame % (7 * 10) // 10 + 1)

                elif self.running:
                    if not active:
                        self.lastActive += 1
                    else:
                        self.lastActive = 0

                    if self.lastActive == 3:
                        self.reset()
                    elif self.lastActive > 0 and self.lastActive % 100 == 0:
                        self.output.none()
                        self.output.start()
                    else:
                        sprites = data['sprites']
                        move, shoot, self.arrows = self.ai.play(sprites)
                        self.output.move_and_shoot(move, shoot)

                try:
                    window = cv2.getWindowProperty('Robotron', 0)
                    if window < 0:
                        return
                except Exception:
                    return

                x, y = (586, 615)
                w, h = (7, 12)
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 1)

                resp = self.showScreen(image, sprite_grid, data)
                self.handleInput(resp)
                end = time.time()
                self.addProfData('all', end - start)

        except KeyboardInterrupt:
            print("Interrupt detected.  Exiting...")


if __name__ == '__main__':
    r = Robotron(0)
    r.run()
