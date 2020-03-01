#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
  ______  _____  ______   _____  _______  ______  _____  __   _
 |_____/ |     | |_____] |     |    |    |_____/ |     | | \  |
 |    \_ |_____| |_____] |_____|    |    |    \_ |_____| |  \_|

 Module designed to play the xbox robotron game.  Reads input via a capture card and sends output over terminal.
"""

import environment
import control
import capture
import ai
import cv2
import sys
import time


class Robotron:
    """ Robotron XBox Player """
    WINDOW_NAME = "Robotron"
    MOVE_KEYS = {ord("w"): 1, ord("d"): 3, ord("s"): 5, ord("a"): 7}
    SHOOT_KEYS = {ord("i"): 1, ord("l"): 3, ord("k"): 5, ord("j"): 7}

    def __init__(self, capDevice=2, arduinoPort="/dev/ttyACM0"):
        self.cap = capture.VideoCapture(capDevice)
        self.output = control.Output(arduinoPort)
        self.env = environment.Environment()
        self.ai = ai.AI()

        cv2.namedWindow(self.WINDOW_NAME)
        self.running = False
        self.loop_time = 0.1

    def __del__(self):
        if self.output:
            self.output.close()
        print("Killing all windows.")
        cv2.destroyAllWindows()

    def reset(self):
        """ Reset the game from an already running game. """
        self.env.reset()
        self.output.reset()

    def handle_input(self, key):
        """ Process either keyboard or controller input """
        if key is None or key == ord("Q"):
            sys.exit(1)
        elif key == ord("`"):
            self.running = not self.running
            self.ai.reset()
        elif self.running:
            pass
        elif key == ord("e"):
            self.output.start()
        elif key == ord("b"):
            self.output.back()
        else:
            move = self.MOVE_KEYS.get(key, 0)
            shoot = self.SHOOT_KEYS.get(key, 0)
            if move or shoot:
                self.output.move_and_shoot(move, shoot)
            else:
                self.output.none()

    def run(self):
        """ Run the player """
        try:
            inactive_frames = 0
            for image in self.cap:
                if image is None:
                    print("No image received.")
                    continue

                start = time.time()
                gamebox, active, score = self.env.process(image)
                if self.running:
                    if active:
                        inactive_frames = 0
                        move, shoot, image = self.ai.get_action(gamebox, image)
                        self.output.move_and_shoot(move, shoot)
                    else:
                        inactive_frames += 1
                        self.ai.reset()
                        if inactive_frames % 5 + 1 == 5:
                            self.output.move_and_shoot(0, 5)
                        else:
                            self.output.none()

                if cv2.getWindowProperty(self.WINDOW_NAME, 0) >= 0:
                    cv2.imshow(self.WINDOW_NAME, image)
                    self.handle_input(cv2.waitKey(1))
                else:
                    return

                end = time.time()
                loop_time = end - start

                if loop_time < self.loop_time:
                    sleep_time = self.loop_time - loop_time
                    time.sleep(sleep_time)
                else:
                    print('Loop took longer than loop_time: ', loop_time)
        except (KeyboardInterrupt, BrokenPipeError):
            print("Interrupt detected.  Exiting...")


def main():
    """ Program Entry """
    player = Robotron(2)
    player.run()


if __name__ == "__main__":
    main()
