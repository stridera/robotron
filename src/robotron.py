#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
  ______  _____  ______   _____  _______  ______  _____  __   _
 |_____/ |     | |_____] |     |    |    |_____/ |     | | \  |
 |    \_ |_____| |_____] |_____|    |    |    \_ |_____| |  \_|

 Module designed to play the xbox robotron game.  Reads input via a capture card and sends output over terminal.
"""

import time
import sys
import concurrent.futures
from enum import Enum
from collections import namedtuple
from statistics import mean
import cv2
import numpy as np
import matplotlib.pyplot as plt

import ai
import capture
import control
import environment


class STATE(Enum):
    """ State enum """
    STOPPED = 1
    RUNNING = 2
    QUITTING = 3

    def __str__(self):
        # pylint: disable=invalid-sequence-index
        return [
            "Stopped",
            "Running",
            "Quitting"
        ][self.value]


class Robotron:
    """ Robotron XBox Player """
    WINDOW_NAME = "Robotron"
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, capDevice=2, arduinoPort='/dev/ttyACM0'):

        self.move_ai = ai.DQN()
        self.shoot_ai = ai.DQN()

        self.cap = capture.VideoCapture(capDevice)
        # self.cap = capture.VideoCapture('/home/strider/Code/robotron/resources/video/robotron-1.mp4')

        self.controller = control.Controller()
        self.output = None  # control.Output(arduinoPort)
        self.env = environment.Environment()

        self.profile_data = {'all': [0], 'ai': [0]}
        self.max_prof = 100
        self.last_active = 0
        self.graph = namedtuple('graph', ['initialized', 'fig', 'line1'])
        self.graph.initialized = False

        self.state = STATE.STOPPED
        self.using_controller = False

        self.executor = concurrent.futures.ProcessPoolExecutor()

        cv2.namedWindow(self.WINDOW_NAME)

    def __del__(self):
        if self.output:
            self.output.close()

        print("Killing all windows.")
        cv2.destroyAllWindows()

    def add_profile_data(self, name, delta):
        """ Add profile data """
        self.profile_data[name].append(delta)
        if len(self.profile_data[name]) > self.max_prof:
            self.profile_data[name] = self.profile_data[name][-self.max_prof:]

    def reset(self):
        """ Reset the game from an already running game. """
        # self.output.reset()
        self.env.reset()

    def handle_input(self, key):
        """ Process either keyboard or controller input """

        if key == ord('`'):
            self.state = STATE.RUNNING if self.state == STATE.STOPPED else STATE.STOPPED
        elif key == ord('q'):
            sys.exit(1)
        elif key == ord('e'):
            if self.output:
                self.output.start()
        elif key == ord('q'):
            if self.output:
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
            if self.output:
                if move or shoot:
                    self.output.move_and_shoot(move, shoot)
                else:
                    self.output.none()

    def get_graph(self, data):
        """ Create a graph and return an image """
        if not self.graph.initialized:
            self.graph.fig = plt.figure(figsize=(5, 4))

            x1 = np.linspace(0.0, 5.0)
            x2 = np.linspace(0.0, 2.0)

            y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
            y2 = np.cos(2 * np.pi * x2)
            self.graph.line1, = plt.plot(x1, y1, 'ko-')
            self.graph.initialized = True

        i = data['frame']
        self.graph.line1.set_ydata(np.cos(2 * np.pi * (x1+i*3.14/2)) * np.exp(-x1))

        self.graph.fig.canvas.draw()
        img = np.fromstring(self.graph.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(self.graph.fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def show_screen(self, image, data=None):
        """ Show the screen and data """
        full_height, _, _ = image.shape

        graph_panel = self.get_graph(data)
        graph_height, graph_width, _ = graph_panel.shape
        data_panel = np.zeros((full_height - graph_height, graph_width, 3), dtype=np.uint8)
        if self.using_controller:
            state = "Using Controller"
        else:
            state = self.state

        datastr = [
            "Score: {}".format(data['score']),
            "Lives: {}".format(data['lives']),
            "Active: {}".format(data['active']),
            "State: {}".format(state),
            "Times:",
            "  - Total: Max: {:.4f}  Average: {:.4f}".format(
                max(self.profile_data['all']), mean(self.profile_data['all'])),
            "  - AI: Max: {:.4f}  Average: {:.4f}".format(
                max(self.profile_data['ai']), mean(self.profile_data['ai'])),
        ]

        for i, line in enumerate(datastr):
            cv2.putText(data_panel, line, (15, (30 * i) + 30),
                        self.FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        image = np.hstack((np.vstack((graph_panel, data_panel)), image))

        cv2.imshow(self.WINDOW_NAME, image)
        return cv2.waitKey(1)

    def run(self):
        """ Run the player """
        try:
            for image in self.cap:
                if image is None:
                    print("No image received.")
                    continue

                start = time.time()

                game_image, data = self.env.process(image)

                if data['game_over']:
                    self.reset()
                    if data['inactive_frame_count'] % 11 == 0:
                        if self.output:
                            self.output.none()
                    elif data['inactive_frame_count'] % 100 == 0:
                        if self.output:
                            self.output.start()
                elif self.state == STATE.RUNNING and data['active']:
                    move_thread = self.executor.submit(self.move_ai.play, game_image)
                    shoot_thread = self.executor.submit(self.shoot_ai.play, game_image)

                    ai_timer = time.time()

                    move = move_thread.result()
                    shoot = shoot_thread.result()
                    self.add_profile_data('ai', shoot_timer - time.time())

                    if self.output:
                        self.output.move_and_shoot(move, shoot)

                # Check if window is closed, if so, quit.
                window = cv2.getWindowProperty('Robotron', 0)
                if window < 0:
                    return

                resp = self.show_screen(image, data)
                self.handle_input(resp)

                end = time.time()
                self.add_profile_data('all', end - start)

        except KeyboardInterrupt:
            print("Interrupt detected.  Exiting...")


def main():
    """ Program Entry """
    player = Robotron(1)
    player.run()


if __name__ == '__main__':
    main()
