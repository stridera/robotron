#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
  ______  _____  ______   _____  _______  ______  _____  __   _
 |_____/ |     | |_____] |     |    |    |_____/ |     | | \  |
 |    \_ |_____| |_____] |_____|    |    |    \_ |_____| |  \_|

 Module designed to play the xbox robotron game.  Reads input via a capture card and sends output over terminal.
"""

import time
import concurrent.futures
from enum import Enum
from statistics import mean
import cv2
import numpy as np

import ai
import capture
import control
import environment
from utils import Graph


class STATE(Enum):
    """ State enum """
    RUNNING = 0
    RESETTING = 1
    QUITTING = 2

    def __str__(self):
        # pylint: disable=invalid-sequence-index
        return [
            "Running",
            "Resetting",
            "Quitting"
        ][self.value]


class Robotron:
    """ Robotron XBox Player """
    WINDOW_NAME = "Robotron"
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, capDevice=2, arduinoPort='/dev/ttyACM0'):

        self.cap = capture.VideoCapture(capDevice)
        # self.cap = capture.VideoCapture('/home/strider/Code/robotron/resources/video/robotron-1.mp4')

        self.output = control.Output(arduinoPort)
        self.controller = control.Controller()
        self.env = environment.Environment()

        self.move_ai = ai.DQNAgent("cuda:0")
        self.shoot_ai = ai.DQNAgent("cuda:1")

        self.profile_data = {'all': [0], 'ai': [0], 'env': [0]}
        self.max_prof = 100
        self.last_active = 0
        self.ai_in_control = False

        self.state = STATE.RUNNING
        self.using_controller = False

        self.graph = Graph((5, 5))
        self.graph.add_graph('ep', 'Score/Q Graph of last 1000 Epoch', 1000)
        self.graph.add_line('ep', 'move', 'g-', 'Move Score')
        self.graph.add_line('ep', 'shoot', 'r-', 'Shoot Score')
        # self.graph.add_line('ep', 'moveq', 'y-', 'Move Q')
        # self.graph.add_line('ep', 'shootq', 'b-', 'Shoot Q')

        self.graph.add_graph('rewards', 'Reward for last 100 actions')
        self.graph.add_line('rewards', 'move', 'g-', 'Movement')
        self.graph.add_line('rewards', 'shoot', 'r-', 'Shooting')

        self.executor = concurrent.futures.ProcessPoolExecutor()

        cv2.namedWindow(self.WINDOW_NAME)

    def __del__(self):
        if self.output:
            self.output.close()

        print("Killing all windows.")
        cv2.destroyAllWindows()

    def add_profile_data(self, name, delta):
        """ Add profile data """
        self.profile_data[name].append(delta*1000.0)
        if len(self.profile_data[name]) > self.max_prof:
            self.profile_data[name] = self.profile_data[name][-self.max_prof:]

    def reset(self):
        """ Reset the game from an already running game. """
        self.env.reset()
        self.output.reset()

    def handle_input(self, key):
        """ Process either keyboard or controller input """
        if key == ord('`'):
            self.ai_in_control = not self.ai_in_control
            # self.state = STATE.RESETTING
        elif key == ord('q'):
            self.state = STATE.QUITTING
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
            else:
                self.output.none()

    def show_screen(self, image, data=None):
        """ Show the screen and data """
        full_height, _, _ = image.shape

        graph_panel = self.graph.get_image()

        graph_height, graph_width, _ = graph_panel.shape
        data_panel = np.zeros((full_height - graph_height, graph_width, 3), dtype=np.uint8)
        if self.using_controller:
            state = "Using Controller"
        else:
            state = self.state

        if data:
            datastr = [
                f"Frame: {data['frame']}   Score: {data['score']} Lives: {data['lives']}",
                f"Movement Reward: {data['movement_reward']} - Shooting Reward: {data['shooting_reward']}",
                f"Active: {data['active']}  Game Over: {data['game_over']}",
                f"AI in Controlled: {self.ai_in_control} State: {state}",
                "Profiling Times: (Last 100 frames)",
                "  - Total: Max: {:.4f}ms  Average: {:.4f}ms".format(
                    max(self.profile_data['all']), mean(self.profile_data['all'])),
                "  - AI: Max: {:.4f}ms  Average: {:.4f}ms".format(
                    max(self.profile_data['ai']), mean(self.profile_data['ai'])),
                "  - Env: Max: {:.4f}ms  Average: {:.4f}ms".format(
                    max(self.profile_data['env']), mean(self.profile_data['env'])),
            ]

            for i, line in enumerate(datastr):
                cv2.putText(data_panel, line, (15, (20 * i) + 20),
                            self.FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        image = np.hstack(
            (np.vstack((graph_panel, data_panel)),
             image[0:full_height, 300:980])
        )

        cv2.imshow(self.WINDOW_NAME, image)
        return cv2.waitKey(1)

    def run(self):
        """ Run the player """

        try:
            wait_frame = 0
            cum_shooting = 0
            cum_movement = 0
            for image in self.cap:
                if image is None:
                    print("No image received.")
                    continue

                start = time.time()

                env_start = time.time()
                game_image, data = self.env.process(image)
                self.add_profile_data('env', time.time() - env_start)

                self.graph.add('rewards', 'move', data['movement_reward'])
                self.graph.add('rewards', 'shoot', data['shooting_reward'])

                move = 0
                shoot = 0

                # print(f"Score: {data['score']}")
                if self.ai_in_control:
                    if self.state == STATE.RESETTING:
                        wait_frame += 1
                        self.output.reset(wait_frame//10)
                        if wait_frame > 50:
                            if data['score'] == 0:
                                self.env.reset()
                                self.state = STATE.RUNNING

                    elif data['game_over']:
                        self.graph.add('ep', 'move', cum_movement)
                        self.graph.add('ep', 'shoot', cum_shooting)
                        cum_shooting = 0
                        cum_movement = 0
                        self.state = STATE.RESETTING

                    elif self.state == STATE.RUNNING and data['active']:
                        cum_shooting += data['shooting_reward']
                        cum_movement += data['movement_reward']
                        wait_frame = 0

                        # move_thread = self.executor.submit(
                        #     self.move_ai.play, move, state, data['movement_reward'], data['game_over'])
                        # shoot_thread = self.executor.submit(
                        #     self.shoot_ai.play, shoot, state, data['shooting_reward'], data['game_over'])

                        ai_timer = time.time()
                        # move = move_thread.result()
                        # shoot = shoot_thread.result()
                        move = self.move_ai.play(game_image, move, data['movement_reward'], data['game_over'])
                        shoot = self.shoot_ai.play(game_image, shoot, data['shooting_reward'], data['game_over'])
                        self.add_profile_data('ai', time.time() - ai_timer)

                        self.output.move_and_shoot(move, shoot)

                if self.state == STATE.QUITTING:
                    return

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
    player = Robotron(2)
    player.run()


if __name__ == '__main__':
    main()
