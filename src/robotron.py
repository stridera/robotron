#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
  ______  _____  ______   _____  _______  ______  _____  __   _
 |_____/ |     | |_____] |     |    |    |_____/ |     | | \  |
 |    \_ |_____| |_____] |_____|    |    |    \_ |_____| |  \_|

 Module designed to play the xbox robotron game.  Reads input via a capture card and sends output over terminal.
"""

import time
import multiprocessing as mp
import queue
from enum import Enum
from statistics import mean


import ai
import capture
import control
import environment
from ui import UI


class STATE(Enum):
    """ State enum """

    RUNNING = 0
    RESETTING = 1
    QUITTING = 2

    def __str__(self):
        # pylint: disable=invalid-sequence-index
        return ["Running", "Resetting", "Quitting"][self.value]


class Robotron:
    """ Robotron XBox Player """

    def __init__(self, capDevice=2, arduinoPort="/dev/ttyACM0"):
        mp.set_start_method("spawn")

        self.cap = capture.VideoCapture(capDevice)
        self.output = control.Output(arduinoPort)
        self.controller = control.Controller()
        self.env = environment.Environment()
        self.ui = UI(capDevice)

        self.move_ai = ai.DQNAgent('move', 'cuda:0', replay_buffer_size=10000)
        self.shoot_ai = ai.DQNAgent('shoot', 'cuda:1', replay_buffer_size=10000)

        self.move_q_out = mp.Queue()
        self.move_q_in = mp.Queue()
        self.shoot_q_out = mp.Queue()
        self.shoot_q_in = mp.Queue()
        self.ui_q_out = mp.Queue()
        self.ui_q_in = mp.Queue()

        self.ui_thread = mp.Process(target=self.ui.loop, args=(self.ui_q_out, self.ui_q_in,))
        self.move_thread = mp.Process(target=self.move_ai.loop, args=(self.move_q_out, self.move_q_in,))
        self.shoot_thread = mp.Process(target=self.shoot_ai.loop, args=(self.shoot_q_out, self.shoot_q_in,))

        self.max_prof = 100
        self.last_active = 0
        self.ai_in_control = False
        self.episode = 0
        self.loop_time = 0.3
        self.profile_data = {"all": [0], "ai": [0], "env": [0]}

        self.state = STATE.RUNNING
        self.using_controller = False

    def __del__(self):
        if self.output:
            self.output.close()

        self.move_q_out.put(None)
        self.shoot_q_out.put(None)
        self.ui_q_out.put(None)

        self.move_thread.join()
        self.shoot_thread.join()
        self.ui_thread.join()

    def reset(self):
        """ Reset the game from an already running game. """
        self.env.reset()
        self.output.reset()

    def add_profile_data(self, name, delta):
        """ Add profile data """
        self.profile_data[name].append(delta * 1000.0)
        if len(self.profile_data[name]) > self.max_prof:
            self.profile_data[name] = self.profile_data[name][-self.max_prof:]

    def handle_input(self, key):
        """ Process either keyboard or controller input """
        if key == ord("`"):
            self.ai_in_control = not self.ai_in_control
            self.state = STATE.RESETTING
        elif key == ord("Q"):
            self.state = STATE.QUITTING
        elif key == ord("e"):
            self.output.start()
        elif key == ord("b"):
            self.output.back()
        elif (self.controller is not None and self.controller.attached() and key == ord("c")):
            self.using_controller = not self.using_controller
            print("Using controller: ", "True" if self.using_controller else "False")
        else:
            move_key = {ord("w"): 1, ord("d"): 3, ord("s"): 5, ord("a"): 7}

            shoot_key = {ord("i"): 1, ord("l"): 3, ord("k"): 5, ord("j"): 7}

            move = move_key.get(key, 0)
            shoot = shoot_key.get(key, 0)
            if move or shoot:
                self.output.move_and_shoot(move, shoot)
            else:
                self.output.none()

    def update_ui_data(self, data):
        data['episode'] = self.episode
        data['state'] = self.state
        data['ai_in_control'] = self.ai_in_control
        data['all_max'] = max(self.profile_data["all"])
        data['all_mean'] = mean(self.profile_data["all"])
        data['ai_max'] = max(self.profile_data["ai"])
        data['ai_mean'] = mean(self.profile_data["ai"])
        data['env_max'] = max(self.profile_data["env"])
        data['env_mean'] = mean(self.profile_data["env"])
        self.ui_q_out.put_nowait(('data', data))
        self.ui_q_out.put_nowait(('rewards', (data['movement_reward'], data['shooting_reward'])))

    def run(self):
        """ Run the player """

        try:
            wait_frame = 0
            cum_shooting = 0
            cum_movement = 0

            self.move_thread.start()
            self.shoot_thread.start()
            self.ui_thread.start()

            move = 0
            shoot = 0

            for image in self.cap:
                if image is None:
                    print("No image received.")
                    continue

                start = time.time()

                env_start = time.time()
                game_image, data = self.env.process(image)
                self.add_profile_data("env", time.time() - env_start)

                if self.ai_in_control:
                    if self.state == STATE.RESETTING:
                        self.output.reset(wait_frame)
                        if wait_frame > 10:
                            if data["score"] == 0:
                                self.env.reset()
                                self.state = STATE.RUNNING
                                self.episode += 1
                        wait_frame += 1

                    elif data["game_over"]:
                        self.ui_q_out.put_nowait(('ep', (cum_movement, cum_shooting)))
                        cum_shooting = 0
                        cum_movement = 0
                        self.state = STATE.RESETTING

                    elif self.state == STATE.RUNNING and data["active"]:
                        cum_shooting += data["shooting_reward"]
                        cum_movement += data["movement_reward"]
                        wait_frame = 0

                        ai_timer = time.time()
                        self.move_q_out.put((game_image, move, data["movement_reward"], data["game_over"]))
                        self.shoot_q_out.put((game_image, shoot, data["shooting_reward"], data["game_over"]))

                        try:
                            move, moveq, move_epsilon = self.move_q_in.get(True, 1)
                            shoot, shootq, shoot_epsilon = self.shoot_q_in.get(True, 1)
                            data['moveq'] = moveq
                            data['shootq'] = shootq
                            data['move_epsilon'] = move_epsilon
                            data['shoot_epsilon'] = shoot_epsilon
                        except queue.Empty:
                            shoot = 0
                            move = 0
                            print("Timeout triggered... moving on")

                        self.add_profile_data("ai", time.time() - ai_timer)

                        self.output.move_and_shoot(move, shoot)

                        while not self.move_q_in.empty():
                            self.move_q_in.get()
                        while not self.shoot_q_in.empty():
                            self.shoot_q_in.get()

                self.update_ui_data(data)

                if self.state == STATE.QUITTING:
                    return

                if not self.ui_q_in.empty():
                    key = self.ui_q_in.get_nowait()
                    if key:
                        print(key)
                        self.handle_input(key)
                        time.sleep(0.1)
                        self.output.none()
                    else:
                        self.state = STATE.QUITTING

                end = time.time()
                loop_time = end - start
                self.add_profile_data("all", loop_time)

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
