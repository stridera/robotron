# -*- coding: utf-8 -*-

import environment
import control
# import nn


class Robotron():
    is_playing = False
    in_controller = None
    out = None

    def __init__(self):
        self.in_controller = control.Controller()
        self.out = control.Output()
        self.env = environment.Environment(True)

    def run(self):
        while True:
            while not self.is_playing:
                (active, playarea, reward, done) = self.env.process()

                if done:
                    print("RESET")
                    self.out.reset()
                    self.env.reset()
                    continue

                (left, right, back, start, xbox) = self.in_controller.read()
                if (xbox):
                    self.is_playing = True
                    self.out.close()
                    self.env.close()
                    return
                else:
                    if start:
                        self.out.start()
                    elif back:
                        self.out.back()
                    else:
                        self.out.move_and_shoot(left, right)

            # while self.is_playing:
            #     try:
            #         print("Playing")
            #     except KeyboardInterrupt:
            #         self.is_playing = False


def main():
    r = Robotron()
    r.run()


if __name__ == '__main__':
    main()
