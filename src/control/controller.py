# -*- coding: utf-8 -*-
import pygame


class Controller():
    UP = 1  # 1 << 0
    DOWN = 2  # 1 << 1
    RIGHT = 4  # 1 << 2
    LEFT = 8  # 1 << 3

    UPDOWNAXIS = 1
    LEFTRIGHTAXIS = 0
    ABUTTON = 0
    BBUTTON = 1
    XBUTTON = 2
    YBUTTON = 3
    BACKBUTTON = 6
    STARTBUTTON = 7
    XBOXBUTTON = 8

    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        self.has_joystick = False

        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)

            print("Using joystick: ", self.joystick.get_name())
            self.has_joystick = True
            self.joystick.init()

    def __bin_to_cardinal(self, b):
        if b == 0:
            return 0

        if b == self.UP:
            return 1
        if b == self.UP | self.RIGHT:
            return 2
        if b == self.RIGHT:
            return 3
        if b == self.DOWN | self.RIGHT:
            return 4
        if b == self.DOWN:
            return 5
        if b == self.DOWN | self.LEFT:
            return 6
        if b == self.LEFT:
            return 7
        if b == self.UP | self.LEFT:
            return 8
        print("Unknown input", b)
        return 0

    def read(self):
        if not self.has_joystick:
            raise Exception('Joystick not attached.')

        pygame.event.pump()

        start = self.joystick.get_button(self.STARTBUTTON)
        back = self.joystick.get_button(self.BACKBUTTON)
        xbox = self.joystick.get_button(self.XBOXBUTTON)
        axis0 = self.joystick.get_axis(self.UPDOWNAXIS)
        axis1 = self.joystick.get_axis(self.LEFTRIGHTAXIS)

        left = 0
        if axis0 > 0.5:
            left |= self.DOWN
        if axis0 < -0.5:
            left |= self.UP
        if axis1 > 0.5:
            left |= self.RIGHT
        if axis1 < -0.5:
            left |= self.LEFT

        a = self.joystick.get_button(self.ABUTTON)
        b = self.joystick.get_button(self.BBUTTON)
        x = self.joystick.get_button(self.XBUTTON)
        y = self.joystick.get_button(self.YBUTTON)

        right = 0
        if a:
            right |= self.DOWN
        if b:
            right |= self.RIGHT
        if y:
            right |= self.UP
        if x:
            right |= self.LEFT

        return (
            self.__bin_to_cardinal(left),
            self.__bin_to_cardinal(right),
            back,
            start,
            xbox
        )

    def run(self, out):
        while True:
            (left, right, back, start, xbox) = self.read()
            if (xbox):
                return
            else:
                if start:
                    out.start()
                elif back:
                    out.back()
                else:
                    out.move_and_shoot(left, right)

    def attached(self):
        return self.has_joystick


if __name__ == '__main__':
    c = Controller()
    running = True
    while running:
        (left, right, back, start, xbox) = c.read()
        print(left, right, back, start, xbox)
        running = not xbox
