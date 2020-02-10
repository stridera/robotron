# -*- coding: utf-8 -*-
""" Module handles controller input """

import pygame


class Controller():
    """ Gets imput from a controller if attached. """
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

    def __bin_to_cardinal(self, direction):
        """ Convert from the int direction to a cardinal direction """
        response = 0

        if direction == 0:
            response = 0
        elif direction == self.UP:
            response = 1
        elif direction == self.UP | self.RIGHT:
            response = 2
        elif direction == self.RIGHT:
            response = 3
        elif direction == self.DOWN | self.RIGHT:
            response = 4
        elif direction == self.DOWN:
            response = 5
        elif direction == self.DOWN | self.LEFT:
            response = 6
        elif direction == self.LEFT:
            response = 7
        elif direction == self.UP | self.LEFT:
            response = 8
        else:
            print("Unknown input", direction)
            response = 0

        return response

    def read(self):
        """ Read value from controller """

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

        a_button = self.joystick.get_button(self.ABUTTON)
        b_button = self.joystick.get_button(self.BBUTTON)
        x_button = self.joystick.get_button(self.XBUTTON)
        y_button = self.joystick.get_button(self.YBUTTON)

        right = 0
        if a_button:
            right |= self.DOWN
        if b_button:
            right |= self.RIGHT
        if y_button:
            right |= self.UP
        if x_button:
            right |= self.LEFT

        return (
            self.__bin_to_cardinal(left),
            self.__bin_to_cardinal(right),
            back,
            start,
            xbox
        )

    def attached(self):
        """ Is controller attached? """
        return self.has_joystick


def test():
    """ Test function """
    controller = Controller()

    running = True
    while running:
        (left, right, back, start, xbox) = controller.read()
        print(left, right, back, start, xbox)
        running = not xbox


if __name__ == '__main__':
    test()
