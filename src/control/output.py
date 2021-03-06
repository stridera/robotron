# -*- coding: utf-8 -*-

import serial
import time


class Output():

    UP = 1  # 1 << 0
    DOWN = 2  # 1 << 1
    RIGHT = 4  # 1 << 2
    LEFT = 8  # 1 << 3

    BTN_Y = 1  # 1 << 0
    BTN_A = 2  # 1 << 1
    BTN_B = 4  # 1 << 2
    BTN_X = 8  # 1 << 3

    START = int('11000000', 2)
    BACK = int('00110000', 2)

    def __init__(self, port):
        try:
            self.ser = serial.Serial(port)
        except serial.SerialException:
            self.ser = None
            print("Unable to connect to arduino.  Simulating output.")

        self.dir = [
            0,
            self.UP,
            self.UP | self.RIGHT,
            self.RIGHT,
            self.DOWN | self.RIGHT,
            self.DOWN,
            self.DOWN | self.LEFT,
            self.LEFT,
            self.UP | self.LEFT
        ]

    def __read(self):
        print('READ:', self.ser.readline())

    def __write(self, val):
        if self.ser:
            self.ser.write(val.to_bytes(1, byteorder='big'))
            self.ser.flush()
            self.ser.flushInput()
        else:
            print(f"Simulated Output: {val}")

    def move_and_shoot(self, left, right):
        both = (self.dir[left] << 4) | self.dir[right]

        self.__write(both)

    def move(self, left):
        self.__write(self.dir[left] << 4)

    def shoot(self, right):
        self.__write(self.dir[right])

    def start(self):
        self.__write(self.START)

    def back(self):
        self.__write(self.BACK)

    def none(self):
        self.__write(0)

    def quick_press(self, key):
        time.sleep(0.3)
        self.__write(key)
        time.sleep(0.3)
        self.none()

    def close(self):
        if self.ser is not None:
            self.ser.close()

    def reset(self):
        self.none()
        self.quick_press(self.START)
        for _ in range(5):
            self.quick_press(self.BTN_A)


if __name__ == "__main__":
    o = Output("/dev/ttyACM0")
    o.shoot(1)
