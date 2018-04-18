# -*- coding: utf-8 -*-

import serial
from bitstring import BitArray
import time


class Output():

    UP = 1  # 1 << 0
    DOWN = 2  # 1 << 1
    RIGHT = 4  # 1 << 2
    LEFT = 8  # 1 << 3

    START = int('11000000', 2)
    BACK = int('00110000', 2)

    def __init__(self):
        self.ser = serial.Serial('/dev/ttyACM0')
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
        if not self.ser:
            raise("Unable to connect to arduino")

    def __write(self, val):
        self.ser.write(val.to_bytes(1, byteorder='big'))
        self.ser.flush()
        self.ser.flushInput()

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

    def a(self):
        self.__write(2)

    def none(self):
        self.__write(0)


    def close(self):
        if self.ser is not None:
            self.ser.close()

    def reset(self):
        print("Bytes Waiting", self.ser.inWaiting())
        self.start()
        time.sleep(0.1)
        self.none()
        time.sleep(0.1)
        self.__write(2)
        time.sleep(0.1)
        self.none()
        time.sleep(0.1)
        self.__write(2)
        time.sleep(0.1)
        self.none()
        time.sleep(0.1)
        self.__write(2)
        time.sleep(0.1)
        self.none()
        time.sleep(1)
        self.__write(2)
        time.sleep(0.1)
        self.none()
        time.sleep(1)
