import numpy as np


class NoiseFilter():
    def __init__(self, size, init=None, debug=False):
        self.__counter = 0
        self.__size = 0
        self.__max_size = size
        self.__debug = debug
        self.__buffer = []
        if init:
            self.set(init)

    def set(self, num):
        if self.__size < self.__max_size:
            self.__buffer.append(int(num))
            self.__size += 1
        else:
            self.__buffer[self.__counter] = num

        self.__counter += 1

        if self.__counter >= self.__max_size:
            self.__counter = 0

        if self.__debug:
            print("DEBUG:", self.__buffer)

    def get(self):
        if self.__size == 0:
            return 0
        return np.bincount(self.__buffer).argmax()

    def reset(self):
        self.__counter = 0
        self.__size = 0
        self.__buffer = []

    def zero(self):
        self.counter = 0
        self.__size == self.__max_size
        self.__buffer = list(np.zeros(self.__max_size))
