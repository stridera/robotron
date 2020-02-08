"""
NoiseFilter package.
"""
import numpy as np


class NoiseFilter():
    """
    Filters out noisy data.  Basically returns the mode of the last x responses.
    """
    def __init__(self, size, init=None, debug=False):
        self.__counter = 0
        self.__size = 0
        self.__max_size = size
        self.__debug = debug
        self.__buffer = []
        if init:
            self.set(init)

    def set(self, num):
        """
        Attempt to update the value.  Basically adds a new value to the buffer.

        Arguments:
            num {variable} -- The value to set
        """
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
        """
        Get the current value that is most often repeated.

        Returns:
            variable -- The most common value in the list.
        """
        if self.__size == 0:
            return 0

        return np.bincount(self.__buffer).argmax()

    def reset(self):
        """
        Reset the module.  (resets all values to None)
        """
        self.__counter = 0
        self.__size = 0
        self.__buffer = []

    def zero(self):
        """
        Zero out all the values.
        """
        self.__counter = 0
        self.__size = self.__max_size
        self.__buffer = list(np.zeros(self.__max_size))
