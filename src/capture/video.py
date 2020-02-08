# -*- coding: utf-8 -*-
import cv2


class VideoCapture:

    def __init__(self, capDeviceOrFile=2):
        self.cap = cv2.VideoCapture(capDeviceOrFile)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

    def __del__(self):
        if self.cap:
            self.cap.release()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.cap and self.cap.isOpened():
            status, image = self.cap.read()
            if status:
                return image
            return None
        else:
            raise StopIteration()
