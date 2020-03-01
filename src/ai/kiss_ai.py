import cv2
import math
import numpy as np
from dataclasses import dataclass
import imagehash
from PIL import Image

GAMEBOX = [117, 310, 608, 974]  # Area to crop
PLAYER_BUFFER = 250


@dataclass
class Point:
    x: int
    y: int

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def distance(self, arr: list):
        dists = []
        for (x, y, w, h) in arr:
            x += w / 2
            y += h / 2
            dists.append(((self.x - x) ** 2 +
                          (self.y - y) ** 2) ** 0.5)
        return np.array(dists)

    def direction(self, x: int, y: int) -> int:
        rad = math.atan2(y-self.y, x-self.x)
        deg = math.degrees(rad) - 180 - 90 + 22
        if deg < 0:
            deg += 360
        # print(self.x, self.y, x, y, deg, int(deg//45))
        direction = int(deg // 45)
        direction = 0 if direction > 7 else direction
        return direction, deg

    def direction_heatmap(self):
        (top, left, bottom, right) = GAMEBOX
        w = right - left
        h = bottom - top
        tb = max(PLAYER_BUFFER - self.y, 0)
        rb = max(PLAYER_BUFFER - (w - self.x), 0)
        bb = max(PLAYER_BUFFER - (h - self.y), 0)
        lb = max(PLAYER_BUFFER - self.x, 0)
        return np.array([
            tb,
            (tb + rb) // 2,
            rb,
            (rb + bb) // 2,
            bb,
            (lb + bb) // 2,
            lb,
            (lb + tb) // 2
        ])

    def keys(self):
        return 'x', 'y'

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        return (getattr(self, x) for x in self.keys())

    def __add__(self, o):
        if isinstance(o, Point):
            self.x += o.x
            self.y += o.y
        elif isinstance(o, tuple):
            x, y = o
            self.x += x
            self.y += y
        return self


class AI:
    CARDINAL = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    DIST_PER_MOVE_CARDINAL = 16
    DISTANCE_PER_MOVE_DIAGONAL = 14
    MOVE_CHART = [
        (0, 0),
        (0, -DIST_PER_MOVE_CARDINAL),  # Up
        (DISTANCE_PER_MOVE_DIAGONAL, -DISTANCE_PER_MOVE_DIAGONAL),  # Up Right
        (DIST_PER_MOVE_CARDINAL, 0),  # Right
        (DISTANCE_PER_MOVE_DIAGONAL, DISTANCE_PER_MOVE_DIAGONAL),  # Down Right
        (0, DIST_PER_MOVE_CARDINAL),  # Down
        (-DISTANCE_PER_MOVE_DIAGONAL, DISTANCE_PER_MOVE_DIAGONAL),  # Down Left
        (-DIST_PER_MOVE_CARDINAL, 0),  # Left
        (-DISTANCE_PER_MOVE_DIAGONAL, -DISTANCE_PER_MOVE_DIAGONAL),  # Up Left
    ]

    def __init__(self):
        self.player_location = None
        self.move_buffer = []
        self.move_buffer_size = 5
        self.warp_wait = 10
        self.it = 0

    def reset(self):
        self.move_buffer = []
        self.player_location = None
        self.it = 0

    def imghash(self, img):
        img = cv2.resize(img, (20, 20))
        return imagehash.average_hash(Image.fromarray(img))

    def crop(self, image, x, y, w, h):
        return image[y:y + h, x:x + w]

    def add_rect(self, image, x, y, w, h, text: str = None):
        (top, left, bottom, right) = GAMEBOX
        x += left
        y += top
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 2)

        if text:
            cv2.putText(image, text, (x-w, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

        return image

    def get_action(self, gamebox, image):
        """ Lets do this """
        if not self.player_location:
            h, w = gamebox.shape
            self.player_location = Point(w//2, h//2)

        if len(self.move_buffer) > self.move_buffer_size:
            move = self.move_buffer.pop(0)
            self.player_location += self.MOVE_CHART[move]

        _, thresh = cv2.threshold(gamebox, 16, 255, 0)
        blur = cv2.GaussianBlur(thresh, (3, 3), 0)
        contours, _ = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rects = []
        for contour in enumerate(contours):
            rect = cv2.boundingRect(contour[1])
            (x, y, w, h) = rect
            if h > 10 and w > 10 and np.count_nonzero(self.crop(thresh, *rect)) > 55:
                rects.append(rect)

        distances = self.player_location.distance(rects)
        for rect, d in zip(rects, distances):
            (x, y, w, h) = rect

        move = 0
        shoot = 1

        move_map = self.player_location.direction_heatmap()
        print(self.player_location, move_map)
        for i, (distance, rect) in enumerate(sorted(zip(distances, rects), key=lambda pair: pair[0])):
            (x, y, w, h) = rect
            dx = x + w // 2
            dy = y + h // 2
            direction, deg = self.player_location.direction(dx, dy)

            if i == 0 and distance < 200 and self.it > self.warp_wait:
                self.player_location = Point(dx, dy)
                self.add_rect(image, x, y, w, h, "Player")
            else:
                if distance < 10:
                    continue
                if i == 1:
                    shoot = direction + 1
                if distance < PLAYER_BUFFER:
                    move_map[direction] += distance

                self.add_rect(image, x, y, w, h, f"{distance:.0f}:{self.CARDINAL[direction]}:{deg:.0f}")

        print(move_map)
        move = move_map.argsort().argmin() + 1

        if self.it < self.warp_wait:
            move = 0

        self.move_buffer.append(move)
        self.it += 1
        return move, shoot, image
