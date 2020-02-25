# -*- coding: utf-8 -*-
""" Creates graph images based on values sent """

import cv2
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np


class Graph:
    @dataclass
    class __graph:
        graph_title: str
        buffer_size: int
        lines: Dict[str, List[int]]
        data: Dict[str, List[int]]
        labels: Dict[str, List[str]]
        limits: Tuple[int, int]
        dirty: False

        def __init__(self, title: str = "", buffer_size: int = 100, initial_limits: Tuple[int, int] = (0, 10)):
            self.graph_title = title
            self.buffer_size = buffer_size
            self.limits = initial_limits
            self.lines = {}
            self.data = {}
            self.labels = {}

    def __init__(self, size: Tuple[int, int]):
        self.graphs = {}
        self.size = size
        self.dirty = True
        self.iteration = 0
        self.fig = None
        self.ax = None

    def update_limits(self) -> None:
        ngraphs = len(self.graphs)
        if self.fig:
            plt.close(self.fig)
        self.fig, self.ax = plt.subplots(ngraphs, 1, figsize=self.size, facecolor="black")

        for i, graph_name in enumerate(self.graphs):
            self.ax[i].clear()
            self.ax[i].set_facecolor("black")
            self.ax[i].set_title(self.graphs[graph_name].graph_title, color="white")
            self.ax[i].tick_params(colors='red')
            for line in self.graphs[graph_name].lines:
                self.graphs[graph_name].lines[line], = self.ax[i].plot(
                    np.linspace(-self.graphs[graph_name].buffer_size, 0, self.graphs[graph_name].buffer_size),
                    np.linspace(self.graphs[graph_name].limits[0],
                                self.graphs[graph_name].limits[1],
                                self.graphs[graph_name].buffer_size),
                    label=self.graphs[graph_name].labels[line]
                )
            self.ax[i].legend(loc='upper left')

    def add_graph(self, name: str, title: Optional[str] = "", buffer_size: int = 100):
        """ Add a graph """
        self.graphs[name] = self.__graph(title)
        self.graphs[name].buffer_size = buffer_size

    def add_line(self, graph_name: str, name: str, format: str, label: Optional[str] = "") -> None:
        """ Add a new line to the graph """
        self.graphs[graph_name].labels[name] = label
        self.graphs[graph_name].lines[name] = None
        self.graphs[graph_name].data[name] = np.zeros(self.graphs[graph_name].buffer_size)
        self.graphs[graph_name].dirty = True

    def add(self, graph_name: str, name: str, val: int) -> None:
        """ Add value to graph """
        if name not in self.graphs[graph_name].data:
            print(f"Name {name} not assigned.  Use add_line({name}) first.")
            return

        self.graphs[graph_name].data[name] = np.append(self.graphs[graph_name].data[name], val)
        self.graphs[graph_name].data[name] = self.graphs[graph_name].data[name][-self.graphs[graph_name].buffer_size:]

        if val < self.graphs[graph_name].limits[0] or val > self.graphs[graph_name].limits[1]:
            self.graphs[graph_name].limits = (min(self.graphs[graph_name].limits[0], val),
                                              max(self.graphs[graph_name].limits[1], val))
            self.update_limits()

    def get_image(self) -> 'np.array[np.int]':
        """ Return an image of the graph """
        # self.iteration += 1

        if self.dirty:  # or self.iteration > 50:
            self.update_limits()
            self.dirty = False
            # self.iteration = 0

        for graph_name in self.graphs:
            for line_name in self.graphs[graph_name].lines:
                self.graphs[graph_name].lines[line_name].set_ydata(self.graphs[graph_name].data[line_name])

        plt.tight_layout()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img


def main():
    """ Program Entry """
    cv2.namedWindow('test')

    graph = Graph((3, 6))
    graph.add_graph('one', 'This is graph one', 1000)
    graph.add_line('one', 'green', 'g-', 'Green Line')
    graph.add_line('one', 'red', 'r-', 'Red Line')

    graph.add_graph('two', 'This is graph two')
    graph.add_line('two', 'blue', 'b-', 'Green Line')

    for i in range(1000):
        graph.add('one', 'green', i % 25)
        graph.add('one', 'red', (i*2) % 5)

        graph.add('two', 'blue', 1)

        cv2.imshow('test', graph.get_image())
        cv2.waitKey(10)


if __name__ == '__main__':
    main()
