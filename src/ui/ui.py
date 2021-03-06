import capture
from .graph import Graph
import numpy as np
import cv2
import time


class UI:
    WINDOW_NAME = "Robotron"
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, capDevice):
        self.capDevice = capDevice
        self.data = {
            'episode': 0,
            'frame': 0,
            'score': 0,
            'lives': 0,
            'dead': 0,
            'reward': 0,
            'active': 0,
            'move': 0,
            'shoot': 0,
            'game_over': 0,
            'ai_in_control': 0,
            'state': 0,
            'epsilon': 0,
            'q_value': 0,
            'all_max': 0,
            'all_mean': 0,
            'ai_max': 0,
            'ai_mean': 0,
            'env_max': 0,
            'env_mean': 0,
        }

        self.graph = None

        self.initialized = False

    def __del__(self):
        print("Killing all windows.")
        cv2.destroyAllWindows()

    def initialize(self):
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        self.graph = Graph((5, 5))
        self.graph.add_graph("score", "Score of last 1000 Epoch", 1000)
        self.graph.add_line("score", "score", "g-", "Score")

        self.graph.add_graph("q", "Q-Values for last 10000 actions", buffer_size=10000, initial_limits=(0, 1))
        self.graph.add_line("q", "q", "g-", "Q Value")

        self.initialized = True
        self.movedesc = ["N", "U", "UR", "R", "DR", "D", "DL", "L", "UL"]

    def show_screen(self, image):
        """ Show the screen and data """

        data = self.data
        full_height, _, _ = image.shape
        graph_panel = self.graph.get_image()

        graph_height, graph_width, _ = graph_panel.shape
        data_panel = np.zeros((full_height - graph_height, graph_width, 3), dtype=np.uint8)

        datastr = [
            f"Episode: {data['episode']} Frame: {data['frame']} Score: {data['score']} Lives: {data['lives']}",
            f"Active: {data['active']}  Episode Over: {data['dead']}",
            f"AI in Controlled: {data['ai_in_control']} State: {data['state']}",
            f"Action: Move: {self.movedesc[data['move']]}  Shoot: {self.movedesc[data['shoot']]}",
            f"Epsilon:  {data['epsilon']:.4f}  Reward:  {data['reward']}",
            "Profiling Times: (Last 100 frames)",
            f"  - Total: Max: {data['all_max']:.4f}ms  Average: {data['all_mean']:.4f}ms",
            f"  - AI: Max: {data['ai_max']:.4f}ms  Average: {data['ai_mean']:.4f}ms",
            f"  - Env: Max: {data['env_max']:.4f}ms  Average: {data['env_mean']:.4f}ms",
        ]

        for i, line in enumerate(datastr):
            cv2.putText(data_panel, line, (15, (20 * i) + 20), self.FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        image = np.hstack(
            (np.vstack((graph_panel, data_panel)), image[0:full_height, 300:980])
        )

        # Check if window is closed, if so, quit.
        window = cv2.getWindowProperty("Robotron", 0)
        if window < 0:
            return None

        cv2.imshow(self.WINDOW_NAME, image)
        return cv2.waitKey(1)

    def loop(self, in_queue, out_queue):
        cap = capture.VideoCapture(self.capDevice)

        self.initialize()

        for image in cap:
            if image is None:
                print("No image received.")
                continue

            if in_queue.empty():
                time.sleep(0.1)
            else:
                data = in_queue.get()
                if data:
                    type, val = data
                    if type == 'data':
                        self.data.update(val)
                        self.graph.add("q", "q", self.data['q_value'])
                    else:
                        self.graph.add(type, type, val)
                else:
                    return

            if cv2.getWindowProperty(self.WINDOW_NAME, 0) >= 0:
                resp = self.show_screen(image)
                if not resp:
                    out_queue.put(None)
                    return

                if resp != -1:
                    out_queue.put_nowait(resp)
            else:
                out_queue.put(None)
                return
