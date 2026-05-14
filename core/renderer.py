from __future__ import annotations

import cv2
import numpy as np
from typing import Iterable, Optional, Sequence, Tuple


class Cv2Renderer:
    """
    Renderer simple con OpenCV para el simulador 2D.

    El mundo del simulador usa coordenadas tipo cartesiano:
    x hacia la derecha, y hacia arriba.

    OpenCV usa coordenadas de imagen:
    x hacia la derecha, y hacia abajo.

    Por eso se invierte el eje Y en world_to_screen().
    """

    def __init__(
        self,
        arena_size: int = 200,
        scale: int = 4,
        window_name: str = "E-puck RL Env",
        show_sensors: bool = True,
        show_path: bool = True,
    ):
        self.arena_size = arena_size
        self.scale = scale
        self.window_name = window_name
        self.show_sensors = show_sensors
        self.show_path = show_path
        self.width = int(arena_size * scale)
        self.height = int(arena_size * scale)

        self.colors = {
            "background": (255, 255, 255),
            "walls": (252, 186, 3)
        }

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        sx = int(round(x * self.scale))
        sy = int(round((self.arena_size - y) * self.scale))
        return sx, sy

    def _draw_circle_world(self, frame, x, y, radius, color, thickness=-1):
        center = self.world_to_screen(x, y)
        cv2.circle(frame, center, max(1, int(radius * self.scale)), color, thickness)

    def render(
        self,
        robot,
        obstacles: Iterable,
        goal: Sequence[float],
        path: Optional[Sequence[Tuple[float, float]]] = None,
        info: Optional[dict] = None,
    ) -> np.ndarray:
        # frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame = np.full((self.height, self.width, 3), self.colors["background"], dtype=np.uint8)

        # Arena border
        cv2.rectangle(frame, (0, 0), (self.width - 1, self.height - 1), (80, 80, 80), 1)

        # Goal
        gx, gy, gr = goal
        self._draw_circle_world(frame, gx, gy, gr, (0, 255, 0), thickness=2)

        # Obstacles
        for obs in obstacles:
            self._draw_circle_world(frame, obs.x, obs.y, obs.radius, self.colors["walls"], thickness=-1)

        # Path
        if self.show_path and path and len(path) > 1:
            pts = np.array([self.world_to_screen(x, y) for x, y in path], dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=False, color=(120, 120, 255), thickness=1)

        # Robot body
        self._draw_circle_world(frame, robot.x, robot.y, robot.radius, (255, 180, 80), thickness=-1)
        self._draw_circle_world(frame, robot.x, robot.y, robot.radius, (255, 255, 255), thickness=1)

        # Robot heading
        hx = robot.x + np.cos(robot.theta) * robot.radius * 1.8
        hy = robot.y + np.sin(robot.theta) * robot.radius * 1.8
        cv2.line(frame, self.world_to_screen(robot.x, robot.y), self.world_to_screen(hx, hy), (0, 0, 255), 2)

        # Sensors
        if self.show_sensors:
            for sensor in robot.prox_sensors:
                sx, sy = self.world_to_screen(sensor.x, sensor.y)
                ex = sensor.x + np.cos(sensor.theta) * sensor.max_range
                ey = sensor.y + np.sin(sensor.theta) * sensor.max_range
                color = (90, 90, 90) if sensor.reading < 0 else (0, 255, 255)
                cv2.circle(frame, (sx, sy), 2, color, -1)
                cv2.line(frame, (sx, sy), self.world_to_screen(ex, ey), color, 1)

        # Text info
        if info:
            y = 18
            for key, value in info.items():
                cv2.putText(
                    frame,
                    f"{key}: {value}",
                    (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                y += 18

        return frame

    def show(self, frame: np.ndarray, delay_ms: int = 1) -> int:
        cv2.imshow(self.window_name, frame)
        return cv2.waitKey(delay_ms) & 0xFF

    def close(self):
        cv2.destroyWindow(self.window_name)