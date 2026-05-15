from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from core.renderer import Cv2Renderer
from core.controller import Controller_c
from core.sim import Robot_c

from scenes.scene_1 import simple_center_obstacle, scene2


@dataclass
class EnvConfig:
    arena_size: int = 200
    max_steps: int = 20000
    robot_start: Tuple[float, float, float] = (50, 50, np.pi / 4)
    objetivo: List[float] = field(default_factory=lambda: [190, 190, 5])
    render_mode: Optional[str] = "human"
    render_every: int = 1


class RobotEnv:
    def __init__(
        self,
        controller,
        config: Optional[EnvConfig] = None,
        renderer: Optional[Cv2Renderer] = None,
    ):
        self.controller = controller
        self.config = config or EnvConfig()

        self.renderer = renderer
        if self.config.render_mode is not None and self.renderer is None:
            self.renderer = Cv2Renderer(arena_size=self.config.arena_size)

        self.robot = None
        self.obstacles = []
        self.path = []
        self.step_count = 0
        self.done = False

        self.reset()

    def reset(self):
        x, y, theta = self.config.robot_start

        self.robot = Robot_c(
            x=x,
            y=y,
            theta=theta,
            objetivo=self.config.objetivo,
        )

        self.robot.theta = theta

        self.obstacles = scene2(arena_size=self.config.arena_size, wall_thickness=2)

        self.path = [(self.robot.x, self.robot.y)]
        self.step_count = 0
        self.done = False

        self._update_sensors_and_collisions()

    def _update_sensors_and_collisions(self):
        for sensor in self.robot.prox_sensors:
            sensor.updateGlobalPosition(
                self.robot.x,
                self.robot.y,
                self.robot.theta,
            )

        for obs in self.obstacles:
            self.robot.updateSensors(obs)
            self.robot.collisionCheck(obs)

        self.robot.updateScore()

    def reached_goal(self) -> bool:
        gx, gy, gr = self.config.objetivo
        return np.hypot(self.robot.x - gx, self.robot.y - gy) <= gr

    def out_of_bounds(self) -> bool:
        r = self.robot.radius
        size = self.config.arena_size

        return not (
            r <= self.robot.x <= size - r and
            r <= self.robot.y <= size - r
        )

    def step(self):
        if self.done:
            return {"reason": "already_done"}

        vl, vr = self.controller.update(
            self.robot,
            self.config.objetivo,
        )

        self.robot.updatePosition(vl, vr)
        self._update_sensors_and_collisions()

        self.step_count += 1
        self.path.append((self.robot.x, self.robot.y))

        goal = self.reached_goal()
        bounds = self.out_of_bounds()
        timeout = self.step_count >= self.config.max_steps

        self.done = goal or bounds or timeout

        info = {
            "step": self.step_count,
            "score": self.robot.score,
            "vl": round(float(vl), 3),
            "vr": round(float(vr), 3),
            "goal": goal,
            "stall": self.robot.stall == 1,
            "reason": (
                "goal"
                if goal
                else "bounds"
                if bounds
                else "timeout"
                if timeout
                else "running"
            ),
        }

        if self.config.render_mode and self.step_count % self.config.render_every == 0:
            self.render(info)

        return info

    def render(self, info=None):
        if self.renderer is None:
            return None

        frame = self.renderer.render(
            robot=self.robot,
            obstacles=self.obstacles,
            goal=self.config.objetivo,
            path=self.path,
            info=info,
        )

        if self.config.render_mode == "human":
            key = self.renderer.show(frame, delay_ms=1)
            if key == ord("q"):
                self.done = True

        return frame

    def run(self):
        info = {}

        while not self.done:
            info = self.step()

        return info

    def close(self):
        if self.renderer:
            self.renderer.close()


if __name__ == "__main__":
    controller = Controller_c()

    env = RobotEnv(controller=controller)

    try:
        final_info = env.run()
        print(final_info)
    finally:
        env.close()