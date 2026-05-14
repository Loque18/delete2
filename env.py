from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from core.renderer import Cv2Renderer
from core.controller import Controller_c

# IMPORTANTE:
# Guarda tu simulador base como simulator.py en la misma carpeta.
# Ese archivo debe contener Robot_c, Obstacle_c, Obstacle_wall y crear_paredes_v2.
from core.sim import Robot_c, Obstacle_c, Obstacle_wall, crear_paredes_v2


@dataclass
class EnvConfig:
    arena_size: int = 200
    max_steps: int = 20000
    robot_start: Tuple[float, float, float] = (50, 50, np.pi / 4)
    objetivo: List[float] = field(default_factory=lambda: [190, 190, 5])
    render_mode: Optional[str] = "human"  # "human", "rgb_array" o None
    render_every: int = 1
    seed: int = 42


class RobotEnv:
    """
    Ambiente ligero para tu robot e-puck simulado.

    Responsabilidades:
    - Crear robot, objetivo y obstáculos.
    - Actualizar sensores.
    - Llamar controller.update(robot, objetivo).
    - Mover el robot.
    - Revisar colisiones y condición de éxito.
    - Renderizar con OpenCV si render_mode != None.

    Este env todavía NO es Gymnasium formal. Es el wrapper limpio para correr
    controladores reactivos o modelos ya entrenados.
    """

    def __init__(
        self,
        controller,
        config: Optional[EnvConfig] = None,
        obstacles: Optional[List] = None,
        renderer: Optional[Cv2Renderer] = None,
    ):
        self.controller = controller
        self.config = config or EnvConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.renderer = renderer
        self.robot = None
        self.obstacles: List = obstacles or []
        self.path: List[Tuple[float, float]] = []
        self.step_count = 0
        self.done = False
        self.last_frame = None

        if self.config.render_mode is not None and self.renderer is None:
            self.renderer = Cv2Renderer(arena_size=self.config.arena_size)

        self.reset(obstacles=obstacles)

    def reset(self, obstacles: Optional[List] = None):
        x, y, theta = self.config.robot_start
        self.robot = Robot_c(x=x, y=y, theta=theta, objetivo=self.config.objetivo)

        # Tu clase Robot_c actualmente ignora el theta recibido y usa pi/4.
        # Lo seteamos otra vez para respetar config.robot_start.
        self.robot.theta = theta

        for sensor in self.robot.prox_sensors:
            sensor.updateGlobalPosition(self.robot.x, self.robot.y, self.robot.theta)

        self.obstacles = obstacles if obstacles is not None else self.default_obstacles()
        self.step_count = 0
        self.done = False
        self.path = [(self.robot.x, self.robot.y)]
        self._update_sensors_and_collisions()
        return self.get_observation()

    def default_obstacles(self) -> List:
        """Escenario inicial mixto: paredes discretizadas + obstáculo circular."""
        obstacles: List = []

        # Pared en L hecha de puntos circulares pequeños.
        wall_points = crear_paredes_v2(
            trayectoria=[(80, 20), (80, 130), (150, 130)],
            num_puntos_por_segmento=45,
        )
        for x, y in wall_points:
            obstacles.append(Obstacle_wall(x, y, arena_size=self.config.arena_size, radius=2.0))

        # Obstáculo circular real.
        obstacles.append(Obstacle_c(125, 75, arena_size=self.config.arena_size, radius=14))
        return obstacles

    def get_sensor_readings(self) -> np.ndarray:
        """
        Retorna 8 sensores normalizados en [0, 1].
        0 = sin detección o muy lejos.
        1 = obstáculo muy cerca.
        """
        values = []
        for sensor in self.robot.prox_sensors:
            if sensor.reading < 0:
                values.append(0.0)
            else:
                values.append(1.0 - np.clip(sensor.reading / sensor.max_range, 0.0, 1.0))
        return np.array(values, dtype=np.float32)

    def get_observation(self) -> np.ndarray:
        """
        Observación compatible con un controlador/RL simple:
        - 8 sensores normalizados.
        - dx objetivo normalizado.
        - dy objetivo normalizado.
        Total: 10 features.
        """
        sensors = self.get_sensor_readings()
        gx, gy, _ = self.config.objetivo
        dx = (gx - self.robot.x) / self.config.arena_size
        dy = (gy - self.robot.y) / self.config.arena_size
        return np.concatenate([sensors, np.array([dx, dy], dtype=np.float32)])

    def _update_sensors_and_collisions(self):
        # Reset readings before scanning all obstacles.
        for sensor in self.robot.prox_sensors:
            sensor.updateGlobalPosition(self.robot.x, self.robot.y, self.robot.theta)

        for obs in self.obstacles:
            self.robot.updateSensors(obs)
            self.robot.collisionCheck(obs)

        self.robot.updateScore()

    def reached_goal(self) -> bool:
        gx, gy, gr = self.config.objetivo
        return np.hypot(self.robot.x - gx, self.robot.y - gy) <= gr

    def out_of_bounds(self) -> bool:
        r = self.robot.radius
        return not (
            r <= self.robot.x <= self.config.arena_size - r
            and r <= self.robot.y <= self.config.arena_size - r
        )

    def step(self):
        if self.done:
            return self.get_observation(), 0.0, True, {"reason": "already_done"}

        # Controller contract: update(robot, objetivo) -> (vl, vr)
        vl, vr = self.controller.update(self.robot, self.config.objetivo)
        self.robot.updatePosition(vl, vr)
        self._update_sensors_and_collisions()

        self.step_count += 1
        self.path.append((self.robot.x, self.robot.y))

        goal = self.reached_goal()
        timeout = self.step_count >= self.config.max_steps
        bounds = self.out_of_bounds()
        self.done = goal or timeout or bounds

        reward = self.compute_reward(goal=goal, bounds=bounds)
        info = {
            "step": self.step_count,
            "score": self.robot.score,
            "vl": round(float(vl), 3),
            "vr": round(float(vr), 3),
            "goal": goal,
            "stall": self.robot.stall == 1,
            "reason": "goal" if goal else "bounds" if bounds else "timeout" if timeout else "running",
        }

        if self.config.render_mode and self.step_count % self.config.render_every == 0:
            self.last_frame = self.render(info=info)

        return self.get_observation(), reward, self.done, info

    def compute_reward(self, goal: bool, bounds: bool) -> float:
        """Reward simple por ahora. Luego lo ajustamos para Gym/RL."""
        if goal:
            return 100.0
        if bounds:
            return -50.0
        if self.robot.stall == 1:
            return -5.0

        gx, gy, _ = self.config.objetivo
        dist = np.hypot(self.robot.x - gx, self.robot.y - gy)
        return -dist / self.config.arena_size

    def render(self, info: Optional[dict] = None):
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
        obs = self.get_observation()
        info = {}
        while not self.done:
            obs, reward, done, info = self.step()
        return obs, info

    def close(self):
        if self.renderer:
            self.renderer.close()


if __name__ == "__main__":
    # Ejemplo mínimo.
    # Cambia este import por tu controller real.
    # from scripts.controllerv2 import Controller_c
    controller = Controller_c()

    env = RobotEnv(controller=controller)
    try:
        final_obs, final_info = env.run()
        print(final_info)
    finally:
        env.close()
