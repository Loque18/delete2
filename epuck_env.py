# ==================================================
# IMPORTS
# ==================================================


from typing import Optional, Tuple, List, Any

# third party
import gymnasium as gym
import numpy as np
from gymnasium import spaces

# core
from core.sim import Robot_c, Obstacle_c, Obstacle_wall, crear_paredes_v2
from core.renderer import Cv2Renderer

class EpuckEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
            self,
            render_mode: Optional[str] = None,
            arena_size: int = 200,
            max_steps: int = 1000,
            goal: Tuple[float, float, float] = (180.0, 180.0, 8.0),
            robot_start: Tuple[float, float, float] = (25.0, 25.0, np.pi / 4),
            obstacle_radius: float = 14.0
        ):
    
        super().__init__()

        self.render_mode = render_mode
        self.arena_size = float(arena_size)
        self.max_steps = max_steps

        self.goal = np.array(goal, dtype=np.float32)
        self.robot_start = robot_start
        self.obstacle_radius = obstacle_radius

        self.robot: Robot_c | None = None
        self.obstacles: list[Obstacle_c] = []
        self.step_count = 0
        self.prev_dist_to_goal = 0.0

        # action space
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )

        # 8 sensors + x,y,theta al objetivo
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(11,),
            dtype=np.float32
        )

        self.renderer = None
        if self.render_mode == "human":
            self.renderer = Cv2Renderer(
                arena_size=self.arena_size,
                scale=4,
                window_name="E-puck Gym Env",
                show_sensors=True,
                show_path=True
            )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        super().reset(seed=seed)

        self.step_count = 0

        x, y, theta = self.robot_start

        self.robot = Robot_c(
            x=x,
            y=y,
            theta=theta,
            objetivo=[self.goal[0], self.goal[1], self.goal[2]]
        )

        self.obstacles = [
            Obstacle_c(
                x_prop=self.arena_size / 2,
                y_prop=self.arena_size / 2,
                arena_size=int(self.arena_size),
                radius=self.obstacle_radius
            )
        ]

        self._update_robot_sensors_and_collisions()

        self.prev_dist_to_goal = self._distance_to_goal()

        obs = self._get_obs()
        info = self._get_info()

        return obs, info
    

    def step(self, action):
        assert self.robot is not None, "Robot no inicializado. Llama a reset() primero."

        self.step_count += 1

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        vl, vr = float(action[0]), float(action[1])

        old_dist = self._distance_to_goal()

        self.robot.updatePosition(vl, vr)
        self._update_robot_sensors_and_collisions()

        new_dist = self._distance_to_goal()

        reached_goal = new_dist <= self.goal[2]
        collided = self.robot.stall == 1
        out_of_bounds = not self._inside_arena()
        timout = self.step_count >= self.max_steps


        progress = old_dist - new_dist
        reward = 5.0 * progress
        reward -= 0.01

        if collided:
            reward -= 5.0

        if out_of_bounds:
            reward -= 20.0

        if reached_goal:
            reward += 100.0

        terminated = bool(reached_goal or out_of_bounds)
        truncated = bool(timout and not terminated)


        obs = self._get_obs()
        info = self._get_info()
        info.update({
            "reached_goal": reached_goal,
            "collided": collided,
            "out_of_bounds": out_of_bounds,
            "distance_to_goal": new_dist,
        })

        if self.render_mode == "human":
            self.render()

        self.prev_dist_to_goal = new_dist

        return obs, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode != "human" or self.renderer is None:
            return None

        assert self.robot is not None

        frame = self.renderer.render(
            robot=self.robot,
            obstacles=self.obstacles,
            goal=self.goal
        )

        self.renderer.show(frame, delay_ms=1)

        return None
    
    def close(self):
        if self.renderer is not None and hasattr(self.renderer, "close"):
            self.renderer.close()


    def _update_robot_sensors_and_collisions(self):
        assert self.robot is not None

        for sensor in self.robot.prox_sensors:
            sensor.updateGlobalPosition(self.robot.x, self.robot.y, self.robot.theta)

        
        for obstacle in self.obstacles:
            self.robot.updateSensors(obstacle)
            self.robot.collisionCheck(obstacle)


    def _get_obs(self):
        assert self.robot is not None

        sensor_values = []

        for sensor in self.robot.prox_sensors:
            if sensor.reading < 0:
                value = 0.0
            else:
                value = 1.0 - (sensor.reading / sensor.max_range)
                value = np.clip(value, 0.0, 1.0)


            sensor_values.append(value)


        dx = self.goal[0] - self.robot.x
        dy = self.goal[1] - self.robot.y

        dist = np.sqrt(dx**2 + dy**2)
        max_dist = np.sqrt(2.0) * self.arena_size
        dist_norm = np.clip(dist / max_dist, 0.0, 1.0)


        goal_angle = np.arctan2(dy, dx)
        relative_angle = self._wrap_angle(goal_angle - self.robot.theta)


        obs = np.array(
            [
                *sensor_values,
                dist_norm,
                np.sin(relative_angle),
                np.cos(relative_angle)
            ],
            dtype=np.float32
        )

        return obs
    
    def _get_info(self) -> dict[str, Any]:
        assert self.robot is not None

        return {
            "robot_x": self.robot.x,
            "robot_y": self.robot.y,
            "robot_theta": self.robot.theta,
            "step_count": self.step_count,
        }
    
    def _distance_to_goal(self) -> float:
        assert self.robot is not None

        dx = self.goal[0] - self.robot.x
        dy = self.goal[1] - self.robot.y

        return np.sqrt(dx**2 + dy**2)
    

    def _inside_arena(self) -> bool:

        r = self.robot.radius

        return (
            r <= self.robot.x <= self.arena_size - r and
            r <= self.robot.y <= self.arena_size - r
        )
    
    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return np.arctan2(np.sin(angle), np.cos(angle))
        

    

        
