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

from scenes.scene_1 import   scene2, scene3, scene4, scene5

def pick_random_scene(arena_size=200):
    scenes = [scene2, scene3, scene4, scene5]
    scene_fn = np.random.choice(scenes)
    return scene_fn(arena_size=arena_size, wall_thickness=2)

class EnvCtx:
    def distance_to_goal(): 
        pass 

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
            low=[-1.0, -1.0],
            high=[1.0, 1.0],
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
        if self.render_mode == "human" or self.render_mode == "rgb_array":
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

        self.obstacles = pick_random_scene(arena_size=self.arena_size)

        self._update_robot_sensors_and_collisions()

        self.prev_dist_to_goal = self._distance_to_goal()

        obs = self._get_obs()
        info = self._get_info()

        return obs, info
    

    def step(self, action):
        assert self.robot is not None, "Robot no inicializado. Llama a reset() primero."

        self.step_count += 1

        vl, vr = self._action_to_wheels(action)

        old_dist = self._distance_to_goal()

        self.robot.updatePosition(vl, vr)
        self._update_robot_sensors_and_collisions()

        new_dist = self._distance_to_goal()

        reached_goal = new_dist <= self.goal[2]
        collided = self.robot.stall == 1
        out_of_bounds = not self._inside_arena()
        timout = self.step_count >= self.max_steps


        progress = old_dist - new_dist
        reward = 2.0 * progress
        reward -= 0.01

        obs = self._get_obs()
        front_danger = float(max(obs[0], obs[1], obs[6], obs[7]))
        reward -= 1.0 * front_danger

        if collided:
            reward -= 50.0

        if out_of_bounds:
            reward -= 30.0

        if reached_goal:
            reward += 100.0

        terminated = bool(reached_goal or out_of_bounds or collided)
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

        return obs, float(reward), terminated, truncated, info
    
    def render(self):
        if self.renderer is None:
            return None

        assert self.robot is not None

        frame = self.renderer.render(
            robot=self.robot,
            obstacles=self.obstacles,
            goal=self.goal
        )

        if self.render_mode == "human":
            self.renderer.show(frame, delay_ms=1)

        return frame
        
    def close(self):
        if self.render_mode == "human" and self.renderer is not None:
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
    
    def _action_to_wheels(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        vl = float(action[0])
        vr = float(action[1])

        return vl, vr
    

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return np.arctan2(np.sin(angle), np.cos(angle))
        

    

        
