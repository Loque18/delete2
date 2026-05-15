from math import inf, atan2, pi
from typing import Tuple, List
import numpy as np
import keyboard

from stable_baselines3 import SAC

from delete2.core.classes import (FMS_STATEV2, FrameCounter, Logger, Perception, SensorMemory, KVMemory)
from delete2.core.utils import (clamp, angle_diff, rad2Deg, deg2rad, safe_sensor, distance)
from delete2.core.constants import ANGLE_ERR_MARGIN, BASE_SPEED, TARGET_DIST

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TYPES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sensors = Tuple[float, float, float, float, float, float, float]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SETUP
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
logger = Logger()

LABELS = {
    0: Perception.FREE_PATH,
    1: Perception.WALL_LEFT,
    2: Perception.WALL_RIGHT,    
    3: Perception.ROUND_OBJECT,
    4: Perception.CORNER,
    5: Perception.TRAPPED,
    6: Perception.FRONT_BLOCKED,
    7: Perception.UNKNOWN,
}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CONTROLLER
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Controller_c:


    # ========================================
    # CONSTRUCTOR
    # ========================================
    def __init__(self):

        # --------------------
        # components
        # --------------------
        self.frame_counter = FrameCounter()
        self.memory = KVMemory()
        
        # --------------------
        # memory
        # --------------------
        self.memory.set("front_clear_count", 0)


        # bug2
        self.memory.set("bug2_initial_dist", None)
        self.memory.set("line_coeffs", None)

        self.memory.set("hit_point", None)
        self.memory.set("hit_point_dist", None)


        # --------------------
        # modes
        # --------------------
        self.manual_mode = False

        self.model = SAC.load("./models/sac_epuck_final.zip", device="cuda")

        # ------------------------------
        # anti-loop / hysteresis memory
        # ------------------------------
        self.state_timer = 0
        self.turn_cooldown = 0
        self.front_blocked_count = 0
        self.front_soft_blocked_count = 0
        self.wall_lost_count = 0
        self.rehook_seen_count = 0

        # --------------------
        # state
        # --------------------
        self.state_timer = 0
        self.state = FMS_STATEV2.START 
        self.perc_state = Perception.UNKNOWN


    def build_observation(self, robot, goal):
        # =========================
        # SENSORES
        # =========================
        sensors = np.array(robot.sensor_distances, dtype=np.float32)

        # normalizar
        sensors = sensors / robot.sensor_range

        # =========================
        # GOAL
        # =========================
        dx = goal[0] - robot.x
        dy = goal[1] - robot.y

        distance = np.sqrt(dx**2 + dy**2)

        goal_angle = np.arctan2(dy, dx) - robot.theta

        # wrap angle
        goal_angle = np.arctan2(
            np.sin(goal_angle),
            np.cos(goal_angle)
        )

        # normalizaciones
        distance = distance / 200.0
        goal_angle = goal_angle / np.pi

        obs = np.concatenate([
            sensors,
            [distance, goal_angle]
        ]).astype(np.float32)

        return obs

  
    # ========================================
    # RAW PERCEPTION
    # ========================================
    def read_sensors(self, robot) -> Tuple[float, float, float, float, float, float, float, List[float]]:
        sensors = []
        for i in range(8):
            reading = robot.prox_sensors[i].reading
            if reading == -1:
                reading = inf
            sensors.append(reading)

        front_left = min(sensors[7], sensors[6])
        left = sensors[5]
        back_left = sensors[4]

        front_right = min(sensors[0], sensors[1])
        right = sensors[2]
        back_right = sensors[3]

        front = min(front_left, front_right)

        return (
            front_left,
            left,
            back_left,
            front_right,
            right,
            back_right,
            front,
            sensors
        )

    def rc_controller(self, robot):
        linear = 0.0
        angular = 0.0

        max_linear = 0.65
        max_turn = 0.45

        if keyboard.is_pressed("w"):
            linear += max_linear
        if keyboard.is_pressed("s"):
            linear -= max_linear

        if keyboard.is_pressed("a"):
            angular -= max_turn
        if keyboard.is_pressed("d"):
            angular += max_turn

        if keyboard.is_pressed("space"):
            return 0.0, 0.0

        # diferencial: angular positivo gira a la derecha
        vl = linear + angular
        vr = linear - angular

        return clamp(vl, -1.0, 1.0), clamp(vr, -1.0, 1.0)

    # ========================================
    # HIGH LEVEL ACTIONS
    # ========================================
    
    def store_initial_state(self, robot, goal):
        x0, y0 = robot.x, robot.y
        x1, y1 = goal[0], goal[1]

        A  = y1 - y0
        B = x0 - x1
        C = x1*y0 - x0*y1

        _dist = distance(x0, y0, x1, y1)

        self.memory.set("line_coeffs", (A, B, C))
        self.memory.set("bug2_initial_dist", _dist)


    def is_on_line(self, robot, threshold=5.0):
        A, B, C = self.memory.get("line_coeffs")
        distance = abs(A*robot.x + B*robot.y + C) / np.sqrt(A**2 + B**2)
        return distance < threshold
        
    def store_hit_point(self, robot, goal):
        _dist = distance(robot.x, robot.y, goal[0], goal[1])

        self.memory.set("hit_point", (robot.x, robot.y))
        self.memory.set("hit_point_dist", _dist)

    
    def start_turn(
        self,
        robot,
        direction=1,
        target_angle=0.0,
        state_after=FMS_STATEV2.GO_TO_GOAL,
    ):
        self.state = FMS_STATEV2.ROTATE
        self.state_timer = 0
        self.turn_direction = direction
        self.turn_target = target_angle
        self.state_after = state_after
        self.turn_cooldown = 12   # avoid immediate re-turn after rotation
    
    def wall_follow_left(
        self,
        front_left,
        left,
        back_left,
        target_dist=10.0,
        base_speed=0.62
    ):


        # 2. Usar left como distancia principal
        if left != inf:
            side_dist = left
        else:
            valid = [v for v in [front_left, back_left] if v != inf]
            side_dist = sum(valid) / len(valid) if valid else target_dist + 4.0

        # 3. Error de distancia
        dist_error = side_dist - target_dist

        # 4. Error angular: positivo = front_left más lejos que back_left
        if front_left != inf and back_left != inf:
            angle_error = front_left - back_left
        else:
            angle_error = 0.0

        # 5. Ganancias
        kp_dist = 0.055
        kp_angle = 0.045

        correction = kp_dist * dist_error + kp_angle * angle_error

        # 6. Seguridad: si el frente izquierdo está muy cerca, alejarse más fuerte
        if front_left != inf and front_left < target_dist * 0.65:
            correction -= 0.35
            base_speed = 0.42

        # 7. Si está muy pegado al lado, bajar velocidad
        if side_dist < target_dist - 2.0:
            base_speed = 0.45

        # 8. Si está muy desalineado, bajar velocidad
        if abs(angle_error) > 3.0:
            base_speed = min(base_speed, 0.45)

        correction = clamp(correction, -0.45, 0.45)

        vl = base_speed - correction
        vr = base_speed + correction

        return clamp(vl, -3.0, 3.0), clamp(vr, -3.0, 3.0)
    

    # ========================================
    # PRIMITIVES
    # ========================================
    def _idle(self):
        return 0.0, 0.0

    def _go_forward(self, speed=0.4):
        return speed, speed

    def _go_backward(self, speed=0.4):
        return -speed, -speed

    def _rotate(self, direction=1, speed=0.7):
        if direction == 1:
            return speed, -speed
        return -speed, speed
    
    def _1w_rotate(self, direction=1, speed=0.7, radius=0.5):

        if radius == 0:
            raise ValueError("Radius cannot be zero for 1-wheel rotation, use _rotate instead")
        
        R = radius 
        v = speed 
        L = 1.0

        vr = v * (R + L/2) / R
        vl = v * (R - L/2) / R
        
        if direction == 1:
            return vl, vr
        return vr, vl
    
    def _get_obs(self, robot, goal):

        sensor_values = []

        for sensor in self.robot.prox_sensors:
            if sensor.reading < 0:
                value = 0.0
            else:
                value = 1.0 - (sensor.reading / sensor.max_range)
                value = np.clip(value, 0.0, 1.0)


            sensor_values.append(value)


        dx = goal[0] - robot.x
        dy = goal[1] - robot.y

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
    
    def _action_to_wheels(self, action):
        speed = 1.0
        turn_speed = 0.5

        if action == 0:  # adelante
            vl, vr = speed, speed
        elif action == 1:  # atrás
            vl, vr = -speed, -speed
        elif action == 2:  # izquierda
            vl, vr = -turn_speed, turn_speed
        elif action == 3:  # derecha
            vl, vr = turn_speed, -turn_speed

        else:
            raise ValueError(f"Acción inválida: {action}")
        
        return vl, vr
    
    # ========================================
    # UPDATES
    # ========================================
    def update( self, robot , goal):

        self.frame_counter.increase()
        self.state_timer += 1

        if self.turn_cooldown > 0:
            self.turn_cooldown -= 1

        if self.frame_counter.frame < 3:
            return self._idle()

        if self.manual_mode:
            return self.rc_controller(robot)
        

        obs = self._get_obs(robot, goal)
        action, _ = self.model.predict(obs, deterministic=True)


        vl, vr = self._action_to_wheels(action)

        return vl, vr

        # --------------------------------------------------
        # PERCEPTION AND INTERPRETATION
        # --------------------------------------------------

        (
            front_left,
            left,
            back_left,
            front_right,
            right,
            back_right,
            front,
            sensors
        ) = self.read_sensors(robot)

        self.perc_state = self.interpret_perception(sensors=sensors)

        # --------------------------------------------------
        # FRAME STATE
        # --------------------------------------------------

        current_x = robot.x
        current_y = robot.y
        current_theta = robot.theta

        vl, vr = self._idle()

        if robot.desire == 1.0:
            self.state = FMS_STATEV2.DONE

    


