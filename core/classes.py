from enum import Enum
from typing import TypedDict, List

class FMS_State(Enum):
    START = "start"
    SEEK_WALL = "seek_wall"
    ROTATE = "rotate"
    FOLLOW_WALL = "follow_wall"
    REHOOK = "rehook"
    IDLE = "idle"
    
class FMS_STATEV2(Enum):    
    START = "start"
    IDLE = "idle"   
    GO_TO_GOAL = "go_to_goal" 
    EXPLORE_OBSTACLE = "explore_obstacle"
    ROUND_OBSTACLE = "round_obstacle"
    FOLLOW_WALL = "follow_wall"
    ROTATE = "rotate"
    DONE = "done"

class Perception(Enum):
    FREE_PATH = "FREE_PATH"
    WALL_LEFT = "WALL_LEFT"
    WALL_RIGHT = "WALL_RIGHT"
    ROUND_OBJECT = "ROUND_OBJECT"
    CORNER = "CORNER"
    TRAPPED = "TRAPPED"
    FRONT_BLOCKED = "FRONT_BLOCKED"
    UNKNOWN = "UNKNOWN"

class LogCounter:
    def __init__(self):
        self.max_logs = 2000

    def decrease(self):
        self.max_logs -= 1
        if self.max_logs < 0:
            self.max_logs = 0


class FrameCounter:
    def __init__(self):
        self.frame = 0

    def increase(self):
        self.frame += 1


class Logger:
    def __init__(self):
        self.counter = LogCounter()
        self.active = True

    def activate(self):
        self.active = True

    def deactivate(self):
        self.active = False

    def log(self, *args, **kwargs):
        if not self.active:
            return
        if self.counter.max_logs > 0:
            print(*args, **kwargs)
            self.counter.decrease()


class MemoryEntry(TypedDict):
    x: float
    y: float
    left_dist: float
    hooked: bool


class Memory:
    def __init__(self, size: int):
        self.memory_size = size
        self.last_hooked_positions: List[MemoryEntry] = []

    def store_entry(self, entry: MemoryEntry):
        self.last_hooked_positions.append(entry)
        if len(self.last_hooked_positions) > self.memory_size:
            self.last_hooked_positions.pop(0)

    def reset(self):
        self.last_hooked_positions = []


class SensorMemory:
    def __init__(self, max_length: int = 5):
        self.data: List[List[float]] = []
        self.max_length = max_length

    def add(self, sensor_readings: List[float]):
        self.data.append(sensor_readings)
        if len(self.data) > self.max_length:
            self.data.pop(0)

    def __len__(self):
        return len(self.data)
    

class KVMemory:
    def __init__(self):
        self.data = dict = {}
    

    def set(self, key: str, value):
        self.data[key] = value

    def get(self, key: str, default: any = None):
        return self.data.get(key, default)
    
