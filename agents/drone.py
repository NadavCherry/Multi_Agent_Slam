import numpy as np
import random

FREE_SPACE = 0
WALL = 1
ENTRY_POINT = 2
DOOR_CLOSED = 3
DOOR_OPEN = 4
WINDOW = 5
OUT_OF_BOUNDS = 6

DIRECTIONS = {
    'UP': (0, -1),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
    'RIGHT': (1, 0),
    'STAY': (0, 0)
}


class Drone:
    def __init__(self, drone_id, start_pos, fov_radius=5, entry_time=0):
        self.id = drone_id
        self.pos = start_pos  # (x, y)
        self.fov_radius = fov_radius
        self.entry_time = entry_time
        self.active = False
        self.local_map = None   # Will be initialized once we get map dimensions
        self.path_history = [start_pos]
        self.collided = False

    def initialize_map(self, map_shape):
        self.local_map = np.full(map_shape, -1, dtype=np.int8)  # -1 = unknown

    def activate(self, current_time):
        if current_time >= self.entry_time:
            self.active = True


    def move(self, direction, env):
        if not self.active:
            return

        dx, dy = DIRECTIONS[direction]
        new_x = self.pos[0] + dx
        new_y = self.pos[1] + dy

        if env.is_collision(new_x, new_y):
            self.collided = True
            return  # Don't move into collision
        else:
            self.pos = (new_x, new_y)
            self.path_history.append(self.pos)
            self.collided = False
            return self.sense(env)

    def sense(self, env):
        if not self.active:
            return []

        def bresenham(x0, y0, x1, y1):
            """Yield integer coordinates on the line from (x0, y0) to (x1, y1)."""
            dx = abs(x1 - x0)
            dy = -abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx + dy
            while True:
                yield x0, y0
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    x0 += sx
                if e2 <= dx:
                    err += dx
                    y0 += sy

        cx, cy = self.pos
        new_discoveries = []

        for dy in range(-self.fov_radius, self.fov_radius + 1):
            for dx in range(-self.fov_radius, self.fov_radius + 1):
                x = cx + dx
                y = cy + dy
                if not (0 <= x < env.width and 0 <= y < env.height):
                    continue
                if dx ** 2 + dy ** 2 > self.fov_radius ** 2:
                    continue

                blocked = False
                for lx, ly in bresenham(cx, cy, x, y):
                    if not (0 <= lx < env.width and 0 <= ly < env.height):
                        break
                    val = env.get_tile(lx, ly)
                    if self.local_map[ly, lx] != val:
                        self.local_map[ly, lx] = val
                        new_discoveries.append((lx, ly, val))
                    if val in {WALL, DOOR_CLOSED}:
                        break  # stop vision beyond this point

        return new_discoveries

    def get_observed_map(self):
        return self.local_map

    def get_position(self):
        return self.pos

    def get_history(self):
        return self.path_history
