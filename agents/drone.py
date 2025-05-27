import numpy as np
import random

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

    def sense(self, env):
        """
        Update local map using simple circular FOV around the drone
        """
        if not self.active:
            return

        cx, cy = self.pos
        for dy in range(-self.fov_radius, self.fov_radius + 1):
            for dx in range(-self.fov_radius, self.fov_radius + 1):
                x = cx + dx
                y = cy + dy

                if 0 <= x < env.width and 0 <= y < env.height:
                    if dx**2 + dy**2 <= self.fov_radius**2:
                        self.local_map[y, x] = env.get_tile(x, y)

    def get_observed_map(self):
        return self.local_map

    def get_position(self):
        return self.pos

    def get_history(self):
        return self.path_history
