import random
import numpy as np

DIRECTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']


class MasterController:
    def __init__(self, drones, env, discoverable_mask):
        self.drones = drones
        self.env = env
        self.global_map = np.full((env.height, env.width), -1, dtype=np.int8)  # unknown
        self.frontiers = set()
        self.discoverable_mask = discoverable_mask

    def step(self, current_time):
        """
        Main controller step. Activates drones and gives them random actions.
        """
        for drone in self.drones:
            if not drone.active:
                drone.activate(current_time)

            if drone.active:
                new_info = self.random_walk(drone)
                if new_info is not None:
                    for x, y, val in new_info:
                        if self.global_map[y, x] == -1:
                            self.global_map[y, x] = val

                            # Remove all newly discovered cells from frontiers
                            for x, y, _ in new_info:
                                self.frontiers.discard((x, y))

                            # Update new frontiers based on all new observations
                            self._update_frontiers(new_info)
        print(self.frontiers)

    def _update_frontiers(self, new_info):
        for x, y, val in new_info:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy

                if not (0 <= nx < self.env.width and 0 <= ny < self.env.height):
                    continue

                if self.global_map[ny, nx] == -1 and self.discoverable_mask[ny, nx]:
                    self.frontiers.add((nx, ny))

    def random_walk(self, drone):
        """
        Simple algorithm: try a random direction; if it fails (collision), try another.
        """
        directions = random.sample(DIRECTIONS, len(DIRECTIONS))  # shuffle

        for direction in directions:
            dx, dy = {
                'UP': (0, -1),
                'DOWN': (0, 1),
                'LEFT': (-1, 0),
                'RIGHT': (1, 0),
                'STAY': (0, 0)
            }[direction]

            new_x = drone.pos[0] + dx
            new_y = drone.pos[1] + dy

            if not self.env.is_collision(new_x, new_y):
                return drone.move(direction, self.env)


        # All directions blocked — stay
        return drone.move('STAY', self.env)
