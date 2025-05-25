import random

DIRECTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']


class MasterController:
    def __init__(self, drones, env):
        self.drones = drones
        self.env = env

    def step(self, current_time):
        """
        Main controller step. Activates drones and gives them random actions.
        """
        for drone in self.drones:
            if not drone.active:
                drone.activate(current_time)

            if drone.active:
                self.random_walk(drone)

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
                drone.move(direction, self.env)
                drone.sense(self.env)
                return

        # All directions blocked — stay
        drone.move('STAY', self.env)
        drone.sense(self.env)
