import numpy as np
import random
from agents.drone import Drone

# Map tile definitions
FREE_SPACE = 0
WALL = 1
ENTRY_POINT = 2
DOOR_CLOSED = 3
DOOR_OPEN = 4
WINDOW = 5
OUT_OF_BOUNDS = 6

TILE_NAME = {
    FREE_SPACE: "Free",
    WALL: "Wall",
    ENTRY_POINT: "Entry Point",
    DOOR_CLOSED: "Door (Closed)",
    DOOR_OPEN: "Door (Open)",
    WINDOW: "Window",
    OUT_OF_BOUNDS: "Out of Bounds"
}


class GridMapEnv:
    def __init__(self, width=32, height=32, randomize=False, map_path=None, num_entry_points=2, num_drones=3, fov=0):
        if map_path:
            self.grid = self.load_map(map_path)
        elif randomize:
            self.grid = self.generate_random_map(width, height, num_entry_points)
        else:
            self.grid = np.zeros((height, width), dtype=np.int8)

        self.height, self.width = self.grid.shape
        self.entry_points = self.find_entry_points()
        # print(self.height, self.width, self.entry_points)

        self.drones = []
        for i in range(num_drones):
            y, x = self.entry_points[i % len(self.entry_points)]
            entry_time = i * 2
            drone = Drone(drone_id=i, start_pos=(x, y), fov_radius=fov, entry_time=entry_time)
            drone.initialize_map(self.grid.shape)
            self.drones.append(drone)


    @staticmethod
    def load_map(path):
        """
        Loads a map from a .txt file with numeric values
        """
        return np.loadtxt(path, dtype=np.int8)

    @staticmethod
    def generate_random_map(width, height, num_entry_points=2):
        """
        Generates a random map with guaranteed entry points and includes all tile types.
        """
        grid = np.zeros((height, width), dtype=np.int8)

        # Border walls
        grid[0, :] = WALL
        grid[-1, :] = WALL
        grid[:, 0] = WALL
        grid[:, -1] = WALL

        # Random internal walls
        for _ in range(int(width * height * 0.1)):
            x = random.randint(1, width - 2)
            y = random.randint(1, height - 2)
            grid[y, x] = WALL

        # Random closed doors
        for _ in range(int(width * height * 0.01)):
            x = random.randint(1, width - 2)
            y = random.randint(1, height - 2)
            if grid[y, x] == FREE_SPACE:
                grid[y, x] = DOOR_CLOSED

        # Random open doors
        for _ in range(int(width * height * 0.01)):
            x = random.randint(1, width - 2)
            y = random.randint(1, height - 2)
            if grid[y, x] == FREE_SPACE:
                grid[y, x] = DOOR_OPEN

        # Random windows
        for _ in range(int(width * height * 0.01)):
            x = random.randint(1, width - 2)
            y = random.randint(1, height - 2)
            if grid[y, x] == FREE_SPACE:
                grid[y, x] = WINDOW

        # Random out-of-bounds areas (blackout zones)
        for _ in range(int(width * height * 0.005)):
            x = random.randint(1, width - 2)
            y = random.randint(1, height - 2)
            if grid[y, x] == FREE_SPACE:
                grid[y, x] = OUT_OF_BOUNDS

        # Force entry points on borders
        entries = set()
        attempts = 0
        max_attempts = 100

        while len(entries) < num_entry_points and attempts < max_attempts:
            attempts += 1
            side = random.choice(['top', 'bottom', 'left', 'right'])

            if side == 'top':
                x, y = random.randint(1, width - 2), 0
            elif side == 'bottom':
                x, y = random.randint(1, width - 2), height - 1
            elif side == 'left':
                x, y = 0, random.randint(1, height - 2)
            else:
                x, y = width - 1, random.randint(1, height - 2)

            grid[y, x] = ENTRY_POINT
            entries.add((y, x))

        # if len(entries) < num_entry_points:
            # print(f"Only {len(entries)} entry points created (requested {num_entry_points}).")

        return grid

    def is_collision(self, x, y):
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True
        if self.grid[y, x] in {WALL, DOOR_CLOSED, OUT_OF_BOUNDS}:
            return True
        for drone in self.drones:
            if drone.get_position() == (x, y) and drone.active:
                return True
        return False

    def get_tile(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y, x]
        return OUT_OF_BOUNDS

    def find_entry_points(self):
        entry_points = [(y, x) for y in range(self.height)
                        for x in range(self.width)
                        if self.grid[y, x] == ENTRY_POINT]

        if not entry_points:
            # Find all cells with values 0, 1, or 2
            candidates = [(y, x) for y in range(self.height)
                          for x in range(self.width)
                          if self.grid[y, x] in [FREE_SPACE, DOOR_OPEN, WINDOW]]

            if candidates:
                y, x = random.choice(candidates)
                original = self.grid[y, x]
                # Invert value cyclically: 0 → 2, 1 → 0, 2 → 1
                inverted = {0: 2, 1: 0, 2: 1}[original]
                self.grid[y, x] = ENTRY_POINT
                # print(f"No entry points found. Converted cell ({y}, {x}) from {original} to ENTRY_POINT (was inverted to {inverted}).")
                entry_points = [(y, x)]

        return entry_points

    @staticmethod
    def print_legend():
        for k, v in TILE_NAME.items():
            print(f"{k}: {v}")
