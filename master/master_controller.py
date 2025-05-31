import random
import numpy as np
import heapq

DIRECTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']


class MasterController:
    def __init__(self, env, discoverable_mask, mode="frontier"):
        self.env = env
        self.global_map = np.full((env.height, env.width), -1, dtype=np.int8)  # unknown
        self.frontiers = set()
        self.discoverable_mask = discoverable_mask
        self.mode = mode  # "random" or "frontier"
        self.goals = {d.id: None for d in env.drones}
        self.paths = {d.id: [] for d in env.drones}
        self.wait_counters = {d.id: 0 for d in env.drones}
        self.max_wait = 3  # maximum steps to wait before replay

    def step(self, current_time):
        for drone in self.env.drones:
            if not drone.active:
                drone.activate(current_time)

        assigned_goals = set()
        for drone in self.env.drones:
            self._update_frontiers()
            if self.mode == "random":
                new_info = self.random_walk(drone)

            elif self.mode == "frontier":
                new_info = self.frontier_plan(drone, assigned_goals, current_time)

            else:
                raise ValueError("Unknown mode")

            if new_info is not None:
                for x, y, val in new_info:
                    if self.global_map[y, x] == -1:
                        self.global_map[y, x] = val

    def _update_frontiers(self):
        self.frontiers = set()
        for y in range(self.env.height):
            for x in range(self.env.width):
                if self.global_map[y, x] == -1:
                    continue
                if self.env.grid[y, x] in {1, 3, 6}:  # WALL, DOOR_CLOSED, OUT_OF_BOUNDS
                    continue
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.env.width and 0 <= ny < self.env.height:
                        if self.global_map[ny, nx] == -1 and self.discoverable_mask[ny, nx]:
                            self.frontiers.add((x, y))
                            break

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

    def frontier_plan(self, drone, assigned_goals, current_time):
        id = drone.id
        current_pos = drone.pos

        # Check if goal is invalid, reached, or path exhausted
        goal = self.goals[id]
        if not goal or self.global_map[goal[1], goal[0]] != -1 or not self.paths[id]:
            available_frontiers = [f for f in self.frontiers if f not in assigned_goals]
            if not available_frontiers:
                print(f"[Warning] No available_frontiers for Drone {id}. Random walk. at time {current_time}")
                return self.random_walk(drone)

            # Step 1: find the closest frontiers
            min_dist = float('inf')
            closest_frontiers = []
            for f in available_frontiers:
                dist = abs(f[0] - current_pos[0]) + abs(f[1] - current_pos[1])
                if dist < min_dist:
                    closest_frontiers = [f]
                    min_dist = dist
                elif dist == min_dist:
                    closest_frontiers.append(f)

            # Step 2: maximize spacing from other drones
            best_goal, best_path = None, []
            max_spacing = -1
            for f in closest_frontiers:
                path = a_star(current_pos, f, self.global_map)
                if not path:
                    continue
                spacing = sum(np.linalg.norm(np.array(f) - np.array(other.pos))
                              for other in self.env.drones if other.id != id)
                if spacing > max_spacing:
                    best_goal = f
                    best_path = path
                    max_spacing = spacing

            if best_goal:
                self.goals[id] = best_goal
                self.paths[id] = best_path
                assigned_goals.add(best_goal)
            else:
                print(f"[Warning] No valid goal for Drone {id}. Random walk. at time {current_time}")
                return self.random_walk(drone)

        # Move along path
        if self.paths[id]:
            next_pos = self.paths[id][0]

            # If next position is occupied, wait
            blocked = any(other.id != id and other.pos == next_pos for other in self.env.drones)

            if blocked:
                self.wait_counters[id] += 1
                if self.wait_counters[id] >= self.max_wait:
                    print(f"[Info] Drone {id} waited too long. Replanting. at time {current_time}")
                    self.goals[id] = None
                    self.paths[id] = []
                    self.wait_counters[id] = 0
                    return self.random_walk(drone)
                else:
                    print(f"[Info] Drone {id} blocked by another. Waiting ({self.wait_counters[id]}/{self.max_wait}).")
                    return drone.move('STAY', self.env)

            # Safe to move
            self.wait_counters[id] = 0  # Reset wait counter
            self.paths[id].pop(0)
            dx, dy = next_pos[0] - current_pos[0], next_pos[1] - current_pos[1]
            direction_map = {(0, -1): 'UP', (0, 1): 'DOWN', (-1, 0): 'LEFT', (1, 0): 'RIGHT'}
            return drone.move(direction_map.get((dx, dy), 'STAY'), self.env)


def a_star(start, goal, grid):
    height, width = grid.shape
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            break

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                continue
            if grid[neighbor[1], neighbor[0]] in {1, 3, 6}:  # WALL, DOOR_CLOSED, OUT_OF_BOUNDS
                continue

            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                heapq.heappush(open_set, (f_score, neighbor))

    path = []
    while goal in came_from:
        path.append(goal)
        goal = came_from[goal]
    path.reverse()
    return path
