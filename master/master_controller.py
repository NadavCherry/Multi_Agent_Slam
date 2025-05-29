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

    def step(self, current_time):
        for drone in self.env.drones:
            if not drone.active:
                drone.activate(current_time)

        self._update_frontiers()

        assigned_goals = set()
        for drone in self.env.drones:
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

        if (not self.goals[id] or
                self.global_map[self.goals[id][1], self.goals[id][0]] != -1 or
                not self.paths[id]):

            available_frontiers = [f for f in self.frontiers if f not in assigned_goals]
            best_goal, best_path = None, []
            best_score = float('inf')

            for f in available_frontiers:
                path = a_star(current_pos, f, self.global_map)
                if not path:
                    continue

                dist = abs(f[0] - current_pos[0]) + abs(f[1] - current_pos[1])
                spacing = sum(np.linalg.norm(np.array(f) - np.array(other.pos))
                              for other in self.env.drones if other.id != id)
                score = dist - 0.5 * spacing

                if score < best_score:
                    best_goal = f
                    best_path = path
                    best_score = score

            if best_goal:
                self.goals[id] = best_goal
                self.paths[id] = best_path
                assigned_goals.add(best_goal)
            else:
                print(f"[Warning] No goal available for Drone {id} at time {current_time}. Using random walk.")
                return self.random_walk(drone)

        if self.paths[id]:
            next_pos = self.paths[id].pop(0)
            dx, dy = next_pos[0] - current_pos[0], next_pos[1] - current_pos[1]
            direction_map = {(0, -1): 'UP', (0, 1): 'DOWN', (-1, 0): 'LEFT', (1, 0): 'RIGHT', (0, 0): 'STAY'}
            return drone.move(direction_map.get((dx, dy), 'STAY'), self.env)
        else:
            print(f"[Warning] Drone {id} path exhausted.")
            return self.random_walk(drone)

def a_star(start, goal, grid):
    height, width = grid.shape
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            break

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                continue
            if grid[neighbor[1], neighbor[0]] in {1, 3, 6}:  # WALL, DOOR_CLOSED, OUT_OF_BOUNDS
                continue

            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + abs(neighbor[0]-goal[0]) + abs(neighbor[1]-goal[1])
                heapq.heappush(open_set, (f_score, neighbor))

    path = []
    while goal in came_from:
        path.append(goal)
        goal = came_from[goal]
    path.reverse()
    return path
