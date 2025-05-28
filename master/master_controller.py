import random
import numpy as np
import heapq


DIRECTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY']


class MasterController:
    def __init__(self, drones, env, discoverable_mask, mode="frontier"):
        self.drones = drones
        self.env = env
        self.global_map = np.full((env.height, env.width), -1, dtype=np.int8)  # unknown
        self.frontiers = set()
        self.discoverable_mask = discoverable_mask
        self.mode = mode  # "random" or "frontier"
        self.goals = {d.id: None for d in drones}
        self.paths = {d.id: [] for d in drones}
        self.first_move_done = {d.id: False for d in drones}

    def step(self, current_time):
        """
        Main controller step. Activates drones and gives them random actions.
        """
        assigned_goals = {goal for goal in self.goals.values() if goal}

        for drone in self.drones:
            if not drone.active:
                drone.activate(current_time)


            if drone.active:
                if self.mode == "random":
                    new_info = self.random_walk(drone)

                elif self.mode == "frontier":
                    if not self.first_move_done[drone.id]:
                        new_info = self.random_walk(drone)
                        self.first_move_done[drone.id] = True
                    else:
                        new_info = self.frontier_plan(drone, assigned_goals)

                else:
                    raise ValueError("Unknown mode")

                if new_info is not None:
                    for x, y, val in new_info:
                        if self.global_map[y, x] == -1:
                            self.global_map[y, x] = val

                            # Remove all newly discovered cells from frontiers
                            for x, y, _ in new_info:
                                self.frontiers.discard((x, y))

                            # Update new frontiers based on all new observations
                            self._update_frontiers(new_info)

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

    def frontier_plan(self, drone, assigned_goals):
        id = drone.id
        current_pos = drone.pos

        # Reassign if no goal or goal is discovered
        current_goal = self.goals[id]
        if (not current_goal or
                self.global_map[current_goal[1], current_goal[0]] != -1 or
                not self.paths[id]):

            candidates = [f for f in self.frontiers if self.discoverable_mask[f[1], f[0]]]
            if not candidates:
                print("no candidates")
                return drone.move('STAY', self.env)

            def score(f):
                dist = abs(f[0] - current_pos[0]) + abs(f[1] - current_pos[1])
                spacing = sum(
                    np.linalg.norm(np.array(f) - np.array(other.pos)) for other in self.drones if other.id != id
                )
                return dist - 0.5 * spacing

            for f in sorted(candidates, key=score):
                if f in assigned_goals:
                    continue
                path = a_star(current_pos, f, self.global_map)
                if path:
                    self.goals[id] = f
                    self.paths[id] = path
                    assigned_goals.add(f)
                    # print(f"Drone {id} assigned goal {f} with path {path}")
                    break
            else:
                print(f"[Warning] Drone {id} has no reachable frontier.")
                return drone.move('STAY', self.env)

        # Move along path
        if not self.paths[id]:
            print("no paths")
            return drone.move('STAY', self.env)

        next_pos = self.paths[id].pop(0)
        dx, dy = next_pos[0] - current_pos[0], next_pos[1] - current_pos[1]
        direction_map = {(0, -1): 'UP', (0, 1): 'DOWN', (-1, 0): 'LEFT', (1, 0): 'RIGHT', (0, 0): 'STAY'}
        if (dx == 0 and dy == 0) or dx>1 or dy>1 or dx<-1 or dy<-1:
            print(f"Drone {id}, paths: {self.paths[id]}, goal: {self.goals[id]}, current pos: {current_pos}")
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
