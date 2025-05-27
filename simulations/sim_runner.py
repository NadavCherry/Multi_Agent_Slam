import pygame
import time
import numpy as np
from envs.grid_map_env import GridMapEnv
from agents.drone import Drone
from master.master_controller import MasterController
from envs.grid_map_env import (
    WALL, FREE_SPACE, ENTRY_POINT, DOOR_CLOSED, DOOR_OPEN, WINDOW, OUT_OF_BOUNDS
)


TILE_SIZE = 20
MAP_WIDTH = 32
MAP_HEIGHT = 32
FPS = 120
NUM_DRONES = 30
ENTRY_POINTS = 1
FOV = 0


def compute_reachable_mask(env, fov):
    """
    Perform BFS from entry points to mark all cells that are discoverable
    considering FOV. Returns a boolean mask.
    """
    reachable = np.zeros_like(env.grid, dtype=bool)
    visited = np.zeros_like(env.grid, dtype=bool)
    queue = list(env.entry_points)

    while queue:
        y, x = queue.pop(0)
        if visited[y, x]:
            continue
        visited[y, x] = True
        reachable[y, x] = True

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < env.height and 0 <= nx < env.width):
                continue
            if visited[ny, nx]:
                continue
            tile = env.grid[ny, nx]
            if tile in {WALL, DOOR_CLOSED, OUT_OF_BOUNDS}:
                continue
            queue.append((ny, nx))

    # Expand reachable mask by FOV
    expanded = np.zeros_like(env.grid, dtype=bool)
    for y in range(env.height):
        for x in range(env.width):
            if reachable[y, x]:
                for dy in range(-fov, fov + 1):
                    for dx in range(-fov, fov + 1):
                        if dx**2 + dy**2 <= fov**2:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < env.height and 0 <= nx < env.width:
                                expanded[ny, nx] = True
    return expanded


def run_simulation():
    pygame.init()
    font = pygame.font.SysFont("Arial", 16)

    screen_width = TILE_SIZE * MAP_WIDTH * 2 + 50
    screen_height = TILE_SIZE * MAP_HEIGHT + 160
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Multi-Agent SLAM Simulation")

    # env = GridMapEnv(width=MAP_WIDTH, height=MAP_HEIGHT, randomize=True, num_entry_points=ENTRY_POINTS)
    env = GridMapEnv(map_path="data/maps/house_map.txt", width=32, height=32, randomize=False,
                         num_entry_points=ENTRY_POINTS, num_drones=NUM_DRONES, fov=FOV)
    # env = GridMapEnv(map_path="data/maps/structured_house_map.txt", width=32, height=32, randomize=False,
    #                  num_entry_points=ENTRY_POINTS, num_drones=NUM_DRONES, fov=FOV)
    # Only count reachable areas

    reachable_mask = compute_reachable_mask(env, FOV)
    master = MasterController(env.drones, env, reachable_mask)

    clock = pygame.time.Clock()
    start_time = time.time()
    tick = 0
    running = True

    completed = False
    completion_time = None

    while running:
        screen.fill((20, 20, 20))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not completed:
            master.step(tick)

        # Draw true map (left)
        for y in range(env.height):
            for x in range(env.width):
                tile = env.grid[y, x]
                color = {
                    WALL: (100, 100, 100),
                    FREE_SPACE: (60, 60, 60),
                    ENTRY_POINT: (0, 255, 255),
                    DOOR_CLOSED: (255, 0, 0),
                    DOOR_OPEN: (0, 200, 0),
                    WINDOW: (0, 0, 255),
                    OUT_OF_BOUNDS: (0, 0, 0)
                }.get(tile, (120, 120, 120))
                pygame.draw.rect(screen, color, (x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE - 1, TILE_SIZE - 1))

        # Observed map (right)
        observed_map = master.global_map

        for y in range(env.height):
            for x in range(env.width):
                tile = observed_map[y, x]
                color = {
                    WALL: (100, 100, 100),
                    FREE_SPACE: (200, 200, 200),
                    ENTRY_POINT: (0, 255, 255),
                    DOOR_CLOSED: (255, 0, 0),
                    DOOR_OPEN: (0, 200, 0),
                    WINDOW: (0, 0, 255),
                    OUT_OF_BOUNDS: (0, 0, 0),
                    -1: (20, 20, 20)  # Unknown
                }.get(tile, (150, 150, 150))
                pygame.draw.rect(screen, color, (MAP_WIDTH * TILE_SIZE + 50 + x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE - 1, TILE_SIZE - 1))

        # Draw drones
        for drone in env.drones:
            if drone.active:
                dx, dy = drone.get_position()

                # Left: yellow circle (on true map)
                pygame.draw.circle(screen, (255, 255, 0),
                                   (dx * TILE_SIZE + TILE_SIZE // 2, dy * TILE_SIZE + TILE_SIZE // 2), 5)

                # Right: draw drone ID on observed map
                drone_id_text = font.render(str(drone.id), True, (0, 0, 0))
                screen.blit(drone_id_text, (MAP_WIDTH * TILE_SIZE + 50 + dx * TILE_SIZE + 5, dy * TILE_SIZE))

        # Progress bar
        # Count all real map cells except those that are -1 in the ground truth (if any)


        known_cells = np.count_nonzero((observed_map != -1) & reachable_mask)
        total_cells = np.count_nonzero(reachable_mask)
        progress_ratio = min(known_cells / total_cells, 1.0)

        if not completed and progress_ratio >= 1.0:
            completed = True
            completion_time = time.time() - start_time

        bar_top = screen_height - 100
        bar_height = 24
        pygame.draw.rect(screen, (80, 80, 80), (50, bar_top, screen_width - 100, bar_height))
        pygame.draw.rect(screen, (0, 255, 0), (50, bar_top, int((screen_width - 100) * progress_ratio), bar_height))
        screen.blit(font.render(f"Progress: {int(progress_ratio * 100)}%", True, (255, 255, 255)), (50, bar_top - 20))

        # Timer
        elapsed = time.time() - start_time
        screen.blit(font.render(f"Time: {elapsed:.2f}s", True, (255, 255, 255)), (screen_width - 140, bar_top - 20))

        # Completion message
        if completed:
            msg = f"Objective Achieved in {completion_time:.2f} seconds"
            rendered_msg = font.render(msg, True, (0, 255, 255))
            screen.blit(rendered_msg, ((screen_width - rendered_msg.get_width()) // 2, bar_top - 50))

        # Legend
        legend_items = [
            ("Free", (200, 200, 200)),
            ("Wall", (100, 100, 100)),
            ("Entry", (0, 255, 255)),
            ("Door (Closed)", (255, 0, 0)),
            ("Door (Open)", (0, 200, 0)),
            ("Window", (0, 0, 255)),
            ("Out of Bounds", (0, 0, 0)),
            ("Drone", (255, 255, 0)),
        ]

        legend_y = screen_height - 36
        box_size = 14
        spacing_x = 140
        total_width = len(legend_items) * spacing_x
        start_x = (screen_width - total_width) // 2

        for i, (label, color) in enumerate(legend_items):
            x = start_x + i * spacing_x
            pygame.draw.rect(screen, color, (x, legend_y, box_size, box_size))
            screen.blit(font.render(label, True, (255, 255, 255)), (x + box_size + 6, legend_y - 2))

        pygame.display.flip()
        tick += 1
        clock.tick(FPS)

    pygame.quit()
