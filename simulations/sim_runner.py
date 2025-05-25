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
MAP_WIDTH = 30
MAP_HEIGHT = 30
FPS = 60
NUM_DRONES = 5


def run_simulation():
    pygame.init()
    font = pygame.font.SysFont("Arial", 16)

    screen_width = TILE_SIZE * MAP_WIDTH * 2 + 50
    screen_height = TILE_SIZE * MAP_HEIGHT + 160
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Multi-Agent SLAM Simulation")

    env = GridMapEnv(width=MAP_WIDTH, height=MAP_HEIGHT, randomize=True, num_entry_points=NUM_DRONES)
    drones = []
    for i, (y, x) in enumerate(env.entry_points[:NUM_DRONES]):
        d = Drone(drone_id=i, start_pos=(x, y), fov_radius=4, entry_time=0)
        d.initialize_map(env.grid.shape)
        drones.append(d)

    master = MasterController(drones, env)

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
        observed_map = np.full(env.grid.shape, -1, dtype=np.int8)
        for drone in drones:
            if drone.local_map is not None:
                observed_map[drone.local_map != -1] = drone.local_map[drone.local_map != -1]

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
        for drone in drones:
            if drone.active:
                dx, dy = drone.get_position()
                pygame.draw.circle(screen, (255, 255, 0), (dx * TILE_SIZE + TILE_SIZE // 2, dy * TILE_SIZE + TILE_SIZE // 2), 5)
                pygame.draw.circle(screen, (0, 255, 0), (MAP_WIDTH * TILE_SIZE + 50 + dx * TILE_SIZE + TILE_SIZE // 2, dy * TILE_SIZE + TILE_SIZE // 2), 5)

        # Progress bar
        explorable_tiles = {FREE_SPACE, ENTRY_POINT, DOOR_CLOSED, DOOR_OPEN, WINDOW}
        explorable_mask = np.isin(env.grid, list(explorable_tiles))
        total_cells = np.count_nonzero(explorable_mask)
        discovered_mask = (observed_map != -1) & explorable_mask
        known_cells = np.count_nonzero(discovered_mask)
        progress_ratio = min(known_cells / total_cells, 1.0) if total_cells else 0

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
