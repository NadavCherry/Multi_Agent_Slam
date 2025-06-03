import time
import os
import logging
import gc
from core.sim_runner import run_simulation
from tqdm import tqdm

# Configuration
MAX_ITERATIONS = 30
MAX_TIME = 50  # seconds
MAP_COUNT = 10
DRONE_COUNTS = [1, 2, 3]

# Set up logging
log_dir = "../data/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "slam_run1.log")

logging.basicConfig(
    filename=log_file,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

total_runs = MAP_COUNT * len(DRONE_COUNTS) * MAX_ITERATIONS

with tqdm(total=total_runs, desc="Running Simulations", ncols=100) as pbar:
    for map_idx in range(MAP_COUNT):
        map_path = f"../data/maps/house_map_{map_idx}.txt"
        for num_drones in DRONE_COUNTS:
            for iteration in range(1, MAX_ITERATIONS + 1):
                try:
                    start = time.time()
                    result = run_simulation(
                        map_path=map_path,
                        width=32,
                        height=32,
                        num_drones=num_drones,
                        num_entry_points=1,
                        fov=1,
                        render=True
                    )
                    elapsed = time.time() - start

                    if result is None or elapsed > MAX_TIME:
                        logging.info(f"Map: {map_idx} | Iteration: {iteration} | Drones: {num_drones} | Time: not solved")
                    else:
                        logging.info(f"Map: {map_idx} | Iteration: {iteration} | Drones: {num_drones} | Time: {result:.2f} seconds")
                except Exception as e:
                    logging.info(f"Map: {map_idx} | Iteration: {iteration} | Drones: {num_drones} | Time: not solved | Error: {e}")

                # Free memory
                import pygame
                pygame.quit()
                gc.collect()

                pbar.update(1)
