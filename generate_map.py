# import numpy as np
# import os
#
# house_map = np.array([
#     [2, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 2],
#     [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
#     [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
#     [1, 0, 3, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 3, 0, 1, 1, 1, 1],
#     [1, 1, 1, 1, 1, 1, 1, 0, 4, 0, 0, 0, 0, 0, 1, 4, 0, 0, 0, 1],
#     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
#     [0, 0, 0, 4, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
#     [0, 0, 0, 0, 0, 0, 1, 0, 3, 0, 1, 0, 3, 0, 1, 0, 0, 0, 0, 1],
#     [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
#     [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
# ] + [[6] * 20 for _ in range(9)])  # Add out-of-bounds padding
#
# os.makedirs("data/maps", exist_ok=True)
#
# # Save the map
# np.savetxt("data/maps/house_map.txt", house_map, fmt="%d")
# print("Saved house_map to data/maps/house_map.txt")
#
import numpy as np
import os

# Tile definitions
FREE = 0
WALL = 1
ENTRY = 2
DOOR_OPEN = 4
WINDOW = 5
OUT_OF_BOUNDS = 6

# Initialize a 32x32 grid fully as walls
house = np.full((32, 32), WALL, dtype=np.int8)

# Carve out the outer structure (house interior)
house[2:30, 2:30] = FREE

# Add exterior entry points
house[16, 1] = ENTRY  # Left entry
house[1, 16] = ENTRY  # Top entry

# Add perimeter walls (already present by default)

# Define rooms with walls
# Bedroom 1
house[3:10, 3:10] = FREE
house[3:10, 3] = WALL
house[3:10, 9] = WALL
house[3, 3:10] = WALL
house[9, 3:10] = WALL
house[9, 6] = DOOR_OPEN  # Door

# Bedroom 2
house[3:10, 12:19] = FREE
house[3:10, 12] = WALL
house[3:10, 18] = WALL
house[3, 12:19] = WALL
house[9, 12:19] = WALL
house[9, 15] = DOOR_OPEN  # Door

# Bedroom 3
house[3:10, 21:28] = FREE
house[3:10, 21] = WALL
house[3:10, 27] = WALL
house[3, 21:28] = WALL
house[9, 21:28] = WALL
house[9, 24] = DOOR_OPEN  # Door

# Living room (central large room)
house[11:20, 3:20] = FREE
# Add windows
house[11, 5] = WINDOW
house[19, 5] = WINDOW
house[15, 3] = WINDOW

# Kitchen (bottom right)
house[21:28, 21:28] = FREE
house[21:28, 21] = WALL
house[21:28, 27] = WALL
house[21, 21:28] = WALL
house[27, 21:28] = WALL
house[21, 24] = DOOR_OPEN  # Door

# Bathroom (near kitchen)
house[15:19, 21:26] = FREE
house[15:19, 21] = WALL
house[15:19, 25] = WALL
house[15, 21:26] = WALL
house[19, 21:26] = WALL
house[19, 23] = DOOR_OPEN  # Door

# Add windows in outside wall
house[2, 10] = WINDOW
house[2, 20] = WINDOW
house[10, 29] = WINDOW
house[20, 29] = WINDOW
house[29, 15] = WINDOW

# Add outer bounds
house[0:2, :] = OUT_OF_BOUNDS
house[:, 0:2] = OUT_OF_BOUNDS
house[-2:, :] = OUT_OF_BOUNDS
house[:, -2:] = OUT_OF_BOUNDS

# Save map
os.makedirs("data/maps", exist_ok=True)
np.savetxt("data/maps/structured_house_map.txt", house, fmt="%d")


