import pandas as pd
import re

# Path to the log file
log_file_path = "../data/logs/slam_run.log"
output_csv_path = "../outputs/slam_results.csv"

# Regular expression to extract values
log_pattern = re.compile(
    r"Map: (\d+) \| Iteration: (\d+) \| Drones: (\d+) \| Time: (not solved|[\d.]+)"
)

# List to store parsed results
results = []

with open(log_file_path, "r") as f:
    for line in f:
        match = log_pattern.search(line)
        if match:
            map_idx = int(match.group(1))
            iteration = int(match.group(2))
            drones = int(match.group(3))
            time_str = match.group(4)
            time_val = None if time_str == "not solved" else float(time_str)
            results.append((map_idx, iteration, drones, time_val))

# Save to CSV
df = pd.DataFrame(results, columns=["map", "iteration", "drones", "time"])
df.to_csv(output_csv_path, index=False)
print(f"Saved parsed results to: {output_csv_path}")
