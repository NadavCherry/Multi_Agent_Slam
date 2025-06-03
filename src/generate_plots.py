import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Load and clean data ===
df = pd.read_csv("outputs/slam_results.csv")

# Remove maps where all runs failed
valid_maps = df.groupby("map")["time"].apply(lambda x: x.notnull().any())
df = df[df["map"].isin(valid_maps[valid_maps].index)]
df = df[df["time"].notnull()]  # Drop remaining NaNs

# Set style
sns.set(style="whitegrid")

# Create figure and subplots grid
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("SLAM Simulation Results by Map and Drone Count", fontsize=16)

# === GRAPH 1: Completion Time per Map by Number of Drones ===
sns.barplot(data=df, x="map", y="time", hue="drones", errorbar="sd", ax=axes[0, 0])
axes[0, 0].set_title("Completion Time per Map by Number of Drones")
axes[0, 0].set_xlabel("Map Number")
axes[0, 0].set_ylabel("Time (seconds)")
axes[0, 0].legend(title="Drones")

# === GRAPH 2: Boxplot of Completion Time by Drones ===
sns.boxplot(data=df, x="drones", y="time", ax=axes[0, 1], showmeans=True,
            meanprops={"marker": "o", "color": "black"})
axes[0, 1].set_title("Distribution of Completion Time by Drones")
axes[0, 1].set_xlabel("Number of Drones")
axes[0, 1].set_ylabel("Time (seconds)")

# === GRAPH 3: Relative Improvement per Map ===
improvement_data = []
for map_id in sorted(df["map"].unique()):
    prev_mean = None
    for drone in sorted(df["drones"].unique()):
        mean_time = df[(df["map"] == map_id) & (df["drones"] == drone)]["time"].mean()
        improvement = ((prev_mean - mean_time) / prev_mean * 100) if prev_mean else None
        improvement_data.append((map_id, drone, improvement))
        prev_mean = mean_time

improvement_df = pd.DataFrame(improvement_data, columns=["map", "drones", "relative_improvement"])

sns.barplot(data=improvement_df[improvement_df["relative_improvement"].notnull()],
            x="map", y="relative_improvement", hue="drones", ax=axes[1, 0])
axes[1, 0].set_title("Relative Improvement per Map by Adding Drones")
axes[1, 0].set_xlabel("Map Number")
axes[1, 0].set_ylabel("Improvement (%)")
axes[1, 0].legend(title="Drones Added")

# === GRAPH 4: Average Time per Drones (mean + std) ===
avg_std = df.groupby("drones")["time"].agg(["mean", "std"]).reset_index()
sns.barplot(data=avg_std, x="drones", y="mean", hue="drones", ax=axes[1, 1], errorbar="sd", legend=False)
axes[1, 1].set_title("Average Completion Time per Drone Count")
axes[1, 1].set_xlabel("Number of Drones")
axes[1, 1].set_ylabel("Average Time (seconds)")

# Final layout
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("outputs/slam_visualization_grid.png")
print("Saved: slam_visualization_grid.png")
# plt.show()  # <-- Optional: use only if not using PyCharm backend
