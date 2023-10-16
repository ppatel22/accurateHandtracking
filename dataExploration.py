"""
This file will be used to view the output .csv files from the trackers.
"""
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
# Load the data
filepath = "dataCombo/2023-10-15-14-47-47.csv"
data_df = pd.read_csv(filepath)

# %%
print(data_df.head())
data_df.describe()
# Count the number of NaN values in Angle L1 and Angle R1
print("Missing in left:", data_df["Angle L1"].isna().sum())
print("Missing in right:", data_df["Angle R1"].isna().sum())
# %%
left_columns = [
    "Angle L1",
    "Angle L2",
    "Angle L3",
    "Angle L4",
    "Angle L5",
    "Angle L6",
    "Angle L7",
    "Angle L8",
    "Handedness L",
    "palmTowards L",
    "Timestamp L (ms)",
]
right_columns = [
    "Angle R1",
    "Angle R2",
    "Angle R3",
    "Angle R4",
    "Angle R5",
    "Angle R6",
    "Angle R7",
    "Angle R8",
    "Handedness R",
    "palmTowards R",
    "Timestamp R (ms)",
]

data_df["Handedness R"] = data_df["Handedness R"].apply(
    lambda x: 1 if x == "Right" else (0 if "Left" else np.nan)
)
data_df["Handedness L"] = data_df["Handedness L"].apply(
    lambda x: 1 if x == "Right" else (0 if "Left" else np.nan)
)
# data_df["palmTowards R"] = data_df["palmTowards R"].apply(
#     lambda x: 1 if x == True else (0 if False else np.nan)
# )
# data_df["palmTowards L"] = data_df["palmTowards L"].apply(
#     lambda x: 1 if x == True else (0 if False else np.nan)
# )
bool_dict = {False: 0, True: 1}
data_df["palmTowards R"] = [
    bool_dict.get(val, np.nan) for val in data_df["palmTowards R"]
]
data_df["palmTowards L"] = [
    bool_dict.get(val, np.nan) for val in data_df["palmTowards L"]
]
# %%
fig, axes = plt.subplots(11, 1, figsize=(10, 14), sharex=True)

# Iterate through the columns and plot left and right camera data together
for i, (left_col, right_col) in enumerate(zip(left_columns, right_columns)):
    axes[i].plot(data_df.index, data_df[left_col], label="Left Camera")
    axes[i].plot(data_df.index, data_df[right_col], label="Right Camera")
    axes[i].set_ylabel(left_col.replace(" (ms)", ""))  # Customize the y-axis label
    axes[i].legend()

# Add common x-axis label
axes[-1].set_xlabel("Index")

plt.tight_layout()
plt.show()

# %% Initialize merged dataframe
new_cols = [f"Angle {i}" for i in range(1, 7)] + ["Handedness", "Timestamp (ms)"]
merged_df = pd.DataFrame(columns=new_cols)
merged_df["Timestamp (ms)"] = np.nanmean(
    [data_df["Timestamp L (ms)"], data_df["Timestamp R (ms)"]], axis=0
)
# %% Merge angles
angle_columns = left_columns[:8] + right_columns[:8]

data_df[angle_columns] = data_df[angle_columns].interpolate(method="linear", axis=0)

for i in range(6):
    left_angle_col = f"Angle L{i+1}"
    right_angle_col = f"Angle R{i+1}"
    merged_angle_col = f"Angle {i+1}"
    merged_df[merged_angle_col] = np.nan

    for idx, row in data_df.iterrows():
        left_angle = row[left_angle_col]
        right_angle = row[right_angle_col]
        palm_towards_l = row["palmTowards L"]

        if pd.notna(left_angle) and pd.notna(right_angle):
            threshold_col = left_angle_col if palm_towards_l else right_angle_col
            threshold = row[threshold_col] * 0.05  # 5% of the threshold
            diff = abs(left_angle - right_angle)

            if diff > threshold:
                merged_df.at[idx, merged_angle_col] = (
                    left_angle if palm_towards_l else right_angle
                )
            else:
                merged_df.at[idx, merged_angle_col] = (left_angle + right_angle) / 2
        else:
            merged_df.at[idx, merged_angle_col] = np.nan
# %% Merge handedness
merged_df["Handedness"] = data_df["Handedness L"].fillna(data_df["Handedness R"])
# %% Plot merged data
fig, axes = plt.subplots(len(merged_df.columns), 1, figsize=(10, 14), sharex=True)

for i, col in enumerate(merged_df.columns):
    axes[i].plot(merged_df.index, merged_df[col], label=col)
    axes[i].set_ylabel(col.replace(" (ms)", ""))
    axes[i].legend()

# Add a common x-axis label
axes[-1].set_xlabel("Index")

plt.tight_layout()
plt.show()
print("Missing in merged:", merged_df["Angle 1"].isna().sum())

"""
Notes from 10/4/23:
- The left and right camera data are not synchronized. The left camera data is
  delayed by about 1 second.
- Need to use a smoothing filter (rolling average possibly) and average values from 
  left and right when both are available.
- Last two angles are useless, only needed by the hardware
- Find a threshold that warrants using one camera over the other (use the one that has palmTowards == True)
"""

# %%
