import pandas as pd

DATA_DIR = "/home1/amadapur/projects/eeg_trait_state_geometry/data"
IN_FILE  = f"{DATA_DIR}/motor_psd_labels.csv"
OUT_FILE = f"{DATA_DIR}/motor_psd_labels_state6.csv"

df = pd.read_csv(IN_FILE)

def map_to_state6(cond):

    # Baselines
    if cond == "baseline_eyes_open":
        return "EO"
    if cond == "baseline_eyes_closed":
        return "EC"

    # Left fist (task1 + task2)
    if "left_fist" in cond:
        return "left_fist"

    # Right fist (task1 + task2)
    if "right_fist" in cond:
        return "right_fist"

    # Both feet (task3 + task4)
    if "both_feet" in cond:
        return "both_feet"

    # Both fists (task3 + task4)
    if "both_fists" in cond:
        return "both_fists"

    return None  # safety

df["condition"] = df["condition"].apply(map_to_state6)

print("Unique classes:")
print(sorted(df["condition"].unique()))

df.to_csv(OUT_FILE, index=False)

print("Saved:", OUT_FILE)
