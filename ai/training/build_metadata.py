import os
import csv

BASE_DIR = os.path.dirname(__file__)

DATASET_DIR = os.path.join(BASE_DIR, "datasets")

OUTPUT_CSV = os.path.join(
    DATASET_DIR,
    "metadata",
    "metadata.csv"
)

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

ravdess_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise"
}

crema_map = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

rows = []

# --------------------------
# RAVDESS
# --------------------------

ravdess_dir = os.path.join(DATASET_DIR, "ravdess")

for root, _, files in os.walk(ravdess_dir):

    for file in files:

        if file.endswith(".wav"):

            parts = file.split("-")

            emotion = ravdess_map.get(parts[2])

            rows.append([
                os.path.join(root, file),
                emotion,
                "ravdess"
            ])

# --------------------------
# CREMA-D
# --------------------------

crema_dir = os.path.join(DATASET_DIR, "crema_d")

for root, _, files in os.walk(crema_dir):

    for file in files:

        if file.endswith(".wav"):

            parts = file.split("_")

            emotion = crema_map.get(parts[2])

            rows.append([
                os.path.join(root, file),
                emotion,
                "crema_d"
            ])

# --------------------------
# Save CSV
# --------------------------

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "audio_path",
        "emotion",
        "dataset"
    ])

    writer.writerows(rows)

print("=" * 50)
print("Metadata created successfully")
print("Samples :", len(rows))
print("Saved to :", OUTPUT_CSV)