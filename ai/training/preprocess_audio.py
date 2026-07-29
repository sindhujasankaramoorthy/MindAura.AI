import os
import csv
import librosa
import soundfile as sf

BASE_DIR = os.path.dirname(__file__)

METADATA_CSV = os.path.join(
    BASE_DIR,
    "datasets",
    "metadata",
    "metadata.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "processed_audio"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def preprocess_audio(input_path, output_path):
    """
    Convert audio to:
    - Mono
    - 16 kHz
    - Normalized
    """

    audio, sr = librosa.load(
        input_path,
        sr=16000,
        mono=True
    )

    peak = max(abs(audio))

    if peak > 0:
        audio = audio / peak

    sf.write(
        output_path,
        audio,
        16000
    )


processed = 0
failed = 0

with open(METADATA_CSV, newline="", encoding="utf-8") as f:

    reader = csv.DictReader(f)

    # ---------- DEBUG MODE ----------
    # Change/remove this later after debugging.
    for i, row in enumerate(reader):

        input_audio = row["audio_path"]
        emotion = row["emotion"]
        dataset = row["dataset"]

        emotion_folder = os.path.join(
            OUTPUT_DIR,
            emotion
        )

        os.makedirs(
            emotion_folder,
            exist_ok=True
        )

        filename = dataset + "_" + os.path.basename(input_audio)

        output_audio = os.path.join(
            emotion_folder,
            filename
        )

        try:

            preprocess_audio(
                input_audio,
                output_audio
            )

            processed += 1

            print(f"Processed: {filename}")

        except Exception as e:

            failed += 1

            print("\n" + "=" * 70)
            print("FAILED FILE:")
            print(input_audio)
            print("\nERROR:")
            print(repr(e))
            print("=" * 70)


print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print(f"Processed : {processed}")
print(f"Failed    : {failed}")
print(f"Saved To  : {OUTPUT_DIR}")