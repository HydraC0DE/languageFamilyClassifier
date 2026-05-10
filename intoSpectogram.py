import os
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
from familyMap import family_map
import random

# =====================
# CONFIG
# =====================

LANG_FOLDERS = sorted(family_map.keys())
FAMILY_CLASSES = sorted(set(family_map.values()))

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR
OUTPUT_DIR = BASE_DIR / "processed_data"

SAMPLE_RATE = 16000
CLIP_DURATION = 4.0              # seconds
MAX_LENGTH = int(SAMPLE_RATE * CLIP_DURATION)

SUPPORTED_EXTS = [".wav", ".mp3", ".flac", ".ogg"]

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512

# =====================
# AUDIO PROCESSING
# =====================

def load_and_fix_length(filepath):
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)

    if len(y) > MAX_LENGTH:
        y = y[:MAX_LENGTH]
    elif len(y) < MAX_LENGTH:
        pad_needed = MAX_LENGTH - len(y)
        start_pos = random.randint(0, pad_needed)
        padded = np.zeros(MAX_LENGTH, dtype=y.dtype)
        padded[start_pos:start_pos + len(y)] = y
        y = padded

    return y


def create_log_mel_spectrogram(audio):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    # Convert power spectrogram -> log scale (better for ML)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Normalize (helps training)
    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)

    # Force fixed time dimension, some inputes were 125 frames, some 126, some 124.
    TARGET_FRAMES = 125
    current_frames = log_mel.shape[1]

    if current_frames > TARGET_FRAMES:
        log_mel = log_mel[:, :TARGET_FRAMES]
    elif current_frames < TARGET_FRAMES:
        pad_width = TARGET_FRAMES - current_frames
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')

    return log_mel.astype(np.float32)


# =====================
# MAIN CONVERSION
# =====================

def process_language(lang):
    clips_dir = BASE_DIR / lang / "clips"

    if not clips_dir.exists():
        print(f"Missing: {clips_dir}")
        return

    family = family_map.get(lang)
    if family is None:
        print(f"No family mapping for language {lang}")
        return

    files = []
    for ext in SUPPORTED_EXTS:
        files.extend(clips_dir.glob(f"*{ext}"))
    files = sorted(files)

    out_dir = OUTPUT_DIR / family
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{lang} -> {family}: {len(files)} files")

    for file in tqdm(files):
        try:
            audio = load_and_fix_length(file)

            features = create_log_mel_spectrogram(audio)

            output_file = out_dir / f"{lang}_{file.stem}.npy"

            np.save(output_file, features)

        except Exception as e:
            print(f"Error processing {file}: {e}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    for lang in LANG_FOLDERS:
        process_language(lang)

    print("\nDone.")
    print("Output saved to:")
    print(OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()