# cleaner.py
import os
import csv

LANG_FOLDER = "es" # others are: es, fr, en, de, pl, ru, fi, hu, ar, in, as in spanish, french, english,
# german, polish, russian, finnish, hungarian, arabic, indonesian respectively
# this also takes crazy long thats why i didnt just iterate through these

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LANG_PATH = BASE_DIR if LANG_FOLDER in (".", "") else os.path.join(BASE_DIR, LANG_FOLDER)

CLIPS_DIR = os.path.join(LANG_PATH, "clips")
VALIDATED_FILE = os.path.join(LANG_PATH, "validated.tsv")
DURATION_FILE = os.path.join(LANG_PATH, "clip_durations.tsv")
MAX_DURATION = 4.0  # seconds

def read_validated():
    validated = set()
    with open(VALIDATED_FILE, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            validated.add(row["path"])
    return validated

def read_durations():
    durations = {}
    with open(DURATION_FILE, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        fieldnames = reader.fieldnames or []
        field_map = {n.lower(): n for n in fieldnames}

        path_key = next((field_map[k] for k in field_map if "path" in k or "clip" in k or "file" in k), None)
        dur_key = next((field_map[k] for k in field_map if "duration" in k), None)

        for row in reader:
            raw_path = row.get(path_key, "")
            raw_dur = row.get(dur_key, "")
            if not raw_path or not raw_dur:
                continue
            try:
                dur = float(raw_dur)
            except ValueError:
                continue

            # néhány valahogy miliszekundumban van
            if dur > 1000:
                dur = dur / 1000.0

            durations[raw_path] = dur
    return durations

def clean_clips():
    validated = read_validated()
    durations = read_durations()
    kept = 0
    deleted = 0

    for file in os.listdir(CLIPS_DIR):
        path = os.path.join(CLIPS_DIR, file)
        if file not in validated:
            os.remove(path)
            deleted += 1
        elif file not in durations:
            os.remove(path)
            deleted += 1
        elif durations[file] > MAX_DURATION:
            os.remove(path)
            deleted += 1
        else:
            kept += 1

    print(f"Finished cleaning clips in {LANG_FOLDER}")
    print(f"Kept: {kept}, Deleted: {deleted}")

if __name__ == "__main__":
    clean_clips()