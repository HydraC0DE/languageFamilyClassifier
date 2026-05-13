LanguageFamilyClassifier

This project builds a CRNN that classifies short speech clips into the correct language family.
Each audio sample is trimmed or padded to 4 seconds, converted into a 128×125 mel‑spectrogram, and fed into a CNN -> GRU architecture that predicts one of six language families.

cleaner.py — Keeps only validated audio clips and deletes anything too long, missing metadata, or low‑quality.

intoSpectogram.py — Converts every cleaned audio file into a fixed‑size 4‑second mel‑spectrogram.

data_loader.py — Loads all spectrogram .npy files, assigns labels, shuffles them, and builds train/val/test datasets.

familyMap.py — Maps each language code to its corresponding language family ID.

model.py — Defines the CRNN model that takes a spectrogram and predicts one of six language families.

run.py — Trains the model if no weights exist, otherwise loads them and evaluates on the test set.

```
LANGUAGEFAMILYCLASSIFIER/
    ar/ (Mozilla Common Voices)
        clips/
        clip_durations.tsv
        dev.tsv
        invalidated.tsv
        other.tsv
        reported.tsv
        test.tsv
        train.tsv
        unvalidated_sentences.tsv
        validated.tsv
        validated_sentences.tsv
    de/
    en/
    es/
    fi/
    fr/
    hu/
    id/
    pl/
    ru/
    (potentially other languages from MCV)
    processed_data/ (data created by intoSpectogram.py)
    saved_weights/
    cleaner.py
    cuda_tester.py (trash)
    data_loader.py
    familyMap.py
    intoSpectogram.py
    model.py
    run.py
    temp_debug.py (trash)
```
