import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from familyMap import family_map

# ============================
# CONFIG
# ============================

SCRIPT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = SCRIPT_DIR / "processed_data"

FAMILIES = sorted(set(family_map.values()))
FAMILY_TO_IDX = {fam: i for i, fam in enumerate(FAMILIES)}

INPUT_SHAPE = (128, 125, 1)   # mel bins × time frames × channel


# ============================
# LOAD SINGLE FILE
# ============================

def load_npy(path):
    spec = np.load(path)  # shape: (128, 125)
    spec = np.expand_dims(spec, axis=-1)  # -> (128, 125, 1)
    return spec.astype(np.float32)


# ============================
# BUILD DATASET LIST
# ============================

def collect_files():
    items = []

    for family in FAMILIES:
        fam_dir = PROCESSED_DIR / family
        if not fam_dir.exists():
            continue

        for file in fam_dir.glob("*.npy"):
            label = FAMILY_TO_IDX[family]
            items.append((str(file), label))

    return items


# ============================
# TF.DATA PIPELINE
# ============================

def make_dataset(batch_size=32, shuffle=True):

    items = collect_files()
    filepaths = [fp for fp, _ in items]
    labels = [lbl for _, lbl in items]

    filepaths = tf.constant(filepaths)
    labels = tf.constant(labels, dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    # load .npy inside tf.data
    def load_fn(path, label):
        spec = tf.numpy_function(load_npy, [path], tf.float32)
        spec.set_shape(INPUT_SHAPE)
        return spec, tf.one_hot(label, depth=len(FAMILIES))

    ds = ds.map(load_fn, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(items))

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# train val test split

def make_splits(batch_size=32, val_ratio=0.1, test_ratio=0.1):
    items = collect_files()
    total = len(items)

    rng = np.random.default_rng(67) 
    rng.shuffle(items)

    n_test = int(total * test_ratio)
    n_val = int(total * val_ratio)

    test_items = items[:n_test]
    val_items = items[n_test:n_test + n_val]
    train_items = items[n_test + n_val:]

    def build_from_list(subset):
        fps = tf.constant([fp for fp, _ in subset])
        lbls = tf.constant([lbl for _, lbl in subset], dtype=tf.int32)

        ds = tf.data.Dataset.from_tensor_slices((fps, lbls))

        def load_fn(path, label):
            spec = tf.numpy_function(load_npy, [path], tf.float32)
            spec.set_shape(INPUT_SHAPE)
            return spec, tf.one_hot(label, depth=len(FAMILIES))

        ds = ds.map(load_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    return (
        build_from_list(train_items),
        build_from_list(val_items),
        build_from_list(test_items)
    )


if __name__ == "__main__":
    train_ds, val_ds, test_ds = make_splits()
    print("Train batches:", len(train_ds))
    print("Val batches:", len(val_ds))
    print("Test batches:", len(test_ds))
