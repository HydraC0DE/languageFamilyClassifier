import os
import tensorflow as tf
from data_loader import make_splits
from model import build_crnn

# ============================
# CONFIG
# ============================

EPOCHS = 30
BATCH_SIZE = 32

WEIGHTS_DIR = "saved_weights"
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "crnn.weights.h5")


# ============================
# TRAINING
# ============================

def train_and_save():
    print("Loading datasets...")
    train_ds, val_ds, test_ds = make_splits(batch_size=BATCH_SIZE)

    print("Building model...")
    model = build_crnn()

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            WEIGHTS_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True
        )
    ]

    print("Starting training...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    print("Training complete.")
    return model, test_ds


# ============================
# EVALUATION
# ============================

def evaluate(model, test_ds):
    print("Evaluating on test set...")
    loss, acc = model.evaluate(test_ds)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test Loss: {loss:.4f}")


# ============================
# MAIN LOGIC
# ============================

if __name__ == "__main__":
    # If weights exist → load model
    if os.path.exists(WEIGHTS_PATH):
        print("Found saved weights. Loading model...")

        model = build_crnn()
        model.load_weights(WEIGHTS_PATH)

        print("Loading datasets for evaluation...")
        _, _, test_ds = make_splits(batch_size=BATCH_SIZE)

        evaluate(model, test_ds)

    else:
        print("No saved weights found. Training new model...")
        model, test_ds = train_and_save()
        evaluate(model, test_ds)
