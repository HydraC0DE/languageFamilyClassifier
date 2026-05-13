import os
import tensorflow as tf
from data_loader import make_splits
from model import build_crnn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


EPOCHS = 10 # way too little, but cuda doesnt work :(
BATCH_SIZE = 32

WEIGHTS_DIR = "saved_weights"
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "crnn.weights.h5")



# TRAINING


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



# EVALUATION


def evaluate(model, test_ds):
    print("Evaluating on test set...")
    loss, acc = model.evaluate(test_ds)
    print(f"\nTest Accuracy: {acc:.4f}")
    print(f"Test Loss: {loss:.4f}")

def plot_confusion_matrix(model, test_ds, class_names):
    # Collect true labels and predictions
    y_true = []
    y_pred = []

    for batch_x, batch_y in test_ds:
        preds = model.predict(batch_x)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(batch_y.numpy(), axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix — Language Families")
    plt.show()



if __name__ == "__main__":
    # If weights exist, then load model, if not, train and save model
    if os.path.exists(WEIGHTS_PATH):
        print("Found saved weights. Loading model...")

        model = build_crnn()
        model.load_weights(WEIGHTS_PATH)

        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        print("Loading datasets for evaluation...")
        _, _, test_ds = make_splits(batch_size=BATCH_SIZE)

        evaluate(model, test_ds)
        FAMILY_NAMES = ["Romance", "Germanic", "Slavic", "FinnoUgric", "Austronesian", "Semitic"]

        plot_confusion_matrix(model, test_ds, FAMILY_NAMES)


    else:
        print("No saved weights found. Training new model...")
        model, test_ds = train_and_save()
        evaluate(model, test_ds)
        FAMILY_NAMES = ["Romance", "Germanic", "Slavic", "FinnoUgric", "Austronesian", "Semitic"]

        plot_confusion_matrix(model, test_ds, FAMILY_NAMES)

