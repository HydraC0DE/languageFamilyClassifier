import tensorflow as tf
from tensorflow.keras import layers, models

# ============================
# CONFIG
# ============================

INPUT_SHAPE = (128, 125, 1)   # mel bins × time frames × channel
NUM_FAMILIES = 6              # Romance, Germanic, Slavic, FinnoUgric, Austronesian


# ============================
# MODEL
# ============================

def build_crnn(input_shape=INPUT_SHAPE, num_families=NUM_FAMILIES):
    inputs = layers.Input(shape=input_shape)

    # --- CNN feature extractor ---
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # --- reshape for GRU ---
    # x.shape = (batch, h, w, c)
    h = x.shape[1]
    w = x.shape[2]
    c = x.shape[3]

    x = layers.Reshape((h, w * c))(x)

    # --- GRU temporal modeling ---
    x = layers.GRU(128)(x)
    x = layers.Dropout(0.3)(x)

    # --- classification head ---
    outputs = layers.Dense(num_families, activation="softmax")(x)

    return models.Model(inputs, outputs)


if __name__ == "__main__":
    model = build_crnn()
    model.summary()
