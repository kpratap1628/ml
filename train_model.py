"""
Train the HAM10000 7-class CNN and export artifacts for the API.

Input:  data/hmnist_28_28_RGB.csv   (Kaggle "skin-cancer-mnist-ham10000")
Output: models/ham10000_cnn.keras
        models/preprocessing.json   (scale/mean/std used for standardization)

Splitting happens on the ORIGINAL data before any resampling or augmentation,
so the validation and test sets stay honest. Oversampling is applied only to
the training partition.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential

SEED = 1
BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "data" / "hmnist_28_28_RGB.csv"
OUT = BASE_DIR / "models"
MODEL_PATH = OUT / "ham10000_cnn.keras"
STATS_PATH = OUT / "preprocessing.json"

if not DATA.exists():
    raise SystemExit(f"Missing dataset: place hmnist_28_28_RGB.csv at {DATA}")

OUT.mkdir(exist_ok=True)

print(f"Loading {DATA} ...")
df = pd.read_csv(DATA)
y = df["label"].astype("int64").to_numpy()
x = df.drop(columns=["label"]).to_numpy(dtype=np.float32).reshape(-1, 28, 28, 3) / 255.0

X_train, X_rest, y_train, y_rest = train_test_split(
    x, y, test_size=0.3, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_rest, y_rest, test_size=2 / 3, random_state=SEED, stratify=y_rest
)
print(f"Splits: train={len(X_train)} val={len(X_val)} test={len(X_test)}")

mean = float(X_train.mean())
std = float(X_train.std())

sampler = RandomOverSampler(random_state=SEED)
X_train, y_train = sampler.fit_resample(X_train.reshape(len(X_train), -1), y_train)
X_train = X_train.reshape(-1, 28, 28, 3)
print(f"After oversampling: train={len(X_train)}")


def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    img = tf.image.rot90(img, k)
    img = tf.image.random_brightness(img, 0.08)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return (img - mean) / std, label


def standardize(img, label):
    return (img - mean) / std, label


def make_ds(features, labels, training):
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    if training:
        ds = ds.shuffle(min(len(features), 20_000), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(standardize, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(128).prefetch(tf.data.AUTOTUNE)


model = Sequential([
    Conv2D(16, (3, 3), input_shape=(28, 28, 3), activation="relu", padding="same"),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPool2D((2, 2)),
    Dropout(0.2),
    Conv2D(32, (3, 3), activation="relu", padding="same"),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPool2D((2, 2), padding="same"),
    Dropout(0.3),
    Flatten(),
    BatchNormalization(),
    Dense(64, activation="relu"),
    Dropout(0.4),
    Dense(32, activation="relu"),
    Dense(7, activation="softmax"),
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH), monitor="val_accuracy", mode="max",
        save_best_only=True, verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5,
    ),
]
model.fit(
    make_ds(X_train, y_train, training=True),
    validation_data=make_ds(X_val, y_val, training=False),
    epochs=40,
    callbacks=callbacks,
)

loss, acc = model.evaluate(make_ds(X_test, y_test, training=False), verbose=0)
y_pred = np.argmax(model.predict(make_ds(X_test, y_test, training=False), verbose=0), axis=1)
codes = ["akiec", "bcc", "bkl", "df", "nv", "vasc", "mel"]
print(f"\nHold-out test loss: {loss:.4f} | accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred, target_names=codes, digits=4))

model.save(MODEL_PATH)
STATS_PATH.write_text(json.dumps({"scale": 255.0, "mean": mean, "std": std}, indent=2))
print(f"Saved {MODEL_PATH} and {STATS_PATH}")
