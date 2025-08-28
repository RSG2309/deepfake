import os, json, numpy as np, tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ====== CONFIG ======
DATA_DIR = r"data/deepfake-detection-kukoh-1"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

AUTOTUNE = tf.data.AUTOTUNE

def make_ds(split):
    path = os.path.join(DATA_DIR, split)
    return tf.keras.utils.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="binary",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=42
    )

train_ds = make_ds("train")
val_ds   = make_ds("valid")
test_ds  = make_ds("test")

# save class names
class_names = train_ds.class_names
with open(os.path.join(ARTIFACTS_DIR, "class_names.json"), "w") as f:
    json.dump(class_names, f)
print("Classes:", class_names)

# preprocessing
def prep(ds, training=False):
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(lambda x, y: (tf.image.random_flip_left_right(x), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    return ds.cache().prefetch(AUTOTUNE)

train_ds = prep(train_ds, training=True)
val_ds   = prep(val_ds)
test_ds  = prep(test_ds)

# model
base = MobileNetV2(input_shape=IMG_SIZE+(3,), include_top=False, weights="imagenet")
base.trainable = False

inputs = layers.Input(shape=IMG_SIZE+(3,))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = models.Model(inputs, outputs)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
cbs = [callbacks.ModelCheckpoint(os.path.join(ARTIFACTS_DIR,"model.keras"), save_best_only=True)]

print("training...")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cbs)

print("evaluating...")
loss, acc = model.evaluate(test_ds)
print(f"Test accuracy: {acc:.4f}")
