import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATA_PATH = "data/features/features.npz"
TFLITE_PATH = "data/features/model_kwsp_int8.tflite"
NORM_PATH = "data/features/norm_stats.npz"

data = np.load(DATA_PATH)
X = data["X"].astype(np.float32)
y = data["y"].astype(np.int32)

# normalize
mean = X.mean(axis=0)
std = X.std(axis=0) + 1e-6
Xn = (X - mean) / std

num_classes = len(np.unique(y))
y_oh = tf.keras.utils.to_categorical(y, num_classes)

X_train, X_val, y_train, y_val = train_test_split(
    Xn, y_oh, test_size=0.2, random_state=42, stratify=y
)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(Xn.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()
model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=16,
)

val_loss, val_acc = model.evaluate(X_val, y_val)
print("Validation accuracy:", val_acc)

# save normalization stats
np.savez(NORM_PATH, mean=mean, std=std)
print("Saved norm stats to", NORM_PATH)

# int8 TFLite conversion
def representative_dataset():
    for i in range(min(100, Xn.shape[0])):
        yield [Xn[i : i + 1].astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print("Saved TFLite model to", TFLITE_PATH)

# After training and evaluating
model.save("data/features/model_kwsp.h5")
print("Saved Keras model to data/features/model_kwsp.h5")


