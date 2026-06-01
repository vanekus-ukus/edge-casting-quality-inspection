from __future__ import annotations

from pathlib import Path


def build_keras_micro_model(input_size: int = 96):
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError("TensorFlow is required for TFLite export. Install with: pip install '.[tflite]'") from exc

    inputs = tf.keras.Input(shape=(input_size, input_size, 3), name="image")
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    for filters in (16, 32, 64):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax", name="probabilities")(x)
    return tf.keras.Model(inputs, outputs)


def train_keras_micro_model(
    train_dir: str | Path,
    val_dir: str | Path,
    output_weights: str | Path,
    input_size: int = 96,
    epochs: int = 5,
    batch_size: int = 64,
) -> Path:
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError("TensorFlow is required for TFLite export. Install with: pip install '.[tflite]'") from exc

    class_names = ["ok", "defective"]
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        class_names=class_names,
        image_size=(input_size, input_size),
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        class_names=class_names,
        image_size=(input_size, input_size),
        batch_size=batch_size,
        shuffle=False,
    )
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    model = build_keras_micro_model(input_size)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=2)
    out = Path(output_weights)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(out))
    return out


def export_micro_edge_fp32(output_path: str | Path, input_size: int = 96, keras_weights: str | Path | None = None) -> Path:
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError("TensorFlow is required for TFLite export. Install with: pip install '.[tflite]'") from exc

    model = build_keras_micro_model(input_size)
    if keras_weights:
        model.load_weights(str(keras_weights))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(tflite_model)
    return out
