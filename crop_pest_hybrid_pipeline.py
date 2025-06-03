
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Channel Attention
def channel_attention(x, ratio=8):
    channel = x.shape[-1]
    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)

    dense1 = layers.Dense(channel // ratio, activation='relu')
    dense2 = layers.Dense(channel)

    avg_out = dense2(dense1(avg_pool))
    max_out = dense2(dense1(max_pool))

    attention = layers.Add()([avg_out, max_out])
    attention = layers.Activation('sigmoid')(attention)
    return layers.Multiply()([x, layers.Reshape((1, 1, channel))(attention)])

# Spatial Attention
def spatial_attention(x):
    avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    attention = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    return layers.Multiply()([x, attention])

# CNN Block
def cnn_attention_block(x, filters):
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = channel_attention(x)
    x = spatial_attention(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    return x

# Vision Transformer Block
def vit_block(x, patch_size=2, projection_dim=64, transformer_layers=2):
    input_shape = x.shape[1:3]
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)

    x = layers.Conv2D(filters=projection_dim, kernel_size=patch_size, strides=patch_size)(x)
    x = layers.Reshape((num_patches, projection_dim))(x)

    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)(positions)
    x = x + position_embedding

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attention_output = layers.MultiHeadAttention(num_heads=4, key_dim=projection_dim)(x1, x1)
        x2 = layers.Add()([attention_output, x])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        ffn = layers.Dense(projection_dim * 2, activation='gelu')(x3)
        ffn = layers.Dense(projection_dim)(ffn)
        x = layers.Add()([ffn, x2])

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return layers.GlobalAveragePooling1D()(x)

# Model Builder
def build_model(input_shape=(224, 224, 3), num_classes=5):
    inputs = Input(shape=input_shape)
    x = preprocess_input(inputs)

    x = layers.RandomFlip('horizontal_and_vertical')(x)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)

    x = cnn_attention_block(x, 32)
    x = cnn_attention_block(x, 64)
    x = cnn_attention_block(x, 128)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Reshape((1, 1, -1))(x)
    x = layers.UpSampling2D(size=(14, 14))(x)

    x = vit_block(x, patch_size=2, projection_dim=64)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)

# Data Loading
def load_data(data_dir, img_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(
        validation_split=0.2,
        preprocessing_function=preprocess_input
    )

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    return train_gen, val_gen

# Evaluation
def evaluate_model(model, val_gen):
    val_gen.reset()
    predictions = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_gen.classes

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=val_gen.class_indices.keys()))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=val_gen.class_indices.keys(), yticklabels=val_gen.class_indices.keys(), cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Metrics Plotting
def plot_metrics(history):
    metrics = ['loss', 'accuracy', 'precision', 'recall']
    for metric in metrics:
        plt.figure()
        plt.plot(history.history[metric], label='Train')
        plt.plot(history.history['val_' + metric], label='Validation')
        plt.title(f"{metric.capitalize()} over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.show()

# Main
if __name__ == "__main__":
    data_dir = "/path/to/your/crop-pest-dataset"  # <-- Replace this with your actual dataset path
    img_size = (224, 224)
    batch_size = 32
    epochs = 30

    train_gen, val_gen = load_data(data_dir, img_size, batch_size)
    model = build_model(input_shape=img_size + (3,), num_classes=train_gen.num_classes)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)
    plot_metrics(history)
    evaluate_model(model, val_gen)
