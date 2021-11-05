import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from loguru import logger

# from linting import Any

logger.add("../reports/debug.log")


def get_class_string_from_index(
    index: int, generator: tf.keras.preprocessing.image.ImageDataGenerator
) -> str:
    for class_string, class_index in generator.class_indices.items():
        if class_index == index:
            return str(class_string)
    return "no class"


def evaluate_image_classifier(
    model: tf.keras.Model,
    generator: tf.keras.preprocessing.image.ImageDataGenerator,
    grid: int = 25,
) -> None:
    """this is to evaluate a trained model by eyeballing some sample predictions"""
    images, label = next(generator)  # gets a batch of images
    prediction_scores = model.predict(images)  # makes predictions

    # translate one-hot to indexes with argmax
    predicted_index = np.argmax(prediction_scores, axis=1)
    true_index = np.argmax(label, axis=1)
    # mismatches = true_index != predicted_index

    plt.figure(figsize=(16, 16))

    gridn = int(np.ceil(np.sqrt(grid)))
    for i in range(grid):
        plt.subplot(gridn, gridn, i + 1)
        plt.imshow(images[i])
        plt.title(
            "predict:"
            + get_class_string_from_index(predicted_index[i], generator)
            + " (true:"
            + get_class_string_from_index(true_index[i], generator)
            + ")"
        )
        plt.axis("off")


def plot_batch(
    generator: tf.keras.preprocessing.image.ImageDataGenerator, grid: int = 9
) -> None:
    inv_map = {v: k for k, v in generator.class_indices.items()}

    plt.figure(figsize=(10, 10))
    image, label = next(generator)
    logger.info(f"image shape: {image.shape}")
    logger.info(f"label shape: {label.shape}")
    gridn = int(np.ceil(np.sqrt(grid)))
    for i in range(grid):
        plt.subplot(gridn, gridn, i + 1)
        plt.imshow(image[i])
        plt.title(inv_map[np.argmax(label[i])])
        plt.axis("off")
