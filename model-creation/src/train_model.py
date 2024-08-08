import os
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL.Image import Image
import yaml
import bentoml

from utils.seed import set_seed
from pathlib import Path
from typing import Tuple


def create_model(
    image_shape: Tuple[int, int, int],
    conv_size: int,
    dense_size: int,
    output_classes: int,
) -> tf.keras.Model:
    """Create a simple CNN model"""

    # Define the layers
    conv_layer = tf.keras.layers.Conv2D(conv_size, (3, 3), activation="relu", input_shape=image_shape)
    maxpool_layer = tf.keras.layers.MaxPool2D((3, 3))
    flatten_layer = tf.keras.layers.Flatten()
    dense_layer = tf.keras.layers.Dense(dense_size, activation="relu")
    output_layer = tf.keras.layers.Dense(output_classes)

    # Create the model
    model = tf.keras.models.Sequential(
        [conv_layer, maxpool_layer, flatten_layer, dense_layer, output_layer],
    )
    return model


def build_model(model: tf.keras.Model, lr: float) -> tf.keras.Model:
    # Buid model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    model.summary()
    return model

def train_model(model:  tf.keras.Model, epochs: int, ds_train: tf.data.Dataset, ds_test: tf.data.Dataset) -> tf.keras.Model:
    # Train the model
    model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_test,
    )
    return model

def get_labels(prepared_dataset_folder: Path) -> dict:
    with open(prepared_dataset_folder / "labels.json") as f:
        labels = json.load(f)
    return labels

def save_model(model: tf.keras.Model, model_folder: Path, model_name: str, image_size: int, grayscale: str, labels: dict) -> None:
    # Save the model
    model_folder.mkdir(parents=True, exist_ok=True)

    def preprocess(x: Image):
        # convert PIL image to tensor
        x = x.convert('L' if grayscale else 'RGB')
        x = x.resize(image_size)
        x = np.array(x)
        x = x / 255.0
        # add batch dimension
        x = np.expand_dims(x, axis=0)
        return x

    def postprocess(x: Image):
        return {
            "prediction": labels[tf.argmax(x, axis=-1).numpy()[0]],
            "probabilities": {
                labels[i]: prob
                for i, prob in enumerate(tf.nn.softmax(x).numpy()[0].tolist())
            },
        }

    # Save the model using BentoML to its model store
    # https://docs.bentoml.com/en/latest/reference/frameworks/keras.html#bentoml.keras.save_model
    bentoml.keras.save_model(
        model_name,
        model,
        include_optimizer=True,
        custom_objects={
            "preprocess": preprocess,
            "postprocess": postprocess,
        }
    )

    # Export the model from the model store to the local model folder
    bentoml.models.export_model(
        model_name + ":latest",
        "model/" + model_name + ".bentomodel",
    )

    # Save the model history
    np.save(model_folder / "history.npy", model.history.history)
    plot_loss(model.history)

    print(f"\nModel saved at {model_folder.absolute()}")


def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    # Save the plot
    Path("evaluation").mkdir(parents=True, exist_ok=True)
    plt.savefig('evaluation/training_loss.png')


def main() -> None:
    # Load parameters
    prepare_params = yaml.safe_load(open("params.yaml"))["prepare"]
    train_params = yaml.safe_load(open("params.yaml"))["train"]

    image_size = prepare_params["image_size"]
    grayscale = prepare_params["grayscale"]
    image_shape = (*image_size, 1 if grayscale else 3)

    seed = train_params["seed"]
    lr = train_params["lr"]
    epochs = train_params["epochs"]
    conv_size = train_params["conv_size"]
    dense_size = train_params["dense_size"]
    output_classes = train_params["output_classes"]

    set_seed(seed)

    # data
    prepared_dataset_folder = Path("data/prepared")
    model_folder = Path("model")
    model_name = "celestial_bodies_classifier_model"

    # Load data
    ds_train = tf.data.Dataset.load(str(prepared_dataset_folder / "train"))
    ds_test = tf.data.Dataset.load(str(prepared_dataset_folder / "test"))

    # Define the model
    model = create_model(image_shape, conv_size, dense_size, output_classes)
    model = build_model(model, lr)
    model = train_model(model, epochs, ds_train, ds_test)

    labels = get_labels(prepared_dataset_folder)
    save_model(model, model_folder, model_name, image_size, grayscale, labels)


if __name__ == "__main__":
    main()
