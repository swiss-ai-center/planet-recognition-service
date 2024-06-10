import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import yaml

from utils.seed import set_seed
from pathlib import Path
from typing import Tuple


def get_model(
    image_shape: Tuple[int, int, int],
    conv_size: int,
    dense_size: int,
    output_classes: int,
) -> tf.keras.Model:
    """Create a simple CNN model"""
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                conv_size, (3, 3), activation="relu", input_shape=image_shape
            ),
            tf.keras.layers.MaxPooling2D((3, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(dense_size, activation="relu"),
            tf.keras.layers.Dense(output_classes),
        ]
    )
    return model


def build_model(image_shape: Tuple[int, int, int]) -> tf.keras.Model:
    '''
    FIXME: refactor
    '''
    train_params = yaml.safe_load(open("params.yaml"))["train"]

    seed = train_params["seed"]
    lr = train_params["lr"]
    # epochs = train_params["epochs"]
    conv_size = train_params["conv_size"]
    dense_size = train_params["dense_size"]
    output_classes = train_params["output_classes"]

    # Define the layers
    conv_layer = tf.keras.layers.Conv2D(conv_size, (3, 3), activation="relu", input_shape=image_shape)
    maxpool_layer = tf.keras.layers.MaxPool2D((3, 3))
    flatten_layer = tf.keras.layers.Flatten()
    dense_layer = tf.keras.layers.Dense(dense_size, activation="relu")
    output_layer = tf.keras.layers.Dense(output_classes)

    # Set seed
    set_seed(seed)

    # Create the model
    model = tf.keras.models.Sequential(conv_layer, maxpool_layer, flatten_layer, dense_layer, output_layer)

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    model.summary()

    return model


def train_model(X_train, X_test, epochs, conv_size, dense_size, output_classes):
    '''
    FIXME: refactor
    '''
    # Fit the model to images data
    model = build_model(X_train.shape[1], conv_size, dense_size, output_classes)
    history = model.fit(X_train, X_train, epochs=10, batch_size=32)

    model.fit(
        X_train,
        epochs=epochs,
        validation_data=X_test,
    )

    # Save the model
    Path("model").mkdir(parents=True, exist_ok=True)
    model.save("model/planet_recognition_model.h5", save_format='h5')

    # Save the model history
    np.save("model/history.npy", model.history.history)

    return history


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

    # Load data
    ds_train = tf.data.Dataset.load(str(prepared_dataset_folder / "train"))
    ds_test = tf.data.Dataset.load(str(prepared_dataset_folder / "test"))

    # Define the model
    model = get_model(image_shape, conv_size, dense_size, output_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    model.summary()

    # Train the model
    model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_test,
    )

    # Save the model
    model_folder.mkdir(parents=True, exist_ok=True)
    model.save(str(model_folder))

    # Save the model history
    np.save(model_folder / "history.npy", model.history.history)

    plot_loss(model.history)

    print(f"\nModel saved at {model_folder.absolute()}")


if __name__ == "__main__":
    main()
