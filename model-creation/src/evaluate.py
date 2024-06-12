import sys
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List


def get_training_plot(model_history: dict) -> plt.Figure:
    """Plot the training and validation loss"""
    epochs = range(1, len(model_history["loss"]) + 1)

    fig = plt.figure(figsize=(10, 4))
    plt.plot(epochs, model_history["loss"], label="Training loss")
    plt.plot(epochs, model_history["val_loss"], label="Validation loss")
    plt.xticks(epochs)
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    return fig


def get_confusion_matrix_plot(
        model: tf.keras.Model, ds_test: tf.data.Dataset, labels: List[str]
    ) -> plt.Figure:
        """Plot the confusion matrix"""
        fig = plt.figure(figsize=(6, 6), tight_layout=True)
        preds = model.predict(ds_test)
    
        conf_matrix = tf.math.confusion_matrix(
            labels=tf.concat([y for _, y in ds_test], axis=0),
            predictions=tf.argmax(preds, axis=1),
            num_classes=len(labels),
        )

        # Plot the confusion matrix
        conf_matrix = conf_matrix / tf.reduce_sum(conf_matrix, axis=1)
        plt.imshow(conf_matrix, cmap="Blues")

        # Plot cell values
        for i in range(len(labels)):
            for j in range(len(labels)):
                value = conf_matrix[i, j].numpy()
                if value == 0:
                    color = "lightgray"
                elif value > 0.5:
                    color = "white"
                else:
                    color = "black"
                plt.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )

        plt.colorbar()
        plt.xticks(range(len(labels)), labels, rotation=90)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion matrix")

        return fig


def evaluate_model(model, X_test, labels, model_folder):
    # Create folders
    evaluation_folder = Path("evaluation")
    plots_folder = Path("plots")
    (evaluation_folder / plots_folder).mkdir(parents=True, exist_ok=True)

    # Load model history
    model_history = np.load(model_folder / "history.npy", allow_pickle=True).item()
    
    # Log metrics
    val_loss, val_acc = model.evaluate(X_test)
    print(f"Validation loss: {val_loss:.2f}")
    print(f"Validation accuracy: {val_acc * 100:.2f}%")
    with open(evaluation_folder / "metrics.json", "w") as f:
        json.dump({"val_loss": val_loss, "val_acc": val_acc}, f)

    # Save training history plot
    fig = get_training_plot(model_history)
    fig.savefig(evaluation_folder / plots_folder / "training_history.png")

    # Save confusion matrix plot
    fig = get_confusion_matrix_plot(model, X_test, labels)
    fig.savefig(evaluation_folder / plots_folder / "confusion_matrix.png")


def main():
    # Parse command-line arguments
    if len(sys.argv) != 3:
        print("Arguments error. Usage:\n")
        print("\tpython3 evaluate.py <model-folder> <prepared-dataset-folder>\n")
        exit(1)

    # Parse arguments
    model_folder = Path(sys.argv[1])
    prepared_dataset_folder = Path(sys.argv[2])

    # Load files
    ds_test = tf.data.Dataset.load(str(prepared_dataset_folder / "test"))
    labels = None
    with open(prepared_dataset_folder / "labels.json") as f:
        labels = json.load(f)

    # Load model
    model = tf.keras.models.load_model(str(model_folder / "planet_recognition_model.h5"))

    # Evaluate the model using the test dataset
    evaluate_model(model, ds_test, labels, model_folder)


if __name__ == "__main__":
    main()
