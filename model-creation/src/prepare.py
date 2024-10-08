import json
import sys
from pathlib import Path
import tensorflow as tf
import yaml

from utils.seed import set_seed


def main() -> None:
    if len(sys.argv) != 3:
        print("Arguments error. Usage:\n")
        print("\tpython3 prepare.py <raw-dataset-folder> <prepared-dataset-folder>\n")
        exit(1)

    # Load parameters
    prepare_params = yaml.safe_load(open("params.yaml"))["prepare"]

    raw_dataset_folder = Path(sys.argv[1])
    prepared_dataset_folder = Path(sys.argv[2])
    seed = prepare_params["seed"]
    split = prepare_params["split"]
    image_size = prepare_params["image_size"]
    grayscale = prepare_params["grayscale"]

    # Set seed for reproducibility
    set_seed(seed)

    # Read data
    ds_train, ds_test = tf.keras.utils.image_dataset_from_directory(
        raw_dataset_folder,
        labels="inferred",
        label_mode="int",
        color_mode="grayscale" if grayscale else "rgb",
        batch_size=32,
        image_size=image_size,
        shuffle=True,
        seed=seed,
        validation_split=split,
        subset="both",
    )
    labels = ds_train.class_names

    if not prepared_dataset_folder.exists():
        prepared_dataset_folder.mkdir(parents=True)

    # Normalize the data
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(
        1.0 / 255
    )
    ds_train = ds_train.map(lambda x, y: (normalization_layer(x), y))
    ds_test = ds_test.map(lambda x, y: (normalization_layer(x), y))

    # Save the prepared dataset
    with open(prepared_dataset_folder / "labels.json", "w") as f:
        json.dump(labels, f)
    tf.data.Dataset.save(ds_train, str(prepared_dataset_folder / "train"))
    tf.data.Dataset.save(ds_test, str(prepared_dataset_folder / "test"))

    print(f"\nDataset saved at {prepared_dataset_folder.absolute()}")


if __name__ == "__main__":
    main()
