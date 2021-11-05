# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Dict, Tuple

import click
import tensorflow as tf
from dotenv import find_dotenv, load_dotenv
from loguru import logger

logger.add("../reports/debug.log")


def get_raw_data(
    data_dir: Path,
    url: str = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",  # noqa: E501
) -> None:
    download = True
    for directory in data_dir.glob("**"):
        if "flower_photos" in str(directory) and download:
            logger.info(f"found flowers in {directory}, not downloading again")
            download = False

    if download:
        logger.info("Data not present, downloading from url")
        tf.keras.utils.get_file(
            origin=url,
            fname="flower_photos",
            cache_dir=data_dir,
            cache_subdir="",
            untar=True,
        )


def create_generators(
    datagen_kwargs: Dict, dataflow_kwargs: Dict, config: Dict, augment: bool = False
) -> Tuple[
    tf.keras.preprocessing.image.ImageDataGenerator,
    tf.keras.preprocessing.image.ImageDataGenerator,
]:
    logger.info("Creating validation set data generator")
    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**datagen_kwargs)

    valid = valid_datagen.flow_from_directory(
        directory=config["data_dir"],
        subset="validation",
        shuffle=False,
        **dataflow_kwargs,
    )

    logger.info("Creating train set data generator")
    if augment:
        # to squeeze more out your data, modify images.
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=40,  # rotate some degrees
            horizontal_flip=True,  # flip
            width_shift_range=0.2,  # shift
            height_shift_range=0.2,
            zoom_range=0.2,  # zoom
            **datagen_kwargs,
        )
    else:
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagen_kwargs
        )

    train = train_datagen.flow_from_directory(
        config["data_dir"], subset="training", shuffle=True, **dataflow_kwargs
    )
    return train, valid


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: Path, output_filepath: Path) -> None:
    """Runs data processing scripts to turn raw data from (../raw) into cleaned
    data ready to be analyzed (saved in ../processed).
    """
    pass


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
