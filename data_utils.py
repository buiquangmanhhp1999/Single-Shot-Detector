import numpy as np
from pathlib import Path
import tensorflow as tf


def create_train_img_file(name="train_img_path.txt"):
    train_dataset_path = Path('./data/train/')
    with open(name, "w") as fi:
        for img_path in train_dataset_path.glob("*.jpeg"):
            fi.write(str(img_path) + "\n")


def create_val_img_file(name="val_img_path.txt"):
    val_dataset_path = Path('./data/validate/')
    with open(name, "w") as fi:
        for img_path in val_dataset_path.glob("*.jpeg"):
            fi.write(str(img_path) + "\n")
