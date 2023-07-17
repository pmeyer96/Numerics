import tensorflow_datasets as tfds

from keras.models import Sequential
from keras.layers import (Flatten, Dense)
from keras.utils import to_categorical

import numpy as np

import pandas as pd

import tensorflow as tf


(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
