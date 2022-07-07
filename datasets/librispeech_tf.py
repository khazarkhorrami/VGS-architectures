

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

ds = tfds.load('mnist', split='test', shuffle_files=True)
