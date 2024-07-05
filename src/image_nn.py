import os

import keras
from keras import layers
from sklearn.model_selection import train_test_split

from load_data import *

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# repro data
data = load_to_file(
    "data/training_labels.RData",
    "data/full-parsed-data.shp",
    "data/full-parsed-pixels.csv",
    "train_labels",
)
data = pd.read_csv("data/full-parsed-pixels.csv")
