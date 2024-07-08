import os

import keras
from keras import layers
from sklearn.model_selection import train_test_split

from load_data import *

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# repro data
# load_raster()

used = pd.read_csv("data/raster-rank.csv")
print(used)
x_keys = list(used.keys())
x_keys.remove("district")
x_keys.remove("rank")
x = used[x_keys]
y = used["rank"]
print(x)
print(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=100, shuffle=True)
