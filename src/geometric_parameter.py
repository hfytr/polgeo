import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import keras
from keras import layers
from sklearn.model_selection import train_test_split

from load_data import *

# repro data
# data = load_to_file(
#     "data/training_labels.RData", "data/full-parsed-data.shp", "train_labels"
# )
data = gp.read_file("data/full-parsed-data.shp")
x = data[
    [
        "area",
        "perimeter",
        "con_hull",
        "reock",
        "len_width",
        "polsby_pop",
    ]
]
y = data["rank"] / 100.0
for key in data.keys():
    print(key + " " + str(type(data[key].iloc[0])))
print(x.describe())
print(y.describe())
model = keras.Sequential(
    [
        layers.Dense(10, activation="relu"),
        layers.Dense(10, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.MeanSquaredLogarithmicError(),
    metrics=[keras.metrics.MeanSquaredError()],
)
print(model.summary())
history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.02)
print(model.summary())
print(history.history)
