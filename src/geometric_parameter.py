import math
import os
import sys

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import keras
from keras import backend as K
from keras import layers
from sklearn.model_selection import train_test_split

from load_data import *


# repro data
# data = load_to_file(
#     "data/training_labels.RData", "data/full-parsed-data.shp", "train_labels"
# )
def train_model():
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
            layers.Dense(128, activation="sigmoid"),
            layers.Dense(64, activation="sigmoid"),
            layers.Dense(64, activation="sigmoid"),
            layers.Dense(32, activation="sigmoid"),
            layers.Dense(6, activation="sigmoid"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        0.6, decay_steps=10000, decay_rate=0.90, staircase=True
    )
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=lr_schedule),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()],
    )
    print(model.summary())
    batch_size = 8
    num_batches = math.ceil(len(x_test) / batch_size)
    cp_callback = keras.callbacks.ModelCheckpoint(
        filepath="models/geoparams.keras", verbose=1, save_freq=100 * num_batches
    )

    def learning_rate(epoch, lr):
        return 0.5

    lr_callback = keras.callbacks.LearningRateScheduler(learning_rate)
    # K.set_value(model.optimizer.learning_rate, 0.1)
    # K.set_vaulue(model.optimizer.lr.assign(0.1)
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=50,
        validation_data=(x_test, y_test),
        callbacks=[cp_callback],
    )
    print(list(history.history["loss"]))
    return list(history.history["val_loss"])


sys.stdout = open(os.devnull, "w")
val_loss = list(train_model())[-1]
sys.stdout = sys.__stdout__
print(val_loss)
while val_loss > 0.06:
    sys.stdout = open(os.devnull, "w")
    val_loss = train_model()[-1]
    sys.stdout = sys.__stdout__
    print(val_loss)
