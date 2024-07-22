import json
import math
import os
import sys

import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split

from load_data import *

SIDE_LENGTH = 30

# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# repro data
# data = load_raster()


def create_model():
    model = keras.Sequential(
        [
            keras.Input(
                shape=(SIDE_LENGTH, SIDE_LENGTH, 1),
            ),
            layers.Conv2D(
                16,
                (3, 3),
                activation="sigmoid",
                strides=(2, 2),
            ),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(
                16,
                (3, 3),
                activation="sigmoid",
                strides=(2, 2),
            ),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(SIDE_LENGTH**2, activation="sigmoid"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.MeanAbsolutePercentageError(),
        metrics=[keras.metrics.MeanAbsolutePercentageError()],
    )
    print(model.summary())
    return model


def load_data():
    used = pd.read_csv("data/raster-rank.csv")
    print(used)
    used["rank"] = used["rank"] / 100.0
    x_keys = list(used.keys())
    x_keys.remove("district")
    x_keys.remove("rank")
    x = used[x_keys]
    y = used["rank"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=100, shuffle=True
    )

    x_train = x_train.iloc[:, : SIDE_LENGTH**2].values
    x_train = x_train.reshape((x_train.shape[0], SIDE_LENGTH, SIDE_LENGTH, 1))
    x_test = x_test.iloc[:, : SIDE_LENGTH**2].values
    x_test = x_test.reshape((x_test.shape[0], SIDE_LENGTH, SIDE_LENGTH, 1))
    print(x_train.shape)
    return (x_train, x_test, y_train, y_test)


def train_model():
    x_train, x_test, y_train, y_test = load_data()

    # sys.stdout = open(os.devnull, "w")
    batch_size = 8
    num_batches = math.ceil(len(x_test) / batch_size)
    cp_callback = keras.callbacks.ModelCheckpoint(
        filepath="models/bad.keras", verbose=1, save_freq=100 * num_batches
    )
    model = create_model()
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=5000,
        validation_data=(x_test, y_test),
        callbacks=[cp_callback],
    )
    # sys.stdout = sys.__stdout__
    json.dump(history.history, open("data/badhist.json", "w"))
    y_pred = model.predict(x_test)
    print(y_pred)
    print(y_test.values)
    mse = ((y_pred - y_test.values) ** 2).mean(axis=None)
    print("hehhhhhhhhhhhhhhhhhhhhhhhhhh")
    print(mse)
    for i in range(len(y_pred)):
        print(f"{y_pred[i]}  {y_test.values[i]}")
    print(model.summary())
    print(history.history)
    return history.history["val_loss"]


def load_model():
    hist = json.load(open("data/badhist.json"))
    model = create_model()
    model.load_weights("models/bad.keras")
    plt.title("loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.plot(range(len(hist["loss"])), hist["loss"])
    plt.savefig("foo.png")


sys.stdout = open(os.devnull, "w")
val_loss = list(train_model())[-1]
sys.stdout = sys.__stdout__
print(val_loss)
while val_loss > 10:
    # sys.stdout = open(os.devnull, "w")
    val_loss = list(train_model())[-1]
    # sys.stdout = sys.__stdout__
    print(val_loss)
