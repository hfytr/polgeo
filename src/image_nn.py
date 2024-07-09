# import os
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import keras
from keras import layers
from sklearn.model_selection import train_test_split

from load_data import *

# repro data
# load_raster()

used = pd.read_csv("data/raster-rank.csv")
print(used)
x_keys = list(used.keys())
x_keys.remove("district")
x_keys.remove("rank")
x = used[x_keys]
y = used["rank"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=100, shuffle=True)

x_train = x_train.iloc[:, :64].values
x_train = x_train.reshape((x_train.shape[0], 8, 8, 1))
x_test = x_test.iloc[:, :64].values
x_test = x_test.reshape((x_test.shape[0], 8, 8, 1))
print(x_train)

model = keras.Sequential(
    [
        layers.Conv2D(64, (2, 2), activation="relu", input_shape=(8, 8, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (2, 2), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="softmax"),
        layers.Dense(1),
    ]
)
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.MeanSquaredLogarithmicError(),
    metrics=[keras.metrics.MeanSquaredError()],
)
print(model.summary())
history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.02)
print(model.summary())
print(history.history)
