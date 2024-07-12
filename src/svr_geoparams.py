import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

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
].to_numpy()
y = (data["rank"] / 100.0).to_numpy().reshape(-1, 1)
sc_x = StandardScaler()
sc_y = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
x_train = sc_x.fit_transform(x_train)
y_train = sc_y.fit_transform(y_train)
model = SVR(kernel="rbf")
model.fit(x_train, y_train)
y_pred_train = sc_y.inverse_transform(model.predict(x_train).reshape(-1, 1))
mse_train = ((y_pred_train - sc_y.inverse_transform(y_train)) ** 2).mean(axis=None)
print(mse_train)
y_pred_test = sc_y.inverse_transform(
    model.predict(sc_x.transform(x_test)).reshape(-1, 1)
)
mse_test = ((y_pred_test - y_test) ** 2).mean(axis=None)
print(mse_test)
for i in range(len(y_pred_test)):
    print(f"{y_pred_test[i]}  {y_test[i]}")
