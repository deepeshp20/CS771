import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle


def model_train(df_test):
    train_data = pd.read_csv("train.csv")
    test_data = df_test
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = (
        train_data[["temp", "humidity", "no2op1", "no2op2", "o3op1", "o3op2"]],
        test_data[["temp", "humidity", "no2op1", "no2op2", "o3op1", "o3op2"]],
        train_data[["NO2", "OZONE"]],
        test_data[["NO2", "OZONE"]],
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Sequential()
    model.add(Dense(64, activation="relu", input_shape=(6,)))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(2, activation="linear"))

    model.compile(loss="mean_absolute_error", optimizer=Adam(lr=0.001))

    history = model.fit(
        X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32
    )
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    return


def my_predict(df_test):
    test_data = df_test
    # Load the saved model
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    # Scale the input features
    scaler = StandardScaler()
    X_test = scaler.fit_transform(
        test_data[["temp", "humidity", "no2op1", "no2op2", "o3op1", "o3op2"]]
    )
    pred = model.predict(X_test)
    return (pred[:, 1], pred[:, 0])
