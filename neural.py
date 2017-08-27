import numpy as np
import keras
import matplotlib.pyplot as plt

def save(model, name):
    model.save("./models/" + name)

def load(name):
    return keras.models.load_model("./models/" + name)

def sequential_basic():
    m = keras.models.Sequential()
    m.add(keras.layers.Dense(1024, input_dim=4, activation="relu"))
    m.add(keras.layers.Dense(1024, activation="relu"))
    m.add(keras.layers.Dense(1024, activation="relu"))
    m.add(keras.layers.Dense(2))
    m.compile(optimizer="sgd", loss="mse")
    return m

def replicate(data, reps=20):
    data = np.tile(np.asarray(data), (reps,1))
    # add uniform [-.1, .1] noise to it
    data = data + (np.random.rand(*data.shape)-.5)/5
    return data

if __name__ == '__main__':
    r = np.load("data/swerve-1493269411.pickle")
    u = replicate(r[0][0], reps=1000)
    x = replicate(r[1][0], reps=1000)
    model = sequential_basic()
    hist = model.fit(x, u)
    #plt.stem(hist.history["norm"])
    #plt.show()
    save(model, "swerve-3layers-1000reps")
