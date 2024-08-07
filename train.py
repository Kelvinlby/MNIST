import jax
import numpy as np
import keras
from keras import layers
import datetime
import model

# Model / data parameters
batch_size = 256
epochs = 20


def main():
    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data('/mnt/Data/Dataset/mnist.npz')

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float16") / 255
    x_test = x_test.astype("float16") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    instance = model.mnist()
    instance.summary()

    instance.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(),
        metrics=[
            keras.metrics.Precision(),
        ]
    )

    instance.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

    score = instance.evaluate(x_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test precision:", score[1])

    instance.save(f"./Model/MNIST {datetime.datetime.now()}.keras")


if __name__ == '__main__':
    main()
