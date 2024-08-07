import keras
import numpy as np


def run(image):
    model = keras.models.load_model('Model/MNIST DATETIME.keras')
    result = model.predict(np.expand_dims(image, axis=0))
    num = 0

    for i in range(10):
        if result[0][i] > result[0][num]:
            num = i

    return num


def main(test_index):
    # Load the data and split it between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data('/mnt/Data/Dataset/mnist.npz')

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    print("Predicted output: ", run(x_test[test_index]))

    num = 0

    for index in range(10):
        if y_test[test_index][index] > y_test[test_index][num]:
            num = index

    print("Expected output:  ", num)


if __name__ == '__main__':
    for i in range(200, 250):
        main(i)
