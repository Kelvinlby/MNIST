# MNIST
A simple MNIST model built with Keras (JAX backend).
Computing precision: `float32`

## Model Architecture
Model: "mnist_model"

| Layer (type) | Output Shape | Param # |
|---|---|---|
| input_layer (InputLayer) | (None, 28, 28, 1) | 0 |
| conv2d (Conv2D) | (None, 26, 26, 32) | 320 |
| max_pooling2d (MaxPooling2D) | (None, 13, 13, 32) | 0 |
| conv2d_1 (Conv2D) | (None, 11, 11, 64) | 18,496 |
| max_pooling2d_1 (MaxPooling2D) | (None, 5, 5, 64) | 0 |
| flatten (Flatten) | (None, 1600) | 0 |
| dropout (Dropout) | (None, 1600) | 0 |
| dense (Dense) | (None, 10) | 16,010 |

Total params: 34,826 (136.04 KB) <br>
Trainable params: 34,826 (136.04 KB)
Non-trainable params: 0 (0.00 B)
