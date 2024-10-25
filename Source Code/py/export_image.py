#!/usr/bin/env python
import tensorflow as tf # type: ignore
import numpy as np # type: ignore
import random

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Select a random image
image_index = random.randint(0, len(x_train) - 1)
image = x_train[image_index]
label = y_train[image_index]

# add zero-padding to the image
image = np.pad(image, pad_width=1, mode='constant', constant_values=0)

# Save the image and label as a text file
with open("../weights/label.txt", "w") as file:
    file.write(f"{label}\n")

# Save the image as a text file
with open("../weights/image.txt", "w") as file:
    np.savetxt(file, image.shape, fmt="%d")
    file.write("\n")
    np.savetxt(file, image, fmt="%d")
