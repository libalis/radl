#!/usr/bin/env python
import tensorflow as tf # type: ignore
import numpy as np # type: ignore
import os, random

# load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# select a random image
image_index = random.randint(0, len(x_train) - 1)
image = x_train[image_index]
label = y_train[image_index]

# normalize image
norm_img = image / 255.0

# add zero padding to the image
pad_img = np.pad(norm_img, pad_width=1, mode='constant', constant_values=0)

# ensure the directory exists
try:
    os.mkdir("./weights")
except:
    pass

# save the label in a file
with open("./weights/label.txt", "w") as f:
    f.write(f"{label}\n")

# save the image in a file
with open("./weights/image.txt", "w") as f:
    np.savetxt(f, pad_img.shape, fmt="%f")
    f.write("\n")
    np.savetxt(f, pad_img, fmt="%f")
