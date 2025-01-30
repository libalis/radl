#!/usr/bin/env python
import tensorflow # type: ignore
import tensorflow_datasets as tfds # type: ignore
import numpy as np # type: ignore
import os

# https://www.tensorflow.org/datasets/keras_example
tf = tensorflow.compat.v1

# disable eager execution
tf.disable_eager_execution()

# download dataset and store reference in variable
(train_ds, test_ds), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# normalizes images
def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.0, label

# tfds provide images of type "tf.uint8", while the model expects "tf.float32", therefore, you need to normalize images
train_ds = train_ds.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)

# reshape datasets to 28 x 28 x 1 pixels (height x width x color channels)
train_ds = train_ds.map(lambda image, label: (tf.reshape(image, [28, 28, 1]), label))

# pad images with 1 row/column of pixels on each side for 3 x 3 filter (border handling)
train_ds = train_ds.map(lambda image, label: (tf.pad(image, [[1, 1], [1, 1], [0, 0]], "CONSTANT"), label))

# cache the modified data in memory
train_ds = train_ds.cache()

# shuffling and dividing in batches
shuffle_size = 60000
batch_size = 128
train_ds = train_ds.shuffle(shuffle_size).batch(batch_size)

# define iterator over batches of data
data_iterator = tf.data.Iterator.from_structure(tf.data.get_output_types(train_ds), tf.data.get_output_shapes(train_ds))

# define graph operation which initializes the iterator with the dataset
train_init_op = data_iterator.make_initializer(train_ds)

# define graph operation which gets the next batch of the iterator over the dataset
next_data_batch = data_iterator.get_next()

# initialize all variables before evaluating the graph
session = tf.Session()
session.run(tf.global_variables_initializer())

epochs = 1
image_batch = np.array([])
label_batch = np.array([])

# train the weights by looping repeatedly over all the data (and shuffling in between)
for i in range(epochs):
    session.run(train_init_op)
    session.run(train_init_op)
    data_batch = session.run(next_data_batch)
    image_batch = data_batch[0]
    label_batch = data_batch[1]

# post-training quantization
quantization_factor = 2**(8-1)-1

# ensure the directory exists
try:
    os.mkdir("./tmp")
except:
    pass

# save how many images there are
with open("./tmp/image_len.txt", "w") as f:
    f.write(f"{batch_size}\n")

# save output
for i in range(batch_size):
    with open(f"./tmp/image_{i}.txt", "w") as f:
        # first two lines are the shape
        quantized_image = (image_batch[i, :, :, 0] * quantization_factor).astype(np.int8)
        np.savetxt(f, quantized_image.shape, fmt="%d")
        f.write("\n")
        np.savetxt(f, quantized_image, fmt="%d")

# save label
for i in range(batch_size):
    with open(f"./tmp/label_{i}.txt", "w") as f:
        f.write(f"{label_batch[i]}\n")
