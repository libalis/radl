#!/usr/bin/env python
import numpy as np # type: ignore
import os

# ensure the directory exists
try:
    os.mkdir("./tmp")
except:
    pass

batch_size = 128
scale_factor = 20

# save how many images there are
with open("./tmp/image_len.txt", "w") as f:
    f.write(f"{batch_size}\n")

# save output
for i in range(batch_size):
    with open(f"./tmp/image_{i}.txt", "w") as f:
        # first two lines are the shape
        xl = np.random.uniform(-1, 1, (30 * scale_factor, 30 * scale_factor))
        np.savetxt(f, xl.shape, fmt="%f")
        f.write("\n")
        np.savetxt(f, xl, fmt="%f")

# save label
for i in range(batch_size):
    with open(f"./tmp/label_{i}.txt", "w") as f:
        xl = np.random.randint(0, 10)
        f.write(f"{xl}\n")

with open("./tmp/conv_bias.txt", "w") as f:
    xl = np.random.uniform(-1, 1, 4)
    np.savetxt(f, xl.shape, fmt='%f')
    f.write("\n")
    np.savetxt(f, xl, fmt='%f')

with open("./tmp/fc_bias.txt", "w") as f:
    xl = np.random.uniform(-1, 1, 10)
    np.savetxt(f, xl.shape, fmt='%f')
    f.write("\n")
    np.savetxt(f, xl, fmt='%f')

with open("./tmp/fc_weights.txt", "w") as f:
    xl = np.random.uniform(-1, 1, (30 * scale_factor * 30 * scale_factor, 10))
    np.savetxt(f, xl.shape, fmt='%f')
    f.write("\n")
    np.savetxt(f, xl, fmt='%f')

# save how many masks there are
with open("./tmp/masks_len.txt", "w") as f:
    f.write(f"{4}\n")

for i in range(4):
    with open(f"./tmp/masks_{i}.txt", "w") as f:
        xl = np.random.uniform(-1, 1, (3, 3))
        np.savetxt(f, xl.shape, fmt='%f')
        f.write("\n")
        np.savetxt(f, xl, fmt='%f')
