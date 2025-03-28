#!/usr/bin/env python
import numpy as np # type: ignore
import os

# post-training quantization
quantization_factor = 2**(8-1)-1

# ensure the directory exists
try:
    os.mkdir("./tmp")
except:
    pass

batch_size = 128
scale_factor = 32

# save how many images there are
with open("./tmp/image_len.txt", "w") as f:
    f.write(f"{batch_size}\n")

# save output
for i in range(batch_size):
    with open(f"./tmp/image_{i}.txt", "w") as f:
        # first two lines are the shape
        xl = (np.random.uniform(-1, 1, (30 * scale_factor, 30 * scale_factor)) * quantization_factor).astype(np.int8)
        np.savetxt(f, xl.shape, fmt="%d")
        f.write("\n")
        np.savetxt(f, xl, fmt="%d")

# save label
for i in range(batch_size):
    with open(f"./tmp/label_{i}.txt", "w") as f:
        xl = np.random.randint(0, 10)
        f.write(f"{xl}\n")

with open("./tmp/conv_bias.txt", "w") as f:
    xl = (np.random.uniform(-1, 1, 4) * quantization_factor).astype(np.int8)
    np.savetxt(f, xl.shape, fmt="%d")
    f.write("\n")
    np.savetxt(f, xl, fmt="%d")

with open("./tmp/fc_bias.txt", "w") as f:
    fc_bias_txt = (np.random.uniform(-1, 1, 10) * quantization_factor).astype(np.int8)
    xl = np.transpose(fc_bias_txt)
    np.savetxt(f, xl.shape, fmt="%d")
    f.write("\n")
    np.savetxt(f, xl, fmt="%d")

with open("./tmp/fc_weights.txt", "w") as f:
    fc_weights_txt = (np.random.uniform(-1, 1, (30 * scale_factor * 30 * scale_factor, 10)) * quantization_factor).astype(np.int8)
    xl = np.transpose(fc_weights_txt)
    np.savetxt(f, xl.shape, fmt="%d")
    f.write("\n")
    np.savetxt(f, xl, fmt="%d")

# save how many masks there are
with open("./tmp/masks_len.txt", "w") as f:
    f.write(f"{4}\n")

for i in range(4):
    with open(f"./tmp/masks_{i}.txt", "w") as f:
        xl = (np.random.uniform(-1, 1, (3, 3)) * quantization_factor).astype(np.int8)
        np.savetxt(f, xl.shape, fmt="%d")
        f.write("\n")
        np.savetxt(f, xl, fmt="%d")
