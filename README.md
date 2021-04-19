# EECS 442 Final Project

## Dataset Generation

Our dataset pulls from the 2017 Test images subset of the [COCO dataset](https://cocodataset.org/).
Each image gets resized into 128x128 and a grayscale copy is saved as input, while the color
is the expected output. We pull 10k random images for the training set, and 2k images
for the test set.

The dataset can be found in `colorize_dataset.zip`.