# EECS 442 Final Project

## Dataset Generation

Our dataset pulls from the 2017 Test images subset of the [COCO dataset](https://cocodataset.org/).
Each image gets resized into 128x128 and a grayscale copy is saved as input, while the color
is the expected output. We pull 10k random images for the training set, and 2k images
for the test set.

The [ADE20K Outdoors dataset](https://www.kaggle.com/residentmario/ade20k-outdoors) was
also used.

## Organization

`generate_dataset.py` is a simple command-line script that generates dataset folders
with 128x128 images, as described above.

`cnn.py` contains the definition for the Zhang CNN architecture.

`dataset.py` contains the dataloader class declarations as well as the instantiations
of the train, test, and val dataloader objects.

`train_nn.py` contains some hyperparameters at the top, and when run, will train the
neural network from the colorize_dataset folder. After it completes, it will save
a state dictionary of the parameters it learned, and save some debug images of the
test set validation. It also contains various code for the quantitative PSNR evaluation
of the output images.

`util.py` contains various helper functions for saving images and interacting with
the LAB colorspace.

`test.py` is a script that will load a pretrained model and a set of test images. It will
run the model on those images, and save some combined output images.

The master branch contains all the code we wrote, including some experimental features
that we were not fully succesful in utilizing. These will cause the training
of the neural net to not fully function if run as-is. The final commit in the sam_changes branch is the
code that was used to train the two models presented in the report.