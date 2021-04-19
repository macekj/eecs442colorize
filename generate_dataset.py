# Generate a test and train dataset for image colorization 
# given a folder of images.

import os
import sys
from PIL import Image

# From Titouan on StackOverflow
# https://stackoverflow.com/questions/43512615/reshaping-rectangular-image-to-square
def resize_image(image: Image, length: int) -> Image:
    """
    Resize an image to a square. Can make an image bigger to make it fit or smaller if it doesn't fit. It also crops
    part of the image.

    :param self:
    :param image: Image to resize.
    :param length: Width and height of the output image.
    :return: Return the resized image.
    """

    """
    Resizing strategy : 
     1) We resize the smallest side to the desired dimension (e.g. 1080)
     2) We crop the other side so as to make it fit with the same length as the smallest side (e.g. 1080)
    """
    if image.size[0] < image.size[1]:
        # The image is in portrait mode. Height is bigger than width.

        # This makes the width fit the LENGTH in pixels while conserving the ration.
        resized_image = image.resize((length, int(image.size[1] * (length / image.size[0]))))

        # Amount of pixel to lose in total on the height of the image.
        required_loss = (resized_image.size[1] - length)

        # Crop the height of the image so as to keep the center part.
        resized_image = resized_image.crop(
            box=(0, required_loss / 2, length, resized_image.size[1] - required_loss / 2))

        # We now have a length*length pixels image.
        return resized_image
    else:
        # This image is in landscape mode or already squared. The width is bigger than the heihgt.

        # This makes the height fit the LENGTH in pixels while conserving the ration.
        resized_image = image.resize((int(image.size[0] * (length / image.size[1])), length))

        # Amount of pixel to lose in total on the width of the image.
        required_loss = resized_image.size[0] - length

        # Crop the width of the image so as to keep 1080 pixels of the center part.
        resized_image = resized_image.crop(
            box=(required_loss / 2, 0, resized_image.size[0] - required_loss / 2, length))

        # We now have a length*length pixels image.
        return resized_image

if len(sys.argv) < 5:
    print("USAGE: python3 generate_dataset.py src_folder dest_folder num_train num_test")
    exit(1)

src = sys.argv[1]
dst = sys.argv[2]
num_train = int(sys.argv[3])
num_test = int(sys.argv[4])

# make train and test directories within dataset folder
if not os.path.exists(os.path.join(dst, "train")):
    os.mkdir(os.path.join(dst, "train"))
if not os.path.exists(os.path.join(dst, "test")):
    os.mkdir(os.path.join(dst, "test"))


i = 0
for file in os.listdir(src):
    img = Image.open(os.path.join(src,file))
    width, height = img.size

    # skip images smaller than 128 x 128
    if width < 128 or height < 128:
        continue

    folder = "train"
    num = i

    if i > num_train:
        folder = "test"
        num -= num_train

    # crop and save BW and original version
    new_img = resize_image(img, 128)
    new_img.save(os.path.join(dst, folder, 'clr_%05d.jpg' % num))
    new_img = new_img.convert("L")
    new_img.save(os.path.join(dst, folder, 'gry_%05d.jpg' % num))

    i += 1
    
    if i > num_train + num_test:
        break

