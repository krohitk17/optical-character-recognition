# Break Captchas Into Separate Characters

# Import Libraries

import cv2                                  # used for image processing
# used to store image data in numpy arrays
import numpy as np
import os                                   # used to read files from directory


# Dataset : https://www.kaggle.com/datasets/greysky/captcha-dataset
# path to the directory containing the images
DIR = os.getcwd() + '/samples/'
# path to the directory where characters will be stored
DATASET = os.getcwd() + '/dataset/'

RESIZED_IMAGE_WIDTH = 20                    # width of resized image
RESIZED_IMAGE_HEIGHT = 30                   # height of resized image


try:
    # create directory to store characters
    os.mkdir(DATASET)
except:
    pass


# Function for Image Preprocessing


def preprocess(img):
    # convert to grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # get binary image
    thresh = cv2.adaptiveThreshold(
        imgGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)

    # remove noise from image
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                             np.ones((3, 3), np.uint8))

    # get dilated image
    dilate = cv2.dilate(close, np.ones((2, 2), np.uint8), iterations=1)

    # invert image colors
    image = cv2.bitwise_not(dilate)

    return image


# Break Captchas


for img in os.listdir(DIR):
    image = cv2.imread(DIR + img)           # read captcha image
    imgname = img.split('.')[0]             # get captcha text
    print(imgname)
    image = preprocess(image)               # preprocess image

    IMAGE_WIDTH = image.shape[1]            # get image width
    IMAGE_HEIGHT = image.shape[0]           # get image height

    # get separate alphabets from captcha
    for i in range(0, IMAGE_WIDTH, IMAGE_WIDTH//5):
        # divide image into 5 parts
        print(img[int(i/(IMAGE_WIDTH//5))])
        # get each part
        letter = image[10:IMAGE_HEIGHT-10, i:i+IMAGE_WIDTH//5]
        # resize each part to 20x30
        letter = cv2.resize(
            letter, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
        # dilate each part
        letter = cv2.dilate(letter, np.ones((2, 2), np.uint8), iterations=1)
        # save each part
        PATH = DATASET + img[int(i/(IMAGE_WIDTH//5))] + '/'
        try:
            os.chdir(PATH)              # change directory to save image
        except FileNotFoundError:
            os.mkdir(PATH)              # create directory to save image
        finally:
            # get number of images
            dirlen = len(os.listdir(PATH))
        cv2.imwrite(PATH + str(dirlen) + '.png', letter)    # save image
