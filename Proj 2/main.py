# Pouya Mohammadi - 9829039
from functools import wraps
from numpy.core.numeric import ones, zeros_like
from utils import *
import numpy as np


def warpPerspective(img, transform_matrix, output_width, output_height):
    wraped_img = np.zeros((output_width, output_height, 3))
    for i in range(len(img)):
        for j in range(len(img[0])):
            pixel = np.array([[i], [j], [1]])
            pixel = np.dot(transform_matrix, pixel)
            pixel = np.array([int(pixel[0][0]/pixel[2][0]), int(pixel[1][0]/pixel[2][0])])
            if pixel[0] > 0 and pixel[0] < output_width and pixel[1] > 0 and pixel[1] < output_height:
                wraped_img[pixel[0]][pixel[1]] = img[i][j]     
    return wraped_img



def grayScaledFilter(img):
    gray_scale_img = np.zeros((len(img), len(img[0]), 3))
    grayscale_transform_matrix = np.array([[0.3], [0.59], [0.11]])
    for i in range(len(img)):
        for j in range(len(img[0])):
            value = np.dot(img[i][j], grayscale_transform_matrix)
            gray_scale_img[i][j] = [value, value, value]
    return gray_scale_img


def crazyFilter(img):
    filtered_img = np.zeros((len(img), len(img[0]), 3))
    crayzy_transform_matrix = np.array([[0, 1, 0], [1, 0, 0], [1, 0, 0]])
    for i in range(len(img)):
        for j in range(len(img[0])):
            value = np.dot(img[i][j], crayzy_transform_matrix)
            filtered_img[i][j] = value
    return filtered_img


def customFilter(img):
    custom_transform_matrix = np.array([[1/6, 5/6, 1/6], [1/5, 2/5 , 2/5], [1/3, 1/3, 1/3]])
    customFilter_img = Filter(img, custom_transform_matrix)
    # showImage(customFilter_img, title="customFilter_img Filter")
    CTM_inv = np.linalg.inv(custom_transform_matrix)
    print(CTM_inv)
    customFilter_inv_img = Filter(customFilter_img, CTM_inv)
    return customFilter_inv_img



def scaleImg(img, scale_width, scale_height):
    scaled_img = np.zeros((len(img) * scale_height, len(img[0]) * scale_width, 3))
    for i in range(len(img) * scale_height):
        for j in range(len(img[0]) * scale_width):
            value = img[int(i / scale_height)][int(j / scale_width)]
            scaled_img[i][j] = value
    return scaled_img


def cropImg(img, start_row, end_row, start_column, end_column):
    croped_img = np.zeros((end_column - start_column, end_row - start_row, 3))
    for i in range(len(img)):
        for j in range(len(img[0])):
            if i >= start_column and i < end_column and j >= start_row and j < end_row:
                croped_img[i- start_column][j - start_row] = img[i][j]
    return croped_img


if __name__ == "__main__":
    image_matrix = get_input('pic.jpg')

    # You can change width and height if you want
    width, height = 300, 400

    showImage(image_matrix, title="Input Image")

    # TODO : Find coordinates of four corners of your inner Image ( X,Y format)
    #  Order of coordinates: Upper Left, Upper Right, Down Left, Down Right
    pts1 = np.float32([[105, 215], [361, 177], [160, 645], [478, 574]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    m = getPerspectiveTransform(pts1, pts2)

    warpedImage = warpPerspective(image_matrix, m, width, height)
    showWarpPerspective(warpedImage)

    grayScalePic = grayScaledFilter(warpedImage)
    showImage(grayScalePic, title="Gray Scaled")

    crazyImage = crazyFilter(warpedImage)
    showImage(crazyImage, title="Crazy Filter")

    customFilter_inv_img = customFilter(warpedImage)
    showImage(customFilter_inv_img, title="CTM_inv_img Filter")

    croppedImage = cropImg(warpedImage, 50, 300, 50, 225)
    showImage(croppedImage, title="Cropped Image")

    scaledImage = scaleImg(warpedImage, 2, 3)
    showImage(scaledImage, title="Scaled Image")
    