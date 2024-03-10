import cv2 as cv
import numpy as np
import math
def convolve(image, kernel, center=None):
    ix, iy = image.shape[0], image.shape[1]
    kx, ky = kernel.shape[0], kernel.shape[1]
    #determining center
    cx = (kx + 1) // 2 - 1
    cy = (ky + 1) // 2 - 1

    if center is not None:
        cx, cy = center #user defined center
    right_padding = ky - cy - 1
    down_padding = kx - cx - 1

    px = ix + cx + down_padding
    py = iy + cy + right_padding
    padded_image = np.zeros((px, py))

    for i in range(ix):
        for j in range(iy):
            padded_image[i + cx, j + cy] = image[i, j]

    output = np.zeros((px, py))
    #convolution
    for i in range(px):
        for j in range(py):
            if i + kx < px and j + ky < py:
                total = 0
                for k in range(kx):
                    for l in range(ky):
                        total += padded_image[i + k, j + l] * kernel[-k - 1, - l - 1]
                output[i + cx, j + cy] = total

    new_image = np.zeros((ix, iy))
    for i in range(cx, cx + ix):
        for j in range(cy, cy + iy):
            new_image[i - cx, j - cy] = output[i, j]

    return new_image

def LoG(x, y, sigma):
    pw = (x ** 2 + y ** 2) / (2 * sigma ** 2)
    res = -1 / (math.pi * sigma ** 4) * (1 - pw) * math.exp(-pw)
    return res

def LoG_kernel(sigma, center=None):
    dim = (round(sigma * 7), round(sigma * 7))
    kernel = np.zeros(dim)
    cx = (dim[0] + 1) // 2 - 1
    cy = (dim[1] + 1) // 2 - 1
    if center is not None:
        cx, cy = center
    for x in range(dim[0]):
        for y in range(dim[1]):
            kernel[x, y] = LoG(x - cx, y - cy, sigma)
    return kernel

image = cv.imread('Lena.jpg', cv.IMREAD_GRAYSCALE)
output = convolve(image, LoG_kernel(1))

cv.imshow('Input image', image)
cv.waitKey(0)

normalized_output = cv.normalize(output, None, 0, 255, cv.NORM_MINMAX)
cv.imshow('Convoluted Image', np.rint(normalized_output).astype(np.uint8))
cv.waitKey(0)

zero_cross_image = np.zeros_like(image).astype(np.uint8)

def sign(n):
    if n >= 0:
        return 1
    else:
        return -1

edge_contained_image = np.zeros_like(image).astype(np.uint8)

for i in range(1, image.shape[0] - 1):
    for j in range(1, image.shape[1] - 1):
        std = np.std(np.array([image[i, j], image[i - 1, j], image[i + 1, j], image[i, j - 1], image[i, j + 1]]))
        '''
            Here, in the classwork, teacher said to consider varience instead of standerd deviation. But the output
            wasn't satisfying.
        '''
        bad = True
        if sign(output[i - 1, j]) != sign(output[i + 1, j]):
            bad = False
        elif sign(output[i, j - 1]) != sign(output[i, j + 1]):
            bad = False
        if not bad and output[i][j] > std:
            edge_contained_image[i][j] = 255

cv.imshow('Edge contained image', edge_contained_image)
cv.waitKey(0)

cv.destroyAllWindows()


