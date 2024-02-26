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

def myKernel():
    return np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]]) / 273

def gauss(x, y, sigma):
    pw = -0.5 * (x * x / (sigma[0] ** 2) + y * y / (sigma[1] ** 2))
    return 1 / (2 * math.pi * sigma[0] * sigma[1]) * math.exp(pw)

def gaussian_kernel(dim, sigma, center=None):
    cx = (dim[0] + 1) // 2 - 1
    cy = (dim[1] + 1) // 2 - 1
    if center is not None:
        cx, cy = center
    kernel = np.zeros(dim)
    for x in range(dim[0]):
        for y in range(dim[1]):
            kernel[x, y] = gauss(x - cx, y - cy, sigma)
    return kernel

def mean_kernel(dim):
    return np.ones(dim) / (dim[0] * dim[1])

def laplacian_kernel():
    return np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

def LoG(x, y, sigma):
    pw = (x ** 2 + y ** 2) / (2 * sigma ** 2)
    res = -1 / (math.pi * sigma ** 4) * (1 - pw) * math.exp(-pw)
    return res

def LoG_kernel(dim, sigma, center=None):
    kernel = np.zeros(dim)
    cx = (dim[0] + 1) // 2 - 1
    cy = (dim[1] + 1) // 2 - 1
    if center is not None:
        cx, cy = center
    for x in range(dim[0]):
        for y in range(dim[1]):
            kernel[x, y] = LoG(x - cx, y - cy, sigma)
    return kernel

def sobel_x_kernel():
    return np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

def sobel_y_kernel():
    return np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

kernel = gaussian_kernel((3,3), (1.5, 1.5))
img = cv.imread('Lena.jpg', 0)

cv.imshow('Input', img)
cv.waitKey(0)

output = convolve(img, kernel)
cv.imshow('Output', np.rint(output).astype(np.uint8))
cv.waitKey(0)

new_img = cv.normalize(output, None, 0, 255, cv.NORM_MINMAX)
cv.imshow('Normalized', np.rint(new_img).astype(np.uint8))
cv.waitKey(0)
cv.destroyAllWindows()
