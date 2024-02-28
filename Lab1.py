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
    pw = (x ** 2 + y ** 2) / (2 * sigma[0] * sigma[1])
    res = -1 / (math.pi * sigma[0] ** 2 * sigma[1] ** 2) * (1 - pw) * math.exp(-pw)
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

def channel_seperator(colored_image):
    first = np.zeros((colored_image.shape[0], colored_image.shape[1]))
    second = np.zeros((colored_image.shape[0], colored_image.shape[1]))
    third = np.zeros((colored_image.shape[0], colored_image.shape[1]))
    for i in range(colored_image.shape[0]):
        for j in range(colored_image.shape[1]):
            first[i, j] = colored_image[i, j, 0]
            second[i, j] = colored_image[i, j, 1]
            third[i, j] = colored_image[i, j, 2]
    return [first, second, third]

def channel_merger(first, second, third):
    image = np.zeros((first.shape[0], first.shape[1], 3))
    for i in range(first.shape[0]):
        for j in range(first.shape[1]):
            image[i, j, 0] = first[i, j]
            image[i, j, 1] = second[i, j]
            image[i, j, 2] = third[i, j]
    return image

def convolve3D(image, kernel, center=None):
    channels = channel_seperator(image)
    channels[0] = convolve(channels[0], kernel, center)
    channels[1] = convolve(channels[1], kernel, center)
    channels[2] = convolve(channels[2], kernel, center)
    return channel_merger(channels[0], channels[1], channels[2])

def hsv_normalizer(image):
    channels = channel_seperator(image)
    channels[0] = cv.normalize(channels[0], None, 0, 179, cv.NORM_MINMAX)
    channels[1] = cv.normalize(channels[1], None, 0, 255, cv.NORM_MINMAX)
    channels[2] = cv.normalize(channels[2], None, 0, 255, cv.NORM_MINMAX)
    return channel_merger(channels[0], channels[1], channels[2])

def output_generator(kernel, center=None, both=None):
    print('''
        Choose an option, e.g. 1 for grayscale image:
        1. Grayscale image
        2. Color image
    ''')
    img_id = int(input('Enter mode:'))
    if img_id == 1:
        print('You choosed grayscale mode. Applying convolution...')
        image = cv.imread('Lena.jpg', cv.IMREAD_GRAYSCALE)
        output = convolve(image, kernel, center)
        if both is not None:
            output = np.sqrt(output ** 2 + convolve(image, both, center) ** 2)
        normalized_output = cv.normalize(output, None, 0, 255, cv.NORM_MINMAX)

        cv.imshow('Input', image)
        cv.waitKey(0)
        cv.imshow('Output before normalization', np.rint(output).astype(np.uint8))
        cv.waitKey(0)
        cv.imshow('Normalized output', np.rint(normalized_output).astype(np.uint8))
        cv.waitKey(0)
        cv.destroyAllWindows()
    elif img_id == 2:
        print('You choosed color image mode. Applying convolution....')
        image = cv.imread('Lena.jpg', 1)
        output = convolve3D(image, kernel, center)
        if both is not None:
            output = np.sqrt(output ** 2 + convolve3D(image, both, center) ** 2)
        normalized_output = cv.normalize(output, None, 0, 255, cv.NORM_MINMAX)
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        #print(hsv_image)
        hsv_output = convolve3D(hsv_image, kernel, center)
        if both is not None:
            hsv_output = np.sqrt(hsv_output ** 2 + convolve3D(hsv_image, both, center) ** 2)
        #print(hsv_output)
        hsv_normalized_output = hsv_normalizer(hsv_output)

        cv.imshow('RGB output', np.rint(normalized_output).astype(np.uint8))
        cv.waitKey(0)
        cv.imshow('HSV output', np.rint(hsv_normalized_output).astype(np.uint8))
        cv.waitKey(0)
        hsv_normalized_output = cv.cvtColor(np.rint(hsv_normalized_output).astype(np.uint8), cv.COLOR_HSV2BGR)
        dif_output = normalized_output - hsv_normalized_output
        dif_output = cv.normalize(dif_output, None, 0, 255, cv.NORM_MINMAX)
        cv.imshow('Output difference', np.rint(dif_output).astype(np.uint8))
        cv.waitKey(0)
        cv.destroyAllWindows()


def main():
    print('Welcome to the CLI of convolution application')
    while True:
        print('''
            Choose a kernel for convolution, e.g. 1 for gaussian kernel:
            1. gaussian 
            2. mean
            3. laplacian
            4. LoG
            5. sobel x 
            6. sobel y
            7. sobel both axis
        ''')
        kernel_id = int(input('Enter a kernel id:'))

        if kernel_id == 1:
            print('You choosed gaussian kernel')
            print('Enter dimension, center_x, center_y in a single line:')
            dim, center_x, center_y = [int(x) for x in input().split()]
            sigma_x, sigma_y = [float(x) for x in input('Enter sigma_x, sigma_y in a single line:').split()]
            output_generator(gaussian_kernel((dim, dim), (sigma_x, sigma_y), (center_x, center_y)))

        elif kernel_id == 2:
            print('You choosed mean kernel')
            print('Enter dimension, center_x, center_y in a single line:')
            dim, center_x, center_y = [int(x) for x in input().split()]
            output_generator(mean_kernel((dim, dim)), (center_x, center_y))

        elif kernel_id == 3:
            print('You choosed laplacian kernel')
            output_generator(laplacian_kernel())

        elif kernel_id == 4:
            print('You choosed LoG kernel')
            print('Enter dimension, center_x, center_y in a single line:')
            dim, center_x, center_y = [int(x) for x in input().split()]
            sigma_x, sigma_y = [float(x) for x in input('Enter sigma_x, sigma_y in a single line:').split()]
            output_generator(LoG_kernel((dim, dim), (sigma_x, sigma_y), (center_x, center_y)))

        elif kernel_id == 5:
            print('You choosed sobel x derivative kernel')
            output_generator(sobel_x_kernel())

        elif kernel_id == 6:
            print('You choosed sobel y derivative kernel')
            output_generator(sobel_y_kernel())

        elif kernel_id == 7:
            print('You choosed sobel kernel on both axis')
            output_generator(sobel_x_kernel(), None, sobel_y_kernel())


if __name__ == '__main__':
    main()
