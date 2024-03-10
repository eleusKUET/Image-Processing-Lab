import cv2 as cv
import numpy as np
import sys

sys.setrecursionlimit(100000000)

img = cv.imread('Lena.jpg', cv.IMREAD_COLOR)

threshold = 25

cv.imshow('input', img)
cv.waitKey(0)

def segmentation(x1, x2, y1, y2):
    if x2 - x1 + 1 <= 2 and y2 - y1 + 1 <= 2:
        return
    img_slice1 = img[x1:x2+1, y1:y2+1, 0]
    img_slice2 = img[x1:x2+1, y1:y2+1, 1]
    img_slice3 = img[x1:x2+1, y1:y2+1, 2]
    std1 = np.std(img_slice1)
    std2 = np.std(img_slice2)
    std3 = np.std(img_slice3)
    if max(std1, std2, std3) < threshold:
        mean1 = np.mean(img_slice1)
        mean2 = np.mean(img_slice2)
        mean3 = np.mean(img_slice3)
        img[x1:x2 + 1, y1:y2 + 1, 0] = np.full_like(img_slice1, mean1)
        img[x1:x2 + 1, y1:y2 + 1, 1] = np.full_like(img_slice2, mean2)
        img[x1:x2 + 1, y1:y2 + 1, 2] = np.full_like(img_slice3, mean3)
        return
    xm = (x1 + x2) // 2
    ym = (y1 + y2) // 2
    segmentation(x1, xm, y1, ym)
    segmentation(xm + 1, x2, y1, ym)
    segmentation(x1, xm, ym + 1, y2)
    segmentation(xm + 1, x2, ym + 1, y2)

segmentation(0, img.shape[0], 0, img.shape[1])

img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
cv.imshow('output', np.rint(img).astype(np.uint8))
cv.waitKey(0)
cv.destroyAllWindows()