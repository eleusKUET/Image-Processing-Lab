import cv2 as cv
import numpy as np

img, threshold = 0, 0
def iterative_dfs():
    x_low, x_high, y_low, y_high = 0, img.shape[0] - 1, 0, img.shape[1] - 1
    stack = [(x_low, x_high, y_low, y_high)]
    while stack:
        x_low, x_high, y_low, y_high = stack.pop()
        if x_high - x_low + 1 <= 2 and y_high - y_low + 1 <= 2:
            continue
        r_channel = img[x_low:x_high+1, y_low:y_high+1, 0]
        g_channel = img[x_low:x_high+1, y_low:y_high+1, 1]
        b_channel = img[x_low:x_high+1, y_low:y_high+1, 2]
        std_r = np.std(r_channel)
        std_g = np.std(g_channel)
        std_b = np.std(b_channel)
        if std_r >= threshold or std_b >= threshold or std_g >= threshold:
            x_mid = (x_low + x_high) // 2
            y_mid = (y_low + y_high) // 2
            stack.append((x_low, x_mid, y_low, y_mid))
            stack.append((x_mid + 1, x_high, y_low, y_mid))
            stack.append((x_low, x_mid, y_mid + 1, y_high))
            stack.append((x_mid + 1, x_high, y_mid + 1, y_high))
        else:
            r_mean = np.mean(r_channel)
            g_mean = np.mean(g_channel)
            b_mean = np.mean(b_channel)
            img[x_low:x_high + 1, y_low:y_high + 1, 0] = np.full_like(r_channel, r_mean)
            img[x_low:x_high + 1, y_low:y_high + 1, 1] = np.full_like(g_channel, g_mean)
            img[x_low:x_high + 1, y_low:y_high + 1, 2] = np.full_like(b_channel, b_mean)

def main():
    while True:
        global img
        global threshold
        threshold = int(input('Enter a threshold value:'))
        print('Processing...')
        img = cv.imread('Lena.jpg', cv.IMREAD_COLOR)
        cv.imshow('input', img)
        cv.waitKey(0)
        iterative_dfs()
        img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
        cv.imshow('output', np.rint(img).astype(np.uint8))
        cv.waitKey(0)
        cv.destroyAllWindows()

if __name__ == '__main__':
    main()
