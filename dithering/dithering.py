import cv2
import sys
import os
import numpy as np
from math import cos, sin, pi, modf


def readAndResize(file):
    img = cv2.imread(file)
    img = cv2.resize(img, (30, 30))
    return img


def toBWSimple(img):
    c = np.copy(img)
    h, w = c.shape
    for y in range(h):
        for x in range(w):
            if img[y, x] < 0.5:
                c[y, x] = 0.0
            else:
                c[y, x] = 1.0
    return c

def closestColor(val, colors):
    dists = [np.linalg.norm(col - val, ord=1) for key, col in colors]
    min_ind = dists.index(min(dists))
    return colors[min_ind][0], np.array(val - colors[min_ind][1])

def to_space_hsvc(pix):
    blank_image = np.zeros((1, 1, 3), np.uint8)
    blank_image[0][0] = pix
    hsv = cv2.cvtColor(blank_image, cv2.COLOR_BGR2HSV)[0][0]
    zilinder = (cos(hsv[0] / 255 * 2 * pi) * hsv[1], sin(hsv[0] / 255 * 2 * pi) * hsv[1], hsv[2])
    return zilinder

def floydSteinberg(img, colors):

    """
    # convert using to_space_hsvc
    img = np.array([[to_space_hsvc(pix) for pix in row] for row in img])
    colors = [(key, to_space_hsvc(col)) for key, col in colors]

    xy_boost = 2.0
    img[:, :, 0] *= xy_boost
    img[:, :, 1] *= xy_boost

    colors = [(key, np.array([col[0] * xy_boost, col[1] * xy_boost, col[2]])) for key, col in colors]
    """
    temp = np.copy(img) / 1.0
    h, w, t = img.shape
    print(h, w, t)
    c = [[0 for x in range(w)] for y in range(h)]

    for y in range(h):
        for x in range(w):
            old_val = temp[y][x]
            new_val, error = closestColor(old_val, colors)
            c[y][x] = new_val

            if (x < w - 1):
                temp[y][x + 1] += error * 7.0 / 16
            if (x > 0 and y < h - 1):
                temp[y + 1][x - 1] += error * 3.0 / 16
            if (y < h - 1):
                temp[y + 1][x] += error * 5.0 / 16
            if (x < w - 1 and y < h - 1):
                temp[y + 1][x + 1] += error * 1.0 / 16

    return c



def longusMongus(img, colors):
    h, w, t = img.shape
    matched = [[closestColor(val, colors)[0] for val in row] for row in img]
    color_name_map = {col[0]: col[1] for col in colors}
    n = 5000
    for i in range(n):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        r = 4
        neighbor_error = np.zeros(3)
        sum_weight= 0
        for j in range(-r, r + 1):
            for k in range(-r, r + 1):
                if (x + j >= 0 and x + j < w and y + k >= 0 and y + k < h):
                    d = np.linalg.norm(np.array([j, k]))
                    if (d > 0) and (d <= r):
                        weight = d**(-1/2)
                        neighbor_error += (color_name_map[matched[y + k][x + j]] - img[y + k][x + j]) * weight
                        sum_weight += weight

        neighbor_error /= sum_weight
        matched[y][x] = closestColor(img[y][x] - neighbor_error, colors)[0]
    return matched


def toBWandRW(img, colors):
    h, w, t = img.shape
    c = np.zeros((2 * h, w))

    for i in range(2):
        col = colors[i + 1]
        for y in range(h):
            for x in range(w):
                if np.all(img[y, x] == col):
                    c[h * i + y, x] = 0
                else:
                    c[h * i + y, x] = 1
    return c


def convertToBinary(img):
    n = 48000 * 2
    binData = np.zeros((n,), dtype=np.uint8)  # Array of bytes

    for y in range(480 * 2):
        for x in range(800):
            byte = y * 100 + x // 8
            if (img[y, x] == 1):
                bit = 7 - (x % 8)
                binData[byte] |= (1 << bit)

    return binData




def from_space_hsvc(pix):
    color_n = [0] * 3
    color_n[0] = (np.arctan2(pix[1], pix[0]) / (2 * pi) % 1) * 255
    color_n[1] = np.linalg.norm(pix[:2])
    color_n[2] = pix[2]
    color = cv2.cvtColor(np.array([[color_n]]).astype(np.uint8), cv2.COLOR_HSV2BGR)[0][0]
    return color

v = 0

def to_space(pix):
    if v == 0:
        return to_space_hsvc(pix)
    if v == 1:
        return (pix / 255) ** 2.2 * 255
    if v == 2:
        return pix
    if v == 3:
        return cv2.cvtColor(np.array([[pix]]).astype(np.uint8), cv2.COLOR_BGR2LAB)[0][0]

def from_space(pix):
    if v == 0:
        return from_space_hsvc(pix)
    if v == 1:
        return (pix / 255) ** (1 / 2.2) * 255
    if v == 2:
        return pix
    if v == 3:
        return cv2.cvtColor(np.array([[pix]]).astype(np.uint8), cv2.COLOR_LAB2BGR)[0][0]
def get_average_color(img):
    img = np.array([[to_space(pix) for pix in row] for row in img])
    color = np.mean(img, axis=(0, 1))
    color = from_space(color)
    return color

if __name__ == "__main__":

    file = ""
    if len(sys.argv) == 2:
        file = sys.argv[1]
    else:
        raise "Verwendung: python3 dithering.py <Dateiname>"
    if not os.path.exists(file):
        raise f"Datei {file} nicht gefunden!"

    test_farben = [(2, 0, 6), (1, 8, 3), (207, 183, 192), (11, 111, 211)]
    for test_farb in test_farben:
        print("test_farb: {}, reconstructed: {}".format(test_farb, from_space(to_space(test_farb))))

    colors = []
    # go through pngs in emojis/
    for png in os.listdir("emojis"):
        # read image
        if not png.endswith(".png"):
            continue
        img = cv2.imread("emojis/" + png)

        average_color = get_average_color(img)

        # add color to dict with filename as key
        colors.append((png[:-4], average_color))

    img = readAndResize(file)

    string = floydSteinberg(img, colors)

    for line in string:
        for emoschi in line:
            print(emoschi, end=" ")
        print()

    # img = toBWandRW(img, colors)
    # binary_data = convertToBinary(img)

    # rw_img = floydSteinberg(img, [255, 0, 0], [255, 255, 255])
    # binary_data = convertToBinary(img)

    # with open("data.bin", "wb") as f:
    #    f.write(binary_data)
