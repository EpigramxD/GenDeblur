import math
import numpy as np
import cv2 as cv


def draw_line(shape, size, angle):
    line_img = np.zeros(shape, np.float32)
    cv.normalize(line_img, line_img, 0.0, 1.0, cv.NORM_MINMAX)
    center_y = int(shape[0] / 2)
    center_x = int(shape[1] / 2)
    angle_rad = angle * math.pi / 180
    angle_cos = math.cos(angle_rad)
    angle_sin = math.sin(angle_rad)
    length = int(angle_cos * size / 2)
    height = int(angle_sin * size / 2)
    point1 = (center_x - length, center_y + height)
    point2 = (center_x + length, center_y - height)
    cv.line(line_img, point1, point2, (1.0, 1.0, 1.0))
    cv.normalize(line_img, line_img, 0.0, 1.0, cv.NORM_MINMAX)
    return line_img