import cv2
import cv2 as cv
import copy

def check_and_convert_to_grayscale(image):
    if len(image.shape) == 3:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        return image


def get_dark_channel(image, size):
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (size, size))
    dark_channel = image

    if len(image.shape) == 3:
        b, g, r = cv.split(image)
        dark_channel = cv.min(cv.min(r, g), b)

    dark_channel = cv.erode(dark_channel, kernel)
    return dark_channel


def to_grayscale(image):
    if image.shape[2] == 3:
        return cv.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        return None


def crop_image(image, crop_width_size, crop_height_size):
    width = image.shape[1]
    height = image.shape[0]

    h_start = crop_height_size
    h_finish = height - crop_height_size

    w_start = crop_width_size
    w_finish = width - crop_width_size

    return copy.deepcopy(image[h_start:h_finish, w_start:w_finish])
