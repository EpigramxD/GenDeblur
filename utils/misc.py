import cv2 as cv


def check_and_convert_to_grayscale(image):
    if image.shape[2] == 3:
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        return image


def get_dark_channel(image, size):
    if len(image.shape) == 2 or image.shape[2] != 3:
        return None
    else:
        b, g, r = cv.split(image)
        dark_channel = cv.min(cv.min(r, g), b);
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (size, size))
        dark_channel = cv.erode(dark_channel, kernel)
        return dark_channel