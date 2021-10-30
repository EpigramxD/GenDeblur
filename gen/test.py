import cv2 as cv
import numpy as np

from utils.deconv import *
from scipy.optimize import minimize
from scipy.optimize import Bounds
from utils.freq_domain_utils import *
from utils.drawing import *
from utils.size_utils import *
from utils.filteringUtils import FilteringUtils

KERNEL_SIZE = 9

BLURRED_IMAGE = cv.imread("../images/blurred/1.jpg", cv.IMREAD_GRAYSCALE)
BLURRED_IMAGE = np.float32(BLURRED_IMAGE)
cv.normalize(BLURRED_IMAGE, BLURRED_IMAGE, 0.0, 1.0, cv.NORM_MINMAX)

SHARP_IMAGE = cv.imread("../images/sharp/1.jpg", cv.IMREAD_GRAYSCALE)
SHARP_IMAGE = np.float32(SHARP_IMAGE)
cv.normalize(SHARP_IMAGE, SHARP_IMAGE, 0.0, 1.0, cv.NORM_MINMAX)

SIZE_PYRAMID = build_double_size_pyramid(SHARP_IMAGE, BLURRED_IMAGE, 23)


def get_bounds(kernel_size):
    bounds_down = []
    bounds_up = []
    for i in range(0, kernel_size**2, 1):
        bounds_down.append(0.0)
        bounds_up.append(1.0)
    return Bounds(bounds_down, bounds_up)


def obj_fun(kernel):
    kernel_2d = np.reshape(kernel, (-1, KERNEL_SIZE))
    #cv.normalize(kernel_2d, kernel_2d, 0.0, 1.0, cv.NORM_MINMAX)
    sharp_image = SIZE_PYRAMID[KERNEL_SIZE][0]
    blurred_image = SIZE_PYRAMID[KERNEL_SIZE][1]
    #deblurred_image = do_weiner_deconv_1c(blurred_image, kernel_2d, 0.0001)
    deblurred_image = do_RL_deconv(blurred_image, kernel_2d, iterations=1)
    #score = -1 * get_quality(deblurred_image, "fourier") #+ np.count_nonzero(deblurred_image)
    score = -1 * get_no_ref_quality(deblurred_image, "brisque")
    return score


def solve():
    bounds = get_bounds(KERNEL_SIZE)
    kernel = np.zeros(KERNEL_SIZE**2, np.float32)
    kernel_2d = np.reshape(kernel, (-1, KERNEL_SIZE))
    for i in range(0, KERNEL_SIZE, 1):
        for j in range(0, KERNEL_SIZE, 1):
            if j == KERNEL_SIZE - 1 - i:
                kernel_2d[i][j] = 1.0
    kernel = kernel_2d.flatten()

    res = minimize(obj_fun, kernel, method='COBYLA', bounds=bounds)
    result_2d = np.reshape(res.x, (-1, KERNEL_SIZE))
    cv.normalize(result_2d, result_2d, 0.0, 1.0, cv.NORM_MINMAX)
    return result_2d


image = cv.imread("../images/sharp/3.png", cv.IMREAD_GRAYSCALE)
image = np.float32(image)
cv.normalize(image, image, 0.0, 1.0, cv.NORM_MINMAX)

psf = cv.imread("../images/psfs/1.png", cv.IMREAD_GRAYSCALE)
psf = np.float32(psf)
cv.normalize(psf, psf, 0.0, 1.0, cv.NORM_MINMAX)
psf = FilteringUtils.pad_to_shape(psf, image.shape)

blurred = FilteringUtils.freq_filter(image, psf)
cv.normalize(blurred, blurred, 0.0, 255.0, cv.NORM_MINMAX)
cv.imwrite("../images/blurred/3.png", blurred)



image = cv.imread("../images/sharp/bstu2.jpg", cv.IMREAD_GRAYSCALE)
cv.imwrite("../images/sharp/bstu2_gs.jpg", image)
# image = np.float32(image)
# cv.normalize(image, image, 0.0, 1.0, cv.NORM_MINMAX)

# psf = cv.imread("../images/psfs/2.png", cv.IMREAD_GRAYSCALE)
# psf = np.float32(psf)
# cv.normalize(psf, psf, 0.0, 1.0, cv.NORM_MINMAX)
# psf = pad_to_shape(psf, image.shape)
#
# blurred = freq_filter_1c(image, psf)
# cv.normalize(blurred, blurred, 0.0, 255.0, cv.NORM_MINMAX)
# cv.imwrite("../images/blurred/4.png", blurred)


#image = np.float32(image)

# line = draw_line((500, 500), 45, 100)
# line = np.float32(line)
# cv.normalize(line, line, 0.0, 1.0, cv.NORM_MINMAX)
# cv.imshow("line", cv.resize(line, None, fx=3, fy=3, interpolation=cv.INTER_AREA))
# cv.waitKey()

#cv.imwrite("../images/psfs/line_22_45.jpg", cv.resize(line, None, fx=10, fy=10, interpolation=cv.INTER_AREA))

#result = cv.filter2D(image, -1, line)
# result = freq_filter_dft_1c(image, line)
# cv.normalize(result, result, 0.0, 255.0, cv.NORM_MINMAX)
# cv.imshow("result", result.astype(np.float32))
# cv.imwrite("../images/blurred/2.jpg", result)
# cv.waitKey()