from utils.deconv import *
from scipy.optimize import minimize
from scipy.optimize import Bounds
from utils.size_utils import *

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
    score = -1 * get_quality(deblurred_image, "brisque")
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


result = solve()
upscaled_result = cv.resize(result, None, fx=10, fy=10, interpolation=cv.INTER_AREA)
cv.imshow("result", upscaled_result)
cv.waitKey()