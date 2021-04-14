import cv2 as cv
import numpy as np
from skimage import color, data, restoration
from utils.drawing import draw_line
from utils.metric import *
from utils.deconv import *

image = cv.imread("./IMGS/motion.jpg", cv.IMREAD_COLOR)
image = np.float32(image)
cv.normalize(image, image, 0.0, 1.0, cv.NORM_MINMAX)


# line1 = draw_line((23, 23), 22, 45)
# line2 = draw_line((23, 23), 21, 163)

for i in range(0, 180, 5):
    print(i)
    line1 = draw_line((23, 23), 22, i)
    restored_image1 = do_RL_deconv_3c(image, line1, iterations=15)
    cv.imshow("restored_image1", restored_image1)
    quality1 = get_quality(restored_image1, type="dark")
    print("quality1: ", quality1)
    cv.waitKey()


# restored_image1 = do_RL_deconv_3c(image, line1, iterations=15)
# cv.normalize(restored_image1, restored_image1, 0, 255, cv.NORM_MINMAX)
# restored_image1 = np.uint8(restored_image1)
# cv.imshow("restored_image1", restored_image1)
# quality1 = get_quality(restored_image1, type="dark")
# print("quality1: ", quality1)
#
#
# restored_image2 = do_RL_deconv_3c(image, line2, iterations=15)
# cv.normalize(restored_image2, restored_image2, 0, 255, cv.NORM_MINMAX)
# restored_image2 = np.uint8(restored_image2)
# cv.imshow("restored_image2", restored_image2)
# quality2 = get_quality(restored_image2, type="dark")
# print("quality2: ", quality2)

#cv.waitKey()
