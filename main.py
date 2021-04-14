import cv2 as cv
import numpy as np
from skimage import color, data, restoration
from utils.drawing import draw_line

image = cv.imread("./IMGS/motion.jpg", cv.IMREAD_GRAYSCALE)
image = np.float32(image)
cv.normalize(image, image, 0.0, 1.0, cv.NORM_MINMAX)
cv.imshow("original", image)

line = draw_line((23, 23), 22, 45)
cv.imshow("line", line)
restored_image = restoration.richardson_lucy(image, line, iterations=10, clip=True)
cv.normalize(restored_image, restored_image, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)
cv.imshow("restored", restored_image)
cv.waitKey()
