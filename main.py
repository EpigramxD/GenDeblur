
import cv2 as cv
import cv2.quality as q
import imquality.brisque as brisque
from skimage.metrics import structural_similarity as ssim
import numpy as np
from skimage import color, data, restoration
from utils.drawing import draw_line
from utils.metric import *
from utils.misc import *
from utils.deconv import *

image = cv.imread("D:\Images\IMGS\pic6.jpg", cv.IMREAD_GRAYSCALE)
image = np.float32(image)
cv.normalize(image, image, 0.0, 1.0, cv.NORM_MINMAX)

line = draw_line((23, 23), 22, 45)
cv.imshow("line", line)
restored_image1 = do_RL_deconv(image, line, iterations=15)
cv.normalize(restored_image1, restored_image1, 0, 255, cv.NORM_MINMAX)
restored_image1 = np.uint8(restored_image1)
cv.imshow("restored_image1", restored_image1)
cv.waitKey()


#
# resized_image = cv.resize(image, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
# cv.imshow("resized_image", resized_image)
#
# resized_line = cv.resize(line, None, fx=10, fy=10, interpolation=cv.INTER_CUBIC)
# cv.imshow("resized_line", resized_line)
# resized_line2 = cv.resize(resized_line, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
# cv.imshow("resized_line2", resized_line2)
# cv.waitKey()

# data frame for writing stats in excel file
# df = {'angle': [], 'length': [], 'quality': []}

# for i in range(0, 180, 5):
#     for j in range(3, 25, 1):
#         print(f"Progress: {i}:{j}")
#         line1 = draw_line((23, 23), j, i)
#         restored_image1 = do_RL_deconv_3c(image, line1, iterations=1)
#         #quality = get_quality(restored_image1, type="dark")
#         #ssim_measure = ssim(to_grayscale(image), to_grayscale(restored_image1))
#         quality = brisque.score(restored_image1)
#         df['angle'].append(i)
#         df['length'].append(j)
#         df['quality'].append(quality)
#
# df = pd.DataFrame(df)
# df.to_excel('D:/dark_test.xlsx')


# line = draw_line((23, 23), best_length, best_angle)
# restored_image1 = do_RL_deconv_1c(image, line, iterations=15)
# cv.normalize(restored_image1, restored_image1, 0, 255, cv.NORM_MINMAX)
# restored_image1 = np.uint8(restored_image1)
# cv.imshow("restored_image1", restored_image1)
# cv.waitKey()


# restored_image2 = do_RL_deconv_3c(image, line2, iterations=15)
# cv.normalize(restored_image2, restored_image2, 0, 255, cv.NORM_MINMAX)
# restored_image2 = np.uint8(restored_image2)
# cv.imshow("restored_image2", restored_image2)
# quality2 = get_quality(restored_image2, type="dark")
# print("quality2: ", quality2)
