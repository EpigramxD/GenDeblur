from utils.deconv import *
from utils.metric import *
from utils.misc import *
from utils.imgUtils import ImgUtils
from utils.imgQuality import ImgQuality

# image = cv.imread("images/blurred/noisy.jpg", cv.IMREAD_COLOR)
# image = im2double(image)

# image = np.array([
#     [[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]],
#     [[16,17,18],[19,20,21],[22,23,24],[25,26,27],[28,29,30]],
#     [[31,32,33],[34,35,36],[37,38,39],[40,41,42],[43,44,45]],
# ])

def display_dft(image, window_name):
    fourier_image = np.fft.fft2(image)
    magnitude = np.float32(np.log(1 + np.abs(fourier_image)))
    magnitude_to_show = np.zeros(magnitude.shape, np.float32)
    cv.normalize(magnitude, magnitude_to_show, 0, 255, cv.NORM_MINMAX)
    magnitude_to_show = np.ubyte(magnitude_to_show)
    magnitude_to_show = np.fft.fftshift(magnitude_to_show)
    cv.imshow(window_name, magnitude_to_show)
    return magnitude_to_show


sharp = cv.imread("images/sharp/bstu2.jpg", cv.IMREAD_GRAYSCALE)
sharp = ImgUtils.im2double(sharp)
sharp_dft = display_dft(sharp, "sharp_dft")

psf = cv.imread("images/psfs/2.png", cv.IMREAD_GRAYSCALE)

# подгон psf под размеры изображения
center_y = int(sharp.shape[0] / 2)
center_x = int(sharp.shape[1] / 2)

half_height = int(psf.shape[0] / 2)
half_width = int(psf.shape[1] / 2)

x_index_from = center_x - half_width
x_index_to = center_x + psf.shape[1] - half_width

y_index_from = center_y - half_height
y_index_to = center_y + psf.shape[0] - half_height

psf_resized = np.zeros(sharp.shape, np.double)
psf_resized[y_index_from:y_index_to, x_index_from:x_index_to] = psf.copy()
psf_resized = ImgUtils.im2double(psf_resized)

# фильтрация
image_fft = np.fft.fft2(sharp)
psf_fft = np.fft.fft2(psf_resized)

fft_mul = image_fft * psf_fft
fft_mul = ifft2(fft_mul).real
fft_mul = np.fft.fftshift(fft_mul)
result = np.zeros(fft_mul.shape)
cv.normalize(fft_mul, result, 0, 255, cv.NORM_MINMAX)
result = np.ubyte(result)

cv.imshow("result", result)
result_dft = display_dft(result, "result_dft")
print(ImgQuality.get_ref_qualiy(result_dft, sharp_dft, "psnr"))
cv.waitKey()