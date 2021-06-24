from numpy.fft import fft2, ifft2
from numpy.fft import fft2, ifft2
from skimage import restoration
from utils.metric import *


def do_RL_deconv_1c(image, psf, iterations, clip=True):
    """
    Метод Люси-Ричардсона для 1 канала
    :param image: изображение
    :param psf: функция искажения
    :param iterations: количество итераций
    :param clip: см. в skimage.restoration.richardson_lucy
    :return: результат
    """
    if len(image.shape) == 2:
        image_ = np.float32(image)
        cv.normalize(image_, image_, 0.0, 1.0, cv.NORM_MINMAX)
        result = restoration.richardson_lucy(image_, psf.copy(), iterations=iterations, clip=clip)
        result = np.float32(result)
        cv.normalize(result, result, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)
        return result
    else:
        raise AttributeError("Wrong image format, image should be grayscale")


def do_RL_deconv_3c(image, psf, iterations, clip=True):
    """
    Метод Люси-Ричардсона для 3 каналов
    :param image: изображение
    :param psf: функция искажения
    :param iterations: количество итераций
    :param clip: см. в skimage.restoration.richardson_lucy
    :return: результат
    """
    if len(image.shape) == 3:
        image_ = np.float32(image)
        channels = list(cv.split(image_))
        channels[:] = [do_RL_deconv_1c(channel, psf.copy(), iterations, clip) for channel in channels]
        result = cv.merge(channels)
        result = np.float32(result)
        cv.normalize(result, result, alpha=0.0, beta=1.0, norm_type=cv.NORM_MINMAX)
        return result
    else:
        raise AttributeError("Wrong image format, image should be RGB")


def do_RL_deconv(image, psf, iterations, clip=True):
    """
    Метод Люси-Ричардсона
    :param image: изображение
    :param psf: функция искажения
    :param iterations: количество итераций
    :param clip: см. в skimage.restoration.richardson_lucy
    :return: результат
    """
    if len(image.shape) == 3:
        return do_RL_deconv_3c(image, psf, iterations, clip)
    else:
        return do_RL_deconv_1c(image, psf, iterations, clip)


def do_weiner_deconv_1c(image, psf, K):
    """
    Фильтр Винера для одного канала
    :param image: фильтруемое изображение
    :param psf: функция искажения
    :param K: константа сигнал-шум
    :return: результат фильтрации
    """
    dummy = np.copy(image)
    dummy = np.fft.fft2(dummy)
    kernel = np.fft.fft2(psf, s=image.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(ifft2(dummy))
    result = np.zeros(dummy.shape)
    result = np.float32(result)
    dummy = np.float32(dummy)
    cv.normalize(dummy, result, 0.0, 1.0, cv.NORM_MINMAX)
    return result

# TODO do_weiner_deconv_3c