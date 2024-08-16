#! /usr/bin/env python
# -*- coding UTF-8 -*-

"""
@Author : tangxx11
@Since  : 2022/12/27 下午7:54
"""
import numpy as np
import cv2 as cv
from perlin_noise import PerlinNoise
from imgaug import augmenters as iaa


TEXTURE = ['wood.ds', 'leather.ds', 'grid.ds', 'carpet.ds']


class DraemSimulator(object):
    """Draem simulator
    """

    def __init__(self):

        self.transform = iaa.Sequential(
            [
                iaa.SomeOf(
                    3,
                    [
                        iaa.Fliplr(),
                        iaa.Rotate(),
                        iaa.AddToBrightness(),
                        iaa.AddToSaturation(),
                        iaa.AddToHue()
                    ]
                ),
                iaa.Jigsaw(nb_rows=8, nb_cols=8)
            ]
        )

    def __call__(self, image: np.ndarray, noise_source: np.ndarray, foreground_enhance: bool = True):
        """
        :param image: (h, w, 3)
        :param noise_source: textural noise from DTD, structural from input data
        :param foreground_enhance: for object category, foreground_enhance should be applied
        :return:
        """
        if foreground_enhance:
            bin_mask = self.image_thresholding(image)
        else:
            bin_mask = np.ones(image.shape[:2], dtype=np.uint8)

        perlin_mask = self.generate_perlin_mask(bin_mask)
        simu_anomaly = self.generate_anomaly(perlin_mask, noise_source)
        simu_image = self.generate_anomaly_image(simu_anomaly, image, perlin_mask)

        return simu_image, perlin_mask, simu_anomaly

    @staticmethod
    def image_thresholding(image):
        """
        :return: binary mask of input image in the shape of (h, w)
        """
        tmp = image.copy()
        tmp = cv.cvtColor(tmp, cv.COLOR_BGR2GRAY)
        _, bin_mask = cv.threshold(tmp, 0, 1, cv.THRESH_OTSU | cv.THRESH_BINARY)
        return bin_mask

    @staticmethod
    def generate_perlin_mask(image_mask: np.ndarray):
        """Generate perlin mask
        """
        # TODO: Low bound of threshold
        thr = np.random.uniform(0.6, 0.9)
        # generate noise
        noise = PerlinNoise(octaves=2, seed=5)
        x_axis, y_axis = 100, 100
        noise_img = np.array([[noise([i / x_axis, j / y_axis]) for j in range(y_axis)] for i in range(x_axis)])

        # scale mask
        min_val, max_val = np.min(noise_img), np.max(noise_img)
        noise_img = (noise_img - min_val) / (max_val - min_val)

        h, w = image_mask.shape
        noise_img = cv.resize(noise_img, (w, h), interpolation=cv.INTER_LINEAR)

        # thresholding
        noise_img[noise_img >= thr] = 1
        noise_img[noise_img < thr] = 0
        noise_img = noise_img.astype(np.uint8)

        noise_img = cv.bitwise_and(image_mask, noise_img)

        return noise_img

    def generate_anomaly(self, perlin_mask, noise_source):
        """Generate anomaly
        """
        mask = np.stack((perlin_mask, ) * 3, axis=-1)
        noise_source = self.transform(image=noise_source)
        noise_source = cv.resize(noise_source, perlin_mask.shape[::-1], cv.INTER_LINEAR)
        noise_source = noise_source * mask

        return noise_source

    @staticmethod
    def generate_anomaly_image(simu_mask, image, perlin_mask):
        """Generate anomaly image
        """
        factor = np.random.uniform(0.1, 1)
        mask = np.stack((perlin_mask, ) * 3, axis=-1)
        tmp = image.copy()
        tmp = tmp * (1 - mask)  # \hat_M * I
        foreground = factor * simu_mask + (1 - factor) * image * mask

        simu_image = tmp + foreground
        return simu_image.astype(np.uint8)
