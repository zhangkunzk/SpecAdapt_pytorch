#!/usr/bin/env python3


import json
import os
from typing import Union, Sequence

import cv2 as cv
import numpy as np

__all__ = ['CameraCalibration']


class CameraCalibration(object):
    """Camera calibration
    """

    def __init__(
            self,
            inner_point_x: int,
            inner_point_y: int,
            image_list: Union[str, Sequence[np.ndarray]] = None,
    ) -> None:
        self.mtx = None
        self.dist = None
        self.inner_point_x = inner_point_x
        self.inner_point_y = inner_point_y

        if image_list is not None:
            self.register(image_list)

    def register(self, image_list: Union[str, Sequence[np.ndarray]]) -> None:
        """Register
        """
        if isinstance(image_list, str):
            path = image_list
            image_list = []
            for filename in os.listdir(path):
                image = cv.imread(os.path.join(path, filename), cv.IMREAD_COLOR)
                image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
                image_list.append(image)

        object_points = []
        image_points = []
        gray = None

        for _, image in enumerate(image_list):
            gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
            objp = np.zeros((self.inner_point_x * self.inner_point_y, 3), np.float32)
            objp[:, :2] = np.mgrid[0: self.inner_point_x, 0: self.inner_point_y].T.reshape(-1, 2)
            ret, corners = cv.findChessboardCorners(
                image=gray,
                patternSize=(self.inner_point_x, self.inner_point_y),
                corners=None,
                flags=cv.CALIB_CB_ADAPTIVE_THRESH
            )
            if ret:
                object_points.append(objp)
                image_points.append(corners)

        _, self.mtx, self.dist, _, _ = cv.calibrateCamera(
            objectPoints=object_points,
            imagePoints=image_points,
            imageSize=gray.shape[::-1],
            cameraMatrix=None,
            distCoeffs=None
        )

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply
        """
        return cv.undistort(image, self.mtx, self.dist, None, self.mtx)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.apply(image)

    def get_params(self):
        """Get parameters
        """
        params = {
            'mtx': self.mtx,
            'dist': self.dist,
            'inner_point_x': self.inner_point_x,
            'inner_point_y': self.inner_point_y,
        }
        return params

    def set_params(self, params):
        """Set parameters
        """
        self.mtx = params['mtx']
        self.dist = params['dist']
        self.inner_point_x = params['inner_point_x']
        self.inner_point_y = params['inner_point_y']

    def load_params(self, path):
        """Load parameters
        """
        with open(path, 'r', encoding='utf-8') as f:
            self.set_params(json.load(f))

    def save_params(self, path):
        """Save parameters
        """
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.get_params(), f)
