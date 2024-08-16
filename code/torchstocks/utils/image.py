#!/usr/bin/env python3

import random
from ast import literal_eval
from typing import Tuple
from typing import Union

import cv2 as cv
import imgaug.augmenters as iaa
import numpy as np
from imgaug import BoundingBoxesOnImage

__all__ = [
    'IMAGENET_MEAN',
    'IMAGENET_STD',
    'read_image',
    'normalize_image',
    'denormalize_image',
    'hwc_to_chw',
    'chw_to_hwc',
    'RandomCropToSquare',
    'ResizedCrop',
    'RandomColor',
    'RandomAffine',
    'RandomResize',
    'RandomCrop',
    'BasicImageAugmenter',
    'Mosaic',
    'ColorJitter'
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def read_image(path_or_data):
    """Read image
    """
    if isinstance(path_or_data, str):
        # Load image from file path.
        image = cv.imread(path_or_data, cv.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f'Failed to load image {path_or_data}')
        cv.cvtColor(image, cv.COLOR_BGR2RGB, image)  # opencv load image as BGR by default
    elif isinstance(path_or_data, bytes):
        # Load image from bytes of the image file.
        image = cv.imdecode(np.frombuffer(path_or_data, np.byte), cv.IMREAD_COLOR)
        if image is None:
            raise RuntimeError('Failed to load image')
        cv.cvtColor(image, cv.COLOR_BGR2RGB, image)  # opencv load image as BGR by default
    elif isinstance(path_or_data, np.ndarray):
        # Image already loaded.
        image = path_or_data
    else:
        raise RuntimeError(f'Invalid input type {type(path_or_data)}.')
    return image


def normalize_image(
        image: np.ndarray,
        mean: Union[np.ndarray, float] = IMAGENET_MEAN,
        std: Union[np.ndarray, float] = IMAGENET_STD,
        transpose=False
) -> np.ndarray:
    """Normalize image
    """
    image = np.array(image, dtype=np.float32)
    image -= mean
    image /= std
    if transpose:
        image = hwc_to_chw(image)
    return image


def denormalize_image(
        image: np.ndarray,
        mean: Union[np.ndarray, float] = IMAGENET_MEAN,
        std: Union[np.ndarray, float] = IMAGENET_STD,
        transpose=False
) -> np.ndarray:
    """Denormalize image
    """
    if transpose:
        image = chw_to_hwc(image)
    image *= std
    image += mean
    np.clip(image, 0, 255, out=image)
    image = np.array(image, dtype=np.uint8)
    return image


def hwc_to_chw(image: np.ndarray) -> np.ndarray:
    """HWC channel to CHW
    """
    if len(image.shape) != 3:
        raise RuntimeError('Image should be a 3-dimensional tensor/ndarray.')
    image = np.transpose(image, (2, 0, 1))
    image = np.ascontiguousarray(image)
    return image


def chw_to_hwc(image: np.ndarray) -> np.ndarray:
    """CHW channel to HWC
    """
    if len(image.shape) != 3:
        raise RuntimeError('Image should be a 3-dimensional tensor/ndarray.')
    image = np.transpose(image, (1, 2, 0))
    image = np.ascontiguousarray(image)
    return image


class RandomCropToSquare(iaa.Sequential):
    """Random crop to square
    """

    def __init__(
            self,
            size: int,
            scale: float = 1.0,
            pad: float = 0.0,
            pad_to_square=False,
            cval: int = 0,
            interpolation: str = 'area',
            train=True
    ) -> None:
        if train:
            size_ = (size, int(size * scale)) if scale is not None and scale != 1.0 else size
            super(RandomCropToSquare, self).__init__([
                iaa.PadToSquare(pad_cval=cval) if pad_to_square else iaa.Identity(),
                iaa.Resize({'shorter-side': size_, 'longer-side': 'keep-aspect-ratio'}, interpolation),
                iaa.Pad(percent=pad, pad_cval=cval,
                        keep_size=False) if pad is not None and pad > 0.0 else iaa.Identity(),
                iaa.CropToFixedSize(size, size)
            ])
        else:
            super(RandomCropToSquare, self).__init__([
                iaa.PadToSquare(pad_cval=cval) if pad_to_square else iaa.Identity(),
                iaa.Resize({'shorter-side': size, 'longer-side': 'keep-aspect-ratio'}, interpolation),
                iaa.CenterCropToSquare()
            ])


class ResizedCrop(iaa.Sequential):
    """Resized crop
    """

    def __init__(
            self,
            width: int,
            height: int,
            scale: float = 1.0,
            ratio: float = 1.33,
            interpolation='linear'
    ) -> None:
        assert scale >= 1.0, f'Invalid scale {scale}. It should >= 1.'
        assert ratio > 0, f'Invalid ratio {ratio}. It should > 0.'
        if ratio < 1.0:
            ratio = 1.0 / ratio
        min_width = int(width * scale)
        max_width = int(min_width * ratio)
        min_height = int(height * scale)
        max_height = int(min_height * ratio)
        super(ResizedCrop, self).__init__([
            iaa.Resize(
                {'width': (min_width, max_width), 'height': (min_height, max_height)},
                interpolation=interpolation
            ),
            iaa.CropToFixedSize(width=width, height=height),
        ])


class RandomColor(iaa.Sequential):
    """Random color
    """

    def __init__(
            self,
            rnd_hue: Union[float, Tuple[float], None] = 0.05,
            rnd_saturation: Union[float, Tuple[float], None] = 0.2,
            rnd_brightness: Union[float, Tuple[float], None] = 0.2,
            rnd_contrast: Union[float, Tuple[float], None] = 0.2
    ) -> None:
        """Randomly change the hue, saturation, brightness and contrast of an image.

        Args:
            rnd_hue: How much to jitter hue.
                hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
                Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
            rnd_saturation: How much to jitter saturation.
                saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
                or the given [min, max]. Should be non negative numbers.
            rnd_brightness: How much to jitter brightness.
                brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
                or the given [min, max]. Should be non negative numbers.
            rnd_contrast: How much to jitter contrast.
                contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
                or the given [min, max]. Should be non negative numbers.
        """
        if isinstance(rnd_hue, float):
            h = (-int(rnd_hue * 255), int(rnd_hue * 255))
        elif isinstance(rnd_hue, (tuple, list)) and len(rnd_hue) == 2:
            h = (int(rnd_hue[0] * 255), int(rnd_hue[1] * 255))
        elif rnd_hue is None:
            h = None
        else:
            raise RuntimeError(f'Invalid hue_shift {rnd_hue}.')

        if isinstance(rnd_saturation, float):
            s = (max(1.0 - rnd_saturation, 0), 1.0 + rnd_saturation)
        elif isinstance(rnd_saturation, (tuple, list)) and len(rnd_saturation) == 2:
            s = rnd_saturation
        elif rnd_saturation is None:
            s = None
        else:
            raise RuntimeError(f'Invalid saturation_factor {rnd_saturation}.')

        if isinstance(rnd_brightness, float):
            v = (max(1.0 - rnd_brightness, 0), 1.0 + rnd_brightness)
        elif isinstance(rnd_brightness, (tuple, list)) and len(rnd_brightness) == 2:
            v = rnd_brightness
        elif rnd_brightness is None:
            v = None
        else:
            raise RuntimeError(f'Invalid brightness_factor {rnd_brightness}.')

        if isinstance(rnd_contrast, float):
            c = (max(1.0 - rnd_contrast, 0), 1.0 + rnd_contrast)
        elif isinstance(rnd_contrast, (tuple, list)) and len(rnd_contrast) == 2:
            c = rnd_contrast
        elif rnd_contrast is None:
            c = None
        else:
            raise RuntimeError(f'Invalid contrast_factor {rnd_contrast}.')

        super(RandomColor, self).__init__([
            iaa.WithColorspace(
                from_colorspace=iaa.CSPACE_RGB,
                to_colorspace=iaa.CSPACE_HSV,
                children=iaa.Sequential([
                    iaa.WithChannels(0, iaa.Add(h)) if h else iaa.Identity(),
                    iaa.WithChannels(1, iaa.Multiply(s)) if s else iaa.Identity(),
                    iaa.WithChannels(2, iaa.Multiply(v)) if v else iaa.Identity()
                ])
            ) if (h and s and v) else iaa.Identity(),
            iaa.LinearContrast(c) if c else iaa.Identity()
        ])


class RandomAffine(iaa.Sequential):
    """Random affine
    """

    def __init__(
            self,
            p_scale: float = 1.0,
            rnd_scale: float = 0.0,
            rnd_scale_x: float = None,
            rnd_scale_y: float = None,
            p_shear: float = 1.0,
            rnd_shear: float = 0.0,
            rnd_shear_x: float = None,
            rnd_shear_y: float = None,
            p_rotate: float = 1.0,
            rnd_rotate: float = 0.0,
            fill_mode: str = 'constant',
            fill_value: float = 127.5
    ) -> None:
        aug_list = []

        if rnd_scale_x is None:
            rnd_scale_x = rnd_scale if rnd_scale is not None else 0.0
        if rnd_scale_y is None:
            rnd_scale_y = rnd_scale if rnd_scale is not None else 0.0
        if rnd_scale_x or rnd_scale_y:
            assert 0 < rnd_scale_x <= 1
            assert 0 < rnd_scale_y <= 1
            scale_aug = iaa.Affine(
                scale={
                    'x': (1 - rnd_scale_x, 1 + rnd_scale_x),
                    'y': (1 - rnd_scale_y, 1 + rnd_scale_y)
                },
                mode=fill_mode,
                cval=fill_value
            )
            aug_list.append(iaa.Sometimes(p_scale, scale_aug))

        if rnd_shear_x is None:
            rnd_shear_x = rnd_shear if rnd_shear is not None else 0.0
        if rnd_shear_y is None:
            rnd_shear_y = rnd_shear if rnd_shear is not None else 0.0
        if rnd_shear_x or rnd_shear_y:
            shear_aug = iaa.Affine(
                shear={
                    'x': (0 - rnd_shear_x, rnd_shear_x),
                    'y': (0 - rnd_shear_y, rnd_shear_y)
                },
                mode=fill_mode,
                cval=fill_value
            )
            aug_list.append(iaa.Sometimes(p_shear, shear_aug))

        if rnd_rotate:
            rotate_aug = iaa.Rotate((-rnd_rotate, rnd_rotate), mode=fill_mode, cval=fill_value)
            aug_list.append(iaa.Sometimes(p_rotate, rotate_aug))
        super(RandomAffine, self).__init__(aug_list)


class RandomResize(iaa.Resize):
    """Random resize
    """

    def __init__(
            self,
            rnd_resize: float = 0.0,
            rnd_resize_x: float = None,
            rnd_resize_y: float = None,
            interpolation='linear'
    ) -> None:
        if rnd_resize_x is None:
            rnd_resize_x = rnd_resize if rnd_resize is not None else 0.0
        if rnd_resize_y is None:
            rnd_resize_y = rnd_resize if rnd_resize is not None else 0.0

        assert rnd_resize_x >= 0
        assert rnd_resize_y >= 0
        super(RandomResize, self).__init__(
            {'width': (1.0, 1.0 + rnd_resize_x), 'height': (1.0, 1.0 + rnd_resize_y)},
            interpolation=interpolation
        )


class RandomCrop(iaa.Sequential):
    """Random crop
    """

    def __init__(
            self,
            size: Union[int, Tuple[int, int]],
            shorter_side: Union[int, float] = None,
            longer_side: Union[int, float] = None,
            pad_position: str = 'center',
            pad_cval: float = 127.5,
            crop_position: str = 'uniform',
            interpolation='area'
    ) -> None:
        aug_list = []

        if isinstance(size, str):
            size = literal_eval(size)
        if isinstance(size, int):
            width = height = size
        elif isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError('"size" should only have 2 elements.')
            width, height = size
        else:
            raise ValueError(f'Invalid size type "{type(size)}".')

        if width == height:
            if isinstance(shorter_side, float):
                shorter_side = int(shorter_side * width)
            if isinstance(longer_side, float):
                longer_side = int(longer_side * width)

        if shorter_side or longer_side:
            if not shorter_side:
                shorter_side = 'keep-aspect-ratio'
            if not longer_side:
                longer_side = 'keep-aspect-ratio'
            aug_list.append(iaa.Resize(
                {'shorter-side': shorter_side, 'longer-side': longer_side},
                interpolation=interpolation
            ))

        aug_list.append(iaa.PadToFixedSize(
            width=width,
            height=height,
            pad_cval=pad_cval,
            position=pad_position
        ))
        aug_list.append(iaa.CropToFixedSize(width=width, height=height, position=crop_position))

        super(RandomCrop, self).__init__(aug_list)


class BasicImageAugmenter(iaa.Sequential):
    """Basic image augmenter
    """

    def __init__(
            self,
            p_flip_lr: float = 0.0,
            p_flip_ud: float = 0.0,
            p_color: float = 1.0,
            rnd_hue: float = 0.0,
            rnd_saturation: float = 0.0,
            rnd_brightness: float = 0.0,
            rnd_contrast: float = 0.0,
            p_grayscale: float = 0.0,
            p_scale: float = 1.0,
            rnd_scale: float = 0.0,
            rnd_scale_x: float = None,
            rnd_scale_y: float = None,
            p_shear: float = 1.0,
            rnd_shear: float = 0.0,
            rnd_shear_x: float = None,
            rnd_shear_y: float = None,
            p_rotate: float = 1.0,
            rnd_rotate: float = 0.0,
            p_resize: float = 1.0,
            rnd_resize: float = 0.0,
            rnd_resize_x: float = None,
            rnd_resize_y: float = None,
            pad_pct: float = 0.0,
            fill_mode: str = 'constant',
            fill_value: float = 127.5,
            interpolation='linear'
    ) -> None:
        aug_list = []

        if p_flip_lr:
            assert 0 < p_flip_lr <= 1
            aug_list.append(iaa.Fliplr(p_flip_lr))
        if p_flip_ud:
            assert 0 < p_flip_ud <= 1
            aug_list.append(iaa.Flipud(p_flip_ud))

        if rnd_hue or rnd_saturation or rnd_brightness or rnd_contrast:
            aug_list.append(iaa.Sometimes(p_color, RandomColor(
                rnd_hue=rnd_hue,
                rnd_saturation=rnd_saturation,
                rnd_brightness=rnd_brightness,
                rnd_contrast=rnd_contrast
            )))

        if p_grayscale:
            assert 0 < p_grayscale <= 1
            aug_list.append(iaa.Sometimes(p_grayscale, iaa.Grayscale()))

        aug_list.append(RandomAffine(
            p_scale=p_scale,
            rnd_scale=rnd_scale,
            rnd_scale_x=rnd_scale_x,
            rnd_scale_y=rnd_scale_y,
            p_shear=p_shear,
            rnd_shear=rnd_shear,
            rnd_shear_x=rnd_shear_x,
            rnd_shear_y=rnd_shear_y,
            p_rotate=p_rotate,
            rnd_rotate=rnd_rotate,
            fill_mode=fill_mode,
            fill_value=fill_value
        ))

        if p_resize:
            aug_list.append(iaa.Sometimes(p_resize, RandomResize(
                rnd_resize=rnd_resize,
                rnd_resize_x=rnd_resize_x,
                rnd_resize_y=rnd_resize_y,
                interpolation=interpolation
            )))

        if pad_pct:
            aug_list.append(iaa.Pad(percent=pad_pct, pad_cval=fill_value, keep_size=False))

        super(BasicImageAugmenter, self).__init__(aug_list)


class Mosaic(iaa.Augmenter):
    """Image mosaic
    """

    def __init__(self, image_size, shrink=0.1, cval=0.0):
        super(Mosaic, self).__init__()
        self.cval = cval
        s = image_size
        c = shrink
        self.aug_list = [
            iaa.Sequential([
                iaa.Resize({'longer-side': s, 'shorter-side': 'keep-aspect-ratio'}, interpolation='area'),
                iaa.Crop(percent=(0, (0, c), (0, c), 0), keep_size=False),
                iaa.ClipCBAsToImagePlanes(),
                iaa.PadToFixedSize(width=s, height=s, position='left-top', pad_cval=cval),
                iaa.PadToFixedSize(width=2 * s, height=2 * s, position='right-bottom', pad_cval=cval)
            ]),
            iaa.Sequential([
                iaa.Resize({'longer-side': s, 'shorter-side': 'keep-aspect-ratio'}, interpolation='area'),
                iaa.Crop(percent=(0, 0, (0, c), (0, c)), keep_size=False),
                iaa.ClipCBAsToImagePlanes(),
                iaa.PadToFixedSize(width=s, height=s, position='right-top', pad_cval=cval),
                iaa.PadToFixedSize(width=2 * s, height=2 * s, position='left-bottom', pad_cval=cval)
            ]),
            iaa.Sequential([
                iaa.Resize({'longer-side': s, 'shorter-side': 'keep-aspect-ratio'}, interpolation='area'),
                iaa.Crop(percent=((0, c), (0, c), 0, 0), keep_size=False),
                iaa.ClipCBAsToImagePlanes(),
                iaa.PadToFixedSize(width=s, height=s, position='left-bottom', pad_cval=cval),
                iaa.PadToFixedSize(width=2 * s, height=2 * s, position='right-top', pad_cval=cval)
            ]),
            iaa.Sequential([
                iaa.Resize({'longer-side': s, 'shorter-side': 'keep-aspect-ratio'}, interpolation='area'),
                iaa.Crop(percent=((0, c), 0, 0, (0, c)), keep_size=False),
                iaa.ClipCBAsToImagePlanes(),
                iaa.PadToFixedSize(width=s, height=s, position='right-bottom', pad_cval=cval),
                iaa.PadToFixedSize(width=2 * s, height=2 * s, position='left-top', pad_cval=cval)
            ])
        ]

    def _augment_batch_(self, batch, random_state, parents, hooks):
        kwargs_list = [{} for _ in range(4)]
        result_dict = {}

        if batch.images is not None:
            image_list = batch.images
            assert len(image_list) == 4
            for i in range(4):
                kwargs_list[i]['image'] = image_list[i]
                result_dict['image'] = []

        if batch.bounding_boxes is not None:
            bboi_list = batch.bounding_boxes
            assert len(bboi_list) == 4
            for i in range(4):
                kwargs_list[i]['bounding_boxes'] = bboi_list[i]
                result_dict['bounding_boxes'] = []

        aug_list = list(self.aug_list)
        random.shuffle(aug_list)
        for i in range(4):
            ret = aug_list[i](**kwargs_list[i])
            if not isinstance(ret, tuple):
                result_dict['image'].append(ret)
            else:
                for result, name in zip(ret, result_dict):
                    result_dict[name].append(result)

        if 'image' in result_dict:
            image_list = result_dict['image']
            image = np.zeros_like(image_list[0], dtype=np.int16)
            for i in range(0, 4):
                image += image_list[i]
            image -= int(3 * self.cval)
            image = np.clip(image, 0, 255).astype(np.uint8)
            batch.images = [image]

        if 'bounding_boxes' in result_dict:
            bboi_list = result_dict['bounding_boxes']
            bbox_list = []
            for bboi in bboi_list:
                bbox_list.extend(bboi.bounding_boxes)
            batch.bounding_boxes = [BoundingBoxesOnImage(
                bounding_boxes=bbox_list,
                shape=bboi_list[0].shape
            )]

        return batch

    def __call__(self, *args, **kwargs):
        ret = super(Mosaic, self).__call__(*args, **kwargs)
        if isinstance(ret, tuple):
            return tuple(value[0] for value in ret)
        else:
            return ret[0]

    def get_parameters(self):
        params = []
        for aug in self.aug_list:
            params.extend(aug.get_parameters())
        return params


ColorJitter = RandomColor

# class AugmenterAdapter(object):
#
#     def __init__(self, augmenter: iaa.Augmenter, data_format='xywh'):
#         self.augmenter = augmenter
#         self.data_format = data_format
#
#     def __call__(self, **kwargs):
#         assert 'image' in kwargs
#         aug_kwargs = {'image': kwargs['image']}
#
#         # Some operations (e.g., coordinate transform for bboxes) may use the information of the image.
#         image = kwargs['image']
#         # Record the position of the image in the arguments, so that the augmented image can be extract from the
#         # augmenter's returns by this position.
#         index = None
#
#         for i, (name, value) in enumerate(kwargs.items()):
#             if name == 'image':
#                 aug_kwargs['image'] = value
#                 index = i
#             elif name == 'mask':
#                 mask = value
#                 aug_kwargs['segmentation_maps'] = SegmentationMapsOnImage(
#                     arr=mask,
#                     shape=mask.shape
#                 )
#             elif name == 'bboxes':
#                 bboxes = value
#                 ih, iw = image.shape[0], image.shape[1]
#                 bbox_objs = []
#                 for bbox in bboxes:
#                     if self.data_format == 'xyxy':
#                         x1, y1, x2, y2, label = bbox
#                     elif self.data_format == 'xywh':
#                         x, y, w, h, label = bbox
#                         x, y, w, h = x * iw, y * ih, w * iw, h * ih
#                         ow, oh = w * 0.5, h * 0.5
#                         x1, y1, x2, y2 = x - ow, y - oh, x + ow, y + oh
#                     else:
#                         raise RuntimeError(f'Unsupported data format "{self.data_format}".')
#                     bbox_obj = BoundingBox(x1, y1, x2, y2, label)
#                     bbox_objs.append(bbox_obj)
#                 aug_kwargs['bounding_boxes'] = BoundingBoxesOnImage(
#                     bounding_boxes=bbox_objs,
#                     shape=image.shape
#                 )
#             else:
#                 raise RuntimeError(f'Unsupported argument "{name}".')
#
#         aug_results = self.augmenter(**aug_kwargs)
#         if not isinstance(aug_results, tuple):
#             return aug_results
#
#         # Image is changed after the augmentation.
#         # So any operation depends on the image should use the new one.
#         image = aug_results[index]
#
#         results = []
#         for name, value in zip(kwargs, aug_results):
#             if name == 'image':
#                 results.append(value)
#             elif name == 'mask':
#                 seg_on_image = value
#                 results.append(seg_on_image.arr.squeeze(2))
#             elif name == 'bboxes':
#                 bbox_objs = value
#                 bbox_objs = bbox_objs.remove_out_of_image_fraction(0.8).clip_out_of_image()
#                 bboxes = np.empty((len(bbox_objs), 5), dtype=np.float32)
#                 ih, iw = image.shape[0], image.shape[1]
#                 for i, bbox_obj in enumerate(bbox_objs):
#                     x1, y1, x2, y2, label = bbox_obj.x1, bbox_obj.y1, bbox_obj.x2, bbox_obj.y2, bbox_obj.label
#                     if self.data_format == 'xyxy':
#                         bboxes[i] = x1, y1, x2, y2, label
#                     elif self.data_format == 'xywh':
#                         x, y, w, h = (x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1
#                         x, y, w, h = x / iw, y / ih, w / iw, h / ih
#                         bboxes[i] = x, y, w, h, label
#                     else:
#                         raise RuntimeError(f'Unsupported data format "{self.data_format}".')
#                 results.append(bboxes)
#             else:
#                 raise RuntimeError(f'Unsupported argument "{name}".')
#         return tuple(results)

# class AugmentedSquareCrop(iaa.Sequential):
#
#     def __init__(
#             self,
#             image_size: int,
#             shorter_side: Optional[Union[int, float]] = None,
#             longer_side: Optional[Union[int, float]] = None,
#             interpolation='area',
#             augmenter: Optional[iaa.Augmenter] = None,
#             train: bool = False
#     ) -> None:
#         warnings.warn('AugmentedSquareCrop will be deprecated. Please use RandomCrop instead.')
#         if train:
#             if isinstance(shorter_side, float):
#                 shorter_side = int(shorter_side * image_size)
#             if isinstance(longer_side, float):
#                 longer_side = int(longer_side * image_size)
#             if augmenter is None:
#                 augmenter = iaa.Identity()
#             crop_position = 'uniform'
#         else:
#             if isinstance(shorter_side, float):
#                 shorter_side = image_size
#             if isinstance(longer_side, float):
#                 longer_side = image_size
#             augmenter = iaa.Identity()
#             crop_position = 'center'
#
#         super(AugmentedSquareCrop, self).__init__([
#             augmenter,
#             RandomCrop(
#                 size=image_size,
#                 shorter_side=shorter_side,
#                 longer_side=longer_side,
#                 interpolation=interpolation,
#                 crop_position=crop_position
#             )
#         ])
