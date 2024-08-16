#! /usr/bin/env python
# -*- coding UTF-8 -*-

"""
@Author : tangxx11
@Since  : 2022/3/10 下午2:35
"""

import numpy as np

from classification import ClassificationMeter
from object_detection import MAPMeter
from semantic_segmentation import IouMeter


def classification_test():
    """Compute accuracy, precision, recall
    :params:
        target: true label with shape (n, ) or (n, 1).
            each number represent for one image's true label
        output: model predict result with shape (n, ) or (n, 1).
            each number represent for the image's predict label
    :return:
    """
    meter = ClassificationMeter()

    # This means there are four images, and the true label of them are 0, 1, 2, 0
    target = np.array([0, 1, 2, 0, 1])  # shape: (n, )
    output = np.array([0, 1, 2, 1, 1])  # shape: (n, )
    meter.update(target, output)  # add array to meter
    acc = meter.accuracy()  # compute accuracy
    print(f"{acc:.2f}")
    # or
    _target = target.reshape(-1, 1)  # reshape target to (n, 1)
    _output = output.reshape(-1, 1)  # reshape output to (n, 1)
    print("target: ", _target)
    meter.update(_target, _output)
    _acc = meter.accuracy()
    print(f"{_acc:.2f}")

    # precision and recall for each class, return a ndarray
    # e.g. precision: [0.5, 1, 1] recall: [1, 0.667, 1]
    # for class 0, precision is 0.5, class 1 is 1 and etc.
    # for class 0, recall is 1, class 1 is 0.667 and etc.
    precision = meter.precision()
    recall = meter.recall()
    print(precision, recall)


def detection_test():
    """
    :param:
        name: image filename, dtype: str
        target: true label. dtype: nested array with shape (n, 5)
                n represents for the number of bboxes in the image
                5 represents [x, y, h, w, class_label], these should be float
        output: model predict result. dtype: nested array with shape (n, 6)
                n represents for the number of bboxes in the image
                6 represents [x, y, h, w, class_label, confidence], these should be float
    :return:
    """
    meter = MAPMeter()

    # image filename
    filename = "007113.jpg"

    # target and output's bboxes are normalized to 0~1
    # target: [x, y, h, w, class] output: [x, y, h, w, class, confidence]
    target = np.array([[0.6038, 0.4989, 0.0692, 0.1674, 0.0000],
                       [0.4981, 0.4954, 0.4129, 0.9193, 1.0000]])
    output = np.array([[0.5038, 0.4589, 0.0612, 0.1174, 0.0000, 0.7800],
                       [0.4281, 0.4754, 0.3129, 0.8193, 1.0000, 0.2508]])
    print("shape: ", target.shape, output.shape)

    meter.update(filename, target, output)
    map, ap50 = meter.m_ap()
    print(map)
    print(ap50)


def segmentation_test():
    """Segmentation meter test
    """
    meter = IouMeter()

    np.random.seed(20)
    target = np.random.normal(0, 1, (5, 256, 256))
    output = np.random.normal(0, 1, (5, 256, 256))
    # target = (target > 0.5).astype(int)
    # output = (output > 0.5).astype(int)
    # print(target, output)
    print(target.shape, output.shape)

    meter.update(target, output, [0, 1, 2, 1, 0])
    miou = meter.m_iou()
    print(miou)


def main():
    """Main
    """
    # classification_test()
    # detection_test()
    segmentation_test()


if __name__ == '__main__':
    main()
