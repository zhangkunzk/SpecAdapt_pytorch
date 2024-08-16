#!/usr/bin/env python3

"""
@author: liying50, lvgy1
@since: 2022-12-27
"""

import abc
import pickle
import random
from typing import Union, Tuple, Literal, Sequence, Optional

import cv2 as cv
import numpy as np
import torch
from matplotlib import pyplot as plt

__all__ = [
    'ImageStitching'
]


class AbstractFilter(abc.ABC):
    """Abstract filter
    """

    @abc.abstractmethod
    def __call__(self, src, dst):
        pass


class FilterByYAxis(AbstractFilter):
    """Filter error matched points by y-axis
    """

    def __init__(self, num_bins=5):
        self.num_bins = num_bins

    def __call__(self, src, dst):
        dist = src[:, 1] - dst[:, 1]
        hist = np.histogram(dist, self.num_bins)
        idx = np.argmax(hist[0])
        big_bin = dist[(hist[1][idx] <= dist) & (dist <= hist[1][idx + 1])]
        mean, sigma = np.mean(big_bin), np.sqrt(np.var(big_bin))
        allow = (dist < mean + sigma) & (dist > mean - sigma)
        src = src[allow]
        dst = dst[allow]
        return src, dst


class FilterByDistance(AbstractFilter):
    """Filter error matched points by distance
    """

    def __init__(self, num_bins=5):
        self.num_bins = num_bins

    def __call__(self, src, dst):
        dist = np.sqrt(np.square(src - dst).sum(1))
        hist = np.histogram(dist, self.num_bins)
        idx = np.argmax(hist[0])
        big_bin = dist[(hist[1][idx] <= dist) & (dist <= hist[1][idx + 1])]
        mean, sigma = np.mean(big_bin), np.sqrt(np.var(big_bin))
        allow = (dist < mean + sigma) & (dist > mean - sigma)
        src = src[allow]
        dst = dst[allow]
        return src, dst


class AbstractTransformEstimator(abc.ABC):
    """Abstract transform estimator
    """

    @abc.abstractmethod
    def __call__(self, src, dst):
        pass


class RANSACEstimator(AbstractTransformEstimator):
    """RANSAC estimator
    """

    def __init__(
            self,
            estimator,
            max_iter=1000,
            min_iter=100,
            num_samples=0.6,
            accept_prob=0.95
    ) -> None:
        self.estimator = estimator
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.num_samples = num_samples
        self.accept_prob = accept_prob

    def __call__(self, src, dst):
        total = len(src)
        assert total >= 4
        one = np.ones((total, 1))
        aug_src = np.concatenate([src, one], 1)
        aug_dst = np.concatenate([dst, one], 1)
        idx_list = [i for i in range(total)]
        best_fit = -1
        best_H = None
        num_samples = self.num_samples
        if isinstance(num_samples, float):
            num_samples = int(total * self.num_samples + 0.5)
        num_samples = max(num_samples, 4)
        for i in range(self.max_iter):
            idx = random.sample(idx_list, num_samples)
            H = self.estimator(src[idx], dst[idx])
            dist = np.abs(aug_src @ H.T - aug_dst)[:, 0:2].mean(1)
            hist = np.histogram(dist, 5)
            num_fit = int(hist[0][0:1].sum())
            if num_fit > best_fit:
                best_fit = num_fit
                best_H = H
            # print(best_fit, num_fit, total)
            if i > self.min_iter and best_fit / total > self.accept_prob:
                break
        return best_H


class ScaleTranslateEstimator(AbstractTransformEstimator):
    """Scale translate estimator
    """

    def __call__(self, src, dst):
        n = len(src)
        v0 = np.zeros((n, 1))
        v1 = np.ones((n, 1))
        A = np.concatenate([
            np.concatenate([src[:, 0:1], v0, v1, v0], 1),
            np.concatenate([v0, src[:, 1:2], v0, v1], 1)
        ], 0)
        B = np.concatenate([dst[:, 0:1], dst[:, 1:2]], 0)
        A_inv = np.linalg.pinv(A)
        X = A_inv @ B
        a = float(X[0])
        b = float(X[1])
        c = float(X[2])
        d = float(X[3])
        return np.array([
            [a, 0, c],
            [0, b, d],
            [0, 0, 1]
        ])


class ImageStitching(object):
    """Image stitching
    """

    def __init__(
            self,
            overlap_factor=0.5,
            num_feats=2000,
            filters: Sequence[AbstractFilter] = None,
            estimator: Optional[AbstractTransformEstimator] = RANSACEstimator(ScaleTranslateEstimator()),
            ransac_reproj_threshold=5,
            orders=None,
            multirow=False,
            device=None
    ) -> None:
        super().__init__()
        self.num_feats = num_feats
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.overlap_factor = overlap_factor
        self.filters = filters
        self.estimator = estimator
        self.orders = orders
        self.multirow = multirow
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.shapes = None
        self.num_images = None
        self.middle_idx = None
        self.H_all = None

        self.canvas = None
        self.pts_list = None
        self.tl_list = None
        self.tr_list = None
        self.bl_list = None
        self.br_list = None
        self.weight_list = None
        self.coords_list = None

    def register(self, image_list: Sequence[np.ndarray]):
        """Register images
        """
        self.shapes = [(*image.shape,) for image in image_list]
        self.num_images = len(self.shapes)
        self.middle_idx = self.num_images // 2

        if __debug__:
            print(f'Registering shapes {self.shapes}')

        H_all = {}
        for i in range(len(image_list) - 1):
            _key = f'H{i}{i + 1}'
            H = find_homography_by_features(
                image_list[i],
                image_list[i + 1],
                num_feats=self.num_feats,
                ransac_reproj_threshold=self.ransac_reproj_threshold,
                overlap_factor=self.overlap_factor,
                filters=self.filters,
                estimator=self.estimator,
                multirow=self.multirow
            )
            assert H is not None
            H_all[_key] = H.astype(np.float32)
        self.H_all = self._compute_H_wrt_middle_img(H_all)

        self._init()

    def _compute_H_wrt_middle_img(self, H_all):
        num_images = len(H_all) + 1
        key = f'H{self.middle_idx}{self.middle_idx}'
        H_all[key] = np.eye(3, dtype=np.float32)

        for i in range(0, self.middle_idx):
            key = f'H{i}{self.middle_idx}'  # H02
            j = i
            temp = np.eye(3, dtype=np.float32)
            while j < self.middle_idx:
                key_t = f'H{j}{j + 1}'
                temp = np.matmul(H_all[key_t], temp)
                j += 1
            H_all[key] = temp

        for i in range(self.middle_idx + 1, num_images):
            key = f'H{i}{self.middle_idx}'  # H32
            j = i - 1
            temp = np.eye(3, dtype=np.float32)
            while j >= self.middle_idx:
                key_t = f'H{j}{j + 1}'
                temp = np.matmul(np.linalg.inv(H_all[key_t]), temp)
                j -= 1
            H_all[key] = temp

        return H_all

    def _init(self):
        if __debug__:
            print('Initializing')

        self.canvas, mask, offset = self._get_blank_canvas(self.shapes, self.H_all)
        self.pts_list = []
        self.tl_list = []
        self.tr_list = []
        self.bl_list = []
        self.br_list = []
        self.weight_list = []
        self.coords_list = []
        for i, shape in enumerate(self.shapes):
            key = f'H{i}{self.middle_idx}'
            H = torch.from_numpy(self.H_all[key]).to(self.device)
            img_src_h, img_src_w = shape[:2]
            pts, tl, tr, bl, br, weight = fit_image_in_target_space(
                img_src_h,
                img_src_w,
                mask,
                torch.linalg.inv(H),
                offset=offset,
                device=self.device
            )  # the inp to fit_image_in_target_space
            self.pts_list.append(pts)
            self.tl_list.append(tl)
            self.tr_list.append(tr)
            self.bl_list.append(bl)
            self.br_list.append(br)
            self.weight_list.append(weight)
            pts = pts.long()

            x1 = pts[:, 0].min()
            y1 = pts[:, 1].min()
            x2 = pts[:, 0].max()
            y2 = pts[:, 1].max()
            mask[torch.where(self.canvas)[0:2]] = 0
            self.coords_list.append((int(x1), int(y1), int(x2), int(y2)))

    def _get_blank_canvas(self, shapes, H_all):
        min_crd_canvas = np.array([np.inf, np.inf, np.inf])
        max_crd_canvas = np.array([-np.inf, -np.inf, -np.inf])

        for i, shape in enumerate(shapes):
            img_h, img_w = shape[:2]
            key = f'H{i}{self.middle_idx}'
            H = H_all[key]
            min_crd, max_crd = self._compute_extent(H, img_w, img_h)

            min_crd_canvas = np.minimum(min_crd, min_crd_canvas)
            max_crd_canvas = np.maximum(max_crd, max_crd_canvas)

        width_canvas = int(np.ceil(max_crd_canvas - min_crd_canvas)[0] + 1)
        height_canvas = int(np.ceil(max_crd_canvas - min_crd_canvas)[1] + 1)

        offset = min_crd_canvas.astype(np.int16)
        offset[2] = 0  # [x_offset, y_offset, 0]
        offset = torch.from_numpy(offset).to(self.device)
        mask = torch.ones((height_canvas, width_canvas), dtype=torch.bool).to(self.device)
        if len(shapes[0]) == 2:
            canvas_img = torch.zeros((height_canvas, width_canvas, 1), dtype=torch.uint8).to(self.device)
        else:
            canvas_img = torch.zeros((height_canvas, width_canvas, 3), dtype=torch.uint8).to(self.device)

        return canvas_img, mask, offset

    @staticmethod
    def _compute_extent(H, img_w, img_h):

        corners_img = np.array([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]])

        t_one = np.ones((corners_img.shape[0], 1))
        t_out_pts = np.concatenate((corners_img, t_one), axis=1)
        canvas_crd_corners = np.matmul(H, t_out_pts.T)
        canvas_crd_corners = canvas_crd_corners / canvas_crd_corners[-1, :]  # cols of [x1, y1, z1]

        min_crd = np.amin(canvas_crd_corners.T, axis=0)  # [x, y, z]
        max_crd = np.amax(canvas_crd_corners.T, axis=0)

        return min_crd, max_crd

    def __call__(self, image_list: Sequence[np.ndarray]) -> Tuple[np.ndarray, Sequence[Tuple]]:
        """
        Returns:
            panorama_img: np.ndarray  # 支持多种格式,如RGB、BGR、Gray,与传入的图像格式相同
            coords_list: List[Tuple(int, int, int, int)]   # (x1, y1, x2, y2)
        """
        if len(image_list) == 1:
            panorama_img = image_list[0]
            _h, _w = panorama_img.shape[:2]
            return panorama_img, [(0, 0, _w, _h)]

        if self.shapes is None:
            self.register(image_list)

        assert len(image_list) == self.num_images
        if self.orders is None:
            order_list = [*enumerate(image_list)]
        else:
            order_list = [*zip(self.orders, enumerate(image_list))]
            order_list = [item[1] for item in sorted(order_list, key=lambda _x: _x[0])]
        for i, image_src in order_list:
            if __debug__:
                print(f'Writing image-{i}')
            pts = self.pts_list[i]
            tl = self.tl_list[i]
            tr = self.tr_list[i]
            bl = self.bl_list[i]
            br = self.br_list[i]
            weight = self.weight_list[i]
            image_src = torch.from_numpy(image_src).to(self.device)
            pts = pts.long()
            if len(image_src.shape) == 2:
                image_src = image_src[:, :, None]
            self.canvas[pts[:, 1], pts[:, 0], :] = self._blend(image_src, tl, tr, bl, br, weight)
        panorama_img = self.canvas.cpu().numpy()
        return panorama_img, self.coords_list

    @staticmethod
    def _blend(image_src, tl, tr, bl, br, weight):
        tl = tl.long()
        tr = tr.long()
        bl = bl.long()
        br = br.long()
        weighted_tl = image_src[tl[:, 0], tl[:, 1], :] * weight[:, 0:1]
        weighted_tr = image_src[tr[:, 0], tr[:, 1], :] * weight[:, 1:2]
        weighted_bl = image_src[bl[:, 0], bl[:, 1], :] * weight[:, 2:3]
        weighted_br = image_src[br[:, 0], br[:, 1], :] * weight[:, 3:4]
        sum_weight = torch.sum(weight, dim=1, keepdim=True)
        return ((weighted_tl + weighted_tr + weighted_bl + weighted_br) / sum_weight).byte()

    def get_params(self) -> dict:
        """Get some reusable parameters
        """
        params = {
            'shapes': self.shapes,
            'num_images': self.num_images,
            'middle_idx': self.middle_idx,
            'H_all': self.H_all
        }
        return params

    def set_params(self, params):
        """Set reusable parameters
        """
        self.shapes = params['shapes']
        self.num_images = params['num_images']
        self.middle_idx = params['middle_idx']
        self.H_all = params['H_all']
        self._init()

    def load_params(self, path):
        """Load saved parameters
        """
        with open(path, 'rb') as f:
            self.set_params(pickle.load(f))

    def save_params(self, path):
        """Save parameters
        """
        with open(path, 'wb') as f:
            pickle.dump(self.get_params(), f)


def find_homography_by_features(
        image1: np.ndarray,
        image2: np.ndarray,
        num_feats: int,
        ransac_reproj_threshold: int,
        overlap_factor: float = 0.5,
        filters: Sequence[AbstractFilter] = None,
        estimator: AbstractTransformEstimator = RANSACEstimator(ScaleTranslateEstimator()),
        multirow: bool = False
) -> Union[np.ndarray, None]:
    """Find homography
    """
    if multirow:
        kpts1, desc1 = _get_features(image1, num_feats, None, 1.0)
        kpts2, desc2 = _get_features(image2, num_feats, None, 1.0)
    else:
        kpts1, desc1 = _get_features(image1, num_feats, 'left', 0.5)
        kpts2, desc2 = _get_features(image2, num_feats, 'right', overlap_factor)

    matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = matcher.match(desc1.astype(np.uint8), desc2.astype(np.uint8))
    good_matches = sorted(matches, key=lambda x: x.distance)

    src = np.array([kpts1[m.queryIdx].pt for m in good_matches], np.float32)
    dst = np.array([kpts2[m.trainIdx].pt for m in good_matches], np.float32)

    # show_matches(image1, image2, src, dst)

    if filters is not None:
        for filter_ in filters:
            src, dst = filter_(src, dst)

    # show_matches(image1, image2, src, dst)

    if len(src) >= 4:
        if estimator is not None:
            H = estimator(src, dst)
        else:
            H, _ = cv.findHomography(
                src.reshape((-1, 1, 2)),
                dst.reshape((-1, 1, 2)),
                method=cv.RANSAC,
                ransacReprojThreshold=ransac_reproj_threshold
            )
        return H
    else:
        return None


def _get_features(
        image: np.ndarray,
        num_feats: int,
        position: Literal['left', 'right', None],
        overlap_factor: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    # 如果传入的是3通道图像, 算法内部会将其转换为灰度图
    if position == 'left':
        h, w = image.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        mask[:, -int(w * overlap_factor):] = 1
    elif position == 'right':
        h, w = image.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        mask[:, :int(w * overlap_factor)] = 1
    else:
        mask = None
    detector = cv.SIFT_create(nfeatures=num_feats)  # 默认nfeatures=0,返回所有keypoints
    keypoints, descriptors = detector.detectAndCompute(image, mask=mask)
    return keypoints, descriptors


def show_matches(image1, image2, src, dst):
    """Show matches
    """
    image = np.concatenate([image1, image2], 1)
    for pt1, pt2 in zip(src, dst):
        pt1 = np.array(pt1, np.int64)
        pt2 = np.array(pt2 + np.array([image1.shape[1], 0]), np.int64)
        cv.circle(image, pt1, 10, (0, 255, 0), 3)
        cv.circle(image, pt2, 10, (0, 255, 0), 3)
        cv.line(image, pt1, pt2, (0, 255, 0), 3)
    plt.imshow(image)
    plt.show()


def fit_image_in_target_space(
        img_src_h: int,
        img_src_w: int,
        mask: torch.Tensor,
        H: torch.Tensor,
        offset=torch.tensor([0, 0, 0]),
        device=None
) -> Tuple:
    """Fit image in target space.

    Args:
        img_src_h: Source image's height.
        img_src_w: Source image's width.
        mask: Mask corresponding to dest image.
        H: pts_in_src_img = H * pts_in_dst_img
        offset: [x_offset, y_offset, 0]. Offset 0,0 in mask to this value
        device: Device to use.

    Returns:
        pts, tl, tr, bl, br, weight
    """
    pts = _get_pixel_coord(mask, device)  # rows of [x1, y1, 1]  # (n,3)
    pts = pts + offset
    out_src = torch.matmul(H, pts.T)  # out_src has cols of [x1, y1, z1]   # (3,n)

    out_src = out_src / out_src[-1, :]  # (3,n)

    # Return only x, y non-homogenous coordinates
    out_src = out_src[0:2, :]  # corresponds to pixels in img_src  # (2,n)
    out_src = out_src.T  # rows of [x1, y1]  # (n,2)

    # Convert pts to out_src convention
    pts = pts[:, 0:2].short()  # Corresponds to pixel locs in img_dst, rows of [x1,y1] #(n,2)

    pts, tl, tr, bl, br, weight = _get_pixel_val(img_src_h, img_src_w, pts, out_src, offset, device)

    return pts, tl, tr, bl, br, weight


def _get_pixel_coord(mask: torch.Tensor, device: str) -> torch.Tensor:
    """Function to get x, y coordinates of white pixels in mask as homogenous coordinates
    """
    y, x = torch.where(mask)
    pts = torch.cat([
        x[:, None],
        y[:, None],
        torch.ones((x.shape[0], 1)).to(device)
    ], dim=1)  # rows of [x1, y1, 1]
    return pts


def _get_pixel_val(img_src_h, img_src_w, pts, out_src, offset, device):
    """
    :param pts: pts for img_dst rows of [x1, y1], shape is (n,2)
    :param out_src: rows of [x1, y1], corresponding pts in src img after homography on dst points
    :return:
    """

    h, w = img_src_h, img_src_w
    tl = torch.floor(
        torch.flip(out_src, dims=[1])).short()  # reverse cols to get row, col notation  # (n,2) rows of (y,x)
    br = torch.ceil(torch.flip(out_src, dims=[1])).short()  # (n,2) rows of (y,x)

    pts = pts - offset[:2]  # (n,2)

    r_lzero = torch.where(~torch.logical_or(torch.any(tl < 0, dim=1), torch.any(br < 0, dim=1)))[0]  # (n1,)
    pts = pts[r_lzero, :]  # (n1, 2)
    out_src = out_src[r_lzero, :]  # (n1,2)
    tl = tl[r_lzero, :]  # (n1,2)
    br = br[r_lzero, :]  # (n1,2)

    r_fl = torch.where(~torch.logical_or(tl[:, 0] >= h - 1, tl[:, 1] >= w - 1))[0]  # (n2,)
    pts = pts[r_fl, :]  # (n2,2)
    out_src = out_src[r_fl, :]  # (n2,2)
    tl = tl[r_fl, :]  # (n2,2)
    br = br[r_fl, :]  # (n2,2)

    r_ce = torch.where(~torch.logical_or(br[:, 0] >= h - 1, br[:, 1] >= w - 1))[0]  # (n3,)
    pts = pts[r_ce, :]  # (n3,2)
    out_src = out_src[r_ce, :]  # (n3,2)
    tl = tl[r_ce, :]  # (n3,2)
    br = br[r_ce, :]  # (n3,2)

    tr = torch.cat((tl[:, 0:1], br[:, 1:2]), dim=1)
    bl = torch.cat((br[:, 0:1], tl[:, 1:2]), dim=1)

    weight = torch.zeros((out_src.shape[0], 4), dtype=torch.float16).to(device)
    weight[:, 0] = torch.linalg.norm(tl - torch.flip(out_src, dims=[1]), dim=1)
    weight[:, 1] = torch.linalg.norm(tr - torch.flip(out_src, dims=[1]), dim=1)
    weight[:, 2] = torch.linalg.norm(bl - torch.flip(out_src, dims=[1]), dim=1)
    weight[:, 3] = torch.linalg.norm(br - torch.flip(out_src, dims=[1]), dim=1)

    weight[torch.all(weight == 0, dim=1)] = 1  # For entries where they exactly overlap
    weight = 1 / weight

    return pts, tl, tr, bl, br, weight

# def main():
#     parser = argparse.ArgumentParser()
#     args = parser.parse_args()
#
#     image_files = [
#         '/data/3.jpg',
#         '/data/1.jpg',
#         '/data/2.jpg',
#     ]
#     output_file = '/data/all.png'
#
#     scale = 0.3
#     images = [
#         cv.resize(cv.cvtColor(cv.imread(path, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB), None, fx=scale, fy=scale)
#         for path in image_files
#     ]
#     image, _ = ImageStitching(
#         overlap_factor=0.3,
#         filters=[FilterByYAxis(5), FilterByDistance(5)],
#         orders=[1, 3, 0],
#     )(images)
#     cv.imwrite(output_file, cv.cvtColor(image, cv.COLOR_RGB2BGR))
#     return 0
#
#
# if __name__ == '__main__':
#     raise SystemExit(main())
