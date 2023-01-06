#!/usr/bin/env python

# ================================================================
#   Copyright (C) 2021 Yufeng Liu (Braintell, Southeast University). All rights reserved.
#
#   Filename     : generic_augmentation.py
#   Author       : Yufeng Liu
#   Date         : 2021-04-02
#   Description  : Some codes are borrowed from ssd.pytorch: https://github.com/amdegroot/ssd.pytorch
#                  And some augmentation implementations are directly copied from nnUNet.
#
# ================================================================

import shutil
import numpy as np
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from skimage import exposure
from batchgenerators.augmentations.utils import create_zero_centered_coordinate_mesh, elastic_deform_coordinates, \
    interpolate_img, rotate_coords_2d, rotate_coords_3d, scale_coords, elastic_deform_coordinates_2, \
    resize_multichannel_image

from NRT.utils import image_util


def get_random_shape(img, scale_range, per_axis):
    if type(img) == np.ndarray and img.size > 1024:
        shape = np.array(img[0].shape)
    else:
        shape = np.array(list(img))
    if per_axis:
        scales = np.random.uniform(*scale_range, size=len(shape))
    else:
        scales = np.array([np.random.uniform(*scale_range)] * len(shape))
    target_shape = np.round(shape * scales).astype(np.int)
    return shape, target_shape


def image_scale_4D(img, tree, shape, target_shape, mode, anti_aliasing):
    if target_shape.prod() / shape.prod() > 1:
        # up-scaling
        order = 0
    else:
        order = 1

    new_img = np.zeros((img.shape[0], *target_shape), dtype=img.dtype)

    for c in range(img.shape[0]):
        new_img[c] = resize(img[c], target_shape, order=order, mode=mode, anti_aliasing=anti_aliasing)

    if tree is not None:
        scales = target_shape / shape
        new_tree = []
        for leaf in tree:
            idx, type_, x, y, z, r, p = leaf
            new_tree.append((idx, type_, x * scales[2], y * scales[1], z * scales[0], r, p))
        tree = new_tree

    return new_img, new_tree


def random_crop_image_4D(img, tree, target_shape):
    new_img = np.zeros((img.shape[0], *target_shape), dtype=img.dtype)
    sz = None
    sy = None
    sx = None
    for c in range(img.shape[0]):
        if c == 0:
            new_img[c], sz, sy, sx = image_util.random_crop_3D_image(img[c], target_shape)
        else:
            new_img[c] = img[c][sz:sz + target_shape[0], sy:sy + target_shape[1], sx:sx + target_shape[2]]

    # processing the gt

    if tree is not None:
        new_tree = []
        for leaf in tree:
            idx, type_, x, y, z, r, p = leaf
            x = x - sx
            # Since the y-coordinate in swc file is mirrored by crop
            # center, the coordinate change should take care!!!
            y = y - sy
            z = z - sz
            new_tree.append((idx, type_, x, y, z, r, p))
        return new_img, new_tree

    return new_img, tree


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, tree=None):
        for t in self.transforms:
            img, tree= t(img, tree)
        return img, tree


class ResizeToDividable(object):
    def __init__(self, divid=2 ** 5):
        self.divid = divid

    def __call__(self, img, tree=None, soma=None, spacing=None):
        shape = np.array(img[0].shape).astype(np.float32)
        target_shape = np.round(shape / self.divid).astype(np.long) * self.divid
        img, tree = image_scale_4D(img, tree, shape, target_shape, mode='edge', anti_aliasing=False)
        return img, tree


class AbstractTransform(object):
    def __init__(self, p=0.5):
        self.p = p


class ConvertToFloat(object):
    """
    Most augmentation assumes the input image of float type, so it is always recommended to
    call this class before all augmentations.
    """

    def __init__(self, dtype=np.float32):
        self.dtype = dtype

    def __call__(self, img, tree=None):
        if not img.dtype.name.startswith('float'):
            img = img.astype(self.dtype)
        return img, tree


# Coordinate-invariant augmentation
class RandomSaturation(AbstractTransform):
    def __init__(self, lower=0.9, upper=1.1, p=0.5):
        super(RandomSaturation, self).__init__(p)
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower"
        assert self.lower > 0, "contrast lower must be positive"

    def __call__(self, img, gt=None):
        if np.random.random() < self.p:
            sf = np.random.uniform(self.lower, self.upper)
            # print(f'RandomSaturation with factor: {sf}')
            img *= sf

        return img, gt


class RandomBrightness(AbstractTransform):
    def __init__(self, dratio=0.1, p=0.5):
        super(RandomBrightness, self).__init__(p)
        assert dratio >= 0. and dratio < 1.
        self.dratio = dratio

    def __call__(self, img, gt=None):
        if np.random.random() < self.p:
            img_flat = img.reshape((img.shape[0], -1))
            mm = img_flat.max(axis=1) - img_flat.min(axis=1)
            dmm = np.random.uniform(-self.dratio, self.dratio) * mm
            # print(f'RandomBrightness with shift: {dmm}')
            img += dmm.reshape((mm.shape[0], 1, 1, 1))

        return img, gt


class RandomGaussianNoise(AbstractTransform):
    def __init__(self, p=0.5, max_var=0.1, max_ratio=0.01):
        super(RandomGaussianNoise, self).__init__(p)
        self.max_var = max_var
        self.max_ratio = max_ratio

    def __call__(self, img, tree=None):
        if np.random.random() < self.p:
            var = np.random.uniform(0, self.max_var)
            img_flat = img.reshape((img.shape[0], -1))
            mm = img_flat.max(axis=1) - img_flat.min(axis=1)

            keep_ratio = np.random.uniform(0, self.max_ratio)
            mask = np.random.random(size=img.shape) < keep_ratio
            noise = np.random.normal(0, var, size=img.shape) * mm.reshape((-1, 1, 1, 1)) * mask.astype(np.float32)
            # print(f'RandomGaussianNoise with var: {var}')

            img += noise
        return img, tree


class RandomGaussianBlur(AbstractTransform):
    def __init__(self, kernels=(0, 1), p=0.5):
        super(RandomGaussianBlur, self).__init__(p)
        self.kernels = kernels

    def __call__(self, img, tree=None):
        if np.random.random() < self.p:
            idx = np.random.randint(len(self.kernels))
            kernel = self.kernels[idx]
            kernel_z = kernel
            kernel_z = max(int(round(kernel_z)) * 2 + 1, 1)
            kernel_xy = kernel * 2 + 1
            sigmas = (kernel_z, kernel_xy, kernel_xy)
            # print(f'RandomGaussianBlur with sigmas: {sigmas}')

            for c in range(img.shape[0]):
                img[c] = gaussian_filter(img[c], sigma=sigmas)
        return img, tree


class RandomResample(AbstractTransform):
    def __init__(self, p=0.5, zoom_range=(0.8, 1), order_down=1, order_up=0, per_axis=True):
        super(RandomResample, self).__init__(p)
        self.zoom_range = zoom_range
        self.order_down = order_down
        self.order_up = order_up
        self.per_axis = per_axis

    def __call__(self, img, tree=None):
        if np.random.random() < self.p:
            shape, target_shape = get_random_shape(img, self.zoom_range, self.per_axis)

            # print(f'RandomSample with zoom factor: {zoom}')
            for c in range(img.shape[0]):
                downsampled = resize(img[c], target_shape, order=self.order_down, mode='edge', anti_aliasing=False)
                img[c] = resize(downsampled, shape, order=self.order_up, mode='edge', anti_aliasing=False)

        return img, tree


class CLAHETransform(AbstractTransform):
    def __init__(self, p=1.0, kernel_size=(16, 32, 32)):
        super(CLAHETransform, self).__init__(p)
        self.kernel_size = kernel_size

    def __call__(self, img, tree=None):
        if np.random.random() < self.p:
            assert img.shape[0] == 1
            vmax, vmin = img.max(), img.min()
            img = ((img - vmin) / (vmax - vmin + 1e-7) * 255).astype(np.uint8)
            img_new = img.copy()
            img = exposure.equalize_adapthist(img[0], kernel_size=self.kernel_size, clip_limit=0.01, nbins=256)
            img = (img - img.min()) / (img.max() - img.min() + 1e-7) * (vmax - vmin) + vmin

        return img[None], tree


class EqHistTransform(AbstractTransform):
    def __init__(self, p=1.0):
        super(EqHistTransform, self).__init__(p)

    def __call__(self, img, tree=None):
        if np.random.random() < self.p:
            assert img.shape[0] == 1
            img = exposure.equalize_hist(img[0], nbins=256)
            vmax = img.max()
            vmin = img.min()
            img = (img - img.min()) / (img.max() - img.min() + 1e-7) * (vmax - vmin) + vmin

        return img[None], tree


class RandomGammaTransform(AbstractTransform):
    def __init__(self, p, gamma_range=(0.5, 2), invert_image=False, per_channel=False, retain_stats=False):
        super(RandomGammaTransform, self).__init__(p)
        self.gamma_range = gamma_range
        self.invert_image = invert_image
        self.per_channel = per_channel
        self.retain_stats = retain_stats

    def __call__(self, img, tree=None):
        if np.random.random() < self.p:
            img = image_util.augment_gamma(img, self.gamma_range,
                                           self.invert_image,
                                           per_channel=self.per_channel,
                                           retain_stats=self.retain_stats)

        return img, tree


class RandomGammaTransformDualModes(AbstractTransform):
    def __init__(self, p, gamma_range=(0.5, 2), per_channel=False, retain_stats=False):
        super(RandomGammaTransformDualModes, self).__init__(p)
        self.gamma_range = gamma_range
        self.per_channel = per_channel
        self.retain_stats = retain_stats

    def __call__(self, img, tree=None):
        if np.random.random() < self.p:
            if np.random.randint(2):
                img = image_util.augment_gamma(img, self.gamma_range,
                                               True,
                                               per_channel=self.per_channel,
                                               retain_stats=self.retain_stats)
            else:
                img = image_util.augment_gamma(img, self.gamma_range,
                                               False,
                                               per_channel=self.per_channel,
                                               retain_stats=self.retain_stats)
        return img, tree


class GammaTransform(AbstractTransform):
    def __init__(self, gamma=1.0, trunc_thresh=0, invert_image=False, per_channel=False, retain_stats=False):
        super(GammaTransform, self).__init__(1.0)
        self.gamma = gamma
        self.invert_image = invert_image
        self.per_channel = per_channel
        self.retain_stats = retain_stats
        self.trunc_thresh = trunc_thresh

    def __call__(self, img, tree=None):
        img = image_util.do_gamma(img, gamma=self.gamma,
                                  trunc_thresh=self.trunc_thresh,
                                  invert_image=self.invert_image,
                                  per_channel=self.per_channel,
                                  retain_stats=self.retain_stats)
        return img, tree


# Coordinate-changing augmentations
class RandomMirror(AbstractTransform):
    def __init__(self, p=0.5):
        super(RandomMirror, self).__init__(p)

    def __call__(self, img, tree=None):
        if np.random.random() < self.p:
            axis = np.random.randint(img.ndim - 1) + 1
            # NOTE: img in (c,z,y,x) order, while coord in tree is (x,y,z)
            if axis == 1:
                img = img[:, ::-1, ...]
            elif axis == 2:
                img = img[:, :, ::-1, ...]
            elif axis == 3:
                img = img[:, :, :, ::-1]
            else:
                raise ValueError('Number of dimension should not exceed 4')

            if tree is not None:
                # processing tree structure
                shape = img[0].shape
                shape_axis = shape[axis - 1]
                new_tree = []
                if axis == 1:
                    for leaf in tree:
                        idx, type_, x, y, z, r, p = leaf
                        z = shape_axis - z
                        new_tree.append((idx, type_, x, y, z, r, p))
                elif axis == 2:
                    for leaf in tree:
                        idx, type_, x, y, z, r, p = leaf
                        y = shape_axis - y
                        new_tree.append((idx, type_, x, y, z, r, p))
                else:
                    for leaf in tree:
                        idx, type_, x, y, z, r, p = leaf
                        x = shape_axis - x
                        new_tree.append((idx, type_, x, y, z, r, p))
                tree = new_tree

        return img, tree


# The following geometric transformation can be composed into an unique geometric transformation.
# But I prefer to use this separate versions, since they are implemented with matrix production,
# which is much more efficient.
class RandomScale(AbstractTransform):
    def __init__(self, p=0.5, scale_range=(0.85, 1.25), per_axis=True, anti_aliasing=False, mode='edge',
                 update_spacing=True):
        super(RandomScale, self).__init__(p)
        self.per_axis = per_axis
        self.anti_aliasing = anti_aliasing
        self.mode = mode
        self.update_spacing = update_spacing

    def __call__(self, img, tree=None):
        if np.random.random() < self.p:
            shape, target_shape = get_random_shape(img, self.scale_range, self.per_axis)
            img, tree = image_scale_4D(img, tree, shape, target_shape, self.mode,
                                                          self.anti_aliasing, self.update_spacing)

        return img, tree


# verified
class ScaleToFixedSize(AbstractTransform):
    def __init__(self, p, target_shape, anti_aliasing=False, mode='edge', update_spacing=True):
        super(ScaleToFixedSize, self).__init__(p)
        self.target_shape = np.array(target_shape)
        self.anti_aliasing = anti_aliasing
        self.mode = mode
        self.update_spacing = update_spacing

    def __call__(self, img, tree=None):
        if np.random.random() < self.p:
            shape = np.array(img[0].shape)
            img, tree = image_scale_4D(img, tree, shape, self.target_shape, self.mode,
                                                      self.anti_aliasing)

        return img, tree


class RandomCrop(AbstractTransform):
    def __init__(self, p=0.5, imgshape=None, crop_range=(0.85, 1), per_axis=True, force_fg_sampling=False):
        super(RandomCrop, self).__init__(p)
        self.imgshape = imgshape
        self.crop_range = crop_range
        self.per_axis = per_axis
        self.force_fg_sampling = force_fg_sampling

    def __call__(self, img, tree=None):
        if np.random.random() > self.p:
            return img, tree

        if self.crop_range[0] == self.crop_range[1]:
            target_shape = self.imgshape
            img, tree = random_crop_image_4D(img, tree, target_shape)
            return img, tree
        else:
            if self.force_fg_sampling:
                num_trail = 0
                while num_trail < 3:
                    shape, target_shape = get_random_shape(self.imgshape, self.crop_range, self.per_axis)
                    new_img, new_tree= random_crop_image_4D(img, tree, target_shape)
                    # check foreground existence
                    has_fg = False
                    if np.sum(new_tree) > 10:
                        has_fg = True
                    if has_fg:
                        break

                    num_trail += 1
                else:
                    print("No foreground found after three random crops!")
            else:
                shape, target_shape = get_random_shape(self.imgshape, self.crop_range, self.per_axis)
                new_img, new_tree = random_crop_image_4D(img, tree, target_shape)

            return new_img, new_tree


# verified
class CenterCropKeepRatio(AbstractTransform):
    def __init__(self, p=1.0, reference_shape=None):
        super(CenterCropKeepRatio, self).__init__(p)
        self.reference_shape = reference_shape

    def __call__(self, img, gt=None):
        if np.random.random() > self.p:
            return img, gt

        shape = np.array(img[0].shape)
        reference_shape = np.array(self.reference_shape)
        scales = shape / self.reference_shape
        min_dim = np.argmin(scales)
        target_shape = np.round(scales[min_dim] * reference_shape).astype(int)
        # do center cropping
        sz, sy, sx = (shape - target_shape) // 2
        img = img[:, sz:sz + target_shape[0], sy:sy + target_shape[1], sx:sx + target_shape[2]]
        gt = gt[:, sz:sz + target_shape[0], sy:sy + target_shape[1], sx:sx + target_shape[2]]

        if gt is not None:
            gt = gt[:, sz:sz + target_shape[0], sy:sy + target_shape[1], sx:sx + target_shape[2]]

            return img, gt
        else:
            return img, gt


class CenterCrop(AbstractTransform):
    def __init__(self, p=1.0, reference_shape=None):
        super(CenterCrop, self).__init__(p)
        self.reference_shape = reference_shape

    def __call__(self, img, gt=None):
        if np.random.random() > self.p:
            return img, gt

        shape = np.array(img[0].shape)
        target_shape = np.array(self.reference_shape)
        # do center cropping
        sz, sy, sx = (shape - target_shape) // 2
        img = img[:, sz:sz + target_shape[0], sy:sy + target_shape[1], sx:sx + target_shape[2]]

        if gt is not None:
            gt = gt[:, sz:sz + target_shape[0], sy:sy + target_shape[1], sx:sx + target_shape[2]]

            return img, gt
        else:
            return img, gt


# NOTE: not verified, take care!
class RandomPadding(AbstractTransform):
    def __init__(self, p=0.5, pad_range=(1, 1.2), per_axis=True, pad_value=None):
        super(RandomPadding, self).__init__(p)
        self.pad_range = pad_range
        assert pad_range[0] >= 1 and pad_range[1] >= pad_range[0]
        self.per_axis = per_axis
        self.pad_value = pad_value

    def __call__(self, img, gt=None):
        if np.random.random() > self.p:
            return img, gt

        shape, target_shape = get_random_shape(img, self.pad_range, self.per_axis)
        for si, ti in zip(shape, target_shape):
            assert si <= ti
        if self.pad_value is None:
            # use lowerest value
            pad_value = img.min()
        else:
            pad_value = self.pad_value

        new_img = np.ones((img.shape[0], *target_shape), dtype=img.dtype) * pad_value
        new_gt = np.ones((gt.shape[0], *target_shape), dtype=gt.dtype) * pad_value

        # import ipdb; ipdb.set_trace()
        sz = np.random.randint(0, target_shape[0] - shape[0])
        sy = np.random.randint(0, target_shape[1] - shape[1])
        sx = np.random.randint(0, target_shape[2] - shape[2])
        for c in range(len(new_img)):
            new_img[c][sz:sz + shape[0], sy:sy + shape[1], sx:sx + shape[2]] = img
        if gt is not None:
            for c in range(len(new_gt)):
                new_gt[c][sz:sz + shape[0], sy:sy + shape[1], sx:sx + shape[2]] = gt
            return new_img, new_gt
        else:
            return new_img, new_gt


class RandomRotation(AbstractTransform):
    def __init__(self, p=0.5):
        super(RandomRotation, self).__init__(p)

    def __call__(self, img, gt=None):
        raise NotImplementedError


# RandomShift is an subset of composition of Crop and Padding, thus we do not need to implement it.
class RandomShift(AbstractTransform):
    def __init__(self, p=0.5):
        super(RandomShift, self).__init__(p)

    def __call__(self, img, gt=None):
        raise NotImplementedError


class InstanceAugmentation(object):
    def __init__(self, p=0.2, imgshape=(48, 96, 96), phase='train', divid=2 ** 5):
        if phase == 'train':
            pre_crop_size = []
            pre_crop_ratio = 1.5

            self.augment = Compose([
                ConvertToFloat(),
                RandomCrop(1.0, imgshape, crop_range=(0.85, 1.0), force_fg_sampling=True),
                RandomGammaTransformDualModes(p=p, gamma_range=(0.7, 1.4), per_channel=False, retain_stats=False),
                RandomGaussianNoise(p=p),
                # RandomSaturation(p=p),
                # RandomBrightness(p=p),
                # RandomGaussianBlur(p=p/2.),
                # RandomResample(p=p),
                RandomMirror(p=p),
                ScaleToFixedSize(1.0, imgshape),
            ])
        elif phase == 'val':
            self.augment = Compose([
                ConvertToFloat(),
                RandomCrop(1.0, imgshape, crop_range=(1.0, 1.0)),
            ])
        elif phase == 'test' or phase == 'par':
            self.augment = Compose([
                ConvertToFloat(),
                # CenterCropKeepRatio(1.0, imgshape),
                # ResizeToDividable(divid),
                # GammaTransform(gamma=0.4, trunc_thresh=0.216, retain_stats=True),  #0.2->0.133
                # GammaTransform(gamma=0.4, trunc_thresh=0, retain_stats=True),  #0.2->0.133
                # CLAHETransform(p=1.0, kernel_size=(16,32,32))
                # EqHistTransform(p=1.0)
            ])
        else:
            raise NotImplementedError

    def __call__(self, img, tree=None):
        return self.augment(img, tree)

