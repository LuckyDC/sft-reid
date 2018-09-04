from __future__ import division
import random
import math
import cv2
import collections
import numbers
import types
import numpy as np


class Lambda:
    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


class Normalize:
    def __init__(self, mean=(0, 0, 0), std=(1, 1, 1)):
        self.mean = np.array(mean).reshape([1, 1, 3])
        self.std = np.array(std).reshape([1, 1, 3])

    def __call__(self, img):
        return (img - self.mean) / self.std


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img = cv2.flip(img, flipCode=1)

        return img


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random < self.p:
            img = cv2.flip(img, flipCode=0)

        return img


class RandomErase:
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(127.0, 127.0, 127.0)):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mean = mean

    def __call__(self, img):
        return self.random_erasing(img)

    def random_erasing(self, img):
        im_h, im_w, num_ch = img.shape

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = im_h * im_w

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < im_w and h < im_h:
                x1 = random.randint(0, im_h - h)
                y1 = random.randint(0, im_w - w)
                if num_ch == 3:
                    img[x1:x1 + h, y1:y1 + w, :] = np.array(self.mean).reshape([1, 1, 3])
                else:
                    img[x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class Resize:
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        assert (isinstance(size, int) or isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

        # mode 0 to resize shorter edge and keep aspect ratio;
        # mode 1 to resize both sides to match (h,w).
        self.mode = 0 if isinstance(size, int) else 1

    def __call__(self, img):
        if self.mode == 0:
            ori_h, ori_w = img.shape[:2]
            if ori_h < ori_w:
                h, w = self.size, int(round(self.size / ori_h * ori_w))
            else:
                h, w = int(round(self.size / ori_w * ori_h)), self.size

            if h != ori_h or w != ori_w:
                img = cv2.resize(img, (w, h), interpolation=self.interpolation)

        else:
            if tuple(img.shape[:2]) != self.size:
                img = cv2.resize(img, tuple(self.size[::-1]), interpolation=self.interpolation)

        return img


class CenterCrop:
    def __init__(self, size):
        assert (isinstance(size, int) or isinstance(size, collections.Iterable) and len(size) == 2)

        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img):
        h, w = self.size
        ori_h, ori_w = img.shape[:2]

        assert ori_h >= h and ori_w >= w

        x = int(round((ori_w - w) / 2))
        y = int(round((ori_h - h) / 2))

        return img[y:y + h, x:x + h, :]


class RandomCrop:
    def __init__(self, size, padding=None, pad_if_needed=False, fill_value=0, padding_mode=cv2.BORDER_CONSTANT):
        assert (isinstance(size, int) or isinstance(size, collections.Iterable) and len(size) == 2)

        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

        self.padding = padding
        if isinstance(self.padding, int):
            self.padding = [padding] * 4
        elif isinstance(self.padding, collections.Iterable) and len(self.padding) == 2:
            self.padding = [padding[0], padding[0], padding[1], padding[1]]

        self.pad_if_needed = pad_if_needed
        self.fill_value = fill_value
        self.padding_mode = padding_mode

    def __call__(self, img):
        h, w = self.size

        ori_h, ori_w = img.shape[:2]

        if w == ori_w and h == ori_h:
            return img

        if self.padding is not None:
            img = cv2.copyMakeBorder(img, *self.padding, borderType=self.padding_mode, value=self.fill_value)

        # pad the width if needed
        if self.pad_if_needed and ori_w < w:
            img = cv2.copyMakeBorder(img, 0, 0, (w - ori_w) // 2, w - (w - ori_w) // 2, borderType=self.padding_mode,
                                     value=self.fill_value)
        # pad the height if needed
        if self.pad_if_needed and ori_h < h:
            img = cv2.copyMakeBorder(img, (h - ori_h) // 2, h - (h - ori_h) // 2, 0, 0, borderType=self.padding_mode,
                                     value=self.fill_value)

        x = random.randint(0, int(ori_w - w))
        y = random.randint(0, int(ori_h - h))

        return img[y:y + h, x:x + w, :]


class PadTo:
    def __init__(self, size, fill_value=0, padding_mode=cv2.BORDER_CONSTANT):
        assert (isinstance(size, int) or isinstance(size, collections.Iterable) and len(size) == 2)

        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

        self.fill_value = fill_value
        self.padding_mode = padding_mode

    def __call__(self, img):
        h, w = self.size
        ori_h, ori_w = img.shape[:2]

        if ori_h > h or ori_w > w:
            ori_aspect = ori_h / ori_w
            aspect = h / w

            if ori_aspect > aspect:
                ori_h, ori_w = h, int(h / ori_aspect)
                img = cv2.resize(img, (ori_w, ori_h))
            elif ori_aspect < aspect:
                ori_h, ori_w = int(w * ori_aspect), w
                img = cv2.resize(img, (ori_w, ori_h))

        left = (w - ori_w) // 2
        right = w - ori_w - left
        top = (h - ori_h) // 2
        bottom = h - ori_h - top

        img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=self.padding_mode, value=self.fill_value)
        return img


class Cast:
    def __init__(self, dtype="float32"):
        self.dtype = dtype

    def __call__(self, img):
        return img.astype(self.dtype)


class ColorJitter:
    """Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        brightness = self._check_input(brightness, 'brightness')
        contrast = self._check_input(contrast, 'contrast')
        saturation = self._check_input(saturation, 'saturation')
        hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

        transforms = []
        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: self.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: self.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: self.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: self.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)

        self.transform = Compose(transforms)

    def __call__(self, img):
        img = self.transform(img)
        return img

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None

        return value

    @staticmethod
    def adjust_brightness(img, brightness_factor):
        """Adjust brightness of an Image.
        """

        degenerate = np.zeros(img.shape, dtype=np.uint8)
        img = degenerate * (1.0 - brightness_factor) + img * brightness_factor
        return img.astype(np.uint8)

    @staticmethod
    def adjust_contrast(img, contrast_factor):
        """Adjust contrast of an Image.
        """
        mean = int(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).mean() + 0.5)
        degenerate = np.full(img.shape, mean, dtype=np.uint8)

        img = degenerate * (1.0 - contrast_factor) + img * contrast_factor
        return img.astype(np.uint8)

    @staticmethod
    def adjust_saturation(img, saturation_factor):
        """Adjust color saturation of an image.
        """
        degenerate = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        degenerate = np.expand_dims(degenerate, 2)

        img = degenerate * (1.0 - saturation_factor) + img * saturation_factor
        return img.astype(np.uint8)

    @staticmethod
    def adjust_hue(img, hue_factor):
        """Adjust hue of an image.
        The image hue is adjusted by converting the image to HSV and
        cyclically shifting the intensities in the hue channel (H).
        The image is then converted back to original image mode.
        `hue_factor` is the amount of shift in H channel and must be in the
        interval `[-0.5, 0.5]`.
        """
        if not (-0.5 <= hue_factor <= 0.5):
            raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

        img = cv2.cvtColor(img.astype(np.uint8), code=cv2.COLOR_RGB2HSV)

        h = img[:, :, 0].astype(np.int16)
        img[:, :, 0] = ((h + hue_factor * 180) % 181).astype(np.uint8)

        return img.astype(np.uint8)


class Pad:
    """Pad the given PIL Image on all sides with the given "pad" value.
    Args:
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the top, bottom, left and right borders
            respectively.
        fill (int or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode (str): Type of padding.
    """

    def __init__(self, padding, fill_value=0, padding_mode=cv2.BORDER_CONSTANT):
        assert isinstance(padding, (int, tuple))
        assert isinstance(fill_value, (numbers.Number, str, tuple))

        if isinstance(self.padding, int):
            self.padding = [padding] * 4
        elif isinstance(self.padding, collections.Iterable) and len(self.padding) == 2:
            self.padding = [padding[0], padding[0], padding[1], padding[1]]

        self.padding = padding
        self.fill_value = fill_value
        self.padding_mode = padding_mode

    def __call__(self, img):
        return cv2.copyMakeBorder(img, *self.padding, borderType=self.padding_mode, value=self.fill_value)
