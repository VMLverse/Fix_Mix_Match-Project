# code in this file is adpated from
# https://github.com/kekmodel/FixMatch-pytorch/blob/master/dataset/randaugment.py
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py

import random
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageOps
from PIL import ImageEnhance

PARAMETER_MAX = 10

#LIST OF RANDOM AUGMENTATION FUNCTIONS:
#1: AutoContrast
def AutoContrast(img, **kwarg):
    return ImageOps.autocontrast(img)

#2: Brightness
def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return ImageEnhance.Brightness(img).enhance(v)

#3: Color
def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return ImageEnhance.Color(img).enhance(v)

#4: Contrast
def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return ImageEnhance.Contrast(img).enhance(v)

#5: Equalize
def Equalize(img, **kwarg):
    return ImageOps.equalize(img)

#6: Identity
def Identity(img, **kwarg):
    return img

#7: Posterize
def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return ImageOps.posterize(img, v)

#8: Rotate
def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v)

#9: Sharpness
def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return ImageEnhance.Sharpness(img).enhance(v)

#10: ShearX
def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, Image.AFFINE, (1, v, 0, 0, 1, 0))

#11: ShearY
def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, v, 1, 0))

#12: Solarize
def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return ImageOps.solarize(img, 256 - v)

#14: TranslateX
def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, Image.AFFINE, (1, 0, v, 0, 1, 0))

#14: TranslateY
def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, v))

#Utility Functions:
def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX

def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)

def random_augment_pool():
    # returns a list of random augmentation functions, Max Value & Bias
    # augm_fn, max_v, bias
    aug_pool_arr = [(AutoContrast, None, None),
                    (Brightness, 0.9, 0.05),
                    (Color, 0.9, 0.05),
                    (Contrast, 0.9, 0.05),
                    (Equalize, None, None),
                    (Identity, None, None),
                    (Posterize, 4, 4),
                    (Rotate, 30, 0),
                    (Sharpness, 0.9, 0.05),
                    (ShearX, 0.3, 0),
                    (ShearY, 0.3, 0),
                    (Solarize, 256, 0),
                    (TranslateX, 0.3, 0),
                    (TranslateY, 0.3, 0)]
    return aug_pool_arr

class RandAugment(object):
    """
    RandAugmentation - Selects 'aug_count' random augmentation operations from pool and applies random augmentation 50% of the time;
    Augmentation Function is applied with an 'aug_value' randomly chosen between 1 & 10.
    """
    def __init__(self, aug_count, aug_value,enable_cutout,cutout_only):
        if (aug_count < 1):
            raise AssertionError("RandAugment n Value is :%d, cannot be less than 1." % (aug_count))
        if (1 > aug_value > 10):
            raise AssertionError("RandAugment m Value is :%d, It has to be between 1 <= m <= 10" % (aug_value))
        self.n = aug_count # number of augmentations to apply
        self.m = aug_value #max value to apply during augmentation
        self.enable_cutout = enable_cutout # decides whether to apply cutout or not
        self.cutout_only = cutout_only  # decides whether to apply cutout augmentation alone during strong augment

    def __call__(self, img):
        if not (self.cutout_only):  # apply rand augment if cutout_only is not required (False)
            # STEP1: Select N random operations from Random Augment Pool
            operations = random.choices(random_augment_pool(), k=self.n)
            # STEP2: loop through all operations, get operation func, max value and bias from Random Augment Pool
            for augm_fn, max_v, bias in operations:
                v = np.random.randint(1, self.m)  # select a random value between 1 & M
                prob = random.random()  # returns a random number between 0.0 and 1.0
                if prob < 0.5:  # apply augmentation function with prob=0.5
                    img = augm_fn(img, v=v, max_v=max_v, bias=bias)
        if self.enable_cutout:  # apply cuout (aka random rect overlay only if enabled)
            img = self.rand_rectangle_overlay(img)  # overlay a random greyrectangle
        return img

    def rand_rectangle_overlay(self, img):
        """
        Random Rectangle Overlay, aka, "CUTOUT" in the paper
        Args:
            img: input image
        Returns: image with a random grey rectangle overlay
        """
        w, h = img.size #get image size=[width=32, height=32]
        mid = int(w * 0.5)
        #uniformly select a random value from max width & max height.
        x_rand = np.random.uniform(0, w)
        y_rand = np.random.uniform(0, h)
        #generate random lower co-ordinates x0,y0 - >=0
        x0 = int(max(0, x_rand - mid / 2.))
        y0 = int(max(0, y_rand - mid / 2.))
        # generate random upper co-ordinates x1,y1 - <=w,<=h
        x1 = int(min(w, x0 + mid))
        y1 = int(min(h, y0 + mid))
        xy = (x0, y0, x1, y1) #generate coordiates for drawing rectangle
        img = img.copy() #create a copy of the image to apply rectangle overlay
        grey_color = (127, 127, 127)
        ImageDraw.Draw(img).rectangle(xy, grey_color)
        return img
