
from PIL import Image
import numpy as np

import torch
from random import shuffle

class ToGPU(object):


    def __call__(self, img):
        return torch.squeeze(img.cuda()*255)



class Augmenter(object):
    # Here besides just applying the list, the class should also apply the scheduling
    # Based on the augmentation

    def __init__(self, composition):
        # TODO: Add scheduler .
        self.transforms = composition
        # print (composition)

    def __call__(self, iteration, img):
        # temp = self.transforms[1:]
        # shuffle(temp)
        # self.transforms = [self.transforms[0]] + temp
        for t in self.transforms:
            img = t(img)

        return img/255.0

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class AugmenterCPU(object):
    """
    This class serve as a wrapper to apply augmentations from IMGAUG in CPU mode in
    the same way augmentations are applyed when using the transform library from pytorch

    """
    # Here besides just applying the list, the class should also apply the scheduling


    def __init__(self, composition):
        self.transforms = composition

    def __call__(self, img):
        #TODO: Check this format issue

        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 1, 2)

        for t in self.transforms:

            img = t.augment_images(img)


        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
