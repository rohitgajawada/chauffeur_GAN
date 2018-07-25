import numpy as np
import scipy

import os
import torch
import torch.nn.functional as torchF
from torchvision import transforms

import random

def multiply(tensor, scalar):

    return torch.clamp(tensor * scalar.expand_as(tensor), 0, 255)


def multiply_element_wise(tensor, prob):
    # prob = 0.2
    x = torch.cuda.FloatTensor(tensor.size())
    # print(x.size())
    # a = x.uniform_()
    # print(a)
    # print(prob)
    # b = a > prob
    # print (prob)

    prob = random.uniform(0,0.1)
    scalar = ((x.uniform_()) > prob).type(torch.cuda.FloatTensor)
    # print("Scalar ratio", scalar.sum() / float(tensor.size(1) * tensor.size(2) * tensor.size(3) * tensor.size(0)))
    # print(scalar.size())
    # print(scalar[0][0], tensor[0][0])

    inter = tensor * scalar.expand_as(tensor)
    # print(inter[0][0][0])
    # image_to_save = transforms.ToPILImage()(inter[0].cpu())
    #
    # image_to_save.save('./DROPOUT.png')
    # exit()

    return torch.clamp(inter, 0, 255)


def add(tensor, scalar):

    return torch.clamp(tensor+scalar.expand_as(tensor), 0, 255)


def add_element_wise(tensor, prob):


    x = torch.cuda.FloatTensor(tensor.shape)
    scalar = (x.normal_())

    return torch.clamp(tensor + scalar.expand_as(tensor), 0, 255)

def contrast(tensor, scalar):


    # Start by subtracting 128

    tensor = tensor + torch.cuda.FloatTensor([-128.0]).expand_as(tensor)

    # Multiply by the alpha ( scalar)
    tensor = scalar.expand_as(tensor)*tensor

    # Then add 128 and return the clamped value

    return torch.clamp(torch.cuda.FloatTensor([128.0]).expand_as(tensor) + tensor, 0, 255)



def grayscale(tensor, scalar):

    # Get the negated probability
    scalar_inv = torch.cuda.FloatTensor([1.0])-scalar


    tensor_gray =  tensor.clone()
    gray = tensor[:, 0, ...] + tensor[:, 1, ...] + tensor[:, 2, ...]
    tensor_gray[:, 0, ...] = gray
    tensor_gray[:, 1, ...] = gray
    tensor_gray[:, 2, ...] = gray


    tensor = torch.clamp((tensor * scalar_inv) \
                         + ( tensor_gray
                         * scalar.expand_as(tensor)), 0, 255)


    return tensor





def blur(tensor, sigma_vector):
    def make_filter(sigma_vector, window=5):

        # Simple function to make the filter that is going to be used by pytorch
        gaussian_filters = []
        for sigma in sigma_vector:
            g = scipy.signal.gaussian(window, sigma).reshape(1, 5)
            g_f = np.dot(g.T, g)
            g_f = g_f/sum(sum(g_f))
            gaussian_filters.append(g_f)


        return gaussian_filters


    gaussian_filters = make_filter(sigma_vector)


    kernel = torch.cuda.FloatTensor(gaussian_filters)

    kernel = torch.stack([kernel])  # this stacks the kernel into 3 identical 'channels' for rgb images


    return (torchF.conv2d(torch.autograd.Variable(tensor.transpose(0, 1)),
                         torch.autograd.Variable(kernel.transpose(0, 1)), padding=2, groups=tensor.shape[0]).data).transpose(0,1)
