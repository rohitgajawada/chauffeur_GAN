from logger import coil_logger
import numpy as np
from torch.nn import functional as F
import torch.nn as nn
import torch


class TaskLoss(object):

    def __init__(self):
        pass

    # TODO: iteration control should go inside the logger, somehow
    def __call__(self, tensor, output_tensor):

        return tensor


    def MSELoss(self, branches, targets, controls, speed_gt, size_average=True, reduce=True, weights=None):

        # Normalize with the maximum speed from the training set (40 km/h)
        speed_gt = speed_gt / 40.0

        # calculate loss for each branch with specific activation
        loss_b1 = (branches[0] - targets) ** 2
        loss_b5 = (branches[1] - speed_gt) ** 2

        # add all branches losses together
        mse_loss = loss_b1

        if reduce:
            if size_average:
                mse_loss = torch.sum(mse_loss)/(mse_loss.shape[0] * mse_loss.shape[1]) + torch.sum(loss_b5)/mse_loss.shape[0]
            else:
                mse_loss = torch.sum(mse_loss) + torch.sum(loss_b5)
        else:
            if size_average:
                raise RuntimeError(" size_average can not be applies when reduce is set to 'False'")
            else:
                mse_loss =torch.cat([mse_loss,loss_b5],1)

        return mse_loss
