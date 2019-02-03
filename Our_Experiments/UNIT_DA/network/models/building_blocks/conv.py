
# from logger import coil_logger
import torch.nn as nn
import torch.nn.init as init
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super(AddCoords, self).__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = Variable(torch.arange(x_dim).repeat(1, y_dim, 1))
        yy_channel = Variable(torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2))

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type(torch.cuda.FloatTensor),
            yy_channel.type(torch.cuda.FloatTensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):

    def __init__(self, params=None, with_r=False, module_name='Default'): #in_channels, out_channels, , **kwargs):
        super(CoordConv, self).__init__()

        if params is None:
            raise ValueError("Creating a NULL fully connected block")
        if 'channel_sizes' not in params:
            raise ValueError(" Missing the channel sizes parameter ")
        if 'kernel_sizes' not in params:
            raise ValueError(" Missing the kernel sizes parameter ")
        if 'strides' not in params:
            raise ValueError(" Missing the strides parameter ")
        if 'dropouts' not in params:
            raise ValueError(" Missing the dropouts parameter ")
        if 'end_layer' not in params:
            raise ValueError(" Missing the end module parameter ")

        if len(params['dropouts']) != len(params['channel_sizes'])-1:
            raise ValueError("Dropouts should be from the len of channel_sizes minus 1")

        self.layers = []

        for i in range(0, len(params['channel_sizes'])-1):
            addcoords = AddCoords(with_r=with_r)
            in_size = params['channel_sizes'][i] + 2
            if with_r:
                in_size += 1
            conv = nn.Conv2d(in_size, out_channels=params['channel_sizes'][i+1],
                                kernel_size=params['kernel_sizes'][i], stride=params['strides'][i])

            dropout = nn.Dropout2d(p=params['dropouts'][i])
            relu = nn.ReLU(inplace=True)

            bn = nn.BatchNorm2d(params['channel_sizes'][i+1])
            layer = nn.Sequential(*[addcoords, conv, bn, dropout, relu])

            self.layers.append(layer)

        self.layers = nn.Sequential(*self.layers)
        self.module_name = module_name

        self.ifendlayer = params['end_layer']

    def forward(self, x):
        # get only the speeds from measurement labels

        # TODO: TRACK NANS OUTPUTS
        # TODO: Maybe change the name
        # TODO: Maybe add internal logs !

        """ addcoords + conv1 + batch normalization + dropout + relu + gap"""
        x = self.layers(x)
        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))

        if self.ifendlayer:
            x = x.view(-1, self.num_flat_features(x))

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Conv(nn.Module):

    def __init__(self, params=None, module_name='Default'):
        # TODO:  For now the end module is a case
        # TODO: Make an auto naming function for this.

        super(Conv, self).__init__()

        if params is None:
            raise ValueError("Creating a NULL fully connected block")
        if 'channel_sizes' not in params:
            raise ValueError(" Missing the channel sizes parameter ")
        if 'kernel_sizes' not in params:
            raise ValueError(" Missing the kernel sizes parameter ")
        if 'strides' not in params:
            raise ValueError(" Missing the strides parameter ")
        if 'dropouts' not in params:
            raise ValueError(" Missing the dropouts parameter ")
        if 'end_layer' not in params:
            raise ValueError(" Missing the end module parameter ")

        if len(params['dropouts']) != len(params['channel_sizes'])-1:
            raise ValueError("Dropouts should be from the len of channel_sizes minus 1")




        """" ------------------ IMAGE MODULE ---------------- """


        self.layers = []

        # TODO: need to log the loaded networks
        for i in range(0, len(params['channel_sizes'])-1):

            conv = nn.Conv2d(in_channels=params['channel_sizes'][i], out_channels=params['channel_sizes'][i+1],
                             kernel_size=params['kernel_sizes'][i], stride=params['strides'][i])

            dropout = nn.Dropout2d(p=params['dropouts'][i])
            relu = nn.ReLU(inplace=True)

            #This was instancenorm before but I made it batchnorm
            bn = nn.BatchNorm2d(params['channel_sizes'][i+1])
            layer = nn.Sequential(*[conv, bn, dropout, relu])

            self.layers.append(layer)

        self.layers = nn.Sequential(*self.layers)
        self.module_name = module_name

        self.ifendlayer = params['end_layer']





    # TODO: iteration control should go inside the logger, somehow

    def forward(self, x):
        # get only the speeds from measurement labels

        # TODO: TRACK NANS OUTPUTS
        # TODO: Maybe change the name
        # TODO: Maybe add internal logs !

        """ conv1 + batch normalization + dropout + relu """
        x = self.layers(x)

        if self.ifendlayer:
            x = x.view(-1, self.num_flat_features(x))

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
