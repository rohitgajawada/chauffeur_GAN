import os
import sys
import random
import time
import traceback

import torch
import torch.optim as optim
import imgauggpu as iag
import numpy as np

from configs import g_conf, set_type_of_process, merge_with_yaml
from network import CoILModel, Loss
from input import CoILDataset, BatchSequenceSampler, splitter
from logger import monitorer, coil_logger
from utils.checkpoint_schedule import is_ready_to_save, get_latest_saved_checkpoint
from torchvision import transforms

import network.models.coil_ganmodules as ganmodels
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.utils as vutils
import torch.nn.init as init
from torchvision.utils import make_grid
# import matplotlib.pyplot as plt
import numpy as np
import random

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):

            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)

            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

# The main function maybe we could call it with a default name
def execute(gpu, exp_batch, exp_alias):

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    merge_with_yaml(os.path.join('configs', exp_batch, exp_alias + '.yaml'))
    set_type_of_process('train')

    coil_logger.add_message('Loading', {'GPU': gpu})
    if not os.path.exists('_output_logs'):
        os.mkdir('_output_logs')
    sys.stdout = open(os.path.join('_output_logs',
                      g_conf.PROCESS_NAME + '_' + str(os.getpid()) + ".out"), "a", buffering=1)
    if monitorer.get_status(exp_batch, exp_alias + '.yaml', g_conf.PROCESS_NAME)[0] == "Finished":
        return

    full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], g_conf.TRAIN_DATASET_NAME)
    dataset = CoILDataset(full_dataset, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([ 0.5,  0.5,  0.5], [ 1.0, 1.0, 1.0])]))

    sampler = BatchSequenceSampler(splitter.control_steer_split(dataset.measurements, dataset.meta_data),
                          g_conf.BATCH_SIZE, g_conf.NUMBER_IMAGES_SEQUENCE, g_conf.SEQUENCE_STRIDE)
    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler,
                                              shuffle=False, num_workers=6, pin_memory=True)


    l1weight = 1.0
    image_size = tuple([88, 200])
    testmode = 1

    if g_conf.GANMODEL_NAME == 'LSDcontrol':
        netD = ganmodels._netD().cuda()
        netG = ganmodels._netG(skip=g_conf.SKIP).cuda()

    checkpoint = torch.load(os.path.join('/datatmp/Experiments/rohitgan/_logs/eccv/experiment_1/checkpoints/1010000.pth'))
    netG.load_state_dict(checkpoint['stateG_dict'])
    netD.load_state_dict(checkpoint['stateD_dict'])

    print(netD)
    print(netG)

    MSE_loss = torch.nn.MSELoss().cuda()
    L1_loss = torch.nn.L1Loss().cuda()

    iteration = 0

    # netG.eval()
    # netD.eval()
    capture_time = time.time()
    for data in data_loader:

        input_data, float_data = data
        inputs = input_data['rgb'].cuda()
        inputs = inputs.squeeze(1)

        #forward pass
        fake_inputs = netG(inputs)

        imgs_to_save = torch.cat((fake_inputs, inputs), 0).cpu().data
        imgs_to_save = (imgs_to_save + 1.0)/2.0
        imgs = [img for img in imgs_to_save]
        vutils.save_image(imgs, 'imgs/' + str(iteration) + '.png')


        outputsD_fake_forD = netD(fake_inputs.detach())

        labsize = outputsD_fake_forD.size()
        labels_fake = torch.zeros(labsize[0], labsize[1], labsize[2], labsize[3]) #Fake labels
        label_fake_noise = torch.rand(labels_fake.size()) * 0.5 - 0.25 #Label smoothing
        labels_fake = labels_fake + label_fake_noise
        labels_fake = Variable(labels_fake).cuda()

        lossD_fake = MSE_loss(outputsD_fake_forD, labels_fake)

        outputsD_real = netD(inputs)

        labsize = outputsD_real.size()
        labels_real = torch.ones(labsize[0], labsize[1], labsize[2], labsize[3]) #Real labels
        label_real_noise = torch.rand(labels_real.size()) * 0.5 - 0.25 #Label smoothing
        labels_real = labels_real + label_real_noise
        labels_real = Variable(labels_real).cuda()

        lossD_real = MSE_loss(outputsD_real, labels_real)

        lossD = (lossD_real + lossD_fake) * 0.5
        lossD /= len(inputs)

        outputsD_fake_forG = netD(fake_inputs)

        lossG_adv = MSE_loss(outputsD_fake_forG, labels_real)
        lossG_smooth = L1_loss(fake_inputs, inputs)
        lossG = lossG_adv + l1weight * lossG_smooth
        lossG /= len(inputs)

        # print("LossD", lossD.data.tolist(), "LossG", lossG.data.tolist(), "BestLossD", best_lossD, "BestLossG", best_lossG, "Iteration", iteration, "Best Loss Iteration", best_loss_iter)

        iteration += 1
        # if iteration == 20:
        #     break
