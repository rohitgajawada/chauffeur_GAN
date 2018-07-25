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
import network.models.coil_ganmodules_task as coil_ganmodules_task
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.utils as vutils
import torch.nn.init as init
from torchvision.utils import make_grid
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
    dataset = CoILDataset(full_dataset, transform=transforms.Compose([transforms.ToTensor()]))

    sampler = BatchSequenceSampler(splitter.control_steer_split(dataset.measurements, dataset.meta_data),
                          g_conf.BATCH_SIZE, g_conf.NUMBER_IMAGES_SEQUENCE, g_conf.SEQUENCE_STRIDE)
    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler,
                                              shuffle=False, num_workers=6, pin_memory=True)


    l1weight = 1.0
    image_size = tuple([88, 200])
    testmode = 1

    modelG = coil_ganmodules_task._netG()
    modelF = coil_ganmodules_task._netF()

    fstd = torch.load('netF_GAN_Pretrained.wts')
    gstd = torch.load('netG_GAN_Pretrained.wts')
    print(fstd.keys())
    print("+++++++++")
    print(gstd.keys())
    print(modelG)

    modelF.load_state_dict(fstd)
    modelG.load_state_dict(gstd)

    print(modelF)
    print(modelG)
    # netG.eval()

    capture_time = time.time()
    for data in data_loader:

        val = 0.5
        input_data, float_data = data
        inputs = input_data['rgb'].cuda()
        inputs = inputs.squeeze(1)
        inputs_in = inputs - val #subtracted by 0.5

        #forward pass
        embeds, branches = modelF(inputs)
        fake_inputs = modelG(embeds)

        imgs_to_save = torch.cat((inputs_in[:2] + val, fake_inputs_in[:2]), 0).cpu().data
        vutils.save_image(imgs_to_save, './imgs' + '/' + str(iteration) + '_real_and_fake.png', normalize=True)
        coil_logger.add_image("Images", imgs_to_save, iteration)
