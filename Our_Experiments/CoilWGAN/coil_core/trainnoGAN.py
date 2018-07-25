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

    # print("helllooooo", g_conf.MODEL_NAME)
    if g_conf.GANMODEL_NAME == 'LSDcontrol':
        # netD = ganmodels._netD().cuda()
        netG = ganmodels._netG(skip=g_conf.SKIP).cuda()
    # else:
    #     netD = ganmodels._oldnetD().cuda()
    #     netG = ganmodels._oldnetG().cuda()

    # init_weights(netD)
    init_weights(netG)
    # print(netD)
    print(netG)

    # optimD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.7, 0.999))
    MSE_loss = torch.nn.MSELoss().cuda()
    L1_loss = torch.nn.L1Loss().cuda()

    iteration = 0
    best_loss_iter = 0
    best_lossD = 1000000.0
    best_lossG = 1000000.0
    accumulated_time = 0

    netG.eval()
    # netD.eval()
    capture_time = time.time()
    for data in data_loader:

        val = 0.5
        input_data, float_data = data
        inputs = input_data['rgb'].cuda()
        inputs = inputs.squeeze(1)
        inputs_in = inputs - val

        #forward pass
        # print(inputs[0][0][0][0], inputs_in[0][0][0][0])
        fake_inputs = netG(inputs_in) #subtracted by 0.5
        fake_inputs_in = fake_inputs
        # print(fake_inputs[0][0][0][0], fake_inputs_in[0][0][0][0])
        if iteration % 200 == 0:
            imgs_to_save = torch.cat((inputs_in[:2] + val, fake_inputs_in[:2]), 0).cpu().data
            vutils.save_image(imgs_to_save, './noganimgs/' + str(iteration) + 'noganreal_samples.png', normalize=True)
            coil_logger.add_image("Images", imgs_to_save, iteration)

        optimG.zero_grad()

        print("~~~~~~~~~__________")
        print(inputs_in[0][0][0][0])
        print(fake_inputs[0][0][0][0])
        lossG_mse = MSE_loss(fake_inputs, inputs)
        print(lossG_mse)
        lossG_mse /= len(inputs_in)

        print("~~~~~~~~~__________--------------")

        lossG_mse.backward() #retain_graph=True needed?
        optimG.step()

        coil_logger.add_scalar('MSE LossG', lossG_mse.data / len(inputs_in), iteration)

        #optimization for one iter done!

        position = random.randint(0, len(float_data)-1)

        # if lossD.data < best_lossD:
        #     best_lossD = lossD.data.tolist()

        if lossG_mse.data < best_lossG:
            best_lossG = lossG_mse.data.tolist()
            best_loss_iter = iteration

        accumulated_time += time.time() - capture_time
        capture_time = time.time()
        # print("LossD", lossD.data.tolist(), "LossG", lossG.data.tolist(), "BestLossD", best_lossD, "BestLossG", best_lossG, "Iteration", iteration, "Best Loss Iteration", best_loss_iter)

        # coil_logger.add_message('Iterating',
        #                         {'Iteration': iteration,
        #                             'LossD': lossD.data.tolist(),
        #                             'LossG': lossG.data.tolist(),
        #                             'Images/s': (iteration*g_conf.BATCH_SIZE)/accumulated_time,
        #                             'BestLossD': best_lossD, 'BestLossIteration': best_loss_iter,
        #                             'BestLossG': best_lossG, 'BestLossIteration': best_loss_iter,
        #                             'GroundTruth': dataset.extract_targets(float_data)[position].data.tolist(),
        #                             'Inputs': dataset.extract_inputs(float_data)[position].data.tolist()},
        #                         iteration)
        # if is_ready_to_save(iteration):
        #
        #     state = {
        #         'iteration': iteration,
        #         'stateD_dict': netD.state_dict(),
        #         'stateG_dict': netG.state_dict(),
        #         'best_lossD': best_lossD,
        #         'best_lossG': best_lossG,
        #         'total_time': accumulated_time,
        #         'best_loss_iter': best_loss_iter
        #
        #     }
        #     torch.save(state, os.path.join('/datatmp/Experiments/rohitgan/_logs', exp_batch, exp_alias
        #                                    , 'checkpoints', str(iteration) + '.pth'))
        # if iteration == best_loss_iter:
        #
        #     state = {
        #         'iteration': iteration,
        #         'stateD_dict': netD.state_dict(),
        #         'stateG_dict': netG.state_dict(),
        #         'best_lossD': best_lossD,
        #         'best_lossG': best_lossG,
        #         'total_time': accumulated_time,
        #         'best_loss_iter': best_loss_iter
        #
        #     }
        #     torch.save(state, os.path.join('/datatmp/Experiments/rohitgan/_logs', exp_batch, exp_alias
        #                                    , 'best_modelG' + '.pth'))
        #
        iteration += 1
