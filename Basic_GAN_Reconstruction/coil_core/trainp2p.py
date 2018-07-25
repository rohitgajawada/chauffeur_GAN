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
from skimage.transform import resize


import network.models.p2p_ganmodules as ganmodels
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.utils as vutils

import torch.nn.init as init

# class Resize(object):
#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         self.output_size = output_size

#     def __call__(self, sample):
#         images = sample['rgb']
#         for i in range(len(images)):
#             sample['rgb'][i] = resize(images[i], (128,128))
        
#         return sample
        


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
    dataset = CoILDataset(full_dataset, transform=transforms.Compose([transforms.ToPILImage(), transforms.Resize(128, 128), transforms.ToTensor(), transforms.Normalize([ 0.5,  0.5,  0.5], [ 1.0, 1.0, 1.0])]))

    sampler = BatchSequenceSampler(splitter.control_steer_split(dataset.measurements, dataset.meta_data),
                          g_conf.BATCH_SIZE, g_conf.NUMBER_IMAGES_SEQUENCE, g_conf.SEQUENCE_STRIDE)
    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler,
                                              shuffle=False, num_workers=6, pin_memory=True)

    transform = transforms.Compose([transforms.Resize((88, 200))])
    l1weight = 1.0
    image_size = tuple([88, 200])
    testmode = 1

    # print("helllooooo", g_conf.MODEL_NAME)
    if g_conf.GANMODEL_NAME == 'LSDcontrol':
        netD = ganmodels._netD().cuda()
        netG = ganmodels._netG(skip=g_conf.SKIP).cuda()
    # else:
    #     netD = ganmodels._oldnetD().cuda()
    #     netG = ganmodels._oldnetG().cuda()

    init_weights(netD)
    init_weights(netG)
    print(netD)
    print(netG)

    optimD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    MSE_loss = torch.nn.MSELoss().cuda()
    L1_loss = torch.nn.L1Loss().cuda()

    iteration = 0
    best_loss_iter = 0
    best_lossD = 1000000.0
    best_lossG = 1000000.0
    accumulated_time = 0

    netG.eval()
    netD.eval()
    capture_time = time.time()
    for data in data_loader:

        input_data, float_data = data
        inputs = input_data['rgb'].cuda()
        inputs = inputs.squeeze(1)
        print ("Inputs", i)

        #forward pass
        fake_inputs = netG(inputs)
        if iteration % 1000 == 0:
            coil_logger.add_image("Images", torch.cat((inputs[:3], fake_inputs[:3]), 0), iteration)

        ##--------------------Discriminator part!!!!!!!!!!-----------------------
        set_requires_grad(netD, True)
        optimD.zero_grad()

        ##fake
        outputsD_fake_forD = netD(fake_inputs.detach())

        labsize = outputsD_fake_forD.size()
        #Create labels of patchgan style with label smoothing
        labels_fake = torch.zeros(labsize[0], labsize[1], labsize[2], labsize[3]) #Fake labels
        label_fake_noise = torch.rand(labels_fake.size()) * 0.5 - 0.25 #Label smoothing
        labels_fake = labels_fake + label_fake_noise
        labels_fake = Variable(labels_fake).cuda()

        lossD_fake = MSE_loss(outputsD_fake_forD, labels_fake)

        ##real
        outputsD_real = netD(inputs)

        labsize = outputsD_real.size()
        #Create labels of patchgan style with label smoothing
        labels_real = torch.ones(labsize[0], labsize[1], labsize[2], labsize[3]) #Real labels
        label_real_noise = torch.rand(labels_real.size()) * 0.5 - 0.25 #Label smoothing
        labels_real = labels_real + label_real_noise
        labels_real = Variable(labels_real).cuda()

        lossD_real = MSE_loss(outputsD_real, labels_real)

        #Discriminator updates

        lossD = (lossD_real + lossD_fake) * 0.5
        lossD /= len(inputs)
        lossD.backward() #retain_graph=True needed?
        optimD.step()


        coil_logger.add_scalar('Total LossD', lossD.data, iteration)
        coil_logger.add_scalar('Real LossD', lossD_real.data / len(inputs), iteration)
        coil_logger.add_scalar('Fake LossD', lossD_fake.data / len(inputs), iteration)

        ##--------------------Generator part!!!!!!!!!!-----------------------

        #TODO change decoder architecture
        #TODO check norms of gradients later
        #TODO add auxiliary regression loss for steering

        set_requires_grad(netD, False)
        optimG.zero_grad()
        outputsD_fake_forG = netD(fake_inputs)
        #Generator updates

        lossG_adv = MSE_loss(outputsD_fake_forG, labels_real)
        lossG_smooth = L1_loss(fake_inputs, inputs)
        lossG = lossG_adv + l1weight * lossG_smooth
        lossG /= len(inputs)

        lossG.backward() #retain_graph=True needed?
        optimG.step()

        coil_logger.add_scalar('Total LossG', lossG.data, iteration)
        coil_logger.add_scalar('Adv LossG', lossG_adv.data / len(inputs), iteration)
        coil_logger.add_scalar('Smooth LossG', lossG_smooth.data / len(inputs), iteration)

        #optimization for one iter done!

        position = random.randint(0, len(float_data)-1)

        if lossD.data < best_lossD:
            best_lossD = lossD.data.tolist()

        if lossG.data < best_lossG:
            best_lossG = lossG.data.tolist()
            best_loss_iter = iteration

        accumulated_time += time.time() - capture_time
        capture_time = time.time()
        print("LossD", lossD.data.tolist(), "LossG", lossG.data.tolist(), "BestLossD", best_lossD, "BestLossG", best_lossG, "Iteration", iteration, "Best Loss Iteration", best_loss_iter)

        coil_logger.add_message('Iterating',
                                {'Iteration': iteration,
                                    'LossD': lossD.data.tolist(),
                                    'LossG': lossG.data.tolist(),
                                    'Images/s': (iteration*g_conf.BATCH_SIZE)/accumulated_time,
                                    'BestLossD': best_lossD, 'BestLossIteration': best_loss_iter,
                                    'BestLossG': best_lossG, 'BestLossIteration': best_loss_iter,
                                    'GroundTruth': dataset.extract_targets(float_data)[position].data.tolist(),
                                    'Inputs': dataset.extract_inputs(float_data)[position].data.tolist()},
                                iteration)
        if is_ready_to_save(iteration):

            state = {
                'iteration': iteration,
                'stateD_dict': netD.state_dict(),
                'stateG_dict': netG.state_dict(),
                'best_lossD': best_lossD,
                'best_lossG': best_lossG,
                'total_time': accumulated_time,
                'best_loss_iter': best_loss_iter

            }
            torch.save(state, os.path.join('/datatmp/Experiments/rohitgan/_logs', exp_batch, exp_alias
                                           , 'checkpoints', str(iteration) + '.pth'))
        if iteration == best_loss_iter:

            state = {
                'iteration': iteration,
                'stateD_dict': netD.state_dict(),
                'stateG_dict': netG.state_dict(),
                'best_lossD': best_lossD,
                'best_lossG': best_lossG,
                'total_time': accumulated_time,
                'best_loss_iter': best_loss_iter

            }
            torch.save(state, os.path.join('/datatmp/Experiments/rohitgan/_logs', exp_batch, exp_alias
                                           , 'best_modelG' + '.pth'))

        iteration += 1
