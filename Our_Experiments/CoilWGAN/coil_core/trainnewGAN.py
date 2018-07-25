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
from input import CoILDataset, BatchSequenceSampler, splitter
from logger import monitorer, coil_logger
from utils.checkpoint_schedule import is_ready_to_save, get_latest_saved_checkpoint
from torchvision import transforms

import network.models.coil_ganmodules as ganmodels
import network.models.coil_ganmodules_nopatch as ganmodels_nopatch
import network.models.coil_ganmodules_nopatch_smaller as ganmodels_nopatch_smaller
import network.models.coil_ganmodules_task as ganmodels_task
from utils.pooler import ImagePool
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.utils as vutils
import torch.nn.init as init
from network.loss_task import TaskLoss

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


    l1weight = g_conf.L1_WEIGHT
    image_size = tuple([88, 200])

    print("Configurations of ", exp_alias)
    print("GANMODEL_NAME", g_conf.GANMODEL_NAME)
    print("LOSS_FUNCTION", g_conf.LOSS_FUNCTION)
    print("LR_G, LR_D, LR", g_conf.LR_G, g_conf.LR_D, g_conf.LEARNING_RATE)
    print("SKIP", g_conf.SKIP)
    print("TYPE", g_conf.TYPE)
    print("L1 WEIGHT", g_conf.L1_WEIGHT)
    print("LAB SMOOTH", g_conf.LABSMOOTH)

    if g_conf.GANMODEL_NAME == 'LSDcontrol':
        netD = ganmodels._netD(loss=g_conf.LOSS_FUNCTION).cuda()
        netG = ganmodels._netG(loss=g_conf.LOSS_FUNCTION, skip=g_conf.SKIP).cuda()
    elif g_conf.GANMODEL_NAME == 'LSDcontrol_nopatch':
        netD = ganmodels_nopatch._netD(loss=g_conf.LOSS_FUNCTION).cuda()
        netG = ganmodels_nopatch._netG(loss=g_conf.LOSS_FUNCTION).cuda()
    elif g_conf.GANMODEL_NAME == 'LSDcontrol_nopatch_smaller':
        netD = ganmodels_nopatch_smaller._netD(loss=g_conf.LOSS_FUNCTION).cuda()
        netG = ganmodels_nopatch_smaller._netG(loss=g_conf.LOSS_FUNCTION).cuda()


    elif g_conf.GANMODEL_NAME == 'LSDcontrol_task':
        netD = ganmodels_task._netD(loss=g_conf.LOSS_FUNCTION).cuda()
        netG = ganmodels_task._netG(loss=g_conf.LOSS_FUNCTION).cuda()
        netF = ganmodels_task._netG(loss=g_conf.LOSS_FUNCTION).cuda()

    init_weights(netD)
    init_weights(netG)

    print(netD)
    print(netG)

    optimD = torch.optim.Adam(netD.parameters(), lr=g_conf.LR_D, betas=(0.7, 0.999))
    optimG = torch.optim.Adam(netG.parameters(), lr=g_conf.LR_G, betas=(0.7, 0.999))
    if g_conf.TYPE =='task':
        optimF = torch.optim.Adam(netG.parameters(), lr=g_conf.LEARNING_RATE, betas=(0.7, 0.999))
        Task_Loss = TaskLoss()

    if g_conf.LOSS_FUNCTION == 'LSGAN':
        Loss = torch.nn.MSELoss().cuda()
    elif g_conf.LOSS_FUNCTION == 'NORMAL':
        Loss = torch.nn.BCELoss().cuda()

    L1_loss = torch.nn.L1Loss().cuda()

    iteration = 0
    best_loss_iter = 0
    best_lossD = 1000000.0
    best_lossG = 1000000.0
    accumulated_time = 0

    netG.train()
    netD.train()
    capture_time = time.time()

    if not os.path.exists('./imgs_' + exp_alias):
        os.mkdir('./imgs_' + exp_alias)

    #TODO add image queue
    #TODO add auxiliary regression loss for steering
    #TODO put family for losses

    fake_img_pool = ImagePool(50)

    for data in data_loader:

        val = 0.5
        input_data, float_data = data
        inputs = input_data['rgb'].cuda()
        inputs = inputs.squeeze(1)
        inputs_in = inputs - val

        fake_inputs = netG(inputs_in) #subtracted by 0.5
        fake_inputs_in = fake_inputs

        if iteration % 200 == 0:
            imgs_to_save = torch.cat((inputs_in[:2] + val, fake_inputs_in[:2]), 0).cpu().data
            vutils.save_image(imgs_to_save, './imgs_' + exp_alias + '/' + str(iteration) + '_real_and_fake.png', normalize=True)
            coil_logger.add_image("Images", imgs_to_save, iteration)

        ##--------------------Discriminator part!!!!!!!!!!-------------------##
        set_requires_grad(netD, True)
        optimD.zero_grad()

        ##fake
        fake_inputs_forD = fake_img_pool.query(fake_inputs)
        outputsD_fake_forD = netD(fake_inputs_forD.detach())

        labsize = outputsD_fake_forD.size()
        labels_fake = torch.zeros(labsize) #Fake labels
        label_fake_noise = torch.rand(labels_fake.size()) * 0.05 - 0.025 #Label smoothing

        if g_conf.LABSMOOTH == 1:
            labels_fake = labels_fake + labels_fake_noise

        labels_fake = Variable(labels_fake).cuda()
        lossD_fake = Loss(outputsD_fake_forD, labels_fake)

        ##real
        outputsD_real = netD(inputs)
        print("some d outputs", outputsD_real[0])

        labsize = outputsD_real.size()
        labels_real = torch.ones(labsize) #Real labels
        label_real_noise = torch.rand(labels_real.size()) * 0.05 - 0.025 #Label smoothing

        if g_conf.LABSMOOTH == 1:
            labels_real = labels_real + labels_real_noise

        labels_real = Variable(labels_real).cuda()
        lossD_real = Loss(outputsD_real, labels_real)

        #Discriminator updates

        lossD = (lossD_real + lossD_fake) * 0.5
        # lossD /= len(inputs)
        lossD.backward()
        optimD.step()


        coil_logger.add_scalar('Total LossD', lossD.data, iteration)
        coil_logger.add_scalar('Real LossD', lossD_real.data, iteration)
        coil_logger.add_scalar('Fake LossD', lossD_fake.data, iteration)

        ##--------------------Generator part!!!!!!!!!!-----------------------

        set_requires_grad(netD, False)
        optimG.zero_grad()
        outputsD_fake_forG = netD(fake_inputs)
        #Generator updates

        lossG_adv = Loss(outputsD_fake_forG, labels_real)
        lossG_smooth = L1_loss(fake_inputs, inputs)
        lossG = (lossG_adv + l1weight * lossG_smooth) / (1.0 + l1weight)
        lossG

        lossG.backward()
        optimG.step()

        coil_logger.add_scalar('Total LossG', lossG.data, iteration)
        coil_logger.add_scalar('Adv LossG', lossG_adv.data, iteration)
        coil_logger.add_scalar('Smooth LossG', lossG_smooth.data, iteration)

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
