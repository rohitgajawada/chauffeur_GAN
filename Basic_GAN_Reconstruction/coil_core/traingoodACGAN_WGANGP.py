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
import network.models.coil_ganmodules_nopatch as ganmodels_nopatch
import network.models.coil_acganmodules_nopatch as acganmodels_nopatch
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.utils as vutils
from torch.autograd import grad
import torch.nn.init as init

def calc_gradient_penalty(netD, inputs, fake_inputs):
    alpha = torch.rand((g_conf.BATCH_SIZE, 1, 1, 1))
    alpha = alpha.cuda()

    x_hat = alpha * inputs.data + (1 - alpha) * fake_inputs.data
    x_hat.requires_grad = True

    pred_hat = netD(x_hat)
    gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = 10 * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()
    return gradient_penalty

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

    manualSeed = 123
    torch.cuda.manual_seed(manualSeed)
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


    l1weight = g_conf.L1
    image_size = tuple([88, 200])
    testmode = 1

    if g_conf.GANMODEL_NAME == 'LSDcontrol':
        netD = ganmodels._netD().cuda()
        netG = ganmodels._netG(skip=g_conf.SKIP).cuda()
    if g_conf.GANMODEL_NAME == 'LSDcontrol_acgan_nopatch':
        netD = acganmodels_nopatch._netD().cuda()
        netG = acganmodels_nopatch._netG(skip=g_conf.SKIP).cuda()

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

    netG.train()
    netD.train()
    capture_time = time.time()
    if not os.path.exists('./imgs_' + exp_alias):
        os.mkdir('./imgs_' + exp_alias)

    for data in data_loader:

        input_data, float_data = data
        inputs = input_data['rgb'].cuda()
        inputs = inputs.squeeze(1)

        fake_inputs = netG(inputs)

        if iteration % 200 == 0:
            imgs_to_save = torch.cat((inputs[:2], fake_inputs[:2]), 0).cpu().data
            vutils.save_image(imgs_to_save, './imgs_' + exp_alias + '/' + str(iteration) + '_real_and_fake.png', normalize=True)
            coil_logger.add_image("Images", imgs_to_save, iteration)

        controls = float_data[:, dataset.controls_position(), :]
        steer = controls[:, 0].cuda()

        ##--------------------Discriminator part!!!!!!!!!!-----------------------
        ##fake
        set_requires_grad(netD, True)
        optimD.zero_grad()

        outputsD_fake_forD, fakeD_steer = netD(fake_inputs.detach())

        labsize = outputsD_fake_forD.size()
        labels_fake = torch.zeros(labsize) #Fake labels
        label_fake_noise = torch.rand(labels_fake.size()) * 0.1 #Label smoothing
        labels_fake = labels_fake + label_fake_noise
        labels_fake = Variable(labels_fake).cuda()

        lossD_fake_aux = MSE_loss(fakeD_steer, steer)

        lossD_fake_total = lossD_fake + lossD_fake_aux
        lossD_fake_total.backward()
        optimD.step()

        ##real
        set_requires_grad(netD, True)
        optimD.zero_grad()

        outputsD_real_forD, realD_steer = netD(inputs)

        labsize = outputsD_real_forD.size()
        labels_real = torch.ones(labsize) #Real labels
        label_real_noise = torch.rand(labels_real.size()) * 0.1 #Label smoothing
        labels_real = labels_real - label_real_noise
        labels_real = Variable(labels_real).cuda()

        lossD_real = torch.mean(outputsD_real_forD)
        lossD_real_aux = MSE_loss(realD_steer, steer)

        #Discriminator updates

        lossD_real_total = lossD_real + lossD_real_aux
        lossD_real_total.backward()
        optimD.step()

        lossD = lossD_real_total + lossD_fake_total

        coil_logger.add_scalar('Aux Real LossD', lossD_real_aux.data, iteration)
        coil_logger.add_scalar('Aux Fake LossD', lossD_fake_aux.data, iteration)
        coil_logger.add_scalar('Total Real LossD', lossD_real_total.data , iteration)
        coil_logger.add_scalar('Total Fake LossD', lossD_fake_total.data , iteration)
        coil_logger.add_scalar('Real LossD', lossD_real.data , iteration)
        coil_logger.add_scalar('Fake LossD', lossD_fake.data , iteration)

        ##--------------------Generator part!!!!!!!!!!-----------------------

        set_requires_grad(netD, False)
        optimG.zero_grad()
        outputsD_fake_forG, G_steer = netD(fake_inputs)
        #Generator updates

        lossG_smooth = L1_loss(fake_inputs, inputs)
        lossG_aux = MSE_loss(G_steer, steer)

        lossG = lossG_adv + lossG_aux + l1weight * lossG_smooth

        lossG.backward()
        optimG.step()

        coil_logger.add_scalar('Total LossG', lossG.data / len(inputs), iteration)
        coil_logger.add_scalar('Adv LossG', lossG_adv.data , iteration)
        coil_logger.add_scalar('Smooth LossG', lossG_smooth.data , iteration)
        coil_logger.add_scalar('Aux LossG', lossG_aux.data , iteration)

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