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
from torch.autograd import grad
from network.loss_task import TaskLoss

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

    if g_conf.TRAIN_TYPE == 'WGAN':
        clamp_value = g_conf.CLAMP
        n_critic = g_conf.N_CRITIC

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
        netF = ganmodels_task._netF(loss=g_conf.LOSS_FUNCTION).cuda()

        if g_conf.PRETRAINED == 'RECON':
            netF_statedict = torch.load('netF_GAN_Pretrained.wts')
            netF.load_state_dict(netF_statedict)
        elif g_conf.PRETRAINED == 'IL':
            model_IL = torch.load('best_loss_20-06_EpicClearWeather.pth')
            model_IL_state_dict = model_IL['state_dict']

            netF_state_dict = netF.state_dict()
            for i, keys in enumerate(zip(netF_state_dict.keys(), model_IL_state_dict.keys())):
                newkey, oldkey = keys
                if newkey.split('.')[0] == "branch" and oldkey.split('.')[0] == "branches":
                    print("No Transfer of ",  newkey, " to ", oldkey)
                else:
                    print("Transferring ", newkey, " to ", oldkey)
                    netF_state_dict[newkey] = model_IL_state_dict[oldkey]
                netF.load_state_dict(netF_state_dict)

    init_weights(netD)
    init_weights(netG)
    #do init for netF also later but now it is in the model code itself

    print(netD)
    print(netF)
    print(netG)

    optimD = torch.optim.Adam(netD.parameters(), lr=g_conf.LR_D, betas=(0.5, 0.999))
    optimG = torch.optim.Adam(netG.parameters(), lr=g_conf.LR_G, betas=(0.5, 0.999))
    if g_conf.TYPE =='task':
        optimF = torch.optim.Adam(netF.parameters(), lr=g_conf.LEARNING_RATE)
        Task_Loss = TaskLoss()

    if g_conf.LOSS_FUNCTION == 'LSGAN':
        Loss = torch.nn.MSELoss().cuda()
    elif g_conf.LOSS_FUNCTION == 'NORMAL':
        Loss = torch.nn.BCEWithLogitsLoss().cuda()

    L1_loss = torch.nn.L1Loss().cuda()

    iteration = 0
    best_loss_iter_F = 0
    best_loss_iter_G = 0
    best_lossF = 1000000.0
    best_lossD = 1000000.0
    best_lossG = 1000000.0
    accumulated_time = 0
    lossF = Variable(torch.Tensor([100.0]))

    lossG_adv = Variable(torch.Tensor([100.0]))
    lossG_smooth = Variable(torch.Tensor([100.0]))
    lossG = Variable(torch.Tensor([100.0]))

    netG.train()
    netD.train()
    netF.train()
    capture_time = time.time()

    if not os.path.exists('./imgs_' + exp_alias):
        os.mkdir('./imgs_' + exp_alias)

    #TODO put family for losses

    fake_img_pool = ImagePool(50)

    for data in data_loader:

        set_requires_grad(netD, True)
        set_requires_grad(netF, True)
        set_requires_grad(netG, True)

        # print("ITERATION:", iteration)

        val = 0.0
        input_data, float_data = data
        inputs = input_data['rgb'].cuda()
        inputs = inputs.squeeze(1)
        inputs_in = inputs - val #subtracted by 0.5

        #TODO: make sure the F network does not get optimized by G optim
        controls = float_data[:, dataset.controls_position(), :]
        embed, branches = netF(inputs_in, dataset.extract_inputs(float_data).cuda())
        print("Branch Outputs:::", branches[0][0])

        embed_inputs = embed
        fake_inputs = netG(embed_inputs)
        fake_inputs_in = fake_inputs

        if iteration % 500 == 0:
            imgs_to_save = torch.cat((inputs_in[:2] + val, fake_inputs_in[:2] + val), 0).cpu().data
            vutils.save_image(imgs_to_save, './imgs_' + exp_alias + '/' + str(iteration) + '_real_and_fake.png', normalize=True)
            coil_logger.add_image("Images", imgs_to_save, iteration)

        ##--------------------Discriminator part!!!!!!!!!!-------------------##
        set_requires_grad(netD, True)
        set_requires_grad(netF, False)
        set_requires_grad(netG, False)
        optimD.zero_grad()

        ##fake
        # fake_inputs_forD = fake_img_pool.query(fake_inputs)
        outputsD_fake_forD = netD(fake_inputs)

        labsize = outputsD_fake_forD.size()
        labels_fake = torch.zeros(labsize) #Fake labels
        label_fake_noise = torch.rand(labels_fake.size()) * 0.1 #Label smoothing

        if g_conf.LABSMOOTH == 1:
            labels_fake = labels_fake + label_fake_noise

        labels_fake = Variable(labels_fake).cuda()
        lossD_fake = torch.mean(outputsD_fake_forD) #Loss(outputsD_fake_forD, labels_fake)

        ##real
        outputsD_real = netD(inputs_in)

        labsize = outputsD_real.size()
        labels_real = torch.ones(labsize) #Real labels
        label_real_noise = torch.rand(labels_real.size()) * 0.1 #Label smoothing

        if g_conf.LABSMOOTH == 1:
            labels_real = labels_real - label_real_noise

        labels_real = Variable(labels_real).cuda()
        lossD_real = -1.0 * torch.mean(outputsD_real) #Loss(outputsD_real, labels_real)

        ### Gradient Penalty ###
        gradient_penalty = calc_gradient_penalty(netD, inputs, fake_inputs)
        # alpha = torch.rand((g_conf.BATCH_SIZE, 1, 1, 1))
        # alpha = alpha.cuda()
        #
        # x_hat = alpha * inputs.data + (1 - alpha) * fake_inputs.data
        # x_hat.requires_grad = True
        #
        # pred_hat = netD(x_hat)
        # gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
        #                 create_graph=True, retain_graph=True, only_inputs=True)[0]
        #
        # gradient_penalty = 10 * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

        #Discriminator updates

        lossD = torch.mean(outputsD_fake_forD - outputsD_real) + gradient_penalty #(lossD_real + lossD_fake) * 0.5
        # lossD /= len(inputs)
        print ("Loss d", lossD)
        lossD.backward(retain_graph=True)
        optimD.step()

        # if g_conf.TRAIN_TYPE == 'WGAN':
        #     for p in netD.parameters():
        #         p.data.clamp_(-clamp_value, clamp_value)


        coil_logger.add_scalar('Total LossD', lossD.data, iteration)
        coil_logger.add_scalar('Real LossD', lossD_real.data / len(inputs), iteration)
        coil_logger.add_scalar('Fake LossD', lossD_fake.data / len(inputs), iteration)


        ##--------------------Generator part!!!!!!!!!!-----------------------##
        set_requires_grad(netD, False)
        set_requires_grad(netF, False)
        set_requires_grad(netG, True)

        if ((iteration + 1) % n_critic) == 0:
            optimG.zero_grad()
            outputsD_fake_forG = netD(fake_inputs)

            #Generator updates
            lossG_adv = -1.0 * torch.mean(outputsD_fake_forG) #Loss(outputsD_fake_forG, labels_real)
            lossG_smooth = L1_loss(fake_inputs, inputs)
            lossG = (lossG_adv + l1weight * lossG_smooth) / (1.0 + l1weight)
            # lossG /= len(inputs)
            print(lossG)
            lossG.backward(retain_graph=True)
            optimG.step()

        #####Task network updates##########################
            set_requires_grad(netD, False)
            set_requires_grad(netF, True)
            set_requires_grad(netG, False)

            optimF.zero_grad()
            lossF = Variable(torch.Tensor())
            lossF = Task_Loss.MSELoss(branches, dataset.extract_targets(float_data).cuda(),
                                        controls.cuda(), dataset.extract_inputs(float_data).cuda())
            coil_logger.add_scalar('Task Loss', lossF.data, iteration)
            lossF.backward()
            optimF.step()



        coil_logger.add_scalar('Total LossG', lossG.data, iteration)
        coil_logger.add_scalar('Adv LossG', lossG_adv.data / len(inputs), iteration)
        coil_logger.add_scalar('Smooth LossG', lossG_smooth.data / len(inputs), iteration)

        #optimization for one iter done!

        position = random.randint(0, len(float_data)-1)
        if lossD.data < best_lossD:
            best_lossD = lossD.data.tolist()
        # print (lossG.item(), best_lossG)
        if lossG.item() < best_lossG:
            best_lossG = lossG.item()
            best_loss_iter_G = iteration

        if lossF.item() < best_lossF:
            best_lossF = lossF.item()
            best_loss_iter_F = iteration

        accumulated_time += time.time() - capture_time
        capture_time = time.time()
        print("LossD", lossD.data.tolist(), "LossG", lossG.data.tolist(), "BestLossD", best_lossD, "BestLossG", best_lossG, "LossF", lossF, "BestLossF", best_lossF, "Iteration", iteration, "Best Loss Iteration G", best_loss_iter_G, "Best Loss Iteration F", best_loss_iter_F)

        coil_logger.add_message('Iterating',
                                {'Iteration': iteration,
                                    'LossD': lossD.data.tolist(),
                                    'LossG': lossG.data.tolist(),
                                    'Images/s': (iteration*g_conf.BATCH_SIZE)/accumulated_time,
                                    'BestLossD': best_lossD,
                                    'BestLossG': best_lossG, 'BestLossIterationG': best_loss_iter_G,
                                    'BestLossF': best_lossF, 'BestLossIterationF': best_loss_iter_F,
                                    'GroundTruth': dataset.extract_targets(float_data)[position].data.tolist(),
                                    'Inputs': dataset.extract_inputs(float_data)[position].data.tolist()},
                                iteration)

        if is_ready_to_save(iteration):

            state = {
                'iteration': iteration,
                'stateD_dict': netD.state_dict(),
                'stateG_dict': netG.state_dict(),
                'stateF_dict': netF.state_dict(),
                'best_lossD': best_lossD,
                'best_lossG': best_lossG,
                'total_time': accumulated_time,
                'best_loss_iter_G': best_loss_iter_G,
                'best_loss_iter_F': best_loss_iter_F

            }
            torch.save(state, os.path.join('/datatmp/Experiments/rohitgan/_logs', exp_batch, exp_alias
                                           , 'checkpoints', str(iteration) + '.pth'))
        if iteration == best_loss_iter_G and iteration > 10000:

            state = {
                'iteration': iteration,
                'stateD_dict': netD.state_dict(),
                'stateG_dict': netG.state_dict(),
                'stateF_dict': netF.state_dict(),
                'best_lossD': best_lossD,
                'best_lossG': best_lossG,
                'total_time': accumulated_time,
                'best_loss_iter_G': best_loss_iter_G

            }
            torch.save(state, os.path.join('/datatmp/Experiments/rohitgan/_logs', exp_batch, exp_alias
                                           , 'best_modelG' + '.pth'))

        if iteration == best_loss_iter_F and iteration > 10000:

            state = {
                'iteration': iteration,
                'stateD_dict': netD.state_dict(),
                'stateG_dict': netG.state_dict(),
                'stateF_dict': netF.state_dict(),
                'best_lossD': best_lossD,
                'best_lossG': best_lossG,
                'best_lossF': best_lossF,
                'total_time': accumulated_time,
                'best_loss_iter_F': best_loss_iter_F

            }
            torch.save(state, os.path.join('/datatmp/Experiments/rohitgan/_logs', exp_batch, exp_alias
                                           , 'best_modelF' + '.pth'))

        iteration += 1
