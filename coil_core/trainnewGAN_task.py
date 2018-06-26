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
import network.models.coil_icra as coil_icra

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

    manualSeed = g_conf.SEED
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
    real_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], "FinalRealWorldDataset")

    #main data loader
    dataset = CoILDataset(full_dataset, real_dataset, transform=transforms.Compose([transforms.ToTensor()]))

    sampler = BatchSequenceSampler(splitter.control_steer_split(dataset.measurements, dataset.meta_data),
                          g_conf.BATCH_SIZE, g_conf.NUMBER_IMAGES_SEQUENCE, g_conf.SEQUENCE_STRIDE)
    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler,
                                              shuffle=False, num_workers=6, pin_memory=True)

    st = lambda aug: iag.Sometimes(aug, 0.4)
    oc = lambda aug: iag.Sometimes(aug, 0.3)
    rl = lambda aug: iag.Sometimes(aug, 0.09)
    augmenter = iag.Augmenter([iag.ToGPU()] + [
        rl(iag.GaussianBlur((0, 1.5))), # blur images with a sigma between 0 and 1.5
        rl(iag.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5)), # add gaussian noise to images
        oc(iag.Dropout((0.0, 0.10), per_channel=0.5)), # randomly remove up to X% of the pixels
        oc(iag.CoarseDropout((0.0, 0.10), size_percent=(0.08, 0.2),per_channel=0.5)), # randomly remove up to X% of the pixels
        oc(iag.Add((-40, 40), per_channel=0.5)), # change brightness of images (by -X to Y of original value)
        st(iag.Multiply((0.10, 2), per_channel=0.2)), # change brightness of images (X-Y% of original value)
        rl(iag.ContrastNormalization((0.5, 1.5), per_channel=0.5)), # improve or worsen the contrast
        rl(iag.Grayscale((0.0, 1))), # put grayscale
        ]# do all of the above in random order
    )


    l1weight = g_conf.L1_WEIGHT
    task_adv_weight = g_conf.TASK_ADV_WEIGHT
    image_size = tuple([88, 200])

    print("Configurations of ", exp_alias)
    print("GANMODEL_NAME", g_conf.GANMODEL_NAME)
    print("LOSS_FUNCTION", g_conf.LOSS_FUNCTION)
    print("LR_G, LR_D, LR", g_conf.LR_G, g_conf.LR_D, g_conf.LEARNING_RATE)
    print("SKIP", g_conf.SKIP)
    print("TYPE", g_conf.TYPE)
    print("L1 WEIGHT", g_conf.L1_WEIGHT)
    print("TASK ADV WEIGHT", g_conf.TASK_ADV_WEIGHT)
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
            print("Loading IL")
            model_IL = torch.load('best_loss_20-06_EpicClearWeather.pth')
            model_IL_state_dict = model_IL['state_dict']

            netF_state_dict = netF.state_dict()

            print(len(netF_state_dict.keys()), len(model_IL_state_dict.keys()))
            for i, keys in enumerate(zip(netF_state_dict.keys(), model_IL_state_dict.keys())):
                newkey, oldkey = keys
                if newkey.split('.')[0] == "branch" and oldkey.split('.')[0] == "branches":
                    print("No Transfer of ",  newkey, " to ", oldkey)
                else:
                    print("Transferring ", newkey, " to ", oldkey)
                    netF_state_dict[newkey] = model_IL_state_dict[oldkey]

            netF.load_state_dict(netF_state_dict)
            print("IL Model Loaded!")

    init_weights(netD)
    init_weights(netG)
    #do init for netF also later but now it is in the model code itself

    print(netD)
    print(netF)
    print(netG)

    optimD = torch.optim.Adam(netD.parameters(), lr=g_conf.LR_D, betas=(0.7, 0.999))
    optimG = torch.optim.Adam(netG.parameters(), lr=g_conf.LR_G, betas=(0.7, 0.999))
    if g_conf.TYPE =='task':
        optimF = torch.optim.Adam(netF.parameters(), lr=g_conf.LEARNING_RATE)
        Task_Loss = TaskLoss()

    if g_conf.LOSS_FUNCTION == 'LSGAN':
        Loss = torch.nn.MSELoss().cuda()
    elif g_conf.LOSS_FUNCTION == 'NORMAL':
        if g_conf.GANMODEL_NAME == 'LSDcontrol_task':
            Loss = torch.nn.CrossEntropyLoss().cuda()
        else:
            Loss = torch.nn.BCELoss().cuda()

    L1_loss = torch.nn.L1Loss().cuda()

    iteration = 0
    best_loss_iter_F = 0
    best_loss_iter_G = 0
    best_lossF = 1000000.0
    best_lossD = 1000000.0
    best_lossG = 1000000.0
    accumulated_time = 0

    gen_iterations = 0

    netG.train()
    netD.train()
    netF.train()
    capture_time = time.time()

    if not os.path.exists('./imgs_' + exp_alias):
        os.mkdir('./imgs_' + exp_alias)

    #TODO check how C network is optimized in LSDSEG
    #TODO put family for losses
    #TODO copy entire network with last branch

    for data in data_loader:

        set_requires_grad(netD, True)
        set_requires_grad(netF, True)
        set_requires_grad(netG, True)

        # print("ITERATION:", iteration)

        val = 0.5
        input_data, float_data, tgt_imgs = data
        if iteration % 2 == 1:
            inputs = augmenter(0, input_data['rgb'])
            tgt_imgs = augmenter(0, tgt_imgs)
        else:
            inputs = input_data['rgb'].cuda()
            tgt_imgs = tgt_imgs.cuda()

        inputs = inputs.squeeze(1)
        inputs_in = inputs - val #subtracted by 0.5
        tgt_imgs_in = tgt_imgs - val #subtracted by 0.5

        controls = float_data[:, dataset.controls_position(), :]

        src_embed_inputs, src_branches = netF(inputs_in, dataset.extract_inputs(float_data).cuda())
        tgt_embed_inputs = netF(tgt_imgs_in, None)

        src_fake_inputs = netG(src_embed_inputs.detach())
        tgt_fake_inputs = netG(tgt_embed_inputs.detach())

        if iteration % 500 == 0:
            imgs_to_save = torch.cat((inputs_in[:1] + val, src_fake_inputs[:1] + val, tgt_imgs_in[:1] + val, tgt_fake_inputs[:1] + val), 0).cpu().data
            vutils.save_image(imgs_to_save, './imgs_' + exp_alias + '/' + str(iteration) + '_real_and_fake.png', normalize=True)
            coil_logger.add_image("Images", imgs_to_save, iteration)

        ##--------------------Discriminator part!!!!!!!!!!-------------------##
        set_requires_grad(netD, True)
        set_requires_grad(netF, False)
        set_requires_grad(netG, False)
        optimD.zero_grad()

        ##source fake
        outputsD_src_fake_forD = netD(src_fake_inputs.detach())
        labsize = outputsD_src_fake_forD.size()
        labels_src_fake = torch.zeros(labsize) #Fake labels
        labels_src_fake = Variable(labels_src_fake).cuda()

        ##source real
        outputsD_src_real_forD = netD(inputs_in) # Pass real domain image here
        labels_src_real = torch.zeros(labsize) + 1 #Real labels
        labels_src_real = Variable(labels_src_real).cuda()

        ##target fake
        outputsD_tgt_fake_forD = netD(tgt_fake_inputs.detach())
        labels_tgt_fake = torch.zeros(labsize) + 2
        labels_tgt_fake = Variable(labels_tgt_fake).cuda()

        ##target real
        outputsD_tgt_real_forD = netD(tgt_imgs_in) # Pass real domain image here
        labels_tgt_real = torch.zeros(labsize) + 3 #Real labels
        labels_tgt_real = Variable(labels_tgt_real).cuda()

        #discriminator losses
        lossD_src_fake = Loss(outputsD_src_fake_forD, labels_src_fake)
        lossD_src_real = Loss(outputsD_src_real_forD, labels_src_real)
        lossD_tgt_fake = Loss(outputsD_tgt_fake_forD, labels_tgt_fake)
        lossD_tgt_real = Loss(outputsD_tgt_real_forD, labels_tgt_real)

        #Discriminator updates

        lossD = (lossD_src_real + lossD_src_fake + lossD_src_real + lossD_src_fake) * 0.25
        lossD /= len(inputs_in)
        lossD.backward()
        optimD.step()


        coil_logger.add_scalar('Total LossD', lossD.data, iteration)
        # coil_logger.add_scalar('Real LossD', lossD_real.data / len(inputs), iteration)
        # coil_logger.add_scalar('Fake LossD', lossD_fake.data / len(inputs), iteration)


        ##--------------------Generator part!!!!!!!!!!-----------------------##
        set_requires_grad(netD, False)
        set_requires_grad(netF, False)
        set_requires_grad(netG, True)
        optimG.zero_grad()

        outputsD_src_fake_forG = netD(src_fake_inputs)
        outputsD_tgt_fake_forG = netD(tgt_fake_inputs)
        #Generator updates

        #source
        lossG_src_adv = Loss(outputsD_src_fake_forG, labels_src_real)
        lossG_src_smooth = L1_loss(src_fake_inputs, inputs_in) # L1 loss with real domain image

        #target
        lossG_tgt_adv = Loss(outputsD_tgt_fake_forG, labels_tgt_real)
        lossG_tgt_smooth = L1_loss(tgt_fake_inputs, tgt_imgs_in) # L1 loss with real domain image

        lossG_Adv = lossG_src_adv + lossG_tgt_adv
        lossG_L1 = lossG_src_smooth + lossG_tgt_smooth

        lossG = (lossG_Adv + l1weight * lossG_L1) / (2*(1.0 + l1weight))
        lossG /= len(inputs_in)
        print(lossG)

        lossG.backward(retain_graph=True)
        optimG.step()

        #####Task network updates##########################
        set_requires_grad(netD, False)
        set_requires_grad(netF, True)
        set_requires_grad(netG, False)

        optimF.zero_grad()
        lossF = Task_Loss.MSELoss(src_branches, dataset.extract_targets(float_data).cuda(),
                                     controls.cuda(), dataset.extract_inputs(float_data).cuda())


        lossF_adv_src_tgt = Loss(outputsD_src_fake_forG, labels_tgt_real)
        lossF_adv_tgt_src = Loss(outputsD_tgt_fake_forG, labels_src_real)
        lossF_adv = lossF_adv_src_tgt + lossF_adv_tgt_src
        lossF_adv /= len(inputs_in)

        lossF = (lossF + task_adv_weight * lossF_adv) / (1.0 + task_adv_weight)

        coil_logger.add_scalar('Task Loss', lossF.data, iteration)
        coil_logger.add_scalar('Adv Task Loss', lossF_adv.data, iteration)
        lossF.backward()
        optimF.step()

        coil_logger.add_scalar('Total LossG', lossG.data, iteration)
        coil_logger.add_scalar('Adv LossG', lossG_Adv.data / len(inputs), iteration)
        coil_logger.add_scalar('Smooth LossG', lossG_L1.data / len(inputs), iteration)

        #optimization for one iter done!

        position = random.randint(0, len(float_data)-1)
        if lossD.data < best_lossD:
            best_lossD = lossD.data.tolist()

        if lossG.data < best_lossG:
            best_lossG = lossG.data.tolist()
            best_loss_iter_G = iteration

        if lossF.data < best_lossF:
                best_lossF = lossF.data.tolist()
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
