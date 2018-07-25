import os
import sys
import random
import time
import traceback

import torch
import torch.optim as optim
import imgauggpu as iag
import numpy as np
import math

from configs import g_conf, set_type_of_process, merge_with_yaml
from input import CoILDataset, BatchSequenceSampler, splitter, real_dataloader
from logger import monitorer, coil_logger
from utils.checkpoint_schedule import is_ready_to_save, get_latest_saved_checkpoint
from torchvision import transforms

import network.models.coil_ganmodules as ganmodels
import network.models.coil_ganmodules_nopatch as ganmodels_nopatch
import network.models.coil_ganmodules_nopatch_smaller as ganmodels_nopatch_smaller
import network.models.coil_ganmodules_task_wdgrl as ganmodels_task
import network.models.coil_ganmodules_taskAC as ganmodels_taskAC
import network.models.coil_icra as coil_icra

from utils.pooler import ImagePool
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.utils as vutils
import torch.nn.init as init
from network.loss_task import TaskLoss
from torch.autograd import grad

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

    from time import gmtime, strftime

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
    real_dataset = '/datatmp/Datasets/July21_oldcodedata/trainB' # os.path.join(os.environ["COIL_DATASET_PATH"], "FinalRealWorldDataset")

    real_trans = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    #main data loader
    dataset = CoILDataset(full_dataset, transform=transforms.Compose([transforms.ToTensor()]))

    sampler = BatchSequenceSampler(splitter.control_steer_split(dataset.measurements, dataset.meta_data),
                          g_conf.BATCH_SIZE, g_conf.NUMBER_IMAGES_SEQUENCE, g_conf.SEQUENCE_STRIDE)
    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler,
                                              shuffle=False, num_workers=6, pin_memory=True)

    real_dl = real_dataloader.RealDataset(real_dataset, g_conf.BATCH_SIZE)

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

    print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
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
                # if newkey.split('.')[0] == "branch" and oldkey.split('.')[0] == "branches":
                #     print("No Transfer of ",  newkey, " to ", oldkey)
                # else:
                print("Transferring ", newkey, " to ", oldkey)
                netF_state_dict[newkey] = model_IL_state_dict[oldkey]

            netF.load_state_dict(netF_state_dict)
            print("IL Model Loaded!")


    elif g_conf.GANMODEL_NAME == 'LSDcontrol_task_2d':
        netD_bin = ganmodels_task._netD_task().cuda()
        netF = ganmodels_task._netF().cuda()

        if g_conf.PRETRAINED == 'IL':
            print("Loading IL")
            model_IL = torch.load('Encoder_IL.pth')
            model_IL_state_dict = model_IL['state_dict']

            netF_state_dict = netF.state_dict()

            print(len(netF_state_dict.keys()), len(model_IL_state_dict.keys()))
            for i, keys in enumerate(zip(netF_state_dict.keys(), model_IL_state_dict.keys())):
                newkey, oldkey = keys
                print("Transferring ", newkey, " to ", oldkey)
                netF_state_dict[newkey] = model_IL_state_dict[oldkey]

            netF.load_state_dict(netF_state_dict)
            print("IL Model Loaded!")

    init_weights(netD_bin)

    print(netD_bin)
    print(netF)

    optimD_bin = torch.optim.Adam(netD_bin.parameters(), lr=g_conf.LR_D, betas=(0.5, 0.999))
    if g_conf.TYPE =='task':
        optimF = torch.optim.Adam(netF.parameters(), lr=g_conf.LEARNING_RATE)
        Task_Loss = TaskLoss()

    if g_conf.GANMODEL_NAME == 'LSDcontrol_task_2d':
        print("Using cross entropy!")
        Loss = torch.nn.CrossEntropyLoss().cuda()

    L1_loss = torch.nn.L1Loss().cuda()

    iteration = 0
    best_loss_iter_F = 0
    best_loss_iter_G = 0
    best_lossF = 1000000.0
    best_lossD = 1000000.0
    best_lossG = 1000000.0
    accumulated_time = 0
    gen_iterations = 0
    n_critic = g_conf.N_CRITIC

    lossF = Variable(torch.Tensor([100.0]))
    lossG_adv = Variable(torch.Tensor([100.0]))
    lossG_smooth = Variable(torch.Tensor([100.0]))
    lossG = Variable(torch.Tensor([100.0]))

    netD_bin.train()
    netF.train()
    capture_time = time.time()

    if not os.path.exists('./imgs_' + exp_alias):
        os.mkdir('./imgs_' + exp_alias)

    #TODO check how C network is optimized in LSDSEG
    #TODO put family for losses
    #IMPORTANT WHILE RUNNING THIS, CONV.PY MUST HAVE BATCHNORMS

    fake_img_pool_src = ImagePool(50)
    fake_img_pool_tgt = ImagePool(50)

    for data in data_loader:

        set_requires_grad(netD_bin, True)
        set_requires_grad(netF, True)

        # print("ITERATION:", iteration)

        val = 0.0
        input_data, float_data = data
        tgt_imgs = real_dl.get_imgs()

        if g_conf.IF_AUG:
            inputs = augmenter(0, input_data['rgb'])
            tgt_imgs = augmenter(0, tgt_imgs)
        else:
            inputs = input_data['rgb'].cuda()
            tgt_imgs = tgt_imgs.cuda()

        inputs = inputs.squeeze(1)
        inputs = inputs
        tgt_imgs = tgt_imgs

        # imgs_to_save = torch.cat((inputs[:1], tgt_imgs[:1]), 0).cpu().data
        # coil_logger.add_image("Images", imgs_to_save, iteration)
        # # imgs_to_save = imgs_to_save.clamp(0.0, 1.0)
        # vutils.save_image(imgs_to_save, './imgs_' + exp_alias + '/' + str(iteration) + '_real_and_fake.png', normalize=False)

        controls = float_data[:, dataset.controls_position(), :]

        camera_angle = float_data[:,26,:]
        camera_angle = camera_angle.cuda()
        steer = float_data[:,0,:]
        steer = steer.cuda()
        speed = float_data[:,10,:]
        speed = speed.cuda()

        time_use =  1.0
        car_length = 3.0
        extra_factor = 2.5
        threshold = 1.0

        pos = camera_angle > 0.0
        pos = pos.type(torch.FloatTensor)
        neg = camera_angle <= 0.0
        neg = neg.type(torch.FloatTensor)
        pos = pos.cuda()
        neg = neg.cuda()

        rad_camera_angle = math.pi*(torch.abs(camera_angle)) / 180.0
        val = extra_factor *(torch.atan((rad_camera_angle*car_length)/(time_use*speed+0.05)))/3.1415
        steer -= pos * torch.min(val , torch.Tensor([0.6]).cuda())
        steer += neg * torch.min(val, torch.Tensor([0.6]).cuda())


        steer = steer.cpu()
        float_data[:,0,:] = steer
        float_data[:, 0, :][float_data[:,0,:] > 1.0] = 1.0
        float_data[:, 0, :][float_data[:,0,:] < -1.0] = -1.0

        src_embed_inputs, src_branches = netF(inputs, dataset.extract_inputs(float_data).cuda())
        tgt_embed_inputs = netF(tgt_imgs, None)

        ##--------------------Discriminator part!!!!!!!!!!-------------------##
        set_requires_grad(netD_bin, True)
        set_requires_grad(netF, False)
        optimD_bin.zero_grad()

        outputsD_real_src_bin = netD_bin(src_embed_inputs)
        outputsD_real_tgt_bin = netD_bin(tgt_embed_inputs)

        gradient_penalty = calc_gradient_penalty(netD_bin, src_embed_inputs, tgt_embed_inputs)
        lossD_bin = torch.mean(outputsD_real_tgt_bin - outputsD_real_src_bin) + gradient_penalty
        lossD_bin.backward(retain_graph=True)
        optimD_bin.step()

        coil_logger.add_scalar('Total LossD Bin', lossD_bin.data, iteration)

        if ((iteration + 1) % n_critic) == 0:
        #####Task network updates##########################
            set_requires_grad(netD_bin, False)
            set_requires_grad(netF, True)
            optimF.zero_grad()

            src_embed_inputs, src_branches = netF(inputs, dataset.extract_inputs(float_data).cuda())
            tgt_embed_inputs = netF(tgt_imgs, None)

            lossF_task = Task_Loss.MSELoss(src_branches, dataset.extract_targets(float_data).cuda(),
                                         controls.cuda(), dataset.extract_inputs(float_data).cuda())

            lossF_adv = netD_bin(src_embed_inputs).mean() - netD_bin(tgt_embed_inputs).mean()
            lossF = (lossF_task + task_adv_weight * lossF_adv)

            coil_logger.add_scalar('Total Task Loss', lossF.data, iteration)
            coil_logger.add_scalar('Adv Task Loss', lossF_adv.data, iteration)
            coil_logger.add_scalar('Only Task Loss', lossF_task.data, iteration)
            lossF.backward(retain_graph=True)
            optimF.step()

            if lossF_task.data < best_lossF:
                best_lossF = lossF_task.data.tolist()
                best_loss_iter_F = iteration

            print ("Iteration", iteration, "Best loss F", best_lossF, "BestLossIteration", best_loss_iter_F)

        #optimization for one iter done!

        position = random.randint(0, len(float_data)-1)

        accumulated_time += time.time() - capture_time
        capture_time = time.time()


        if is_ready_to_save(iteration):

            state = {
                'iteration': iteration,
                'stateD_bin_dict': netD_bin.state_dict(),
                'stateF_dict': netF.state_dict(),
                'best_lossD': best_lossD,
                'total_time': accumulated_time,
                'best_loss_iter_F': best_loss_iter_F

            }
            torch.save(state, os.path.join('/datatmp/Datasets/rohitgan/_logs', exp_batch, exp_alias
                                           , 'checkpoints', str(iteration) + '.pth'))


        if iteration == best_loss_iter_F and iteration > 10000:

            state = {
                'iteration': iteration,
                'stateD_bin_dict': netD_bin.state_dict(),
                'stateF_dict': netF.state_dict(),
                'best_lossD': best_lossD,
                'best_lossF': best_lossF,
                'total_time': accumulated_time,
                'best_loss_iter_F': best_loss_iter_F
            }
            torch.save(state, os.path.join('/datatmp/Datasets/rohitgan/_logs', exp_batch, exp_alias
                                           , 'best_modelF' + '.pth'))

        iteration += 1
