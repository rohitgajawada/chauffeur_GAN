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
    real_dataset = g_conf.TARGET_DOMAIN_PATH
    # real_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], "FinalRealWorldDataset")

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
                # if newkey.split('.')[0] == "branch" and oldkey.split('.')[0] == "branches":
                #     print("No Transfer of ",  newkey, " to ", oldkey)
                # else:
                print("Transferring ", newkey, " to ", oldkey)
                netF_state_dict[newkey] = model_IL_state_dict[oldkey]

            netF.load_state_dict(netF_state_dict)
            print("IL Model Loaded!")


    elif g_conf.GANMODEL_NAME == 'LSDcontrol_task_2d':
        netD_bin = ganmodels_taskAC._netD_bin().cuda()
        netD_aux = ganmodels_taskAC._netD_aux().cuda()
        netG = ganmodels_taskAC._netG().cuda()
        netF = ganmodels_taskAC._netF().cuda()

        if g_conf.PRETRAINED == 'IL':
            print("Loading IL")
            model_IL = torch.load(g_conf.IL_AGENT_PATH)
            model_IL_state_dict = model_IL['state_dict']

            netF_state_dict = netF.state_dict()

            print(len(netF_state_dict.keys()), len(model_IL_state_dict.keys()))
            for i, keys in enumerate(zip(netF_state_dict.keys(), model_IL_state_dict.keys())):
                newkey, oldkey = keys
                print("Transferring ", newkey, " to ", oldkey)
                netF_state_dict[newkey] = model_IL_state_dict[oldkey]

            netF.load_state_dict(netF_state_dict)
            print("IL Model Loaded!")

            #####
            if g_conf.IF_AUG:
                print("Loading Aug Decoder")
                model_dec = torch.load(g_conf.DECODER_RECON_PATH)
            else:
                print("Loading Decoder")
                model_dec = torch.load(g_conf.DECODER_RECON_PATH)
            model_dec_state_dict = model_dec['stateG_dict']

            netG_state_dict = netG.state_dict()

            print(len(netG_state_dict.keys()), len(model_dec_state_dict.keys()))
            for i, keys in enumerate(zip(netG_state_dict.keys(), model_dec_state_dict.keys())):
                newkey, oldkey = keys
                print("Transferring ", newkey, " to ", oldkey)
                netG_state_dict[newkey] = model_dec_state_dict[oldkey]

            netG.load_state_dict(netG_state_dict)
            print("Decoder Model Loaded!")

    init_weights(netD_bin)
    init_weights(netD_aux)

    print(netD_bin)
    print(netD_aux)
    print(netF)
    print(netG)

    optimD_bin = torch.optim.Adam(netD_bin.parameters(), lr=g_conf.LR_D, betas=(0.5, 0.999))
    optimD_aux = torch.optim.Adam(netD_aux.parameters(), lr=g_conf.LR_D, betas=(0.5, 0.999))
    optimG = torch.optim.Adam(netG.parameters(), lr=g_conf.LR_G, betas=(0.5, 0.999))
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

    netG.train()
    netD_bin.train()
    netD_aux.train()
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
        set_requires_grad(netD_aux, True)
        set_requires_grad(netF, True)
        set_requires_grad(netG, True)

        # print("ITERATION:", iteration)

        val = 0.0
        input_data, float_data, tgt_imgs = data

        if g_conf.IF_AUG:
            inputs = augmenter(0, input_data['rgb'])
            tgt_imgs = augmenter(0, tgt_imgs)
        else:
            inputs = input_data['rgb'].cuda()
            tgt_imgs = tgt_imgs.cuda()

        inputs = inputs.squeeze(1)
        inputs = inputs - val #subtracted by 0.5
        tgt_imgs = tgt_imgs - val #subtracted by 0.5

        controls = float_data[:, dataset.controls_position(), :]

        src_embed_inputs, src_branches = netF(inputs, dataset.extract_inputs(float_data).cuda())
        tgt_embed_inputs = netF(tgt_imgs, None)

        src_fake_inputs = netG(src_embed_inputs.detach())
        tgt_fake_inputs = netG(tgt_embed_inputs.detach())

        if iteration % 100 == 0:
            imgs_to_save = torch.cat((inputs[:1] + val, src_fake_inputs[:1] + val, tgt_imgs[:1] + val, tgt_fake_inputs[:1] + val), 0).cpu().data
            coil_logger.add_image("Images", imgs_to_save, iteration)
            imgs_to_save = imgs_to_save.clamp(0.0, 1.0)
            vutils.save_image(imgs_to_save, './imgs_' + exp_alias + '/' + str(iteration) + '_real_and_fake.png', normalize=False)

        ##--------------------Discriminator part!!!!!!!!!!-------------------##
        set_requires_grad(netD_aux, True)
        set_requires_grad(netD_bin, False)
        set_requires_grad(netF, False)
        set_requires_grad(netG, False)
        optimD_aux.zero_grad()

        ##source fake
        if g_conf.IF_POOL:
            src_fake_inputs_forD = fake_img_pool_src.query(src_fake_inputs)
            tgt_fake_inputs_forD = fake_img_pool_tgt.query(tgt_fake_inputs)
        else:
            src_fake_inputs_forD = src_fake_inputs
            tgt_fake_inputs_forD = tgt_fake_inputs

        outputsD_src_fake_forD_aux = netD_aux(src_fake_inputs_forD.detach())
        labsize = outputsD_src_fake_forD_aux.size()
        print("Discriminator aux label size HERE", labsize)

        labsize = labsize[0]

        labels_src_fake_aux = torch.zeros(labsize).type(torch.LongTensor) #Fake labels
        labels_src_fake_aux = Variable(labels_src_fake_aux).cuda()

        ##source real
        outputsD_src_real_forD_aux = netD_aux(inputs) # Pass real domain image here
        labels_src_real_aux = torch.zeros(labsize).type(torch.LongTensor) + 1 #Real labels
        labels_src_real_aux = Variable(labels_src_real_aux).cuda()

        ##target fake
        outputsD_tgt_fake_forD_aux = netD_aux(tgt_fake_inputs_forD.detach())
        labels_tgt_fake_aux = torch.zeros(labsize).type(torch.LongTensor) + 2
        labels_tgt_fake_aux = Variable(labels_tgt_fake_aux).cuda()

        ##target real
        outputsD_tgt_real_forD_aux = netD_aux(tgt_imgs) # Pass real domain image here
        labels_tgt_real_aux = torch.zeros(labsize).type(torch.LongTensor) + 3 #Real labels
        labels_tgt_real_aux = Variable(labels_tgt_real_aux).cuda()

        #discriminator losses
        print(outputsD_src_fake_forD_aux, labels_src_fake_aux)
        lossD_src_fake_aux = Loss(outputsD_src_fake_forD_aux, labels_src_fake_aux)
        lossD_src_real_aux = Loss(outputsD_src_real_forD_aux, labels_src_real_aux)
        lossD_tgt_fake_aux = Loss(outputsD_tgt_fake_forD_aux, labels_tgt_fake_aux)
        lossD_tgt_real_aux = Loss(outputsD_tgt_real_forD_aux, labels_tgt_real_aux)

        print("Some discriminator outputs:: ", outputsD_src_fake_forD_aux[0], outputsD_src_real_forD_aux[0], outputsD_tgt_fake_forD_aux[0], outputsD_tgt_real_forD_aux[0])

        #Discriminator updates

        lossD_aux = (lossD_src_real_aux + lossD_src_fake_aux + lossD_tgt_real_aux + lossD_tgt_fake_aux) * 0.25
        lossD_aux.backward(retain_graph=True)
        optimD_aux.step()

        coil_logger.add_scalar('Total LossD Aux', lossD_aux.data, iteration)
        coil_logger.add_scalar('Src Real LossD Aux', lossD_src_real_aux.data, iteration)
        coil_logger.add_scalar('Src Fake LossD Aux', lossD_src_fake_aux.data, iteration)
        coil_logger.add_scalar('Tgt Real LossD Aux', lossD_tgt_real_aux.data, iteration)
        coil_logger.add_scalar('Tgt Fake LossD Aux', lossD_tgt_fake_aux.data, iteration)


        #2nd discriminator part
        set_requires_grad(netD_aux, False)
        set_requires_grad(netD_bin, True)
        set_requires_grad(netF, False)
        set_requires_grad(netG, False)
        optimD_bin.zero_grad()

        outputsD_fake_src_bin = netD_bin(src_fake_inputs_forD.detach())
        outputsD_fake_tgt_bin = netD_bin(tgt_fake_inputs_forD.detach())

        outputsD_real_src_bin = netD_bin(inputs)
        outputsD_real_tgt_bin = netD_bin(tgt_imgs)

        gradient_penalty_src = calc_gradient_penalty(netD_bin, inputs, src_fake_inputs_forD)
        lossD_bin_src = torch.mean(outputsD_fake_src_bin - outputsD_real_src_bin) + gradient_penalty_src

        gradient_penalty_tgt = calc_gradient_penalty(netD_bin, tgt_imgs, tgt_fake_inputs_forD)
        lossD_bin_tgt = torch.mean(outputsD_fake_tgt_bin - outputsD_real_tgt_bin) + gradient_penalty_tgt

        lossD_bin = (lossD_bin_src + lossD_bin_tgt) * 0.5
        lossD_bin.backward(retain_graph=True)
        optimD_bin.step()

        coil_logger.add_scalar('Total LossD Bin', lossD_bin.data, iteration)
        coil_logger.add_scalar('Src LossD Bin', lossD_bin_src.data, iteration)
        coil_logger.add_scalar('Tgt LossD Bin', lossD_bin_tgt.data, iteration)

        ##--------------------Generator part!!!!!!!!!!-----------------------##
        set_requires_grad(netD_aux, False)
        set_requires_grad(netD_bin, False)
        set_requires_grad(netF, False)
        set_requires_grad(netG, True)
        optimG.zero_grad()

        #fake outputs for bin
        outputsD_bin_src_fake_forG = netD_bin(src_fake_inputs)
        outputsD_bin_tgt_fake_forG = netD_bin(tgt_fake_inputs)

        #fake outputs for aux
        outputsD_aux_src_fake_forG = netD_aux(src_fake_inputs)
        outputsD_aux_tgt_fake_forG = netD_aux(tgt_fake_inputs)

        #Generator updates

        if ((iteration + 1) % n_critic) == 0:
            #for netD_bin

            optimG.zero_grad()
            outputsD_bin_fake_forG = netD_bin(tgt_imgs)

            #Generator updates
            lossG_src_adv_bin = -1.0 * torch.mean(outputsD_bin_src_fake_forG)
            lossG_tgt_adv_bin = -1.0 * torch.mean(outputsD_bin_tgt_fake_forG)

            lossG_adv_bin = 0.5 * (lossG_src_adv_bin + lossG_tgt_adv_bin)

            #######################
            #for netD_aux
            #source
            lossG_src_adv_aux = Loss(outputsD_aux_src_fake_forG, labels_src_real_aux)
            lossG_src_smooth = L1_loss(src_fake_inputs, inputs) # L1 loss with real domain image

            #target
            lossG_tgt_adv_aux = Loss(outputsD_aux_tgt_fake_forG, labels_tgt_real_aux)
            lossG_tgt_smooth = L1_loss(tgt_fake_inputs, tgt_imgs) # L1 loss with real domain image

            lossG_adv_aux = 0.5 * (lossG_src_adv_aux + lossG_tgt_adv_aux)

            #######################
            lossG_Adv = 0.5 * (lossG_adv_bin + lossG_adv_aux)
            lossG_L1 = 0.5 * (lossG_src_smooth + lossG_tgt_smooth)

            lossG = (lossG_Adv + l1weight * lossG_L1) / (1.0 + l1weight)

            lossG.backward(retain_graph=True)
            optimG.step()

            coil_logger.add_scalar('Total LossG', lossG.data, iteration)
            coil_logger.add_scalar('LossG Adv', lossG_Adv.data, iteration)
            coil_logger.add_scalar('Adv Bin LossG', lossG_adv_bin.data , iteration)
            coil_logger.add_scalar('Adv Aux LossG', lossG_adv_aux.data, iteration)
            coil_logger.add_scalar('Smooth LossG', lossG_L1.data, iteration)

        #####Task network updates##########################
            set_requires_grad(netD_aux, False)
            set_requires_grad(netD_bin, False)
            set_requires_grad(netF, True)
            set_requires_grad(netG, False)

            optimF.zero_grad()
            lossF_task = Task_Loss.MSELoss(src_branches, dataset.extract_targets(float_data).cuda(),
                                         controls.cuda(), dataset.extract_inputs(float_data).cuda())


            lossF_adv_src_tgt = Loss(outputsD_aux_src_fake_forG, labels_tgt_real_aux)
            lossF_adv_tgt_src = Loss(outputsD_aux_tgt_fake_forG, labels_src_real_aux)
            lossF_adv = lossF_adv_src_tgt + lossF_adv_tgt_src

            lossF = (lossF_task + task_adv_weight * lossF_adv) / (1.0 + task_adv_weight)

            coil_logger.add_scalar('Total Task Loss', lossF.data, iteration)
            coil_logger.add_scalar('Adv Task Loss', lossF_adv.data, iteration)
            coil_logger.add_scalar('Only Task Loss', lossF_task.data, iteration)
            lossF.backward(retain_graph=True)
            optimF.step()

            if lossG.data < best_lossG:
                best_lossG = lossG.data.tolist()
                best_loss_iter_G = iteration

            if lossF.data < best_lossF:
                    best_lossF = lossF.data.tolist()
                    best_loss_iter_F = iteration

        #optimization for one iter done!

        position = random.randint(0, len(float_data)-1)

        if lossD_aux.data + lossD_bin.data < best_lossD:
            best_lossD = lossD_aux.data.tolist()

        accumulated_time += time.time() - capture_time
        capture_time = time.time()


        if is_ready_to_save(iteration):

            state = {
                'iteration': iteration,
                'stateD_aux_dict': netD_aux.state_dict(),
                'stateD_bin_dict': netD_bin.state_dict(),
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


        if iteration == best_loss_iter_F and iteration > 10000:

            state = {
                'iteration': iteration,
                'stateD_aux_dict': netD_aux.state_dict(),
                'stateD_bin_dict': netD_bin.state_dict(),
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
