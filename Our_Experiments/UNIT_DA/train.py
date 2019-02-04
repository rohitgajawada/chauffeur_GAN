"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utilsUNIT import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images
import argparse
from torch.autograd import Variable
from trainer import UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil
from input import CoILDataset, BatchSequenceSampler, splitter, real_dataloader
from torchvision import transforms
import torchvision.utils as vutils
from random import shuffle
from skimage import io
from skimage.transform import resize
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs_UNIT/da_t20.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='/scratch/ROHITCVC', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--coordconv", action="store_true")
parser.add_argument("--nospeed", action="store_true")
parser.add_argument('--trainer', type=str, default='UNIT', help="MUNIT|UNIT")
parser.add_argument('--gpu', type=str, default='0', help="gpu id")

opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']
config['vgg_model_path'] = opts.output_path

# Setup model and data loader

if opts.trainer == 'UNIT':
    if opts.coordconv:
        trainer = UNIT_Trainer(config, coordconv=True, no_speed=opts.nospeed)
    else:
        trainer = UNIT_Trainer(config, coordconv=False, no_speed=opts.nospeed)
else:
    sys.exit("Only support MUNIT|UNIT")
trainer.cuda()

#############################COIL STUFF###########################
from time import gmtime, strftime

manualSeed = config['seed']
torch.cuda.manual_seed(manualSeed)
os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu

print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
if opts.nospeed:
    print("Wayve style with no speed")

full_dataset = config['train_dataset_name']
real_dataset = config['target_domain_path']

dataset = CoILDataset(full_dataset, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))

real_trans = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

sampler = BatchSequenceSampler(splitter.control_steer_split(dataset.measurements, dataset.meta_data),
                      config['batch_size'], 1, 1, config)
data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler,
                                          shuffle=False, num_workers=6, pin_memory=True)

real_dl = real_dataloader.RealDataset(real_dataset, config['batch_size'], real_trans)

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]
train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder


# Start training
iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if opts.resume else 0
a_black_count = 0
b_black_count = 0
while True:
    for it, data in enumerate(data_loader):
        print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
        trainer.update_learning_rate()
        skip = 0

        imgs_a, float_data = data
        imgs_b = real_dl.get_imgs()

        imgs_a = imgs_a['rgb']
        imgs_a = imgs_a.squeeze(1)

        # if torch.sum(imgs_a) == 0 or torch.sum(imgs_b) == 0:
        imgs_a_sum = imgs_a.sum(dim=3).sum(dim=2).sum(dim=1)
        imgs_b_sum = imgs_b.sum(dim=3).sum(dim=2).sum(dim=1)

        for img in imgs_a_sum:
            if img == 0 or img == float('inf'):
                print("A:", imgs_a_sum)
                print("B:", imgs_b_sum)
                a_black_count += 1
                print("black boy found", a_black_count)
                skip = 1

                imgs_to_save = imgs_a
                vutils.save_image(imgs_to_save, image_directory + '/' + str(it) + '_source.png', normalize=True)
                break

        for img in imgs_b_sum:
            if img == 0 or img == float('inf'):
                print("A:", imgs_a_sum)
                print("B:", imgs_b_sum)
                b_black_count += 1
                print("black boy found", b_black_count)
                skip = 1

                imgs_to_save = imgs_b
                vutils.save_image(imgs_to_save, image_directory + '/' + str(it) + '_target.png', normalize=True)
                break

        if skip == 1:
            continue

        images_a, images_b = Variable(imgs_a.cuda()), Variable(imgs_b.cuda())
        # Main training code
        trainer.dis_update(images_a, images_b, config)
        trainer.gen_update(images_a, images_b, float_data, config)

        # Dump training stats in log file
        if (iterations + 1) % config['log_iter'] == 0:
            write_loss(iterations, trainer, train_writer)

        # Write images
        train_display_images_a = Variable(imgs_a.cuda(), requires_grad=False, volatile=True)
        train_display_images_b = Variable(imgs_b.cuda(), requires_grad=False, volatile=True)

        sys.stdout.flush()

        if (iterations + 1) % config['image_save_iter'] == 0:
            image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
            write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

        if (iterations + 1) % config['image_display_iter'] == 0:
            image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(image_outputs, display_size, image_directory, 'train_current')

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')
