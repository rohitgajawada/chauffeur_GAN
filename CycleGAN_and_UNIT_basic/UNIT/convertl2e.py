from __future__ import print_function

import glob
import h5py
import numpy as np


from utils import get_config, get_data_loader_folder
from trainer import MUNIT_Trainer, UNIT_Trainer
import argparse
from torch.autograd import Variable
import torchvision.utils as vutils
import sys
import torch
import os
from torchvision import transforms
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help="net configuration")
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and others for b2a")
parser.add_argument('--num_style',type=int, default=10, help="number of styles to sample")
parser.add_argument('--seed', type=int, default=10, help="random seed")
parser.add_argument('--style', type=str, default='', help="style image path")
parser.add_argument('--output_path', type=str, default='.', help="path for logs, checkpoints, and VGG model weight")
parser.add_argument('--output_folder', type=str, help="output image path")

opts = parser.parse_args()

torch.manual_seed(opts.seed)
torch.cuda.manual_seed(opts.seed)



config = get_config(opts.config)
opts.num_style = 1 if opts.style != '' else opts.num_style

config['vgg_model_path'] = opts.output_path

trainer = UNIT_Trainer(config)

state_dict = torch.load(opts.checkpoint)
trainer.gen_a.load_state_dict(state_dict['a'])
trainer.gen_b.load_state_dict(state_dict['b'])
trainer.cuda()
trainer.eval()
encode = trainer.gen_a.encode if opts.a2b else trainer.gen_b.encode # encode function
style_encode = trainer.gen_b.encode if opts.a2b else trainer.gen_a.encode # encode function
decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode # decode function

if 'new_size' in config:
    new_size = config['new_size']
else:
    if opts.a2b==1:
        new_size = config['new_size_a']
    else:
        new_size = config['new_size_b']

transform = transforms.Compose([transforms.Resize(new_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset_path = '/datatmp/Datasets/rohitrishabh/HighQuality30k-12-06/SeqTrain'

dataset = glob.glob(dataset_path + '/*.h5')

number_images_per_file = 1000
number_rewards = 28
image_size1 = 200
image_size2 = 88

for i, data in enumerate(dataset):
    ldata = h5py.File(data)
    print (len(ldata['rgb']))
    print (data)
    hf = h5py.File('/datatmp/Datasets/rohitrishabh/FakeDataset-RoughCVC/SeqTrain/' + data[-13:], 'w')
    # print(self._image_size2, self._image_size1)
    data_center= hf.create_dataset('rgb', (number_images_per_file,image_size2,image_size1,3),dtype=np.float64)
    # segs_center= hf.create_dataset('labels', (number_images_per_file,image_size2,image_size1,1),dtype=np.uint8)
    # depth_center= hf.create_dataset('depth', (number_images_per_file,image_size2,image_size1,3),dtype=np.float64)
    data_rewards  = hf.create_dataset('targets', (number_images_per_file, number_rewards),'f')
    
    # segs_center = ldata['labels']# = ldata['labels']
    # depth_center = ldata['depth']
    # data_rewards.write_direct(ldata['targets'].value)

    for j in range(len(ldata['rgb'])):
        # segs_center[j] = ldata['labels'][j]
        # depth_center[j] = ldata['depth'][j]
        data_rewards[j] = ldata['targets'][j]
        ldata_img = np.uint8(ldata['rgb'][j]*255)
        pil_img = Image.fromarray(ldata_img)

        image = Variable(transform(pil_img.convert('RGB')).unsqueeze(0).cuda(), volatile=True)
        
        content, _ = encode(image)
        
        outputs = decode(content)
        outputs = (outputs + 1) / 2.
        outputs = outputs.data.squeeze(0).cpu().numpy()
        # print (outputs.shape)
        outputs_1 = np.transpose(outputs, (1,2,0))
        # print (outputs_1[0,:,0])
        outputs_1img = Image.fromarray(np.uint8(outputs_1*255)).convert('RGB')
        transform2 = transforms.Resize((88,200))
        outputs_resize = transform2(outputs_1img)
        # outputs_resize.show('1.png')
        finalimg_np = np.asarray(outputs_resize)/255.
        data_center[j] = finalimg_np
        # print (finalimg_np[0,:,0])
        # path = os.path.join(opts.output_folder, 'output.jpg')
        # vutils.save_image(outputs.data, path, padding=0, normalize=True)
    hf.close()
    
    
