import os
import time
import sys
import numpy as np
import torch
import glob
from network import CoILModel, Loss

from input.coil_dataset_onlyil import CoILDataset
import network.models.coil_ganmodules_taskAC as ganmodels_taskAC
from network.loss_task import TaskLoss
from torchvision import transforms

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

full_dataset = '/datatmp/Datasets/JulyRohitRishabh/EpicWeather12_60k_June21_Straight+Turn/SeqVal'
dataset = CoILDataset(full_dataset, transform=transforms.Compose([transforms.ToTensor()]))
data_loader = torch.utils.data.DataLoader(dataset, batch_size=50,
                                          shuffle=False, num_workers=12, pin_memory=True)

ckpts = glob.glob('/datatmp/Experiments/rohitgan/_logs/eccv/all_da_aug_orig_E1E12/checkpoints/*.pth')

netF = ganmodels_taskAC._netF().cuda()
Task_Loss = TaskLoss()

best_loss = 1000
best_loss_ckpt = "none"

for ckpt in ckpts:

    iter = 0
    current_loss = 0
    current_loss_total = 0

    print(ckpt)
    model_IL = torch.load(ckpt)
    model_IL_state_dict = model_IL['stateF_dict']
    netF.load_state_dict(model_IL_state_dict)

    for data in data_loader:

        input_data, float_data = data
        inputs = input_data['rgb'].cuda()
        inputs = inputs.squeeze(1)

        print(inputs.size())
        tgt_embed_inputs, tgt_branches = netF(inputs, dataset.extract_inputs(float_data).cuda())

        controls = float_data[:, dataset.controls_position(), :]

        lossF_task = Task_Loss.MSELoss(tgt_branches, dataset.extract_targets(float_data).cuda(), controls.cuda(), dataset.extract_inputs(float_data).cuda())

        current_loss += lossF_task
        iter += 1

    current_loss_total = current_loss / (iter * 1.0)

    if current_loss_total < best_loss:

        best_loss = current_loss_total
        best_loss_ckpt = ckpt

print(best_loss_ckpt, "::::", best_loss)
