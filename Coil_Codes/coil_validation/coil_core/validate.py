import os
import time
import sys

import numpy as np
import torch
import traceback
import torch.optim as optim
import random
import glob
import math
# What do we define as a parameter what not.

from configs import g_conf, set_type_of_process, merge_with_yaml
from network import CoILModel, Loss
from input import CoILDataset, Augmenter
from logger import monitorer, coil_logger
from utils.checkpoint_schedule import get_latest_evaluated_checkpoint, is_next_checkpoint_ready,\
    maximun_checkpoint_reach, get_next_checkpoint
from torchvision import transforms
from torch.autograd import Variable


def write_waypoints_output(iteration, output):

    for i in range(g_conf.BATCH_SIZE):
        steer = 0.8 * (output[i][3] + output[i][4])/0.5

        if steer > 0:
            steer = min(steer, 1)
        else:
            steer = max(steer, -1)

        coil_logger.write_on_csv(iteration, [steer,
                                            output[i][1],
                                            output[i][2]])


def write_regular_output(iteration, output):
    for i in range(g_conf.BATCH_SIZE):
        coil_logger.write_on_csv(iteration, [output[i][0],
                                            output[i][1],
                                            output[i][2]])


# The main function maybe we could call it with a default name
def execute(gpu, exp_batch, exp_alias, dataset_name, architecture, suppress_output):

    try:
        # We set the visible cuda devices
        torch.manual_seed(2)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

        # Validation available for:
            # coil_unit (UNIT + task combined)
            # coil_icra (Also used for finetuned models)
            # wgangp_lsd (Our architecture)

        architecture_name = architecture
        # At this point the log file with the correct naming is created.
        if architecture_name == 'coil_unit':
            pass
        elif architecture_name == 'wgangp_lsd':
            merge_with_yaml(os.path.join('/home/rohitrishabh/CoilWGAN/configs', exp_batch, exp_alias+'.yaml'))
            set_type_of_process('validation', dataset_name)
        elif architecture_name == 'coil_icra':
            merge_with_yaml(os.path.join('/home/adas/CleanedCode/CoIL_Codes/coil_20-06/configs', exp_batch, exp_alias+'.yaml'))
            set_type_of_process('validation', dataset_name)

            if monitorer.get_status(exp_batch, exp_alias + '.yaml', g_conf.PROCESS_NAME)[0] == "Finished":
                # TODO: print some cool summary or not ?
                return

        if not os.path.exists('_output_logs'):
            os.mkdir('_output_logs')

        if suppress_output:
            sys.stdout = open(os.path.join('_output_logs',
                            g_conf.PROCESS_NAME + '_' + str(os.getpid()) + ".out"), "a", buffering=1)



        #Define the dataset. This structure is has the __get_item__ redefined in a way
        #that you can access the HDFILES positions from the root directory as a in a vector.
        if dataset_name != []:
            full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], dataset_name)
        else:
            full_dataset = os.environ["COIL_DATASET_PATH"]

        augmenter = Augmenter(None)

        dataset = CoILDataset(full_dataset, transform=augmenter)

        # Creates the sampler, this part is responsible for managing the keys. It divides
        # all keys depending on the measurements and produces a set of keys for each bach.

        # The data loader is the multi threaded module from pytorch that release a number of
        # workers to get all the data.
        # TODO: batch size an number of workers go to some configuration file
        batchsize=30
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize,
                                                  shuffle=False,
                                                  num_workers=1,
                                                  pin_memory=True)


        # TODO: here there is clearly a posibility to make a cool "conditioning" system.

        if architecture_name == 'coil_unit':
            model_task, model_gen = CoILModel('coil_unit')
            model_task, model_gen = model_task.cuda(), model_gen.cuda()
        else:
            model = CoILModel(architecture_name)
            model.cuda()

        latest = 0

        # print (dataset.meta_data)
        best_loss = 1000
        best_error = 1000
        best_loss_mini = 1000
        best_loss_iter = 0
        best_error_iter = 0
        batch_size = 30
        best_loss_ckpt = ''

        if architecture_name == 'coil_unit':
            ckpts = glob.glob('/home/rohitrishabh/UNIT_DA/outputs/' + exp_alias + '/checkpoints/gen*.pt')
        else:
            ckpts = glob.glob(os.path.join('/home/adas/CleanedCode/CoIL_Codes/coil_20-06/_logs', exp_batch, exp_alias) + '/*.pth')

        if architecture_name == 'coil_unit':
            model_task.eval()
            model_gen.eval()
        else:
            model.eval()
        ckpts = sorted(ckpts)
        # TODO: refactor on the getting on the checkpoint organization needed
        for ckpt in ckpts:

            # if is_next_checkpoint_ready(g_conf.TEST_SCHEDULE):

            # latest = get_next_checkpoint(g_conf.TEST_SCHEDULE)
            # ckpt = os.path.join('/datatmp/Experiments/rohitgan/_logs', exp_batch, exp_alias
            #                         , 'checkpoints', str(latest) + '.pth')
            checkpoint = torch.load(ckpt)
            print ("Validation loaded ", ckpt)
            if architecture_name == 'wgangp_lsd':
                print(ckpt, checkpoint['best_loss_iter_F'])
                model.load_state_dict(checkpoint['stateF_dict'])
                model.eval()
            elif architecture_name == 'coil_unit':
                model_task.load_state_dict(checkpoint['task'])
                model_gen.load_state_dict(checkpoint['b'])
                model_task.eval()
                model_gen.eval()
            elif architecture_name == 'coil_icra':
                model.load_state_dict(checkpoint['state_dict'])
                model.eval()

            accumulated_loss = 0
            accumulated_error = 0
            iteration_on_checkpoint = 0
            datacount = 0
            for data in data_loader:

                input_data, float_data = data

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

                datacount += 1
                control_position = 24
                speed_position = 10

                if architecture_name == 'wgangp_lsd':
                    embed, output = model(torch.squeeze(input_data['rgb']).cuda(),
                                                    float_data[:, speed_position, :].cuda())

                    loss = torch.sum((output[0] - dataset.extract_targets(float_data).cuda())**2).data.tolist()
                    mean_error = torch.sum(torch.abs(output[0] - dataset.extract_targets(float_data).cuda())).data.tolist()

                elif architecture_name == 'coil_unit':
                    embed, n_b = model_gen.encode(torch.squeeze(input_data['rgb']).cuda())
                    output = model_task(embed, Variable(float_data[:, speed_position, :]).cuda())

                    loss = torch.sum((output[0].data - dataset.extract_targets(float_data).cuda())**2)
                    mean_error = torch.sum(torch.abs(output[0].data - dataset.extract_targets(float_data).cuda()))

                elif architecture_name == 'coil_icra':
                    output = model.forward_branch(torch.squeeze(input_data['rgb']).cuda(),
                                                float_data[:, speed_position, :].cuda(),
                                                float_data[:, control_position, :].cuda())

                    loss = torch.sum((output - dataset.extract_targets(float_data).cuda())**2).data.tolist()
                    mean_error = torch.sum(torch.abs(output - dataset.extract_targets(float_data).cuda())).data.tolist()

                if loss < best_loss_mini:
                    best_loss_mini = loss

                accumulated_error += mean_error
                accumulated_loss += loss
                # error = torch.abs(output[0] - dataset.extract_targets(float_data).cuda())


                # Log a random position
                position = random.randint(0, len(float_data) - 1)
                iteration_on_checkpoint += 1

            print(datacount, len(data_loader), accumulated_loss)
            checkpoint_average_loss = accumulated_loss/float(datacount * batchsize)
            checkpoint_average_error = accumulated_error/float(datacount * batchsize)

            if checkpoint_average_loss < best_loss:
                best_loss = checkpoint_average_loss
                best_loss_iter = latest
                best_loss_ckpt = ckpt

            if checkpoint_average_error < best_error:
                best_error = checkpoint_average_error
                best_error_iter = latest

            print("current loss", checkpoint_average_loss)
            print ("best_loss", best_loss)

            coil_logger.add_message('Iterating',

                    {'Summary':
                        {
                        'Error': checkpoint_average_error,
                        'Loss': checkpoint_average_loss,
                        'BestError': best_error,
                        'BestLoss': best_loss,
                        'BestLossCheckpoint': best_loss_iter,
                        'BestErrorCheckpoint': best_error_iter
                        },

                    'Checkpoint': latest},
                                    latest)
            latest += 2000

        coil_logger.add_message('Finished', {})
        print ("Best Validation Loss ckpt:", best_loss_ckpt)

        # TODO: DO ALL THE AMAZING LOGGING HERE, as a way to very the status in paralell.
        # THIS SHOULD BE AN INTERELY PARALLEL PROCESS

    except KeyboardInterrupt:
        coil_logger.add_message('Error', {'Message': 'Killed By User'})

    except:
        traceback.print_exc()

        coil_logger.add_message('Error', {'Message': 'Something Happened'})
