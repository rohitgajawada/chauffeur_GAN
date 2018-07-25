import os
import sys
import random
import time
import traceback

import torch
import torch.optim as optim
import imgauggpu as iag
import math
# import matplotlib.pyplot as plt

# What do we define as a parameter what not.

from configs import g_conf, set_type_of_process, merge_with_yaml
from network import CoILModel, Loss
from input import CoILDataset, BatchSequenceSampler, splitter
from logger import monitorer, coil_logger
from utils.checkpoint_schedule import is_ready_to_save, get_latest_saved_checkpoint
from torchvision import transforms


def adjust_learning_rate(optimizer, num_iters):
	"""
	Adjusts the learning rate every epoch based on the selected schedule
	"""
	cur_iters = num_iters
	lr = g_conf.LEARNING_RATE
	minlr = g_conf.MIN_LEARNING_RATE
	scheduler = "normal"
	decayinterval = g_conf.LEARNING_RATE_DECAY_INTERVAL
	decaylevel = g_conf.LEARNING_RATE_DECAY_LEVEL
	if scheduler == "normal":
		while cur_iters >= decayinterval:
			lr = lr * decaylevel
			cur_iters = cur_iters - decayinterval
		lr = max(lr, minlr)

	for param_group in optimizer.param_groups:
		print("New Learning rate is ", lr)
		param_group['lr'] = lr


# The main function maybe we could call it with a default name
def execute(gpu, exp_batch, exp_alias):
    # We set the visible cuda devices

    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu

        # At this point the log file with the correct naming is created.
        merge_with_yaml(os.path.join('configs', exp_batch, exp_alias + '.yaml'))
        set_type_of_process('train')

        coil_logger.add_message('Loading', {'GPU': gpu})

        if not os.path.exists('_output_logs'):
            os.mkdir('_output_logs')

        sys.stdout = open(os.path.join('_output_logs',
                          g_conf.PROCESS_NAME + '_' + str(os.getpid()) + ".out"), "a", buffering=1)



        if monitorer.get_status(exp_batch, exp_alias + '.yaml', g_conf.PROCESS_NAME)[0] == "Finished":
            # TODO: print some cool summary or not ?
            return

        #Define the dataset. This structure is has the __get_item__ redefined in a way
        #that you can access the HDFILES positions from the root directory as a in a vector.
        full_dataset = os.path.join(os.environ["COIL_DATASET_PATH"], g_conf.TRAIN_DATASET_NAME)

        #augmenter_cpu = iag.AugmenterCPU(g_conf.AUGMENTATION_SUITE_CPU)

        dataset = CoILDataset(full_dataset, transform=transforms.Compose([transforms.ToTensor()]))

        # Creates the sampler, this part is responsible for managing the keys. It divides
        # all keys depending on the measurements and produces a set of keys for each bach.
        sampler = BatchSequenceSampler(splitter.control_steer_split(dataset.measurements, dataset.meta_data),
                              g_conf.BATCH_SIZE, g_conf.NUMBER_IMAGES_SEQUENCE, g_conf.SEQUENCE_STRIDE)

        # The data loader is the multi threaded module from pytorch that release a number of
        # workers to get all the data.
        # TODO: batch size an number of workers go to some configuration file
        data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler,
                                                  shuffle=False, num_workers=12, pin_memory=False)
        # By instanciating the augmenter we get a callable that augment images and transform them
        # into tensors.
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
        # augmenter = iag.Augmenter(g_conf.AUGMENTATION_SUITE)
        # TODO: here there is clearly a posibility to make a cool "conditioning" system.

        model = CoILModel(g_conf.MODEL_NAME)
        model.cuda()
        print(model)

        criterion = Loss()

        # TODO: DATASET SIZE SEEMS WEIRD
        optimizer = optim.Adam(model.parameters(), lr=0.0002)


        checkpoint_file = get_latest_saved_checkpoint()
        if checkpoint_file != None:
            checkpoint = torch.load(os.path.join('_logs', exp_batch, exp_alias,
                                     'checkpoints', str(get_latest_saved_checkpoint())))
            iteration = checkpoint['iteration']
            accumulated_time = checkpoint['total_time']
            best_loss = checkpoint['best_loss']
            best_loss_iter = checkpoint['best_loss_iter']
        else:
            iteration = 0
            best_loss = 10000.0
            accumulated_time = 0  # We accumulate iteration time and keep the average speed
            best_loss_iter = 0

        # TODO: The checkpoint will continue, so it should erase everything up to the iteration

        best_loss_save = 10000.0
        best_loss_save_iter = 0
        curr_loss_save = 0.0

        print (dataset.meta_data)

        print (model)
        capture_time = time.time()
        model.train()
        for data in data_loader:


            input_data, float_data = data


            #TODO, ADD ITERATION SCHEDULE
            input_rgb_data = augmenter(0, input_data['rgb'])
            augment_for_controls = 1
            adjustlr = 1

            if augment_for_controls: #and self._config.targets_names[j] == "Steer":
                camera_angle = float_data[:,26,:]
                camera_angle = camera_angle.cuda()#self._config.variable_names.index('Angle'),i]
                print ("Camera angle", camera_angle[0])
                steer = float_data[:,0,:]
                # print("Original", steer[0])
                steer = steer.cuda()
                speed = float_data[:,10,:]
                speed = speed.cuda()
                # print (steer)

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
                # print(val)
                steer -= pos * torch.min(val , torch.tensor([0.6]).cuda())

                steer += neg * torch.min(val, torch.tensor([0.6]).cuda())

                print("val", val[0])
                print ("speed", speed[0])

                steer = steer.cpu()
                float_data[:,0,:] = steer

                float_data[:, 0, :][float_data[:,0,:] > 1.0] = 1.0
                float_data[:, 0, :][float_data[:,0,:] < -1.0] = -1.0
            #coil_logger.add_images(input_rgb_data)

            # get the control commands from float_data, size = [120,1]

            controls = float_data[:, dataset.controls_position(), :]
            # print(" CONTROLS  ", controls.shape)
            # The output(branches) is a list of 5 branches results, each branch is with size [120,3]

            model.zero_grad()
            # print ( 'INPUTS', dataset.extract_inputs(float_data).shape )
            branches = model(input_rgb_data, dataset.extract_inputs(float_data).cuda())

            #print ("len ",len(branches))



            #targets = torch.cat([steer_gt, gas_gt, brake_gt], 1)
            # print ("Extracted targets ", dataset.extract_targets(float_data).shape[0])
            loss = criterion.MSELoss(branches, dataset.extract_targets(float_data).cuda(),
                                     controls.cuda(), dataset.extract_inputs(float_data).cuda())

            # TODO: All these logging things could go out to clean up the main
            if loss.data < best_loss:
                best_loss = loss.data.tolist()
                best_loss_iter = iteration

            curr_loss_save += loss.data

            # Log a random position
            position = random.randint(0, len(float_data)-1)

            output = model.extract_branch(torch.stack(branches[0:4]), controls)
            error = torch.abs(output - dataset.extract_targets(float_data).cuda())



            # TODO: For now we are computing the error for just the correct branch, it could be multi- branch,

            coil_logger.add_scalar('Loss', loss.data, iteration)


            loss.backward()
            optimizer.step()

            accumulated_time += time.time() - capture_time
            capture_time = time.time()


            # TODO: Get only the  float_data that are actually generating output
            # TODO: itearation is repeating , and that is dumb
            coil_logger.add_message('Iterating',
                                    {'Iteration': iteration,
                                     'Loss': loss.data.tolist(),
                                     'Images/s': (iteration*g_conf.BATCH_SIZE)/accumulated_time,
                                     'BestLoss': best_loss, 'BestLossIteration': best_loss_iter,
                                     'BestLossSave': best_loss_save,
                                     'Output': output[position].data.tolist(),
                                     'GroundTruth': dataset.extract_targets(float_data)[position].data.tolist(),
                                     'Error': error[position].data.tolist(),
                                     'Inputs': dataset.extract_inputs(float_data)[position].data.tolist()},
                                    iteration)

            # TODO: For now we are computing the error for just the correct branch, it could be multi-branch,


            # TODO: save also the optimizer state dictionary
            if is_ready_to_save(iteration):

                state = {
                    'iteration': iteration,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'total_time': accumulated_time,
                    'best_loss_iter': best_loss_iter

                }
                # TODO : maybe already summarize the best model ???
                torch.save(state, os.path.join('_logs', exp_batch, exp_alias
                                               , 'checkpoints', str(iteration) + '.pth'))
            print ("before best save")
            if iteration % 5 == 0 and iteration > 4:
                curr_loss_save /= 5000.0
                if curr_loss_save < best_loss_save:

                    best_loss_save = curr_loss_save
                    curr_loss_save = 0
                    state = {
                        'iteration': iteration,
                        'state_dict': model.state_dict(),
                        'best_loss': best_loss_save,
                        'total_time': accumulated_time,
                        'best_loss_iter': best_loss_save_iter

                    }
                    # TODO : maybe already summarize the best model ???
                    torch.save(state, os.path.join('_logs', exp_batch, exp_alias
                                                , 'best_loss_save'+ '.pth'))
            print ("after best save")
            if iteration == best_loss_iter:

                state = {
                    'iteration': iteration,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'total_time': accumulated_time,
                    'best_loss_iter': best_loss_iter

                }
                # TODO : maybe already summarize the best model ???
                torch.save(state, os.path.join('_logs', exp_batch, exp_alias
                                               , 'best_loss'+ '.pth'))

            iteration += 1

            if adjustlr and iteration % 1000:
                adjust_learning_rate(optimizer, iteration)

    except KeyboardInterrupt:
        coil_logger.add_message('Error', {'Message': 'Killed By User'})

    except:
        traceback.print_exc()

        coil_logger.add_message('Error', {'Message': 'Something Happened'})
