import sys
import os
import time
import socket

import re
import math
import numpy as np
import copy
import random

#from sklearn import preprocessing

import scipy

from skimage import transform


#from carla.autopilot.autopilot import Autopilot
#from carla.autopilot.pilotconfiguration import ConfigAutopilot

from carla.agent import Agent
from PIL import Image


#TODO: The network is defined and toguether there is as forward pass operation to be used for testing, depending on the configuration

from network import CoILModel
from configs import g_conf
from logger import coil_logger
from torchvision import transforms
import imgauggpu as iag
import torch
import matplotlib.pyplot as plt


try:
    from carla import carla_server_pb2 as carla_protocol
except ImportError:
    raise RuntimeError(
        'cannot import "carla_server_pb2.py", run the protobuf compiler to generate this file')

number_of_seg_classes = 5
classes_join = {0: 2, 1: 2, 2: 2, 3: 2, 5: 2, 12: 2, 9: 2, 11: 2, 4: 0, 10: 1, 8: 3, 6: 3, 7: 4}

# TODO: implement this as a torch operation.
def join_classes(labels_image):
    compressed_labels_image = np.copy(labels_image)
    for key, value in classes_join.iteritems():
        compressed_labels_image[np.where(labels_image == key)] = value

    return compressed_labels_image


class CoILAgent(Agent):

    def __init__(self, checkpoint, architecture_name):



        #experiment_name='None', driver_conf=None, memory_fraction=0.18,
        #image_cut=[115, 510]):

        # use_planner=False,graph_file=None,map_file=None,augment_left_right=False,image_cut = [170,518]):

        Agent.__init__(self)
        # This should likely come from global
        #config_gpu = tf.ConfigProto()
        #config_gpu.gpu_options.visible_device_list = '0'

        #config_gpu.gpu_options.per_process_gpu_memory_fraction = memory_fraction
        #self._sess = tf.Session(config=config_gpu)

        # THIS DOES NOT WORK FOR FUSED PLUS LSTM
        #if self._config.number_frames_sequenced > self._config.number_frames_fused:
        #    self._config_train.batch_size = self._config.number_frames_sequenced
        #else:
        #    self._config_train.batch_size = self._config.number_frames_fused

        #self._train_manager = load_system(self._config_train)
        #self._config.train_segmentation = False
        self.architecture_name = architecture_name

        if architecture_name == 'coil_unit':
            self.model_task, self.model_gen = CoILModel('coil_unit')
            self.model_task, self.model_gen = self.model_task.cuda(), self.model_gen.cuda()
        elif architecture_name == 'unit_task_only':
            self.model_task, self.model_gen = CoILModel('unit_task_only')
            self.model_task, self.model_gen = self.model_task.cuda(), self.model_gen.cuda()
        else:
            self.model = CoILModel(architecture_name)
            self.model.cuda()

        if architecture_name == 'wgangp_lsd':
            # print(ckpt, checkpoint['best_loss_iter_F'])
            self.model.load_state_dict(checkpoint['stateF_dict'])
            self.model.eval()
        elif architecture_name == 'coil_unit':
            self.model_task.load_state_dict(checkpoint['task'])
            self.model_gen.load_state_dict(checkpoint['b'])
            self.model_task.eval()
            self.model_gen.eval()
        elif architecture_name == 'coil_icra':
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.eval()
        elif architecture_name == 'unit_task_only':
            self.model_task.load_state_dict(checkpoint['task_state_dict'])
            self.model_gen.load_state_dict(checkpoint['enc_state_dict'])
            self.model_task.eval()
            self.model_gen.eval()

        #self.model.load_network(checkpoint)

        #self._sess.run(tf.global_variables_initializer())

        #self._control_function = getattr(machine_output_functions,
        #                                 self._train_manager._config.control_mode)
        # More elegant way to merge with autopilot
        #self._agent = Autopilot(ConfigAutopilot(driver_conf.city_name))

        #self._image_cut = driver_conf.image_cut
        #self._auto_pilot = driver_conf.use_planner

        #self._recording = False
        #self._start_time = 0


    def run_step(self, measurements, sensor_data, directions, target):

        #control_agent = self._agent.run_step(measurements, None, target)
        print (" RUnning STEP ")
        speed = torch.cuda.FloatTensor([measurements.player_measurements.forward_speed]).unsqueeze(0)
        print("Speed is", speed)
        print ("Speed shape ", speed)
        directions_tensor = torch.cuda.LongTensor([directions])

        # model_outputs = self.model.forward_branch(self._process_sensors(sensor_data), speed,
        # 										  directions_tensor)
        if self.architecture_name == 'wgangp_lsd':
            embed, model_outputs = self.model(self._process_sensors(sensor_data), speed)

        elif self.architecture_name == 'coil_unit':
            embed, n_b = self.model_gen.encode(self._process_sensors(sensor_data))
            model_outputs = self.model_task(embed, speed)
        
        elif self.architecture_name == 'unit_task_only':
            embed, n_b = self.model_gen.encode(self._process_sensors(sensor_data))
            model_outputs = self.model_task(embed, speed)

        elif self.architecture_name == 'coil_icra':
            model_outputs = self.model.forward_branch(self._process_sensors(sensor_data), speed, directions_tensor)

        print (model_outputs)

        if self.architecture_name == 'coil_icra':
            steer, throttle, brake = self._process_model_outputs(model_outputs[0],
                                            measurements.player_measurements.forward_speed)
        else:
            steer, throttle, brake = self._process_model_outputs(model_outputs[0][0],
                                            measurements.player_measurements.forward_speed)

        control = carla_protocol.Control()
        control.steer = steer
        control.throttle = throttle
        control.brake = brake
        # if self._auto_pilot:
        #    control.steer = control_agent.steer
        # TODO: adapt the client side agent for the new version. ( PROBLEM )
        #control.throttle = control_agent.throttle
        #control.brake = control_agent.brake

        # TODO: maybe change to a more meaningfull message ??
        return control


    def _process_sensors(self, sensors):


        iteration = 0
        for name, size in g_conf.SENSORS.items():

            sensor = sensors[name].data[140: 260, ...] #300*800*3

            image_input = transform.resize(sensor, (128, 128))

            # transforms.Normalize([ 0.5315,  0.5521,  0.5205], [ 0.1960,  0.1810,  0.2217])

            image_input = np.transpose(image_input, (2, 0, 1))
            image_input = torch.from_numpy(image_input).type(torch.FloatTensor).cuda()
            image_input = image_input #normalization
            print ("torch size", image_input.size())

            img_np = np.uint8(np.transpose(image_input.cpu().numpy() * 255, (1 , 2, 0)))

            # plt.figure(1)
            # plt.subplot(1, 2, 1)
            # plt.imshow(sensor)
            #
            # plt.subplot(1,2,2)
            # plt.imshow(img_np)
            # #
            # plt.show()

            iteration += 1

        # print (image_input.shape)
        image_input  = image_input.unsqueeze(0)
        print (image_input.shape)

        return image_input


    def _process_model_outputs(self,outputs, speed):
        """
         A bit of heuristics in the control, to eventually make car faster, for instance.
        Returns:

        """

        print("OUTPUTS", outputs)
        steer, throttle, brake = outputs[0], outputs[1], outputs[2]
        # if steer > 0.5:
        # 	throttle *= (1 - steer + 0.3)
        # 	steer += 0.3
        # 	if steer > 1:
        # 		steer = 1
        # if steer < -0.5:
        # 	throttle *= (1 + steer + 0.3)
        # 	steer -= 0.3
        # 	if steer < -1:
        # 		steer = -1

        # if brake < 0.2:
        # 	brake = 0.0
        #
        if throttle > brake:
            brake = 0.0
        # else:
        # 	throttle = throttle * 2
        # if speed > 35.0 and brake == 0.0:
        # 	throttle = 0.0


        return steer, throttle, brake
