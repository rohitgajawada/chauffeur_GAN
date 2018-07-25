import traceback
import sys
import logging
import json
import datetime
import numpy as np
import os
import time
import subprocess
import socket

import torch
from contextlib import closing

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from carla.tcp import TCPConnectionError
from carla.client import make_carla_client
from carla.driving_benchmark import run_driving_benchmark

from drive import CoILAgent
from drive import ECCVTrainingSuite
from drive import ECCVGeneralizationSuite
from testing.unit_tests.test_drive.test_suite import TestSuite
from logger import coil_logger
from logger import monitorer

from carla.settings import CarlaSettings

from configs import g_conf, merge_with_yaml, set_type_of_process

from utils.checkpoint_schedule import  maximun_checkpoint_reach, get_next_checkpoint, is_next_checkpoint_ready

def frame2numpy(frame, frameSize):
    return np.resize(np.fromstring(frame, dtype='uint8'), (frameSize[1], frameSize[0], 3))


def find_free_port():

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

def start_carla_simulator(gpu, exp_batch, exp_alias, city_name):

    port = find_free_port()
    carla_path = os.environ['CARLA_PATH']

#'-DisableCentralRoadlines','-DisableBorderRoadlines'
    sp = subprocess.Popen([carla_path + '/CarlaUE4/Binaries/Linux/CarlaUE4', '/Game/Maps/' + city_name,'-windowed', '-benchmark', '-DisableCentralRoadlines','-DisableBorderRoadlines','-fps=10', '-world-port='+str(port)], shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return sp, port

#
#'

def execute(gpu, exp_batch, exp_alias, city_name='Town01', memory_use=0.2, host='127.0.0.1'):
    # host,port,gpu_number,path,show_screen,resolution,noise_type,config_path,type_of_driver,experiment_name,city_name,game,drivers_name
    #drive_config.city_name = city_name
    # TODO Eliminate drive config.

    print("Running ", __file__, " On GPU ", gpu, "of experiment name ", exp_alias)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    sys.stdout = open(str(os.getpid()) + ".out", "a", buffering=1)

    carla_process, port = start_carla_simulator(gpu, exp_batch, exp_alias, city_name)
    merge_with_yaml(os.path.join(exp_batch, exp_alias+'.yaml'))
    set_type_of_process('test')
    experiment_suite = TestSuite()

    while True:
        try:
            with make_carla_client(host, port) as client:

                checkpoint = torch.load(os.path.join('./best_loss.pth'))
                coil_agent = CoILAgent(checkpoint)
                run_driving_benchmark(coil_agent, experiment_suite, city_name,
                                      exp_batch + '_' + exp_alias + 'iteration', False,
                                      host, port)

                break

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)
            carla_process.kill()

        except KeyboardInterrupt:
            carla_process.kill()
        except:
            traceback.print_exc()
            carla_process.kill()

    carla_process.kill()

execute("0", "eccv", "experiment_1", 'SantQuirze')
