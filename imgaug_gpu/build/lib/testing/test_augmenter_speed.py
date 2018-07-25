import os
import numpy as np
import unittest


import time


import torch


from .coil_dataset import CoILDataset

from torchvision import transforms
from .augmenter_compositions import *

class testAugmenter(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(testAugmenter, self).__init__(*args, **kwargs)
        self.root_test_dir = 'testing/data'
        self.test_images_write_path = 'testing/_test_images_'


    def get_data_loader(self):

        # TODO: FIND A solution for this TO TENSOR
        dataset = CoILDataset(self.root_test_dir, transform=transforms.Compose([transforms.ToTensor()]))
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=120,
                                                  shuffle=True, num_workers=12, pin_memory=True)

        return data_loader

    def test_gpu_vs_cpu_speed_all(self):


        dataset_cpu = CoILDataset(self.root_test_dir, transform=benchmarking_cpu)
        data_loader_cpu = torch.utils.data.DataLoader(dataset_cpu, batch_size=120,
                                                  shuffle=True, num_workers=12, pin_memory=True)

        count = 0
        capture_time = time.time()
        for data in data_loader_cpu:
            image, labels = data

            result = image['rgb']

            count += 1
        print ("CPU Time =  ", time.time() - capture_time)
        count = 0
        capture_time = time.time()
        for data in data_loader_cpu:
            image, labels = data

            result = image['rgb']

            count += 1
        print ("CPU Time =  ", time.time() - capture_time)

        data_loader = self.get_data_loader()

        count = 0
        capture_time = time.time()
        for data in data_loader:
            image, labels = data

            result = benchmarking_gpu(0, image['rgb'])

            count += 1

        print("Gpu Time =  ", time.time() - capture_time)


    def test_gpu_vs_cpu_speed_dropout(self):


        dataset_cpu = CoILDataset(self.root_test_dir, transform=dropout_random_cpu)
        data_loader_cpu = torch.utils.data.DataLoader(dataset_cpu, batch_size=120,
                                                  shuffle=True, num_workers=12, pin_memory=True)

        count = 0
        capture_time = time.time()
        for data in data_loader_cpu:
            image, labels = data

            result = image['rgb']

            count += 1
        print ("CPU Time =  ", time.time() - capture_time)

        capture_time = time.time()
        for data in data_loader_cpu:
            image, labels = data

            result = image['rgb']

            count += 1
        print ("CPU Time =  ", time.time() - capture_time)

        data_loader = self.get_data_loader()

        count = 0
        capture_time = time.time()
        for data in data_loader:
            image, labels = data

            result = dropout_random(0, image['rgb'])

            count += 1

        print("Gpu Time =  ", time.time() - capture_time)

