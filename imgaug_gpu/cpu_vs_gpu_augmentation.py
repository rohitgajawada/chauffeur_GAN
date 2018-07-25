import os
import time


import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt



def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def prepare_data(data_dir):

    # Check if the dataset is already there

    if not os.path.exists(data_dir):
        raise ValueError("Directory does not exist.")


    # Join the data in a single folder
    if os.path.exists(os.path.join(data_dir, 'train')):
        os.mkdir(os.path.join(data_dir, 'images'))
        train_folder_name = os.path.join(data_dir, 'train')
        files_train = listdir_fullpath(os.path.join(train_folder_name, 'bees')) + listdir_fullpath(os.path.join(train_folder_name, 'ants'))
        for file in files_train:
            shutil.move(file, os.path.join(data_dir, 'images'))

        shutil.rmtree(train_folder_name)

        val_folder_name = os.path.join(data_dir, 'val')
        files_val = listdir_fullpath(os.path.join(val_folder_name, 'bees')) + listdir_fullpath(os.path.join(val_folder_name, 'ants'))
        for file in files_val:
            shutil.move(file, os.path.join(data_dir, 'images'))

        shutil.rmtree(val_folder_name)



class ToGPU(object):

    def __call__(self, img):
        return img.cuda()



class Multiply(object):

    def __init__(self, multiply):
        self.multiply = multiply

    def __call__(self, tensor):
        array = torch.cuda.FloatTensor([self.multiply]).expand_as(tensor)
        return tensor * array

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class MultiplyCPU(object):

    def __init__(self, multiply):
        self.multiply = multiply

    def __call__(self, tensor):
        array = torch.FloatTensor([self.multiply]).expand_as(tensor)
        return tensor * array


    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



if __name__ == '__main__':


    # We define parameters for the classes
    batch_size = 12
    num_workers = 12
    data_path = 'hymenoptera_data'
    # Number of repetitions to comp
    repetitions = 3

    no_aug_trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = datasets.ImageFolder(data_path, transform=no_aug_trans)
    # We input this image folder dataset in a dataloader
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers,
                                              pin_memory=True)

    # We load the data with no augmentations in the begining just to cache some data.
    for data in data_loader:
        image, labels = data


    gpu_time_vec = []
    cpu_time_vec = []
    for n_mult in range(5, 60, 5):

        multiply_gpu = transforms.Compose([ToGPU()] + [Multiply(1.01)] * n_mult)
        gpu_time = 0
        for rep in range(0, 3):
            capture_time = time.time()
            for data in data_loader:
                image, labels = data
                result = multiply_gpu(image)
            gpu_time += time.time() - capture_time

        last_gpu_image = result[0, ...]
        gpu_time_vec.append(gpu_time/3.0)
        print("Gpu Time =  ", gpu_time_vec[-1])


        aug_trans = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor()] + [MultiplyCPU(1.01)] * n_mult)

        dataset_aug = datasets.ImageFolder(data_path, transform=aug_trans)
        data_loader_aug = torch.utils.data.DataLoader(dataset_aug, batch_size=batch_size,
                                                      shuffle=False, num_workers=num_workers,
                                                      pin_memory=True)
        cpu_time = 0
        for rep in range(0, 3):
            capture_time = time.time()
            for data in data_loader_aug:
                image, labels = data
            cpu_time += time.time() - capture_time

        last_cpu_image = image[0, ...]

        #print(last_cpu_image-last_gpu_image.cpu())
        cpu_time_vec.append(cpu_time/3.0)
        print("CPU Time =  ", cpu_time_vec[-1])


    plt.plot(range(5, 60, 5), gpu_time_vec, range(5, 60, 5), cpu_time_vec)
    plt.show()



