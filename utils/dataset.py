import torch.utils.data as data
import os
from PIL import Image
import numpy as np


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def img_loader(path, num_channels):
    if num_channels == 1:
        img = Image.open(path)
    else:
        img = Image.open(path).convert('RGB')

    return img


# get the image list pairs
def get_imgs_list(dir_list, client, post_fix=None):
    img_list = []
    dir_list[0] = './data_for_train/{:s}/images/train'.format(client)
    dir_list[1] = './data_for_train/{:s}/labels_voronoi/train'.format(client)
    dir_list[2] = './data_for_train/{:s}/labels_cluster/train'.format(client)


    img_filename_list = [os.listdir(dir_list[i]) for i in range(len(dir_list))]

    for img in img_filename_list[0]:
        if not is_image_file(img):
            continue
        img1_name = os.path.splitext(img)[0]
        item = [os.path.join(dir_list[0], img),]
        for i in range(1, len(img_filename_list)):
            img_name = '{:s}{:s}'.format(img1_name, post_fix[i-1])
            if img_name in img_filename_list[i]:
                img_path = os.path.join(dir_list[i], img_name)
                item.append(img_path)

        if len(item) == len(dir_list):
            img_list.append(tuple(item))

    return img_list


class DataFolder(data.Dataset):
    def __init__(self, dir_list, post_fix, num_channels, client, data_transform=None, loader=img_loader):
        super(DataFolder, self).__init__()

        self.img_list = get_imgs_list(dir_list, client, post_fix)

        self.data_transform = data_transform
        self.num_channels = num_channels
        self.loader = loader

    def __getitem__(self, index):
        img_paths = self.img_list[index]
        sample = [self.loader(img_paths[i], self.num_channels[i]) for i in range(len(img_paths))]

        if self.data_transform is not None:
            sample = self.data_transform(sample)
        return sample

    def __len__(self):
        return len(self.img_list)

