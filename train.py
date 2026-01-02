# -*- coding: utf-8 -*-
import sys, os
import time
import shutil
from torch.utils import data
from torch.utils.data import dataset

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from federated.configs import set_configs

from federated.fl_api import *
from federated.model_trainer_segmentation import ModelTrainerSegmentation

from utils.dataset import DataFolder
from utils.my_transforms import get_transforms
from utils.readVal import getVal
from torch.utils.data import DataLoader
from nets.model import ResUNet34_StyleAdvSG
import test

def rename_and_cleanup_round_dirs(base_path):
    round_dirs = []

    for d in os.listdir(base_path):
        full_path = os.path.join(base_path, d)

        if os.path.isdir(full_path) and d.startswith('round_'):
            round_dirs.append(d)

    round_numbers = []
    for d in round_dirs:
        round_numbers.append(int(d.split('_')[-1]))

    round_numbers.sort(reverse=True)
    max_round_dir = 'round_' + str(round_numbers[0])

    max_round_path = os.path.join(base_path, max_round_dir)
    best_round_path = os.path.join(base_path, 'round_best')

    os.rename(max_round_path, best_round_path)
    print(f"已将 {max_round_dir} 重命名为 round_best")

    round_numbers = round_numbers[1:]

    for _, d in enumerate(round_numbers):
        dir_path = os.path.join(base_path, 'round_' + str(d))
        shutil.rmtree(dir_path)
        print("已删除目录: " + 'round_' + str(d))

def count_tar_files(directory):
    all_files = os.listdir(directory)  # 列出目录中的所有文件和子目录
    tar_files = [f for f in all_files if f.endswith('.tar')]  # 过滤出所有以 .tar 结尾的文件
    return len(tar_files)

def deterministic(seed):
    cudnn.benchmark = False
    cudnn.deterministic = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def set_paths(args):
    args.save_path = './experiments/{}'.format(args.sonName)
    exp_folder = '{}'.format(args.mode)

    if args.balance:
        exp_folder = exp_folder + '_balanced'

    print(exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)


def custom_model_trainer(args, model=ResUNet34(pretrained=True)):
    model_trainer = ModelTrainerSegmentation(model, args)
    return model_trainer


def custom_federated_api(args, model_trainer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    federated_api = FedAvg_StyleAdvPerturb_AdjIN3(device, args, model_trainer)
    return federated_api


def main(curTime=None, curmodel=ResUNet34(pretrained=True), notes=None):
    args = set_configs()
    args.generalize = False
    args.source = ['Sconsep', 'Spannuke']
    args.notes = notes
    args.transform = dict()
    args.transform['train'] = {
        'random_resize': [0.8, 1.25],
        'horizontal_flip': True,
        'vertical_flip': True,
        'random_affine': 0.3,
        'random_rotation': 90,
        'random_crop': args.input_size,
        'label_encoding': 2,
        'to_tensor': 1  #
    }
    args.transform['test'] = {
        'to_tensor': 1
    }

    deterministic(args.seed)
    if curTime:
        args.sonName = curTime
    else:
        args.sonName = time.strftime('%y%m%d-%H%M', time.localtime())
    set_paths(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model_trainer = custom_model_trainer(args, curmodel)

    model = curmodel
    model = model.cuda()

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    args.num_params = num_params

    cudnn.benchmark = True

    log_path = os.path.join(args.save_path, 'log.txt')
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    optimizer = torch.optim.Adam(model.parameters(), 0.0001, betas=(0.9, 0.99),
                                 weight_decay=1e-4)
    criterion = torch.nn.NLLLoss(ignore_index=2).cuda()

    data_transforms = {'train': get_transforms(args.transform['train']),
                       'test': get_transforms(args.transform['test'])}

    dir_list = ['images', 'labels_cluster', 'labels_voronoi']
    post_fix = ['_label_vor.png', '_label_cluster.png']
    num_channels = [3, 3, 3]
    datasets = []
    for client in args.source:
        train_set = DataFolder(dir_list, post_fix, num_channels, client, data_transforms['train'])
        train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=1)
        datasets.append(train_loader)

    valSets = []
    for curClient in args.source:
        valSet = getVal(client=curClient, type='val')
        valSets.append(valSet)

    federated_manager = custom_federated_api(args, model_trainer)
    federated_manager.train(datasets, model, optimizer, criterion, args, valSets)

    rename_and_cleanup_round_dirs(args.save_path)

    pthPath = args.save_path + '/' + 'round_best'
    pthNums = count_tar_files(pthPath)
    ndarray_list = []

    for pid in range(pthNums):
        pthName = 'client_' + str(pid + 1)
        curResult = test.main(curTime, curmodel, path=pthName, testSet=args.source[pid])
        ndarray_list.append(curResult)

    avg_list = [float() for _ in range(len(ndarray_list[0]))]

    for curList in ndarray_list:
        for idx in range(len(curList)):
            avg_list[idx] = avg_list[idx] + curList[idx]

    for idx in range(len(avg_list)):
        avg_list[idx] = avg_list[idx] / len(ndarray_list)

    logging.info('Average Acc: {r[0]:.4f}\nF1: {r[1]:.4f}\nIoU: {r[2]:.4f}\nDice: {r[3]:.4f}\nAJI: {r[4]:.4f}\n'.format(r=avg_list))


if __name__ == "__main__":

    curTime = time.strftime('%y%m%d-%H%M', time.localtime())  # 借助时间给log 命名

    notes = 'model = ResUNet34_StyleAdvSG()；'
    name = notes.split('model = ')[-1].split('()')[0]

    curTime = curTime + '_' + name

    main(curTime, ResUNet34_StyleAdvSG(), notes)
