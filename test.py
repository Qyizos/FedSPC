# -*- coding: utf-8 -*-
import sys, os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import skimage.morphology as morph
import scipy.ndimage.morphology as ndi_morph
from skimage import measure
from scipy import misc
import imageio
import logging
import utils.utils as utils
from utils.accuracy import compute_metrics
import time
import matplotlib.pyplot as plt
from utils.my_transforms import get_transforms
from federated.configs import set_configs

def main(curTime=None, curmodel=ResUNet34_StyleAdvSG(), path='best_val_239',testSet='Spannuke'):
    args = set_configs()
    min_area = 20
    patch_size = 224
    overlap = 80
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)
    datsSource = testSet
    pathName = path + '.pth.tar'

    model_path = './experiments/' + curTime + '/round_best/' + pathName
    parent_path = os.path.dirname(model_path)
    log_path = os.path.join(parent_path, 'test_log.txt')
    logging.basicConfig(filename=log_path, level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    args.transform = dict()
    args.transform['train'] = {
        'random_resize': [0.8, 1.25],
        'horizontal_flip': True,
        'vertical_flip': True,
        'random_affine': 0.3,
        'random_rotation': 90,
        'random_crop': args.input_size,
        'label_encoding': 2,
        'to_tensor': 1
    }
    args.transform['test'] = {
        'to_tensor': 1
    }

    test_transform = get_transforms(args.transform['test'])

    model = curmodel
    model = model.cuda()
    cudnn.benchmark = True

    print("=> loading trained model")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded model at epoch {}".format(checkpoint['epoch']))

    model.eval()
    counter = 0
    print("=> Test begins:")

    curData = datsSource
    img_dir = './data_for_train/{:s}/images/test'.format(curData)
    testPath = curData.split('-')[0]
    label_dir = './data/{:s}/labels_instance'.format(testPath)
    save_dir = os.path.join(parent_path, 'test_results')

    logging.info("############ curData：{:s}, {:s}#############".format(curData, path))

    img_names = os.listdir(img_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    curSaveDir = '{:s}/{:s}'.format(save_dir, curData)
    if not os.path.exists(curSaveDir):
        os.mkdir(curSaveDir)
    strs = img_dir.split('/')
    prob_maps_folder = '{:s}/{:s}/{:s}_prob_maps'.format(save_dir, curData, strs[-1])
    seg_folder = '{:s}/{:s}/{:s}_segmentation'.format(save_dir, curData, strs[-1])
    if not os.path.exists(prob_maps_folder):
        os.mkdir(prob_maps_folder)
    if not os.path.exists(seg_folder):
        os.mkdir(seg_folder)

    metric_names = ['acc', 'p_F1', 'p_iou', 'dice', 'aji', 'dq', 'sq', 'pq']
    test_results = dict()
    all_result = utils.AverageMeter(len(metric_names))

    for img_name in img_names:
        print('=> Processing image {:s}'.format(img_name))
        img_path = '{:s}/{:s}'.format(img_dir, img_name)
        img = Image.open(img_path)

        ori_h = img.size[1]
        ori_w = img.size[0]
        name = os.path.splitext(img_name)[0]
        label_path = '{:s}/{:s}_label.png'.format(label_dir, name)
        gt = imageio.imread(label_path)
        input = test_transform((img,))[0].unsqueeze(0)

        start_time = time.time()

        print('\tComputing output probability maps...')
        prob_maps = get_probmaps(input, model, patch_size, overlap)
        pred = np.argmax(prob_maps, axis=0)

        pred_labeled = measure.label(pred)
        pred_labeled = morph.remove_small_objects(pred_labeled, min_area)
        pred_labeled = ndi_morph.binary_fill_holes(pred_labeled > 0)
        pred_labeled = measure.label(pred_labeled)

        end_time = time.time()

        inference_time = end_time - start_time
        logging.info(f"inference_time: {inference_time}")

        print('\tComputing metrics...')
        metrics = compute_metrics(pred_labeled, gt, metric_names)

        test_results[name] = [metrics['acc'], metrics['p_F1'], metrics['p_iou'], metrics['dice'], metrics['aji'], metrics['dq'], metrics['sq'], metrics['pq']]

        all_result.update([metrics['acc'], metrics['p_F1'], metrics['p_iou'], metrics['dice'], metrics['aji'], metrics['dq'], metrics['sq'], metrics['pq']])

        print('\tSaving image results...')

        plt.imsave('{:s}/{:s}_pred.png'.format(prob_maps_folder, name), pred.astype(np.uint8) * 255)
        plt.imsave('{:s}/{:s}_prob.png'.format(prob_maps_folder, name), prob_maps[1, :, :])

        final_pred = Image.fromarray(pred_labeled.astype(np.uint16))
        final_pred.save('{:s}/{:s}_seg.tiff'.format(seg_folder, name))

        pred_colored_instance = np.zeros((ori_h, ori_w, 3))
        for k in range(1, pred_labeled.max() + 1):
            pred_colored_instance[pred_labeled == k, :] = np.array(utils.get_random_color())
        filename = '{:s}/{:s}_seg_colored.png'.format(seg_folder, name)
        plt.imsave(filename, pred_colored_instance)

        counter += 1
        if counter % 10 == 0:
            print('\tProcessed {:d} images'.format(counter))

    logging.info('=> Processed all {:d} images'.format(counter))
    logging.info('Average Acc: {r[0]:.4f}\nF1: {r[1]:.4f}\nIoU: {r[2]:.4f}\nDice: {r[3]:.4f}\nAJI: {r[4]:.4f}\ndq: {r[5]:.4f}\nsq: {r[6]:.4f}\npq: {r[7]:.4f}\n'.format(r=all_result.avg))

    header = metric_names
    utils.save_results(header, all_result.avg, test_results, '{:s}/test_results.txt'.format(save_dir))

    return all_result.avg

def get_probmaps(input, model, patch_size, overlap):
    size = patch_size
    overlap = overlap

    if size == 0:
        with torch.no_grad():
            output , _ , _ = model(input.cuda())
    else:
        output = utils.split_forward(model, input, size, overlap)

    output = output.squeeze(0)
    prob_maps = F.softmax(output, dim=0).cpu().numpy()

    return prob_maps


if __name__ == '__main__':
    main()
