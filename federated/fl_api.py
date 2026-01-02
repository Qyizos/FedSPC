# -*- coding: utf-8 -*-
import copy, os, glob
import logging
import numpy as np
import pandas as pd
import torch
import math
import utils.utils as utils
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from utils.accuracy import compute_metrics
import skimage.morphology as morph
import scipy.ndimage.morphology as ndi_morph
from skimage import measure
from scipy import misc
from torch.optim.lr_scheduler import StepLR
from utils.loss import *
from utils.tool_func import *
from torch.distributions import Normal, Dirichlet
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering as Agg
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import SpectralClustering
import torch.fft
import pynvml


class FedAvg_StyleAdvPerturb_AdjIN3(object):
    def __init__(self, device, args, model_trainer):
        """
        dataset: data loaders and data size info
        """
        self.device = device
        self.args = args

        client_num = len(args.source)
        self.client_num_in_total = client_num
        self.client_num_per_round = int(self.client_num_in_total * self.args.percent)

        self.client_list = args.source
        self.model_trainer = model_trainer

    def train(self, datasets, model, optimizer, criterion, args, valSet):
        w_global = self.model_trainer.get_model_params()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        best_metrics = 0
        tb_writer = SummaryWriter('{:s}/tb_logs'.format(args.save_path))
        wm_locals = [w_global for _ in range(len(datasets))]
        sL1_loss = nn.SmoothL1Loss(reduction='mean')

        for round_idx in range(self.args.comm_round):

            logging.info("============ Communication round : {}".format(round_idx))
            w_locals = []
            sum_metricsList = [0] * len(datasets)
            curLoss = 0
            lossVar = 0
            lossCluster = 0
            lossConsist = 0

            metric_names = ['aji']
            val_result = utils.AverageMeter(len(metric_names))
            val_results = dict()

            for idx, CurData in enumerate(datasets):
                model.load_state_dict(copy.deepcopy(wm_locals[idx]))

                model.to(device)
                model.train()

                results = utils.AverageMeter(4)

                for curEpoch in range(self.args.iterEpochs):
                    for i, sample in enumerate(CurData):
                        input, target1, target2 = sample

                        if target1.dim() == 4:
                            target1 = target1.squeeze(1)
                        if target2.dim() == 4:
                            target2 = target2.squeeze(1)

                        target1 = target1.cuda()
                        target2 = target2.cuda()

                        input_var = input.cuda()

                        # -------------------------------------------------------------------------------------
                        model.block1.eval()
                        model.block2.eval()
                        model.block3.eval()
                        model.block4.eval()
                        model.decoder_G.eval()
                        model.decoder_L.train()
                        model, _, style_mean3, style_std3 = self.adversarial_attack_Incre(model, input_var, target1, target2, criterion,
                                                                 optimizer)
                        model.train()
                        output_L, output = model(input_var, style_mean3, style_std3)

                        # -------------------------------------------------------------------------------------

                        log_prob_maps_G = F.log_softmax(output, dim=1)
                        log_prob_maps_L = F.log_softmax(output_L, dim=1)
                        loss_vor_G = criterion(log_prob_maps_G, target1)
                        loss_cluster_G = criterion(log_prob_maps_G, target2)

                        loss_vor_L = criterion(log_prob_maps_L, target1)
                        loss_cluster_L = criterion(log_prob_maps_L, target2)

                        loss_consist = sL1_loss(log_prob_maps_L, log_prob_maps_G)

                        loss = loss_vor_G + loss_cluster_G + loss_vor_L + loss_cluster_L + loss_consist

                        result = [loss.item(), (loss_vor_G.item() + loss_vor_L.item()),
                                  (loss_cluster_G.item() + loss_cluster_L.item()), loss_consist.item()]

                        results.update(result, input.size(0))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        if i % args.log_interval == 0:
                            logging.info('\tIteration: [{:d}/{:d}]'
                                         '\tLoss {r[0]:.4f}'
                                         '\tLoss_vor {r[1]:.4f}'
                                         '\tLoss_cluster {r[2]:.4f}'
                                         '\tLoss_consist {r[3]:.4f}'.format(i, len(CurData), r=results.avg))

                    logging.info(
                        '\t---------------------------------' + args.source[idx] + '---------------------------------')

                    logging.info('\t=> Train Avg: Loss {r[0]:.4f}'
                                 '\tloss_vor {r[1]:.4f}'
                                 '\tloss_cluster {r[2]:.4f}'
                                 '\tloss_consist {r[3]:.4f}'.format(r=results.avg))

                    curLoss = curLoss + results.avg[0]
                    lossVar = curLoss + results.avg[1]
                    lossCluster = curLoss + results.avg[2]
                    lossConsist = curLoss + results.avg[3]

                model_copy = copy.deepcopy(model)
                model_copy.decoder_G.load_state_dict(model_copy.decoder_L.state_dict())
                w = model_copy.cpu().state_dict()

                w_locals.append((len(CurData), copy.deepcopy(w)))

            curLoss = curLoss / len(datasets)
            lossVar = lossVar / len(datasets)
            lossCluster = lossCluster / len(datasets)
            lossConsist = lossConsist / len(datasets)

            w_global = self._aggregate(w_locals)
            model.load_state_dict(copy.deepcopy(w_global))


            for wm_idx in range(len(wm_locals)):
                model_copy = copy.deepcopy(model)
                model_copy.load_state_dict(copy.deepcopy(w_locals[wm_idx][-1]))
                wm_locals[wm_idx] = copy.deepcopy(model.cpu().state_dict())

            if (round_idx + 1) > 100:
                model = model.cuda()

                for wm_idx in range(len(datasets)):
                    model.load_state_dict(copy.deepcopy(wm_locals[wm_idx]))

                    model.eval()
                    min_area = 20
                    patch_size = 224
                    overlap = 80

                    for valIdx, curVal in enumerate(valSet[wm_idx]):
                        input, gt = curVal

                        prob_maps = self.get_probmaps(input, model, patch_size, overlap)
                        pred = np.argmax(prob_maps, axis=0)

                        pred_labeled = measure.label(pred)
                        pred_labeled = morph.remove_small_objects(pred_labeled, min_area)
                        pred_labeled = ndi_morph.binary_fill_holes(pred_labeled > 0)
                        pred_labeled = measure.label(pred_labeled)

                        metrics = compute_metrics(pred_labeled, gt, metric_names)
                        sum_metricsList[wm_idx] = sum_metricsList[wm_idx] + metrics['aji']

                        val_results[valIdx] = [metrics['aji']]
                        val_result.update([metrics['aji']])

                    sum_metricsList[wm_idx] = sum_metricsList[wm_idx] / len(valSet[wm_idx])
                    logging.info(self.client_list[wm_idx] + ': ' + str(sum_metricsList[wm_idx]))

            sum_metrics = sum(sum_metricsList) / len(sum_metricsList)
            if sum_metrics != 0:
                if sum_metrics > best_metrics:
                    best_metrics = sum_metrics

                    dirName = 'round_' + str(round_idx + 1)
                    save_mode_path_0 = os.path.join(self.args.save_path, dirName)
                    if not os.path.exists(save_mode_path_0):
                        os.mkdir(save_mode_path_0)

                    for wm_idx in range(len(datasets)):
                        saveName = 'client_' + str(wm_idx + 1) + '.pth.tar'
                        save_mode_path = os.path.join(save_mode_path_0, saveName)
                        state = {
                            'epoch': round_idx + 1,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                        }
                        torch.save(state, save_mode_path)

                    logging.info(('best_val——————Average Acc: {}\n'.format(sum_metrics)))
                else:
                    logging.info(('curVal——————Average Acc: {}\n'.format(sum_metrics)))

                # ----------------------------------------------------------------------------------------------------------------------

            tb_writer.add_scalars('epoch_losses',
                                  {'train_loss': curLoss, 'train_loss_vor': lossVar,
                                   'train_loss_cluster': lossCluster,
                                   'train_loss_consisten': lossConsist}, round_idx + 1)
        tb_writer.close()
        print('训练阶段完成！')

    def adversarial_attack_Incre(self, model, x_ori, target1, target2, criterion, optimizer):
        epsilon_list = [0.8, 0.08, 0.008]

        adv_style_mean_block1, adv_style_std_block1 = 'None', 'None'
        adv_style_mean_block2, adv_style_std_block2 = 'None', 'None'
        adv_style_mean_block3, adv_style_std_block3 = 'None', 'None'

        blocklist = 'block123r'

        if ('1' in blocklist and epsilon_list[0] != 0):
            x_ori_block1, c1 = model.forward_block1(x_ori)
            feat_size_block1 = x_ori_block1.size()
            ori_style_mean_block1, ori_style_std_block1 = calc_mean_std(x_ori_block1)

            ori_style_mean_block1 = torch.nn.Parameter(ori_style_mean_block1)
            ori_style_std_block1 = torch.nn.Parameter(ori_style_std_block1)
            ori_style_mean_block1.requires_grad_()
            ori_style_std_block1.requires_grad_()

            x_normalized_block1 = (x_ori_block1 - ori_style_mean_block1.detach().expand(
                feat_size_block1)) / ori_style_std_block1.detach().expand(feat_size_block1)
            x_ori_block1 = x_normalized_block1 * ori_style_std_block1.expand(
                feat_size_block1) + ori_style_mean_block1.expand(feat_size_block1)

            x_ori_block2, c1, c2 = model.forward_block2(x_ori_block1, c1)
            x_ori_block3, c1, c2, c3 = model.forward_block3(x_ori_block2, c1, c2)
            x_ori_block4, c1, c2, c3, c4 = model.forward_block4(x_ori_block3, c1, c2, c3)
            x_ori_block5, c1, c2, c3, c4 = model.forward_block5(x_ori_block4, c1, c2, c3, c4)
            x_ori_output_G = model.forward_rest_G(x_ori_block5, c1, c2, c3, c4)
            x_ori_output_L = model.forward_rest_L(x_ori_block5, c1, c2, c3, c4)

            log_prob_maps = F.log_softmax(x_ori_output_L, dim=1)
            loss_vor = criterion(log_prob_maps, target1)
            loss_cluster = criterion(log_prob_maps, target2)

            ori_loss = loss_vor + loss_cluster

            optimizer.zero_grad()
            ori_loss.backward()
            optimizer.step()

            grad_ori_style_mean_block1 = ori_style_mean_block1.grad.detach()
            grad_ori_style_std_block1 = ori_style_std_block1.grad.detach()

            index = torch.randint(0, len(epsilon_list), (1,))[0]
            epsilon = epsilon_list[index]

            adv_style_mean_block1 = fgsm_attack_addPerturb(ori_style_mean_block1, epsilon, grad_ori_style_mean_block1)
            adv_style_std_block1 = fgsm_attack_addPerturb(ori_style_std_block1, epsilon, grad_ori_style_std_block1)

        if ('2' in blocklist and epsilon_list[1] != 0):
            x_ori_block1, c1 = model.forward_block1(x_ori)
            x_adv_block1 = changeNewAdvStyle(x_ori_block1, adv_style_mean_block1, adv_style_std_block1,
                                             p_thred=0)

            x_ori_block2, c1, c2 = model.forward_block2(x_adv_block1, c1)
            feat_size_block2 = x_ori_block2.size()
            ori_style_mean_block2, ori_style_std_block2 = calc_mean_std(x_ori_block2)

            ori_style_mean_block2 = torch.nn.Parameter(ori_style_mean_block2)
            ori_style_std_block2 = torch.nn.Parameter(ori_style_std_block2)
            ori_style_mean_block2.requires_grad_()
            ori_style_std_block2.requires_grad_()

            x_normalized_block2 = (x_ori_block2 - ori_style_mean_block2.detach().expand(
                feat_size_block2)) / ori_style_std_block2.detach().expand(feat_size_block2)
            x_ori_block2 = x_normalized_block2 * ori_style_std_block2.expand(
                feat_size_block2) + ori_style_mean_block2.expand(feat_size_block2)

            x_ori_block3, c1, c2, c3 = model.forward_block3(x_ori_block2, c1, c2)
            x_ori_block4, c1, c2, c3, c4 = model.forward_block4(x_ori_block3, c1, c2, c3)
            x_ori_block5, c1, c2, c3, c4 = model.forward_block5(x_ori_block4, c1, c2, c3, c4)
            x_ori_output_G = model.forward_rest_G(x_ori_block5, c1, c2, c3, c4)
            x_ori_output_L = model.forward_rest_L(x_ori_block5, c1, c2, c3, c4)

            log_prob_maps = F.log_softmax(x_ori_output_L, dim=1)
            loss_vor = criterion(log_prob_maps, target1)
            loss_cluster = criterion(log_prob_maps, target2)

            ori_loss = loss_vor + loss_cluster

            optimizer.zero_grad()
            ori_loss.backward()
            optimizer.step()

            grad_ori_style_mean_block2 = ori_style_mean_block2.grad.detach()
            grad_ori_style_std_block2 = ori_style_std_block2.grad.detach()

            index = torch.randint(0, len(epsilon_list), (1,))[0]
            epsilon = epsilon_list[index]
            adv_style_mean_block2 = fgsm_attack_addPerturb(ori_style_mean_block2, epsilon, grad_ori_style_mean_block2)
            adv_style_std_block2 = fgsm_attack_addPerturb(ori_style_std_block2, epsilon, grad_ori_style_std_block2)

        if ('3' in blocklist and epsilon_list[2] != 0):
            x_ori_block1, c1 = model.forward_block1(x_ori)
            x_adv_block1 = changeNewAdvStyle(x_ori_block1, adv_style_mean_block1, adv_style_std_block1, p_thred=0)
            x_ori_block2, c1, c2 = model.forward_block2(x_adv_block1, c1)
            x_adv_block2 = changeNewAdvStyle(x_ori_block2, adv_style_mean_block2, adv_style_std_block2, p_thred=0)
            x_ori_block3, c1, c2, c3 = model.forward_block3(x_adv_block2, c1, c2)

            feat_size_block3 = x_ori_block3.size()
            ori_style_mean_block3, ori_style_std_block3 = calc_mean_std(x_ori_block3)

            ori_style_mean_block3 = torch.nn.Parameter(ori_style_mean_block3)
            ori_style_std_block3 = torch.nn.Parameter(ori_style_std_block3)
            ori_style_mean_block3.requires_grad_()
            ori_style_std_block3.requires_grad_()

            x_normalized_block3 = (x_ori_block3 - ori_style_mean_block3.detach().expand(
                feat_size_block3)) / ori_style_std_block3.detach().expand(feat_size_block3)
            x_ori_block3 = x_normalized_block3 * ori_style_std_block3.expand(
                feat_size_block3) + ori_style_mean_block3.expand(feat_size_block3)

            x_ori_block4, c1, c2, c3, c4 = model.forward_block4(x_ori_block3, c1, c2, c3)
            x_ori_block5, c1, c2, c3, c4 = model.forward_block5(x_ori_block4, c1, c2, c3, c4)
            x_ori_output_G = model.forward_rest_G(x_ori_block5, c1, c2, c3, c4)
            x_ori_output_L = model.forward_rest_L(x_ori_block5, c1, c2, c3, c4)

            log_prob_maps = F.log_softmax(x_ori_output_L, dim=1)
            loss_vor = criterion(log_prob_maps, target1)
            loss_cluster = criterion(log_prob_maps, target2)

            ori_loss = loss_vor + loss_cluster

            optimizer.zero_grad()
            ori_loss.backward()
            optimizer.step()

            grad_ori_style_mean_block3 = ori_style_mean_block3.grad.detach()
            grad_ori_style_std_block3 = ori_style_std_block3.grad.detach()

            index = torch.randint(0, len(epsilon_list), (1,))[0]
            epsilon = epsilon_list[index]
            adv_style_mean_block3 = fgsm_attack_addPerturb(ori_style_mean_block3, epsilon, grad_ori_style_mean_block3)
            adv_style_std_block3 = fgsm_attack_addPerturb(ori_style_std_block3, epsilon, grad_ori_style_std_block3)

        if ('r' in blocklist and epsilon_list[2] != 0):
            x_ori_block1, c1 = model.forward_block1(x_ori)
            x_adv_block1 = changeNewAdvStyle(x_ori_block1, adv_style_mean_block1, adv_style_std_block1, p_thred=0)
            x_ori_block2, c1, c2 = model.forward_block2(x_adv_block1, c1)
            x_adv_block2 = changeNewAdvStyle(x_ori_block2, adv_style_mean_block2, adv_style_std_block2, p_thred=0)
            x_ori_block3, c1, c2, c3 = model.forward_block3(x_adv_block2, c1, c2)
            x_adv_block3 = changeNewAdvStyle(x_ori_block3, adv_style_mean_block3, adv_style_std_block3, p_thred=0)

            ori_style_mean_block3, ori_style_std_block3 = calc_mean_std(x_adv_block3)

            x_ori_block4, c1, c2, c3, c4 = model.forward_block4(x_adv_block3, c1, c2, c3)

            x_ori_block5, c1, c2, c3, c4 = model.forward_block5(x_ori_block4, c1, c2, c3, c4)
            x_ori_output_G = model.forward_rest_G(x_ori_block5, c1, c2, c3, c4)
            x_ori_output_L = model.forward_rest_L(x_ori_block5, c1, c2, c3, c4)

            log_prob_maps = F.log_softmax(x_ori_output_L, dim=1)
            loss_vor = criterion(log_prob_maps, target1)
            loss_cluster = criterion(log_prob_maps, target2)

            ori_loss = loss_vor + loss_cluster

            optimizer.zero_grad()
            ori_loss.backward()
            optimizer.step()

        return model, x_ori_output_L, ori_style_mean_block3, ori_style_std_block3

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def get_probmaps(self, input, model, patch_size, overlap):
        size = patch_size
        overlap = overlap

        if size == 0:
            with torch.no_grad():
                output, _, _ = model(input.cuda())
        else:
            output = utils.split_forward_o2(model, input, size, overlap)
        output = output.squeeze(0)
        prob_maps = F.softmax(output, dim=0).cpu().numpy()

        return prob_maps