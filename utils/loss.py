
"""
Loss for brain segmentaion (not used)
"""
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)


from torch.autograd import Function
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from medpy import metric
import numpy as np

def KL_divergence(mu1, logvar1, mu2=None, logvar2=None, eps=1e-8):
    " KLD(p1 || p2)"
    if mu2 is None:
        mu2 = mu1.new_zeros(mu1.shape)  # prior
        logvar2 = torch.log(mu1.new_ones(mu1.shape))
        eps = 0
    var1 = logvar1.exp()
    var2 = logvar2.exp()  # default : 1
    #     KLD = 0.5*torch.mean(torch.sum(-1  + logvar2 - logvar1 + (var1 + (mu1 - mu2).pow(2)) / (var2 + eps), axis=1))
    KLD = 0.5 * torch.mean(-1 + logvar2 - logvar1 + (var1 + (mu1 - mu2).pow(2)) / (var2 + eps))

    return KLD

class DiffLoss2(nn.Module):

    def __init__(self):
        super(DiffLoss2, self).__init__()

    def forward(self, input1, input2):
        eps = 1e-6
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + eps)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + eps)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

class KL_Loss(nn.Module):
    def __init__(self):
        super(KL_Loss, self).__init__()

    def forward(self, features_t, features_s):
        kl_loss = F.kl_div(F.log_softmax(features_s, dim=1), F.softmax(features_t, dim=1), reduction='mean')
        return kl_loss


def entropy_loss(p, c=3):
    # p N*C*W*H*D
     p = F.softmax(p, dim=1)
     y1 = -1 * torch.sum(p * torch.log(p + 1e-6), dim=0) / torch.tensor(np.log(c)).cuda()
     ent = torch.mean(y1)
     return ent


def js_kl(source, target):
    '''
    Calculate the JS/KL divergence and obtain the conditional distribution using joint distribution and marginal distribution
    '''

    # Use weight redistribution to calculate the joint distribution of source and target domains, avoiding the inability to calculate due to inconsistent quantities
    def compute_joint_distribution(source_samples, target_samples, num_bins=10):
        n_source = source_samples.shape[0]
        n_target = target_samples.shape[0]

        # Compute normalized weights for source and target samples
        source_weights = torch.ones(n_source) / n_source
        target_weights = torch.ones(n_target) / n_target

        # Concatenate source and target samples
        joint_samples = torch.cat((source_samples, target_samples))
        joint_weights = torch.cat((source_weights, target_weights))

        # Compute joint distribution of source and target samples
        joint_samples_np = joint_samples.cpu().detach().numpy()
        joint_weights_np = joint_weights.cpu().detach().numpy()
        bin_edges = [np.linspace(np.min(joint_samples_np[:, 0]), np.max(joint_samples_np[:, 0]), num_bins + 1),
                     np.linspace(np.min(joint_samples_np[:, 1]), np.max(joint_samples_np[:, 1]), num_bins + 1)]
        joint_distribution, _ = np.histogramdd(joint_samples_np, bins=bin_edges, weights=joint_weights_np)
        joint_distribution /= np.sum(joint_distribution)
        joint_distribution = torch.tensor(joint_distribution).float()

        return joint_distribution

    def compute_marginal_distribution(samples, num_bins=10):
        samples_np = samples.detach().cpu()
        print("device", samples_np.device)  # cpu
        samples_np = samples_np.numpy()

        marginal_distribution, _ = np.histogramdd(samples_np, bins=num_bins,
                                                  range=[(np.min(samples_np[:, 0]), np.max(samples_np[:, 0])),
                                                         (np.min(samples_np[:, 1]), np.max(samples_np[:, 1]))])

        marginal_distribution /= len(samples_np)

        marginal_distribution = torch.tensor(marginal_distribution).float()

        return marginal_distribution

    def compute_kl_divergence(p_distribution, q_distribution):
        kl_divergence = F.kl_div(q_distribution.log(), p_distribution)
        # print("kl_divergence",kl_divergence)
        #
        # kl_divergence = torch.sum(p_distribution * torch.log(p_distribution / q_distribution))

        return kl_divergence

    # If the softmax function has been used to convert values between 0-1, there is no need to convert them again.
    source = F.softmax(source, dim=1)
    target = F.softmax(target, dim=1)

    source_distribution = compute_marginal_distribution(source)

    target_distribution = compute_marginal_distribution(target)

    joint_distribution = compute_joint_distribution(source, target)

    # Compute JS divergence using KL divergence
    M = 0.5 * (joint_distribution + source_distribution[:, None] + target_distribution[None, :])

    # Here can obtain two kl divergence
    kl_divergence_source = compute_kl_divergence(source_distribution, M.mean(dim=1))
    kl_divergence_target = compute_kl_divergence(target_distribution, M.mean(dim=0))
    # print("kl",kl_divergence_source)

    return 0.5 * (kl_divergence_source + kl_divergence_target)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, activation='sigmoid'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.activation = activation

    def dice_coef(self, pred, gt):
        """ computational formula
        """
       
        softmax_pred = torch.nn.functional.softmax(pred, dim=1)
        seg_pred = torch.argmax(softmax_pred, dim=1)
        all_dice = 0
        gt = gt.squeeze(dim=1)
        batch_size = gt.shape[0]
        num_class = softmax_pred.shape[1]
        for i in range(num_class):

            each_pred = torch.zeros_like(seg_pred)
            each_pred[seg_pred==i] = 1

            each_gt = torch.zeros_like(gt)
            each_gt[gt==i] = 1            

        
            intersection = torch.sum((each_pred * each_gt).view(batch_size, -1), dim=1)
            
            union = each_pred.view(batch_size,-1).sum(1) + each_gt.view(batch_size,-1).sum(1)
            dice = (2. *  intersection )/ (union + 1e-5)
         
            all_dice = all_dice + torch.mean(dice)
 
        return all_dice * 1.0 / num_class


    def forward(self, pred, gt):
        sigmoid_pred = F.softmax(pred,dim=1)
        

        batch_size = gt.shape[0]
        num_class = sigmoid_pred.shape[1]
        
        # conver label to one-hot
        bg = torch.zeros_like(gt)
        bg[gt==0] = 1
        label1 = torch.zeros_like(gt)
        label1[gt==1] = 1
        label2 = torch.zeros_like(gt)
        label2[gt == 2] = 1
        label = torch.cat([bg, label1, label2], dim=1)
        
        loss = 0
        smooth = 1e-5


        for i in range(num_class):
            intersect = torch.sum(sigmoid_pred[:, i, ...] * label[:, i, ...])
            z_sum = torch.sum(sigmoid_pred[:, i, ...] )
            y_sum = torch.sum(label[:, i, ...] )
            loss += (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss * 1.0 / num_class
        return loss


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n

        return mse


class SIMSE(nn.Module):

    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data)
        simse = torch.sum(diffs).pow(2) / (n ** 2)

        return simse


# 鼓励 input1 和 input2 在特征空间中具有不同的表示，通过最大化 input1和 input2 之间的差异性，模型可以学习到更具有区分性的特征表示。
class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss


class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1 - x2, 2)
        summed = torch.sum(power)
        sqrt = summed ** (0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)
