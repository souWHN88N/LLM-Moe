'''
AAMsoftmax loss function copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/loss/aamsoftmax.py
'''
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, time
import torch.nn.functional as F

class AAMsoftmax(nn.Module):
    def __init__(self, n_class, hdim, m, s):
        super(AAMsoftmax, self).__init__()
        self.m = m
        self.s = s
        self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, hdim), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.FloatTensor(n_class, 1), requires_grad=True)
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.1, )
        nn.init.xavier_normal_(self.weight, gain=1)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label=None):
        cosine = x
        # cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cosine = cosine + self.bias
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s
        loss = self.ce(output, label)
        return loss


class logSoftMaxLoss(nn.Module):
    def __init__(self):
        super(logSoftMaxLoss, self).__init__()
        self.log_SoftMax_Loss = nn.LogSoftmax(dim=1)

    def forward(self, x):
        loss = self.log_SoftMax_Loss(x)
        return loss

class Control_Contrastive(nn.Module):
    def __init__(self, n_class, hdim, m, s):
        super(Control_Contrastive, self).__init__()
        self.pc = 0
        self.m = m
        self.m_center = m  # 0.2  11°
        self.s = s
        self.s_neg = s
        self.n_class = n_class

        # self.H_weight = torch.nn.Parameter(torch.FloatTensor(7, 2708), requires_grad=True)
        # nn.init.xavier_normal_(self.H_weight, gain=1)

        self.aam = AAMsoftmax(n_class=n_class, hdim=hdim, m=m, s=s)
        self.aam_c = AAMsoftmax(n_class=n_class, hdim=hdim, m=m, s=s)
        self.cos_m = math.cos(self.m)
        self.cos_m_center = math.cos(self.m_center)
        self.sin_m = math.sin(self.m)
        self.sin_m_center = math.sin(self.m_center)

        self.th = math.cos(self.m)
        # self.th = 0.95
        self.th_center = math.cos(self.m_center)
        # self.th_center = 0.92
        self.mm = math.sin(math.pi - self.m) * self.m
        # self.mm = math.sin(math.pi - self.m) * self.m
        self.mm_center = math.sin(math.pi - self.m_center) * self.m_center
        # self.mm_center = math.sin(math.pi - self.m_center) * (self.m_center ** 2)

        self.soft_plus = nn.Softplus()

    def forward(self, x, label=None, L=None):
        # pairwise_dist = torch.cdist(x, x)   # 计算节点间欧氏距离
        # K = torch.exp(-pairwise_dist ** 2 / torch.sqrt((torch.mean(pairwise_dist)** 2 + 1e-16)) )
        #
        # Q = torch.linalg.cholesky(x.T @ x)
        # H_ortho = x @ torch.inverse(Q).T
        #
        # loss1 = -torch.trace(H_ortho.T @ K @ H_ortho) + 0.5 * torch.trace(H_ortho.T @ L @ H_ortho)
        # loss1 = 0.5 * torch.trace(H_ortho.T @ L @ H_ortho)
        # loss1 = 0

        aam_center_loss = 0
        x_clone = x.clone()
        avg_neg = []
        avg_x = []

        labels_center, full_unique_index = label.unique(return_inverse=True)
        for labels in labels_center:
            avg_neg.append(torch.cat((x, x_clone), dim=1)[labels == label].mean(dim=0))

            #todo neg(no cat x)
            avg_x.append(x_clone[labels == label].mean(dim=0))
        avg_x = torch.stack(avg_x)
        logit_neg = torch.stack(avg_neg)

        aam_sample_loss = 0.5 * self.aam(x, label)
        aam_center_loss = 0.5 * self.aam_c(avg_x, labels_center)

        logit_neg = torch.matmul(logit_neg, logit_neg.T)  # [-2, 2] [-1, 1]
        logit_neg = logit_neg * (1 - torch.eye(logit_neg.size(0), device=logit_neg.device))
        logit_neg = logit_neg[logit_neg != 0]
        max_logit_neg = max(logit_neg)
        median_logit_neg = logit_neg.median()
        min_logit_neg = logit_neg.min()
        logit_neg = torch.topk(logit_neg, min(len(logit_neg), 7))[0]
        logit_neg_loss = 0

        strip_epoch = 1
        if self.pc % (2 * strip_epoch) == 0:
            sys.stderr.write(time.strftime(
                "%m-%d %H:%M:%S") + "\t Loss n: %.4f, c: %.4f, s: %.4f\n neg: max_n: %.4f, median_n:%.4f, min_n:%.4f\n\n" %
                             (logit_neg_loss, aam_center_loss, aam_sample_loss, max_logit_neg, median_logit_neg,
                              min_logit_neg))
        self.pc += 1
        return logit_neg_loss + aam_center_loss + aam_sample_loss
class KNN_Loss(nn.Module):
    def __init__(self, init_w=1.0, init_b=0.0, hdim=7, n_class=7, **kwargs):  # No temp param
        super(KNN_Loss, self).__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.weight = torch.nn.Parameter(torch.FloatTensor(140, 140), requires_grad=True)
        nn.init.xavier_normal_(self.weight, gain=1)

    def forward(self, features, A, D):
        features = F.normalize(features, p=2)
        sim = features @  features.T
        # A是对称阵  D对角阵
        dot_A_feature = A @ sim
        dot_D_feature = D @ sim
        # exp_A_logits = torch.exp(dot_A_feature)
        # exp_D_logits = torch.exp(dot_D_feature)
        # loss = exp_A_logits.sum(1) / exp_D_logits.sum(1)
        loss = F.kl_div(dot_A_feature.softmax(dim=-1).log(), dot_D_feature.softmax(dim=-1), reduction='sum')
        return loss

