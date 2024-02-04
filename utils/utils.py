import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import shutil
import logging
from logging import handlers
import numpy as np


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

#save model
def save_checkpoint(state, is_best, save_root, name):
    save_path = os.path.join(save_root, name + 'checkpoint.pth.tar')
    torch.save(state, save_path)
    if is_best:
        best_save_path = os.path.join(save_root, name + 'model_best.pth.tar')
        shutil.copyfile(save_path, best_save_path)


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, filename, level='info', when='D', backCount=3,
                 fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        if not self.logger.handlers:
            sh = logging.StreamHandler()
            sh.setFormatter(format_str)
            th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount,
                                                   encoding='utf-8')
            th.setFormatter(format_str)
            self.logger.addHandler(sh)
            self.logger.addHandler(th)



def count_parameters_in_MB(model):
    return sum(np.prod(v.size()) for name, v in model.named_parameters()) / 1e6


def create_exp_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

#metric
def mce(com, base):
    r = np.ones(15) * 100
    ce = (r - com) / (r - base)
    mce = ce.mean() * 100
    return mce