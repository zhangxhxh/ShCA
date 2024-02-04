from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import sys
import time
import argparse
import random
import shutil
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import dataset, dataloader
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
from utils.utils import AverageMeter, accuracy, save_checkpoint, Logger, count_parameters_in_MB, create_exp_dir
from models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnet110, resnet20, resnet56, resnet8
import warnings
from utils.utils import mce
from models.preprocess import AAMixDataset

warnings.filterwarnings('ignore')
import models.augmentations as augmentations

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# Set parameters
parser = argparse.ArgumentParser(description='ShCA-main')
# path
parser.add_argument('--save_root', type=str, default='./results', help='models and logs are saved here')
parser.add_argument('--img_root', type=str, default='./data', help='path name of image dataset')

# training hyper parameters
parser.add_argument('--epochs', type=int, default=240, help='number of total epochs to run')
parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--num_class', type=int, default=100, help='number of classes')
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--mixture-width', default=3, type=int,
                    help='Number of augmentation chains to mix per augmented example')
parser.add_argument('--mixture-depth', default=-1, type=int,
                    help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
parser.add_argument('--aug-severity', default=3, type=int, help='Severity of base augmentation operators')
# others
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--dataset', default='CIFAR100', type=str, help='Input the name of dataset: default(CIFAR10)')

args = parser.parse_args()

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

log_pos = os.path.join(args.save_root, 'print_information.log')
log = Logger(log_pos, level='info')

small_model = resnet8().cuda()
later_model = resnet20().cuda()

triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)

base_c_path = './data/CIFAR-100-C/'
CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]
preprocess = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# define loss function
criterion = nn.CrossEntropyLoss().cuda()

#baseline
base = np.array([21.73, 26.74, 25.68, 54.43, 24.35, 47.37, 47.71, 46.60, 41.40, 55.20, 64.02, 45.06, 52.29, 44.85, 45.95])

def base_main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        cudnn.enabled = True
        cudnn.benchmark = True

    # getdata
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.RandomCrop(32, padding=4)])
    preprocess = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transform = preprocess

    train_data = torchvision.datasets.CIFAR100('./data/', train=True, transform=train_transform, download=True)
    test_data = torchvision.datasets.CIFAR100('./data/', train=False, transform=test_transform, download=True)

    train_data = AAMixDataset(train_data, preprocess)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                               pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False, num_workers=4, pin_memory=False)
    torch.save(test_loader, 'clean.pth')

    # optimizer
    small_optimizer = optim.SGD(small_model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
    later_optimizer = optim.SGD(later_model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=args.weight_decay)

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # train and test and save
    log.logger.info('start training!')
    best_small = 0
    best_later = 0
    best_cor = np.zeros(15)
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(small_optimizer, epoch, args.epochs)
        adjust_learning_rate(later_optimizer, epoch, args.epochs)
        test_loader = torch.load('clean.pth')

        epoch_start_time = time.time()
        train_small_losses, train_small1, train_small5, train_later_losses, train_later1, train_later5 = train(train_loader,
                                                                                                       small_model, later_model,
                                                                                                       small_optimizer,
                                                                                                       later_optimizer,
                                                                                                       criterion)
        test_losses, test_small1, test_small5, test_later1, test_later5 = test(test_loader, small_model, later_model, criterion)

        epoch_duration = time.time() - epoch_start_time
        log.logger.info('Epoch time: {}s'.format(int(epoch_duration)))
        log_train = ('Epoch[{0}]'
                     'train_small_losses:  {small_losses:.4f} '
                     'train_small1:  {small_top1:.2f}  '
                     'train_later_losses:  {later_losses:.4f} '
                     'train_later1:  {later_top1:.2f}  '.format(
            epoch, small_losses=train_small_losses.avg, small_top1=train_small1.avg, later_losses=train_later_losses.avg,
            later_top1=train_later1.avg))
        log.logger.info(log_train)
        log_test = ('Epoch[{0}]'
                    'test_loss:  {losses:.4f} '
                    'test_small1:  {small_top1:.2f}  '
                    'test_later1:  {later_top1:.2f}  '.format(
            epoch, losses=test_losses.avg, small_top1=test_small1.avg, later_top1=test_later1.avg, ))
        log.logger.info(log_test)

        corruption_accs = test_c(later_model, test_loader, base_c_path, criterion)

        # save model
        is_best = False
        if test_small1.avg > best_small:
            best_small = test_small1.avg
            is_best = True
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': small_model.state_dict(),
            'prec@1': test_small1,
            'prec@5': test_small5,
        }, is_best, args.save_root, 'small')
        #
        # save model
        is_best = False
        if test_later1.avg > best_later:
            best_later = test_later1.avg
            is_best = True
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': later_model.state_dict(),
            'prec@1': test_later1,
            'prec@5': test_later5,
        }, is_best, args.save_root, 'later')

        for i in range(0, 15):
            is_best = False
            if corruption_accs[i] > best_cor[i]:
                best_cor[i] = corruption_accs[i]
                is_best = True
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': later_model.state_dict(),
                'prec@1': corruption_accs[i],
            }, is_best, args.save_root, 'corr' + str(i))

    log.logger.info(best_later)
    log.logger.info(best_cor)
    get_mce = mce(best_cor, base)
    log.logger.info(get_mce)

def train(train_loader, small_model, later_model, small_optimizer, later_optimizer, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    later_losses = AverageMeter()
    later_top1 = AverageMeter()
    later_top5 = AverageMeter()

    small_model.train()
    later_model.train()
    for i, (img_all, target) in enumerate(train_loader, start=0):
        img_int = img_all[1].cuda()
        img = img_all[0].cuda()
        target = target.cuda()

        out, out_feature = small_model(img)
        loss = criterion(out, target)

        k = 1
        neg_rank, pos_rank = check_rank(out_feature, target, k)

        pos_img, pos_aug, neg_img, neg_aug = pos_neg_fun(img_int, neg_rank, pos_rank, preprocess, k)

        images = torch.cat([img, pos_img, pos_aug, neg_img, neg_aug], 0)
        outputs, features = later_model(images)

        ori_out, pos_out, pos_aug_out, neg_out, neg_ori_out = torch.chunk(outputs, 5, 0)
        ori_feature, pos_feature, _, neg_feature, _, = torch.chunk(features, 5, 0)

        #label for negative image
        neg_criterion=0
        k_ratio=1/k
        if k==1:
            neg_criterion=criterion(neg_out,target[neg_rank[:,0].type(torch.long)])
        else:
            for m in range(0, k):
                neg_criterion=neg_criterion+k_ratio*criterion(neg_out,target[neg_rank[:,m].type(torch.long)])

        ori_loss = (criterion(ori_out, target) + criterion(pos_out, target) + criterion(pos_aug_out,target) + neg_criterion) / 4
        con_loss = (nn.KLDivLoss(reduction='batchmean')(F.log_softmax(pos_out, dim=1),
                                                        F.softmax(Variable(ori_out), dim=1)) +
                    nn.KLDivLoss(reduction='batchmean')(F.log_softmax(pos_aug_out, dim=1),
                                                        F.softmax(Variable(ori_out), dim=1)) +
                    nn.KLDivLoss(reduction='batchmean')(F.log_softmax(neg_out, dim=1),
                                                        F.softmax(Variable(neg_ori_out), dim=1))) / 3
        tri_loss = triplet_loss(ori_feature, pos_feature, neg_feature)
        later_loss = ori_loss + con_loss + tri_loss

        prec1, prec5 = accuracy(out, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        small_optimizer.zero_grad()
        loss.backward()
        small_optimizer.step()

        prec1, prec5 = accuracy(ori_out, target, topk=(1, 5))
        later_losses.update(later_loss.item(), img.size(0))
        later_top1.update(prec1.item(), img.size(0))
        later_top5.update(prec5.item(), img.size(0))

        later_optimizer.zero_grad()
        later_loss.backward()
        later_optimizer.step()
    return losses, top1, top5, later_losses, later_top1, later_top5

# each anchor image and corresponding positive and negative samples
def check_rank(out_feature, target, k):
    dis_len = list(range(out_feature.shape[0]))

    #rank
    def f(x, *y):
        each_dis = torch.sqrt(
            torch.sum(torch.square(out_feature[x, :].repeat(out_feature.shape[0], 1) - out_feature[:, :]),
                      dim=1))
        return each_dis

    dis_dis = torch.stack(list(map(f, dis_len)))

    com_target = target.unsqueeze(1).repeat(1, out_feature.shape[0])

    neg_value, neg_index = torch.sort(dis_dis, dim=1)
    pos_value, pos_index = torch.sort(dis_dis, dim=1, descending=True)

    #index
    neg_num = search_index(out_feature, target, com_target, dis_len, neg_index, 'neg', k)
    pos_num = search_index(out_feature, target, com_target, dis_len, pos_index, 'pos', k)
    return neg_num, pos_num

#corresponding index of samples
def search_index(out_feature, target, com_target, dis_len, neg_index, name, k):
    def g(x, *y):
        index_target = target[neg_index[x, :]]
        return index_target

    com_index = torch.stack(list(map(g, dis_len)))
    if name == 'neg':
        fake_k = ~com_target.eq(com_index) + 0
    elif name == 'pos':
        fake_k = com_target.eq(com_index) + 0

    idx = reversed(torch.Tensor(range(1, int(out_feature.shape[0] + 1)))).cuda()
    unknown = torch.einsum("ab,b->ab", (fake_k, idx))

    indice_value, indice_index = torch.topk(unknown, k, dim=1)
    all_index = torch.zeros(out_feature.shape[0], k)
    for j in range(0, k):
        image_indice = indice_index[:, j]
        num_index = neg_index[torch.arange(0, out_feature.shape[0]), image_indice]
        all_index[:, j] = num_index
    return all_index

#augmented images
def pos_neg_fun(img_int, neg_rank, pos_rank, preprocess, k):
    aug_list = augmentations.augmentations_all

    dis_len = list(range(img_int.shape[0]))

    def h(x, *y):
        img_ori = np.array(img_int[x, :].cpu())
        fft_1 = np.fft.fftshift(np.fft.fftn(img_ori))
        abs_1, angle_1 = np.abs(fft_1), np.angle(fft_1)

        pos_abs = np.full_like(abs_1, 0)
        neg_fft = np.full_like(fft_1, 0)
        neg_ori_avg = np.full_like(img_ori, 0)
        #spectral
        for m in range(0, k):
            pos_img_k = img_int[pos_rank[:, m].type(torch.long), :]
            img_pos_k = np.array(pos_img_k[x, :].cpu())

            fft_2 = np.fft.fftshift(np.fft.fftn(img_pos_k))
            abs_2, angle_2 = np.abs(fft_2), np.angle(fft_2)
            pos_abs = pos_abs + abs_2

            neg_img_k = img_int[neg_rank[:, m].type(torch.long), :]
            img_neg_ori_k = np.array(neg_img_k[x, :].cpu())

            fft_3 = np.fft.fftshift(np.fft.fftn(img_neg_ori_k))
            abs_3, angle_3 = np.abs(fft_3), np.angle(fft_3)

            fft_3 = (0.5 * abs_3 + 0.5 * abs_1) * np.exp((1j) * angle_3)
            neg_fft = neg_fft + fft_3

            neg_ori_avg = neg_ori_avg + img_neg_ori_k
        pos_abs = pos_abs / k

        angle_1 = angle_1 + torch.normal(mean=0.0, std=0.5, size=(32, 32, 3)).numpy()
        fft_2 = pos_abs * np.exp((1j) * angle_1)
        img_pos = fft_2.astype(np.uint8)

        neg_fft = neg_fft / k
        img_neg = neg_fft.astype(np.uint8)

        neg_ori_avg = neg_ori_avg / k
        neg_ori_avg = neg_ori_avg.astype(np.uint8)

        p = random.uniform(0, 1)
        if p > 0.5:
            img_pos = img_ori
        p = random.uniform(0, 1)
        if p > 0.5:
            img_neg = neg_ori_avg

        ws = np.float32(np.random.dirichlet([1] * args.mixture_width))
        m = np.float32(np.random.beta(1, 1))

        mix_pos = torch.zeros_like(preprocess(img_ori))
        mix_neg = torch.zeros_like(preprocess(img_ori))
        for i in range(args.mixture_width):
            img_aug = Image.fromarray(img_ori.copy())
            neg_aug = Image.fromarray(neg_ori_avg.copy())
            depth = args.mixture_depth if args.mixture_depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(aug_list)
                img_aug = op(img_aug, args.aug_severity)
                neg_aug = op(neg_aug, args.aug_severity)
            t = np.float32(np.random.beta(1, 1))
            mix_pos += ws[i] * ((1 - t) * preprocess(img_aug) + t * preprocess(img_pos))
            mix_neg += ws[i] * ((1 - t) * preprocess(neg_aug) + t * preprocess(img_neg))

        mixed_pos = m * preprocess(img_ori) + (1 - m) * mix_pos
        mixed_neg = preprocess(img_neg)
        return mixed_pos, preprocess(img_aug), mixed_neg, preprocess(neg_ori_avg)

    all_afterimg = list(map(h, dis_len))
    each_mode_img = []
    for i in range(0, 4):
        def g(x, *y):
            each_img = all_afterimg[x][i]
            return each_img

        each_mode_img.append(torch.stack(list(map(g, dis_len))).cuda())
    after_pos_img = each_mode_img[0]
    pos_ap = each_mode_img[1]
    after_neg_img = each_mode_img[2]
    neg_ap = each_mode_img[3]
    return after_pos_img, pos_ap, after_neg_img, neg_ap

#test with two models
def test(test_loader, small_model, later_model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    later_top1 = AverageMeter()
    later_top5 = AverageMeter()

    small_model.eval()
    later_model.eval()

    for i, (img, target) in enumerate(test_loader, start=0):
        if args.cuda:
            img = img.cuda()
            target = target.cuda()

        with torch.no_grad():
            out, _ = small_model(img)
            later_out, _ = later_model(img)

        loss = criterion(later_out, target)

        prec1, prec5 = accuracy(out, target, topk=(1, 5))
        later_prec1, later_prec5 = accuracy(later_out, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))
        later_top1.update(later_prec1.item(), img.size(0))
        later_top5.update(later_prec5.item(), img.size(0))
    return losses, top1, top5, later_top1, later_top5


#test with corrupted images
def test_c(later_model, test_loader, base_c_path, criterion):
    test_data_c = test_loader
    corruption_accs = []
    for corruption in CORRUPTIONS:
        test_data_c.dataset.data = np.load(base_c_path + corruption + '.npy')
        test_data_c.dataset.targets = np.load(base_c_path + 'labels.npy').tolist()

        test_data_c = torch.utils.data.DataLoader(test_data_c.dataset, batch_size=100, shuffle=False, num_workers=4)
        test_loss, test_acc1, test_acc5 = test_one(test_data_c, later_model, criterion)
        corruption_accs.append(test_acc1.avg)
    return corruption_accs

#test with one model
def test_one(test_loader, later_model, criterion):
    losses = AverageMeter()
    later_top1 = AverageMeter()
    later_top5 = AverageMeter()

    later_model.eval()

    for i, (img, target) in enumerate(test_loader, start=0):
        if args.cuda:
            img = img.cuda()
            target = target.cuda()

        with torch.no_grad():
            later_out, _ = later_model(img)

        loss = criterion(later_out, target)

        later_prec1, later_prec5 = accuracy(later_out, target, topk=(1, 5))
        losses.update(loss.item(), img.size(0))
        later_top1.update(later_prec1.item(), img.size(0))
        later_top5.update(later_prec5.item(), img.size(0))
    return losses, later_top1, later_top5

#learning rate
def adjust_learning_rate(optimizer, epoch, epochs):
    if epoch < int(epochs * (5 / 8)):
        lr = 0.1
    elif epoch < int(epochs * (3 / 4)):
        lr = 0.1 * 0.1
    elif epoch < int(epochs * (7 / 8)):
        lr = 0.1 * 0.01
    else:
        lr = 0.1 * 0.001
        # update optimizer's learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    base_main(args)
