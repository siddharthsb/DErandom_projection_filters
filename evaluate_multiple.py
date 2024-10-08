
import argparse
import copy
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.wide_resnet import WideResNet32
from model.resnet import ResNet18, ResNet50

from utils import evaluate_standard, evaluate_standard_rp, evaluate_pgd, evaluate_pgd_rp

from utils import get_loaders

import torchattacks

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--network', default='ResNet18', type=str)
    parser.add_argument('--worker', default=4, type=int)
    parser.add_argument('--epsilon', default=8, type=int)

    parser.add_argument('--pretrain', default=None, type=str, help='path to load the pretrained model')
    parser.add_argument('--save_dir', default=None, type=str, help='path to save log')

    parser.add_argument('--attack_type', default='fgsm')

    parser.add_argument('--tau', default=0.1, type=float, help='tau in cw inf')

    parser.add_argument('--max_iterations', default=100, type=int, help='max iterations in cw attack')

    parser.add_argument('--c', default=1e-4, type=float, help='c in torchattacks')
    parser.add_argument('--steps', default=1000, type=int, help='steps in torchattacks')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--device', type=int, default=0)  # gpu
    parser.add_argument('--num_models', type=int, default=1)
    parser.add_argument('--start_from', type=int, default=0)

    # random setting
    parser.add_argument('--rp', action='store_true', help='if random projection')
    parser.add_argument('--rp_block', default=None, type=int, nargs='*',
                        help='block schedule of rp')
    parser.add_argument('--rp_out_channel', default=0, type=int, help='number of rp output channels')

    return parser.parse_args()


def evaluate_attack_at_once(models, test_loader, args, atk, atk_name, logger):
    test_accs = [0 for i in range(14)]
    n = 0

    for model in models:
      model.eval()

    for i, (X, y) in enumerate(test_loader):
        X, y = X.to('cuda'), y.to('cuda')
        X_adv = atk(X, y)  # advtorch

        final_outputs = [0 for i in range(14)]
        for i in range(14):
            for model in range(i+1):
                with torch.no_grad():
                    output = models[model](X_adv)
                final_outputs[i] += output
            final_outputs[i] = torch.div(final_outputs[i], i)
            test_accs[i] += (final_outputs[i].max(1)[1] == y).sum().item()
        n += y.size(0)

    for i in range(14):
        pgd_acc = test_accs[i] / n
        logger.info('Attack_type: [{:s}] done, numModels: {:.4f}, acc: {:.4f} \t'.format(atk_name,i, pgd_acc))

def evaluate_attack(models, test_loader, args, atk, atk_name, logger):
    test_loss = 0
    test_acc = 0
    n = 0

    for model in models:
      model.eval()

    for i, (X, y) in enumerate(test_loader):
        X, y = X.to('cuda'), y.to('cuda')
        X_adv = atk(X, y)  # advtorch

        final_output = 0
        for model in models:
          with torch.no_grad():
              output = model(X_adv)
          final_output += output
        final_output = torch.div(final_output, args.num_models)
        loss = F.cross_entropy(final_output, y)
        test_loss += loss.item() * y.size(0)
        test_acc += (final_output.max(1)[1] == y).sum().item()
        n += y.size(0)

    pgd_acc = test_acc / n
    logger.info('Attack_type: [{:s}] done, acc: {:.4f} \t'.format(atk_name, pgd_acc))

def evaluate_standard(test_loader, models):
    """Evaluate on clean images"""
    test_loss = 0
    test_acc = 0
    n = 0
    for model in models:
      model.eval()

    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            #model.module.random_rp_matrix()
            X, y = X.to('cuda'), y.to('cuda')
            final_output = 0
            for model in models:
              output = model(X)
              final_output += output
            final_output = torch.div(final_output, len(models))
            loss = F.cross_entropy(final_output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (final_output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss / n, test_acc / n

def main():
    args = get_args()

    args.save_dir = os.path.join('logs', args.save_dir)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logfile = os.path.join(args.save_dir, 'output.log')
    handlers = [logging.FileHandler(logfile, mode='a+'),
                logging.StreamHandler()]

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=handlers)

    # set current device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    assert type(args.pretrain) == str and os.path.exists(args.pretrain)

    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    else:
        print('Wrong dataset:', args.dataset)
        exit()

    train_loader, test_loader, dataset_normalization = get_loaders(args.data_dir, args.batch_size, dataset=args.dataset,
                                                                   worker=args.worker, norm=False)

    # setup network
    if args.network == 'ResNet18':
        net = ResNet18
    elif args.network == 'ResNet50':
        net = ResNet50
    elif args.network == 'WideResNet32':
        net = WideResNet32
    else:
        print('Wrong network:', args.network)
    
    models = [None] * (args.num_models)
    for i in range(args.num_models):

      j = i + args.start_from

      models[i] = net(num_classes=args.num_classes, rp=args.rp, rp_block=args.rp_block, rp_out_channel=args.rp_out_channel,
                  normalize=dataset_normalization).cuda()

      models[i] = torch.nn.DataParallel(models[i])
      logger.info(models[i])

      # load pretrained models
      pretrained_model = torch.load(args.pretrain + "weight_%s_latest.pth" % j )
      models[i].load_state_dict(pretrained_model, strict=False)
      models[i].eval()

    logger.info('Evaluating with standard images...')
    _, nature_acc = evaluate_standard(test_loader, models)
    logger.info('Nature Acc: %.4f \t', nature_acc)

    if args.attack_type == 'pgd':
        atk = torchattacks.PGD(models[0], eps=8 / 255, alpha=2 / 255, steps=20, random_start=True)
        evaluate_attack(models, test_loader, args, atk, 'pgd', logger)
    elif args.attack_type == 'fgsm':
        atk = torchattacks.FGSM(models[0], eps=8/255)
        evaluate_attack(models, test_loader, args, atk, 'fgsm', logger)
    elif args.attack_type == 'mifgsm':
        atk = torchattacks.MIFGSM(models[0], eps=8 / 255, alpha=2 / 255, steps=5, decay=1.0)
        evaluate_attack(models, test_loader, args, atk, 'mifgsm', logger)
    elif args.attack_type == 'deepfool':
        atk = torchattacks.DeepFool(models[0], steps=50, overshoot=0.02)
        evaluate_attack(models, test_loader, args, atk, 'deepfool', logger)
    elif args.attack_type == 'cwl2':
        atk = torchattacks.CW(models[0], c=args.c, kappa=0, steps=args.steps, lr=0.01)
        evaluate_attack(models, test_loader, args, atk, 'cwl2', logger)
    elif args.attack_type == 'autoattack':
        atk = torchattacks.AutoAttack(models[0], norm='Linf', eps=8/255, version='standard', n_classes=args.num_classes)
        evaluate_attack(models, test_loader, args, atk, 'autoattack', logger)
    elif args.attack_type == 'bb':
        atk = torchattacks.Pixle(models[0])
        evaluate_attack(models, test_loader, args, atk, 'pixle', logger)
        atk = torchattacks.Square(models[0], norm='Linf', eps=8/255, n_queries=5000)
        evaluate_attack(models, test_loader, args, atk, 'square', logger)
    else:
        print('Wrong attack method:', args.attack_type)

    logger.info('Testing done.')


if __name__ == "__main__":
    main()
