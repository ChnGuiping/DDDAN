#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import os
import time
from datetime import datetime
from utils.logger import setlogger
import logging
from utils.train_utils_combines import train_utils
import torch
import warnings
print(torch.__version__)
warnings.filterwarnings('ignore')

args = None


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    # model and data parameters
    parser.add_argument('--model_name', type=str, default='WCNN_EWK', help='the name of the model', choices=['WCNN','WCNN_EWK'])
    parser.add_argument('--data_name', type=str, default='PHM', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default=r'D:\Pycharm\Datas\PHM', help='the directory of the data')
    parser.add_argument('--transfer_task', type=str, default=[[0], [3]], help='transfer learning tasks')
    parser.add_argument('--normlizetype', type=str, default='mean-std', help='nomalization type')

    # training parameters
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
    parser.add_argument("--pretrained", type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')

    parser.add_argument('--bottleneck', type=bool, default=True, help='whether using the bottleneck layer')
    parser.add_argument('--bottleneck_num', type=int, default=256, help='whether using the bottleneck layer')
    parser.add_argument('--last_batch', type=bool, default=True, help='whether using the last batch')

    # Distance Loss
    parser.add_argument('--distance_metric', type=bool, default=True, help='whether use distance metric')
    parser.add_argument('--distance_loss', type=str, default='JDA_W', help='which distance loss you use')
    parser.add_argument('--trade_off_distance', type=str, default='Step', help='')
    parser.add_argument('--lam_distance', type=float, default=1, help='this is used for Cons')

    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['step', 'exp', 'stepLR', 'fix'], help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='50, 120', help='the learning rate decay for step and stepLR')

    # save, load and display information
    parser.add_argument('--middle_epoch', type=int, default=0, help='max number of epoch')
    parser.add_argument('--max_epoch', type=int, default=200, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=5, help='the interval of log training information')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()

    # Prepare the saving path for the model
    sub_dir = args.model_name + '_' + datetime.strftime(datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'DDDAN.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    trainer = train_utils(args, save_dir)
    trainer.setup()
    start_time = time.time()
    trainer.train()











