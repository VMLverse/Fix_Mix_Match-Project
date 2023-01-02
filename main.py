# code in this file is adpated from
# https://github.com/kekmodel/FixMatch-pytorch/blob/master/train.py
# CS7643 - Assignment 2 Code

#Utility Imports
import argparse
import yaml
import math
import os
import time
import shutil
import numpy as np

#Framework/Packages Imports
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


#Local Imports
from data.dataset_cifar import  obtain_cifar_data
from training import run_training
from utils.misc import AverageMeter, accuracy, de_interleave, get_cosine_schedule_with_warmup, interleave, torch_check
from models.ema import ModelEMA
from models.wide_resnet import WideResNet



def load_yaml_args():

    #import config.yaml file
    parser = argparse.ArgumentParser(description='CS7643 Final Project')
    parser.add_argument('--config', default='./config/config_fixmatch_cifar10red40_1000_cutoutonly.yaml')

    #Setup Parser & Load yaml
    global args
    args = parser.parse_args(args=[])
    with open(args.config) as f:
        config = yaml.full_load(f)
    #Load config_*.yaml into args
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    print("Config Parameters Loaded:", args)

    #setup parameters for Tensorboard using SummaryWriter
    RaiseException(args.out)

    os.makedirs(args.out, exist_ok=True)
    args.writer = SummaryWriter(args.out)

    #setup num of classes for each dataset
    if args.dataset == 'cifar10':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100

    #check if cuda is available and set cuda
    if torch.cuda.is_available():
        args.device = 'cuda'
    else:
        args.device = 'cpu'

def single_run():
  
    #Obtain train (labeled & unlabeled) & test data
    labeled_dataset, unlabeled_dataset, test_dataset = obtain_cifar_data(args)

    #initialize samples for train
    train_random_sampler = RandomSampler

    # if reduced dataset, then
    if hasattr(args, 'reduced_data'):
        # create a list of reduced data for test data
        test_redset = len(test_dataset) * (args.reduced_data / 100)  # eg: 50k * 40% = 20k
        test_redidx = list(range(0, int(test_redset)))  # sequential list entries for test data
        test_dataset = torch.utils.data.Subset(test_dataset, test_redidx)  # get reduced test dataset

    #prepare dataloaders
    #randomly sample for train dataset per batch size
    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_random_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_random_sampler(unlabeled_dataset),
        batch_size=args.batch_size * args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)


    #create model
    model = WideResNet(depth=28, widen_factor=2, dropout_rate=0.0, num_classes=args.num_classes)

    #calc & print total number of params for the model
    total_parameters = 0
    for p in model.parameters():
        total_parameters = total_parameters + p.numel() # returns the total number of elements in the input tensor
    # get total number of parameters in Million
    total_parameters_mil = total_parameters/1000000 #1e6 = 1 million
    print("Total params: {:.2f}M".format(total_parameters_mil))

    #move model to cpu or gpu
    model.to(args.device)

    #create grouped parameters with weight decay and without weight decay
    # create params with decay - avoid bias & bn
    params_wdecay = []
    for n, p in model.named_parameters():
        if not any(nd in n for nd in ['bias', 'bn']):
            params_wdecay.append(p)

    # create params without decay - include bias & bn
    params_nodecay = []
    for n, p in model.named_parameters():
        if any(nd in n for nd in ['bias', 'bn']):
            params_nodecay.append(p)
    # create grouped parameters:
    grouped_parameters = [
        {'params': params_wdecay, 'weight_decay': float(args.wdecay)},
        {'params': params_nodecay, 'weight_decay': 0.0}
    ]
    
    if hasattr(args, 'use_adam'): #for Adam Ablation
        # Best params from the Fixmatch ablation study (Table 7)
        optimizer = optim.Adam(params=model.parameters(), lr=0.0003, betas=(0.9, 0.999))
    else: # Use SGD Optim as Default
        # create SGD optimizer for backprop -
        optimizer = optim.SGD(grouped_parameters, lr=float(args.lr),
                              momentum=0.9, nesterov=float(args.nesterov))

    #calc total number of epochs
    # epochs = total_steps/eval_step = 2**20/1024 = 1024
    args.epochs = math.ceil(args.total_steps / args.eval_step)

    #get scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    #if use EMA, setup ema model
    if args.use_ema:
        ema_model = ModelEMA(args, model, args.ema_decay)

    #reset start epoch to 0
    args.start_epoch = 0

    #if resume, load checkpoint file from path
    if args.resume:
        print("==> Resuming from checkpoint..")
        #Raise Error if no checkpoint directory found!
        assert (os.path.isfile(args.resume))
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    print("***** Running training *****")
    print(f"  Task = {args.dataset}@{args.num_labeled}")
    print(f"  Num Epochs = {args.epochs}")
    print(f"  Total Batch Size = {args.batch_size}")
    print(f"  Total optimization steps = {args.total_steps}")

    #reset all gradients
    model.zero_grad()

    #run train method
    run_training(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler)


def run_all_experiments():
    # Load yaml file(s?), and walk through experimental setup 
    
    # FixMatch 
    # Fixed hyperparameters from paper: (λu = 1, η = 0.03, β = 0.9, τ = 0.95, µ = 7, B = 64, K = 220) Table 4
    num_labels_main = [40, 250, 4000]


    # Threshold Values (250-Label + CIFAR10 from paper Table 5)
    threshold_values = [0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.97, 0.99]

    # Learning rate decay schedules (250-Label + CIFAR10 from paper Table 6)
    decay_schedules = ["cosine", "linear_01", "linear_02", "none"]

    # Optimizers (skip this - too much tuning?)
    # Table 7 has SGD vs Adam with many learning rates and momentum

    load_yaml_args()
    single_run()

if __name__ == "__main__":
    torch_check()
    run_all_experiments()
