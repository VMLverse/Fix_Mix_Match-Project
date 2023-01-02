# code in this file is adpated from
# https://github.com/kekmodel/FixMatch-pytorch/blob/master/train.py
# CS7643 - Assignment 2 Code

import time
import torch
from utils.misc import AverageMeter, accuracy
import torch.nn.functional as F

def run_testing(args, test_loader, model, epoch):
    #setup average meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    # iterate over test dataset
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            #set model to evaluate mode
            model.eval()
            #move test inputs & targets to device
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            #execute model for input and get output predictions
            outputs = model(inputs)
            #get cross entrophy = H(p,q)
            loss = F.cross_entropy(outputs, targets)

            #get accuraccy values for top 1 and top5
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))

            #update Average meters
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()

    print("top-1 acc: {:.2f}".format(top1.avg))
    print("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg