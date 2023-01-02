# code in this file is adpated from
# https://github.com/kekmodel/FixMatch-pytorch/blob/master/train.py
# CS7643 - Assignment 2 Code

#Utility Imports
import os
import time
import shutil
import numpy as np

#Framework/Packages Imports
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


#Local Imports
from data.dataset_cifar import  obtain_cifar_data
from testing import run_testing
from utils.misc import AverageMeter, accuracy, de_interleave, get_cosine_schedule_with_warmup, interleave, torch_check
from models.mixup import mixup
from models.ema import ModelEMA
from models.wide_resnet import WideResNet



def run_training(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer,  ema_model, scheduler):
    #setup required parameters
    global best_acc
    test_accs = []
    end = time.time()

    #get iterators for labeled and unlabeled data
    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    #set model to training
    model.train()

    #loop through all epochs
    for epoch in range(args.start_epoch, args.epochs):
        #initialize meters to record various variables
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()

        #loop through all the evaluation steps
        for batch_idx in range(args.eval_step):
            try:
                #get input data & targets for labeled data
                inputs_x, targets_x = labeled_iter.next()
            except:
                #if exception error, reset iterator and get inputs & targets
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_iter.next()

            try:
                #get weakly augmented & strongly augmented input data from unlabeled iterator
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
            except:
                # if exception error, reset iterator and get weak & strong inputs
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()

            #update data time
            data_time.update(time.time() - end)

            #get batch size from labeled input shape
            batch_size = inputs_x.shape[0]

            targets_cat = None
            inputs_cat = None
            # mixup here
            if args.run_mixup:
                inputs_cat, targets_cat = mixup(
                    inputs_x, targets_x, 
                    inputs_u_s, inputs_u_w,
                    model, args
                )
            else:
                #concatenate [labeled input=64, unlabeled weak=448, unlabeled strong=448] = 960
                #torch.cat() - Concatenates the given sequence of seq tensors in the given dimension.
                # All tensors must either have the same shape (except in the concatenating dimension) or be empty.
                inputs_cat = torch.cat((inputs_x, inputs_u_w, inputs_u_s))

            inputs = interleave(inputs_cat, 2*args.mu+1)

           
            #move inputs& targets_x to args devices
            inputs = inputs.to(args.device)
            targets_x = targets_x.type(torch.LongTensor).to(args.device)

            #input size = [960,3,32,32]
            logits = model(inputs)
            #logits size = [960,10]
            logits = de_interleave(logits, 2*args.mu+1)

            #extract logits for labeled, unlabeled data (weak & strong)
            logits_x = logits[:batch_size] # logits_x.shape = [64,10]
            logits_u_w_s = logits[batch_size:] #logits unlabeled weak & strong
            # logits_u_w_s.shape = [448*2,10]
            logits_u_w, logits_u_s = logits_u_w_s.chunk(2) # logits_u_w.shape = [448,10]
            #delete logits variable
            del logits

            if args.run_mixup:
                xe_labeled = -F.log_softmax(logits_x, dim=1) * targets_cat[:batch_size]
                Lx = torch.mean(torch.sum(xe_labeled, dim=1))

                xe_unlabeled = -F.log_softmax(logits_u_w_s, dim=1) * targets_cat[batch_size:]
                Lu = torch.mean(torch.sum(xe_unlabeled, dim=1))
            else:
                #cross entropy loss for labeled data - H(xb, pb)
                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

                #detatch weak unlabeled logits, divide by temperature
                #take softmax along last dim (-1)
                pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
                #pseudo labels are probabilities now of shape = [448,10]

                #take max probabilities and their corresponding targets along last dimension
                max_probs, targets_u = torch.max(pseudo_label, dim=-1) #max_probs.shape=[448], targets_u.shape=[448]
                #torch.ge func reference: https://linuxhint.com/torch-gt-torch-ge-pytorch/#:~:text=The%20torch.ge()%20function,greater%20than%20or%20equal%20to).
                #check if max probs >= (greater than equal to (ge)) threshold, and get mask
                mask_boolean = torch.ge(max_probs,args.threshold)
                #convert mask boolean to mask float
                mask = mask_boolean.float()

                # calculate unsupervised loss between
                # targets_u - weakly aug, unlabeled data labels
                # mask -  mask to allow values only over threshold T
                # logits_u_s - predicted prob of strongly aug unlabeled data - A(ub)
                Lu = (F.cross_entropy(logits_u_s, targets_u,
                                    reduction='none') * mask).mean()

            #calculate total loss = supervised loss + (unlabeled losss term * unsupervised loss)
            loss = Lx + args.lambda_u * Lu

            # back propagate the prediciton loss
            loss.backward()

            #update average meters
            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            # once we have gradients, adjust parameters of the model through optimizer step
            optimizer.step()
            scheduler.step()

            #if EMA is required, update ema model with current model
            if args.use_ema:
                ema_model.update(model)

            #reset gradients of the model parameters
            model.zero_grad()

            #update AverageMeters
            batch_time.update(time.time() - end)
            end = time.time()

            if not args.run_mixup:
                mask_probs.update(mask.mean().item())

            if batch_idx % 20 == 0:
                print(('Train Epoch: [{0}/{1}]\t'
                       'Iteration: [{2}/{3}]\t'
                       'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t')
                      .format(epoch+1, args.epochs, batch_idx+1, args.eval_step, iter_time=batch_time, loss=losses))

        #if EMA, use ema model
        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        #run test and get test loss & accuracy
        test_loss, test_acc = run_testing(args, test_loader, test_model, epoch)

        #setup summary writer to write to tensorboard
        args.writer.add_scalar('train/train_loss_total', losses.avg, epoch)
        args.writer.add_scalar('train/train_loss_ce_labeled', losses_x.avg, epoch)
        args.writer.add_scalar('train/train_loss_ce_unsupervised', losses_u.avg, epoch)
        args.writer.add_scalar('train/mask_probs_avg', mask_probs.avg, epoch)
        args.writer.add_scalar('test/test_acc', test_acc, epoch)
        args.writer.add_scalar('test/test_loss', test_loss, epoch)

        #check if bestt accuracy is achieved
        if epoch > 1:
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
        else:
            best_acc = test_acc
            is_best = False

        #save checkpoint
        model_to_save = model.module if hasattr(model, "module") else model
        if args.use_ema:
            ema_to_save = ema_model.ema.module if hasattr(
                ema_model.ema, "module") else ema_model.ema
        #set variables to save checkpoint
        state = {
            'epoch': epoch + 1,
            'state_dict': model_to_save.state_dict(),
            'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        filename = 'checkpoint.pth.tar'
        checkpoint = args.out
        filepath = os.path.join(checkpoint, filename)
        torch.save(state, filepath)
        #if is best, save the check point
        if is_best:
            shutil.copyfile(filepath, os.path.join(checkpoint,
                                                   'model_best.pth.tar'))

        #append test accuracy and get mean
        test_accs.append(test_acc)
        print('Best top-1 acc: {:.2f}'.format(best_acc))
        print('Mean top-1 acc: {:.2f}\n'.format(np.mean(test_accs[-20:])))

    #close argument writer at the end of loop
    args.writer.close()