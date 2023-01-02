import torch
import math
from torch.optim.lr_scheduler import LambdaLR

class AverageMeter(object):
    """Computes and stores the average and current value"""

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

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def torch_check():
    x = torch.rand(5, 3)
    # print(x)
    # Check device availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("You are using device: %s" % device)
    print("Pytorch version:" + torch.__version__)

def interleave(x, size):
    #size = 2*mu+1; eg: mu=7, (2*mu)+1= 15
    #create an empty list with x.shape
    s = list(x.shape) #s= [960,3,32,32]
    t1 = [-1, size]  # t2=[-1, 15]
    t2 = s[1:] #t1 = [3,32,32]
    reshape_size1 = t1 + t2 #reshape_size = [-1,15,3,32,32]
    reshape_size2 = [-1] + t2  # reshape_size = [-1,3,32,32]
    #reshape output
    out1 = x.reshape(reshape_size1) #out1.shape = [64,15,3,32,32]
    out2 = out1.transpose(0, 1) #out2.shape = [15,64,3,32,32]
    out3 = out2.reshape(reshape_size2) #out3.shape = [960,3,32,32]
    return out3

def de_interleave(x, size):
    #size = 2*mu+1 = 15
    s = list(x.shape)  #s = [960,10]
    t1 = [size, -1] #t1 = [15,-1]
    t2 = s[1:] #t2 = [10]
    reshape_out1 = t1 + t2 #reshape_out1 = [15, -1, 10]
    reshape_out2 = [-1] + t2 #reshape_out1 = [-1, 10]
    out1 = x.reshape(reshape_out1) #out1.shape = [15, 64, 10]
    out2 = out1.transpose(0, 1) #out2.shape = [64, 15, 10]
    out3 = out2.reshape(reshape_out2) #out3.shape = [960, 10]
    return out3

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    #function which computes a cosine multiplicative factor given an integer parameter epoch
    def _lr_lambda(current_step):
        #if current step is LESS than num of warmup steps
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # else if current step is MORE than num of warmup steps, calc no progress factor
        no_progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        multiplicative_factor = max(0., math.cos(math.pi * num_cycles * no_progress))
        return multiplicative_factor

    #Sets the learning rate of each parameter group to the initial lr times a given function.
    # When last_epoch=-1, sets initial lr as lr.
    cosine_scheduler =  LambdaLR(optimizer, _lr_lambda, last_epoch)
    return cosine_scheduler