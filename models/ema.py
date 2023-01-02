# code in this file is implemented from
# https://github.com/kekmodel/FixMatch-pytorch/blob/master/models/ema.py

from copy import deepcopy

import torch


class ModelEMA(object):
    def __init__(self, args, model, decay):
        #setup decay to internal variable
        self.decay = decay
        #copy the entire model and save as ema
        self.ema = deepcopy(model)
        #move ema model to same device as rest
        self.ema.to(args.device)
        #run the ema model in evaluation mode (no train mode, skips batchdropnorm, dropout etc)
        self.ema.eval()
        #check if ema module has 'module'
        self.ema_has_module = hasattr(self.ema, 'module')

        #populate parameters
        self.param_keys = []
        for k, _ in self.ema.named_parameters():
            self.param_keys.append(k)

        #populate buffer keys
        self.buffer_keys = []
        for k, _ in self.ema.named_buffers():
            self.buffer_keys.append(k)

        #in parameters to requires grad to false
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                model_v = msd[j].detach()
                ema_v = esd[k]
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)

            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])
