import torch
import torch.nn.functional as F
import numpy as np

def mixup(x_labeled, y_labeled, x_unlabeled_strong, 
          x_unlabeled_weak, model, args):
    # predicted labels of unlabeled samples
    with torch.no_grad():
        x_unlabeled_strong = x_unlabeled_strong.to(args.device)
        x_unlabeled_weak = x_unlabeled_weak.to(args.device)
        preds_strong = torch.softmax(model(x_unlabeled_strong), dim=1)
        preds_weak = torch.softmax(model(x_unlabeled_weak), dim=1)

        # avg prediction
        p = ((preds_strong + preds_weak) / 2).detach()

    p_power = p**(1/args.mixup_temperature)

    y_unlabeled = p_power / p_power.sum(dim=1, keepdim=True)
    y_labeled = F.one_hot(y_labeled.long(), num_classes=10)

    concat_data = torch.cat([x_labeled.to(args.device), x_unlabeled_weak, x_unlabeled_strong])

    concat_labels = torch.cat([y_labeled.to(args.device), y_unlabeled, y_unlabeled])
    # shuffle indices

    shuffled_indices = torch.randperm(concat_labels.size(0))

    # shuffle labels and data

    shuffled_data = concat_data[shuffled_indices]
    shuffled_labels = concat_labels[shuffled_indices]

    # MixUp

    gamma = np.random.beta(args.mixup_alpha, args.mixup_alpha)
    gamma = max(gamma, 1-gamma)

    # can apply to entire concat without splitting because
    # it remains lined up
    mixedup_x = gamma * concat_data + (1-gamma)*shuffled_data
    mixedup_y = gamma * concat_labels + (1-gamma)*shuffled_labels

    return mixedup_x, mixedup_y