# code in this file is adpated from
# https://github.com/kekmodel/FixMatch-pytorch/blob/master/dataset/cifar.py

from torchvision import datasets
import numpy as np
from torchvision import transforms
from PIL import Image
from .randaugment import RandAugment
from torch.utils.data import Subset


def train_split_labeled(args, labels, num_classes):
    """
        Splits Training labels into Labeled Training Indices & Unlabelled Training Indices based on Labelled Number
    Args:
        args: input arguments from config
        labels: CIFAR train labels
        num_classes: CIFAR number of classes
    Returns:
        labeled_idx: list of labeled indices
        unlabeled_idx: list of unlabeled indices
    """
    #STEP1: Get labels per class eg: example label_per_class = 4000/10 = 400
    label_per_class = args.num_labeled // num_classes
    labels = np.array(labels) #convert labels list to numpy array
    labeled_idx = [] #create labeled index list

    #STEP2: create unlabeled index numpy array - eg. for CIFAR10, [0 to 50k]
    label_len = len(labels)
    # create unlabeled labels based on unlabeled_pct %.
    if hasattr(args, 'reduced_data'):
        label_len = len(labels) * (args.reduced_data / 100)  # eg: 50k * 40% = 20k
    # get index of unlabeled data
    unlabeled_idx = np.array(range(int(label_len)))

    #STEP3: populate labeled_idx with balanced labels from each class based on number of labels.
    # #loop through all classes
    for i in range(num_classes):
        #find all the indexes of each class -  eg for CIFAR10, 5000 indices*10 classes = 50k labels
        idx = np.where(labels == i)[0]
        #create a random list of index - based on label_per_class = 4000/10 = num-labeled/num_classes = 400
        idx = np.random.choice(idx, label_per_class, False)
        #add list of index to labeled_idx - extend() used to add multiple elements to the array
        labeled_idx.extend(idx)
    #convert labeled_idx to numpy array
    labeled_idx = np.array(labeled_idx)

    if (args.num_labeled < args.batch_size):
        raise AssertionError("Argument num_labeled:%d cannot be less than argument batch_size:%d." % (args.num_labeled,args.batch_size))

    #STEP4: shuffle the labeled index
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx

class Transformers(object):
    """
        Transformers Class returns transforms.Compose() methods for the provided type of dataset = train_labeled, train_unlabeled, test
    """
    def __init__(self, dataset_type, mean, std, enable_cutout, cutout_only):
        self.dataset_type = dataset_type

        #initialize weak_augmentation tranformer
        self.weak_augmentation = transforms.Compose([
            transforms.RandomCrop(size=32,padding=4,padding_mode='reflect'),
            transforms.RandomHorizontalFlip()])

        # initialize strong_augmentation tranformer
        #NEXT STEP - work on RandAug
        self.strong_augmentation = transforms.Compose([
            transforms.RandomCrop(size=32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            RandAugment(aug_count=2,aug_value=10,enable_cutout=enable_cutout, cutout_only=cutout_only)])

        #initialize normalize transfomer
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, imgs):
        # apply weak transformer layers - T=[Crop, Rand-H-Flip, Tensor, Normalize]
        if self.dataset_type == 'train_labeled':
            weak_tranformed_imgs = self.weak_augmentation(imgs)
            normalized_imgs = self.normalize(weak_tranformed_imgs)
            return normalized_imgs

        # apply weak transformer layers - T=[Crop, Rand-H-Flip, Tensor, Normalize]
        # apply strong transformer layers - T=[Crop, Rand-H-Flip, Rand-Aug, Tensor, Normalize]
        elif self.dataset_type == 'train_unlabeled':
            weak_tranformed_imgs = self.weak_augmentation(imgs)
            strong_tranformed_imgs = self.strong_augmentation(imgs)
            return self.normalize(weak_tranformed_imgs),self.normalize(strong_tranformed_imgs)

        # apply normalizer layers - T=[Tensor, Normalize]
        elif self.dataset_type == 'test':
            normalized_imgs = self.normalize(imgs)
            return normalized_imgs


class CIFAR10_INDEXED(datasets.CIFAR10):
    """
    CIFAR10_INDEXED Class extends CIFAR10 datasets and returns indexed images(data) & labels(targets)
    """
    def __init__(self, root, indices, train=True,transform=None, target_transform=None,download=False):
        #call super() on original CIFAR10 dataset passing parameters
        super().__init__(root, train=train,transform=transform,target_transform=target_transform,download=download)
        #If indices are provided, the subset images(data) & labels(targets) to provided indices.
        if indices is not None:
            self.data = self.data[indices]
            self.targets = np.array(self.targets)[indices]

    #override __getitem__()
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        #if tranform exists, apply transformation
        if self.transform is not None:
            img = self.transform(img)

        # if target_transform exists, apply target_transformation
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

class CIFAR100_INDEXED(datasets.CIFAR100):
    """
        CIFAR100_INDEXED Class extends CIFAR100 datasets and returns indexed images(data) & labels(targets)
    """
    def __init__(self, root, indices, train=True,transform=None, target_transform=None,download=False):
        #call super() on original CIFAR10 dataset passing parameters
        super().__init__(root, train=train,transform=transform,target_transform=target_transform,download=download)
        #If indices are provided, the subset images(data) & labels(targets) to provided indices.
        if indices is not None:
            self.data = self.data[indices]
            self.targets = np.array(self.targets)[indices]

    #override __getitem__()
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        #if tranform exists, apply transformation
        if self.transform is not None:
            img = self.transform(img)

        # if target_transform exists, apply target_transformation
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

def obtain_cifar_data(args, root='./data'):
    """ Get CIFAR10 train (labeled, unlabeled) & test datatests
    Args:
        args: input arguments from config
        root: root directory of data
    Returns:
        train_labeled_dataset: CIFAR10 type, <labeled#> labeled data
        train_unlabeled_dataset: CIFAR10 type, 500000 unlabeled data
        test_dataset: CIFAR10 type, 100000 test images
    """
    # STEP0: Setup Mean & Std for CIFAR10 & CIFAR100
    if (args.dataset == 'cifar10'):
        cifar_mean = (0.4914, 0.4822, 0.4465)  # RGB Mean of CIFAR 10
        cifar_std = (0.2471, 0.2435, 0.2616)   # RGB Std of CIFAR 10
    elif (args.dataset == 'cifar100'):
        cifar_mean = (0.5071, 0.4867, 0.4408)  # RGB Mean of CIFAR 100
        cifar_std = (0.2675, 0.2565, 0.2761)  # RGB Std of CIFAR 100

    #STEP1: Download CIFAR Train dataset
    train_orig_dataset = datasets.CIFAR10(root, train=True, download=True)

    targets = train_orig_dataset.targets

    if "use_subset" in args and  args.use_subset:
        subset_indices = list(range(0, int( len(train_orig_dataset)/5), 1))

        # Subset doesn't seem to work as expected
        train_orig_dataset2 = Subset(dataset=train_orig_dataset, indices=subset_indices)
        targets = targets[0:int(len(targets)/5)]

    #STEP2: Split training labels into labeled train index & unlabeled train index
    train_labeled_idxs, train_unlabeled_idxs = train_split_labeled(args, targets, len(train_orig_dataset.classes))

    # check if enable checkout is sett, if not default to true:
    if hasattr(args, 'cutout'):
        enable_cutout = args.cutout
    else:
        enable_cutout = True  # activate cutout for RandAugment default

    # check if cutout_only is set, if not default to False:
    if hasattr(args, 'cutout_only'):
        cutout_only = args.cutout_only
    else:
        cutout_only = False  # by default set cutout_only to False so StrongAug = RandAug + CutOut

    #STEP3: Setup Transformers
    #step3a: create transformer for labeled dataset - T=[Crop, Rand-H-Flip, Tensor, Normalize]
    train_labeled_transformer = Transformers(dataset_type='train_labeled', mean=cifar_mean, std=cifar_std, enable_cutout=enable_cutout, cutout_only=cutout_only)
    # step3b: create weak & strong transformer for unlabeled dataset
    # weak transformer layers - T=[Crop, Rand-H-Flip, Tensor, Normalize]
    # strong transformer layers - T=[Crop, Rand-H-Flip, Rand-Aug, Tensor, Normalize]
    train_unlabeled_transformer = Transformers(dataset_type='train_unlabeled', mean=cifar_mean, std=cifar_std, enable_cutout=enable_cutout, cutout_only=cutout_only)
    # step3c: create transformer for test dataset - T=[Tensor, Normalize]
    test_transformer = Transformers(dataset_type='test', mean=cifar_mean, std=cifar_std, enable_cutout=enable_cutout, cutout_only=cutout_only)

    # STEP4: Get dataset: train (labeled & unlabeled) & test
    #step4a: get train labeled dataset for given train_labeled_idxs after applying labeled transformer
    #step4b: get train ulabeled dataset for given train_ulabeled_idxs after applying weak & strong transformers
    #step4c: get test dataset after applying test transformer
    if (args.dataset == 'cifar10'):
        train_labeled_dataset = CIFAR10_INDEXED(root, train_labeled_idxs, train=True,transform=train_labeled_transformer)
        train_unlabeled_dataset = CIFAR10_INDEXED(root, train_unlabeled_idxs, train=True, transform=train_unlabeled_transformer)
        test_dataset = datasets.CIFAR10(root, train=False, transform=test_transformer, download=False)
    elif (args.dataset == 'cifar100'):
        train_labeled_dataset = CIFAR100_INDEXED(root, train_labeled_idxs, train=True, transform=train_labeled_transformer)
        train_unlabeled_dataset = CIFAR100_INDEXED(root, train_unlabeled_idxs, train=True,
                                                  transform=train_unlabeled_transformer)
        test_dataset = datasets.CIFAR100(root, train=False, transform=test_transformer, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset