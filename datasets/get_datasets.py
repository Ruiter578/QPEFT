
import os
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
import torch

from .food101 import Food101

_DATASET_NAME = (
    'cifar',
    'caltech101',
    'dtd',
    'oxford_flowers102',
    'oxford_iiit_pet',
    'svhn',
    'sun397',
    'patch_camelyon',
    'eurosat',
    'resisc45',
    'diabetic_retinopathy',
    'clevr_count',
    'clevr_dist',
    'dmlab',
    'kitti',
    'dsprites_loc',
    'dsprites_ori',
    'smallnorb_azi',
    'smallnorb_ele',
)

_CLASSES_NUM = (100, 102, 47, 102, 37, 10, 397, 2, 10, 45, 5, 8, 6, 6, 4, 16, 16, 18, 9)

NAME_CLS = {name: num for name, num in zip(_DATASET_NAME, _CLASSES_NUM)}

class GeneralDataset(ImageFolder):
    def __init__(self, root, train=True, transform=None, target_transform=None,**kwargs):
        self.dataset_root = root
        self.loader = default_loader
        self.target_transform = None
        self.transform = transform

        train_list_path = os.path.join(self.dataset_root, 'train800val200.txt')
        test_list_path = os.path.join(self.dataset_root, 'test.txt')

        self.samples = []
        if train:
            with open(train_list_path, 'r') as f:
                for line in f:
                    img_name = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    self.samples.append((os.path.join(root,img_name), label))
        else:
            with open(test_list_path, 'r') as f:
                for line in f:
                    img_name = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    self.samples.append((os.path.join(root,img_name), label))

class PreTransformedTestSet(ImageFolder):
    def __init__(self, root, transformed_root, train=False, transform=None, target_transform=None,**kwargs):
        assert not train
        self.dataset_root = root
        self.transformed_root = transformed_root
        self.loader = default_loader
        self.target_transform = None
        self.transform = transform

        train_list_path = os.path.join(self.dataset_root, 'train800val200.txt')
        test_list_path = os.path.join(self.dataset_root, 'test.txt')

        self.samples = []
        if train:
            with open(train_list_path, 'r') as f:
                for line in f:
                    img_name = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    self.samples.append((os.path.join(root,img_name), label))
        else:
            with open(test_list_path, 'r') as f:
                for line in f:
                    img_name = line.split(' ')[0]
                    label = int(line.split(' ')[1])
                    self.samples.append((os.path.join(root,img_name), label))

    def __getitem__(self, index):
        ori_path, _ = self.samples[index]
        transformed_path = ori_path.replace(self.dataset_root, self.transformed_root) + '.pt'
        assert os.path.exists(transformed_path)
        data = torch.load(transformed_path)
        sample, target = data['sample'], data['target']
        return sample, target

def build_dataset(is_train, args, folder_name=None):
    transform = build_transform(is_train, args)

    assert args.dataset in _DATASET_NAME
    nb_classes = NAME_CLS[args.dataset]
    data_path = os.path.join(args.data_path, args.dataset)
    assert os.path.exists(data_path), data_path

    dataset = GeneralDataset(data_path, train=is_train, transform=transform)

    return dataset, nb_classes

def build_pretransformed_testset(args, folder_name=None):
    transform = build_transform(False, args)

    assert args.dataset in _DATASET_NAME
    nb_classes = NAME_CLS[args.dataset]
    if args.inception:
        tf_data_root = args.data_path + '/testset_pretransformed_inception'
    else:
        tf_data_root = args.data_path + '/testset_pretransformed'
    assert os.path.exists(tf_data_root)
    
    data_path = os.path.join(args.data_path, args.dataset)
    assert os.path.exists(data_path)

    tf_data_path = os.path.join(tf_data_root, args.dataset)
    assert os.path.exists(tf_data_path)

    dataset = PreTransformedTestSet(data_path, transformed_root=tf_data_path, train=False, transform=transform)

    return dataset, nb_classes

def build_transform(is_train, args):
    if args.inception:
        _mean = IMAGENET_INCEPTION_MEAN
        _std = IMAGENET_INCEPTION_STD
    else:
        _mean = IMAGENET_DEFAULT_MEAN
        _std = IMAGENET_DEFAULT_STD
        
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=_mean, std=_std)])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=_mean, std=_std)])
    return transform
# def build_transform(is_train, args):
#     if not args.no_aug and is_train and args.mode != 'search':
#         transform = create_transform(
#             input_size=args.input_size,
#             is_training=True,
#             color_jitter=args.color_jitter,
#             auto_augment=args.aa,
#             interpolation=args.train_interpolation,
#             re_prob=args.reprob,
#             re_mode=args.remode,
#             re_count=args.recount,
#         )
#         return transform

#     t = []
#     if args.direct_resize:
#         size = args.input_size
#     else:
#         size = int((256 / 224) * args.input_size)

#     t.append(
#         transforms.Resize((size,size), interpolation=3)  # to maintain same ratio w.r.t. 224 images
#     )
#     t.append(transforms.CenterCrop(args.input_size))

#     t.append(transforms.ToTensor())
#     if args.inception:
#         t.append(transforms.Normalize(IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD))
#     else:
#         t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
#     return transforms.Compose(t)


def build_full_datasets(is_train, args):
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    from .crop import RandomResizedCrop

    if args.inception:
        _mean = IMAGENET_INCEPTION_MEAN
        _std = IMAGENET_INCEPTION_STD
    else:
        _mean = IMAGENET_DEFAULT_MEAN
        _std = IMAGENET_DEFAULT_STD
        
    transform_train = transforms.Compose([  # following DyT.
        RandomResizedCrop(224, interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mean, std=_std)])
    transform_val = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mean, std=_std)])
    if args.dataset == 'cifar100_full':
        dataset_train = datasets.CIFAR100(os.path.join('./data/complete_datasets', 'cifar100'), transform=transform_train, train=True, download=True)
        dataset_val = datasets.CIFAR100(os.path.join('./data/complete_datasets', 'cifar100'), transform=transform_val, train=False, download=True)
        nb_classes = 100
    elif args.dataset == 'svhn_full':
        dataset_train = datasets.SVHN(os.path.join('./data/complete_datasets', 'svhn'), transform=transform_train, download=True)
        dataset_val = datasets.SVHN(os.path.join('./data/complete_datasets', 'svhn'), transform=transform_val, download=True)
        nb_classes = 10
    elif args.dataset == 'food101_full':
        dataset_train = Food101(os.path.join('./data/complete_datasets', 'food101'), split='train', transform=transform_train, download=True)
        dataset_val = Food101(os.path.join('./data/complete_datasets', 'food101'), split='test', transform=transform_val, download=True)
        nb_classes = 101
        
    else:
        raise ValueError

    return dataset_train, dataset_val, nb_classes