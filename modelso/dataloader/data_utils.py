from common.utils import set_seed


def dataset_builder(args):
    set_seed(args.seed)  # fix random seed for reproducibility

    if args.dataset == 'miniimagenet':
        from modelso.dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'cub':
        from modelso.dataloader.cub import CUB as Dataset
    elif args.dataset == 'dog':
        from modelso.dataloader.dog import DOG as Dataset
    elif args.dataset == 'car':
        from modelso.dataloader.car import CAR as Dataset
    else:
        raise ValueError('Unkown Dataset')
    return Dataset
