import numpy as np
import sklearn.model_selection as ms
from torch.utils.data import DataLoader
from datautil.imgdata.imgdataload import ImageDataset
import datautil.imgdata.util as imgutil
from datautil.infdataloader import InfiniteDataLoader, FastDataLoader

def get_img_dataloader(args):
    rate = args.TRAIN.VALID_RATE
    trdatalist, tedatalist = [], []
    names = args.img_dataset
    for i in range(len(names)):
        if i in args.test_envs:
            tedatalist.append(ImageDataset(args.dataset, args.DATA.DATA_PATH,names[i], 
                                transform=imgutil.img_test(args.dg_aug)))
        else:
            tmpdatay = ImageDataset(args.dataset, args.DATA.DATA_PATH, names[i], 
                            transform=imgutil.image_train(args.dg_aug)).labels
            l = len(tmpdatay)
            if args.split_style == 'strat':
                lslist = np.arange(l)
                stsplit = ms.StratifiedShuffleSplit(
                    2, test_size= rate, train_size=1 - rate, random_state= args.SEED
                )
                stsplit.get_n_splits(lslist, tmpdatay)
                indextr, indexte = next(stsplit.split(lslist, tmpdatay))
            else:
                raise Exception('the split style is not strat')
            trdatalist.append(ImageDataset(args.dataset, args.DATA.DATA_PATH, 
                            names[i], transform=imgutil.image_train(args.dg_aug), 
                            indices=indextr))
            tedatalist.append(ImageDataset(args.dataset, args.DATA.DATA_PATH, 
                            names[i], transform=imgutil.img_test(args.dg_aug), 
                            indices=indexte))
    train_loader = [FastDataLoader(
        dataset = env,
        batch_size = args.DATA.BATCH_SIZE,
        num_workers = args.DATA.NUM_WORKERS
        ) for env in trdatalist]
    eval_loaders = [FastDataLoader(
        dataset = env, 
        batch_size = args.DATA.TEST_BATCH_SIZE, 
        num_workers = args.DATA.NUM_WORKERS)
        for env in trdatalist + tedatalist]
    return train_loader, eval_loaders


def get_tgt_img_dataloader(args):
    trdatalist, tedatalist = [], []
    names = args.img_dataset
    tedatalist.append(ImageDataset(args.dataset, args.DATA.DATA_PATH, names[args.target_env],
                    transform=imgutil.img_test(args.dg_aug), 
                    return_idx=True))
    trdatalist.append(ImageDataset(args.dataset, args.DATA.DATA_PATH, names[args.target_env],
                    transform=imgutil.image_train(args.dg_aug), 
                    return_idx=True))
    train_loader = [FastDataLoader(
        dataset = env, 
        batch_size = args.DATA.BATCH_SIZE, 
        num_workers = args.DATA.NUM_WORKERS)
        for env in trdatalist]
    test_loaders = [DataLoader(
        dataset = env, 
        batch_size = args.DATA.TEST_BATCH_SIZE, 
        num_workers = args.DATA.NUM_WORKERS,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True)
        for env in tedatalist]
    
    return train_loader, test_loaders

