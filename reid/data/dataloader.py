from __future__ import print_function
import os.path as osp
import numpy as np
from torch.utils.data import DataLoader
from reid.dataset import get_sequence
from reid.data import seqtransforms
from reid.data import SeqPreprocessor
from reid.data.sampler import RandomIdentitySampler

def get_data(dataset_name, split_id, data_dir, batch_size, seq_len, seq_srd, workers,
             num_instances):

    root = osp.join(data_dir, dataset_name)

    dataset = get_sequence(dataset_name, root, split_id=split_id,
                           seq_len=seq_len, seq_srd=seq_srd, num_val=1, download=True)

    train_set = dataset.trainval
    num_classes = dataset.num_trainval_ids

    normalizer = seqtransforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    train_processor = SeqPreprocessor(train_set, dataset, seq_len, transform=seqtransforms.Compose([seqtransforms.RectScale(256, 128), seqtransforms.RandomHorizontalFlip(),
                                    seqtransforms.ToTensor(), normalizer]))

    val_processor = SeqPreprocessor(dataset.val, dataset, seq_len, transform=seqtransforms.Compose([seqtransforms.RectScale(256, 128),
                                       seqtransforms.ToTensor(), normalizer]))

    test_processor = SeqPreprocessor(list(set(dataset.query) | set(dataset.gallery)), dataset, seq_len, transform=seqtransforms.Compose([seqtransforms.RectScale(256, 128),
                                       seqtransforms.ToTensor(), normalizer]))


    if num_instances >0:
        train_loader = DataLoader(
        train_processor, batch_size=batch_size, num_workers=workers,  sampler=RandomIdentitySampler(train_set, num_instances),
        pin_memory=True)
    else:
        train_loader = DataLoader(train_processor, batch_size=batch_size, num_workers=workers, shuffle=True, pin_memory=True)


    val_loader = DataLoader(
        val_processor, batch_size=batch_size, num_workers=workers, shuffle=False,
        pin_memory=True)

    test_loader = DataLoader(
        test_processor, batch_size=batch_size, num_workers=workers, shuffle= False,
        pin_memory=True)

    return dataset, num_classes, train_loader, val_loader, test_loader