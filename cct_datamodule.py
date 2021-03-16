import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from argparse_utils import from_argparse_args
from cct_dataset import CCTDataset


def collate_fn(batch):
    images, targets = [], []

    for x in batch:
        img, t = x
        images.append(img)
        targets.append(t)

    return images, targets


class CCTDataModule(pl.LightningDataModule):
    def __init__(self, dataset_root="./dataset/cct",
                 train_batch_size=8, val_batch_size=8, num_workers=0):
        super().__init__()

        self.dataset_root = dataset_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

    def prepare_data(self, *args, **kwargs):
        pass

    def train_dataloader(self, *args, **kwargs):
        cct_train = CCTDataset(
            dataset_root=self.dataset_root, split="train")
        return DataLoader(cct_train,
                          shuffle=True,
                          num_workers=self.num_workers,
                          batch_size=self.train_batch_size,
                          collate_fn=collate_fn)

    def val_dataloader(self, *args, **kwargs):
        cct_val = CCTDataset(
            dataset_root=self.dataset_root, split="val")
        return DataLoader(cct_val,
                          shuffle=False,
                          num_workers=self.num_workers,
                          batch_size=self.val_batch_size,
                          collate_fn=collate_fn)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--train_batch_size", type=int,
                            default=32, help="batch size for training")
        parser.add_argument("--val_batch_size", type=int,
                            default=32, help="batch size for validation")
        parser.add_argument("--num_workers", type=int, default=4,
                            help="number of processes for dataloader")
        parser.add_argument("--dataset_root", type=str,
                            default="./dataset/cct", help="root path of dataset")

        return parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return from_argparse_args(cls, args, **kwargs)


if __name__ == '__main__':
    dm = CCTDataModule()
    ld = dm.train_dataloader()
    it = iter(ld)
    batch = next(it)
    print(batch)
