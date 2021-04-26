import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets.coco import CocoDetection

import globals
from argparse_utils import from_argparse_args


class CocoDataModule(pl.LightningDataModule):
    def __init__(self, train_batch_size=8, val_batch_size=8, num_workers=0):
        super().__init__()

        self.dataset_root = globals.coco_dataset_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

    def prepare_data(self, *args, **kwargs):
        pass

    def train_dataloader(self, *args, **kwargs):
        coco_train = CocoDetection(
            root=os.path.join(self.dataset_root, "train2017"),
            annFile=os.path.join(
                self.dataset_root, "annotations", "instances_train2017.json")
        )

        return DataLoader(coco_train,
                          shuffle=True,
                          num_workers=self.num_workers,
                          batch_size=self.train_batch_size)

    def val_dataloader(self, *args, **kwargs):
        coco_val = CocoDetection(
            root=os.path.join(self.dataset_root, "val2017"),
            annFile=os.path.join(
                self.dataset_root, "annotations", "instances_val2017.json")
        )

        return DataLoader(coco_val,
                          shuffle=False,
                          num_workers=self.num_workers,
                          batch_size=self.val_batch_size)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--train_batch_size", type=int,
                            default=32, help="batch size for training")
        parser.add_argument("--val_batch_size", type=int,
                            default=32, help="batch size for validation")
        parser.add_argument("--num_workers", type=int, default=8,
                            help="number of processes for dataloader")

        return parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return from_argparse_args(cls, args, **kwargs)


if __name__ == '__main__':
    dm = CocoDataModule()
    ld = dm.train_dataloader()
    it = iter(ld)
    batch = next(it)
    print(batch)
