import pytorch_lightning as pl
from torch.utils.data import DataLoader

import globals
from argparse_utils import from_argparse_args
from cct_dataset import CCTDataset


def collate_fn(batch):
    return list(zip(*batch))


class CCTDataModule(pl.LightningDataModule):
    def __init__(self, train_batch_size=8, val_batch_size=8, num_workers=0):
        super().__init__()

        self.dataset_root = globals.dataset_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

    def prepare_data(self, *args, **kwargs):
        pass

    def train_dataloader(self, *args, **kwargs):
        cct_train = CCTDataset(split="train")

        return DataLoader(cct_train,
                          shuffle=True,
                          num_workers=self.num_workers,
                          batch_size=self.train_batch_size,
                          collate_fn=collate_fn)

    def val_dataloader(self, *args, **kwargs):
        cct_val = CCTDataset(split="val")

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
