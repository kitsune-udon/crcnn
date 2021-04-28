import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.coco import CocoDetection
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

import globals
from argparse_utils import from_argparse_args


def collate_fn(batch):
    return list(zip(*batch))


class CocoTransform:
    def __init__(self, split="train"):
        self.img_transform = Compose([
            Resize((globals.image_height, globals.image_width)),
            ToTensor(),
            Normalize(mean=globals.image_mean,
                      std=globals.image_std)
        ])

    def __call__(self, img, target):
        def scaling(box):
            eps = 0.001
            x_scaler = globals.image_width / img.width
            y_scaler = globals.image_height / img.height
            x0, y0 = box[0] * x_scaler, box[1] * y_scaler
            w, h = box[2] * x_scaler + eps, box[3] * y_scaler + eps
            return [x0, y0, x0 + w, y0 + h]

        boxes = [scaling(x["bbox"]) for x in target]
        labels = [x["category_id"] + 1 for x in target]
        t = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape((-1, 4)), "labels": torch.tensor(
            labels, dtype=torch.int64)}

        return self.img_transform(img), t


class CocoDataModule(pl.LightningDataModule):
    def __init__(self, train_batch_size=8, val_batch_size=8, num_workers=0):
        super().__init__()

        self.dataset_root = globals.coco_dataset_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.train_transforms = CocoTransform(split="train")
        self.val_transforms = CocoTransform(split="val")

    def prepare_data(self, *args, **kwargs):
        pass

    def train_dataloader(self, *args, **kwargs):
        coco_train = CocoDetection(
            root=os.path.join(self.dataset_root, "train2017"),
            annFile=os.path.join(
                self.dataset_root, "annotations", "instances_train2017.json"),
            transforms=self.train_transforms
        )

        return DataLoader(coco_train,
                          shuffle=True,
                          num_workers=self.num_workers,
                          batch_size=self.train_batch_size,
                          collate_fn=collate_fn)

    def val_dataloader(self, *args, **kwargs):
        coco_val = CocoDetection(
            root=os.path.join(self.dataset_root, "val2017"),
            annFile=os.path.join(
                self.dataset_root, "annotations", "instances_val2017.json"),
            transforms=self.val_transforms
        )

        return DataLoader(coco_val,
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
