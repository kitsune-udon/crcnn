import pytorch_lightning as pl
import torch
from numpy.random import randint
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.transforms import Compose, Normalize, ToTensor

from argparse_utils import from_argparse_args


class SimpleDataset(Dataset):
    def __init__(self, split="train", transform=None):
        super().__init__()
        self.split = split
        self.transform = transform
        self.length = 2000
        self.width = 320
        self.height = 320
        self.max_object = 1
        self.n_classes = 3
        self.color_table = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    def __len__(self):
        if self.split == "train":
            return 2000
        elif self.split == "val":
            return 200
        else:
            raise ValueError("invalid split mode")

    def __getitem__(self, index):
        def generate_rectangle():
            table = [32, 64, 128]
            offs = 128
            r_w, r_h = table[randint(3)], table[randint(3)]
            x, y = randint(self.width - offs), randint(self.height - offs)
            return [x, y, x + r_w, y + r_h]

        def generate_targets():
            rects, labels = [], []
            for _ in range(self.max_object):
                rect = generate_rectangle()
                label = randint(self.n_classes)
                rects.append(rect)
                labels.append(label)
            return rects, labels

        img = Image.new("RGB", (self.width, self.height), (128, 128, 128))
        draw = ImageDraw.Draw(img)
        rects, labels = generate_targets()

        for i in range(len(rects)):
            draw.rectangle(rects[i], fill=self.color_table[labels[i]], outline=(0, 0, 0), width=1)

        labels = list(map(lambda x: x + 1, labels))

        target = {"boxes": torch.tensor(
            rects, dtype=torch.float32).reshape((-1, 4)), "labels": torch.tensor(labels, dtype=torch.int64)}

        if self.transform:
            img = self.transform(img)

        return img, target


# List[Tuple[Image, Target]] -> Tuple[List[Image], List[Target]]
def collate_fn(batch):
    images, targets = [], []

    for x in batch:
        img, t = x
        images.append(img)
        targets.append(t)

    return images, targets


class SimpleDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_batch_size=8, val_batch_size=8, num_workers=0):
        super().__init__()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

        mean = (0.5, 0.5, 0.5)
        std = (0.25, 0.25, 0.25)

        self.transform = Compose([
            ToTensor(),
            Normalize(mean=mean,
                      std=std)
        ])

    def prepare_data(self, *args, **kwargs):
        pass

    def train_dataloader(self, *args, **kwargs):
        simple_train = SimpleDataset(split="train", transform=self.transform)
        return DataLoader(simple_train,
                          shuffle=True,
                          num_workers=self.num_workers,
                          batch_size=self.train_batch_size,
                          collate_fn=collate_fn)

    def val_dataloader(self, *args, **kwargs):
        simple_val = SimpleDataset(split="val", transform=self.transform)
        return DataLoader(simple_val,
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
        parser.add_argument("--num_workers", type=int, default=0,
                            help="number of processes for dataloader")

        return parser

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return from_argparse_args(cls, args, **kwargs)


if __name__ == '__main__':
    dataset = SimpleDataset()
    img, target = dataset[0]
    img.save("sample.png")
    print(target)
    #dm = SimpleDataModule()
    #ld = dm.train_dataloader()
    #it = iter(ld)
    #batch = next(it)
    #print(batch)
