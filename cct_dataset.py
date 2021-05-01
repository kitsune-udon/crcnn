from datetime import datetime

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.transforms import (Compose, Normalize, Resize,
                                               ToTensor)

import globals
from cct_annotation_handler import CCTAnnotationHandler


class CCTDataset(Dataset):
    stage3 = False
    use_horizontal_flip = True

    def __init__(self, split="train"):
        super().__init__()
        self.dataset_root = globals.dataset_root
        self.split = split
        self.out_width = globals.image_width
        self.out_height = globals.image_height
        self.mean = globals.image_mean
        self.std = globals.image_std
        self.transform = Compose([
            Resize((self.out_height, self.out_width)),
            ToTensor(),
            Normalize(mean=self.mean,
                      std=self.std)
        ])
        self.handler = CCTAnnotationHandler()
        if CCTDataset.stage3:
            self._memory_long_table = torch.load(
                "memory_long.pt", map_location=torch.device('cpu'))
            self._memory_long_date_table = torch.load(
                "memory_long_date.pt", map_location=torch.device('cpu'))

    def __len__(self):
        return len(self.handler.annotated_images[self.split])

    def __getitem__(self, index):
        def flip_target(target):
            src = target["boxes"]
            dst = torch.clone(src)
            w = globals.image_width
            t = src[:, 2] - src[:, 0]
            dst[:, 0] = w - src[:, 0] - t
            dst[:, 2] = w - src[:, 2] + t
            return {"boxes": dst, "labels": target["labels"]}

        def flip_image(img):
            return torch.flip(img, [2])

        def flip_memory_long(memory_long):
            x = torch.clone(memory_long)
            x[:, 5] = 1 - x[:, 5]  # 5 is index of x-center value
            return x

        image_info, annot_list = self.handler.annotated_images[self.split][index]
        image_path = self.handler.get_image_path(image_info["file_name"])
        img = Image.open(image_path).convert("RGB")
        target = self._get_target(annot_list, image_info)
        img = self.transform(img)

        if CCTDataset.use_horizontal_flip and self.split == "train":
            do_flip = np.random.rand() < globals.horizontal_flip_rate
        else:
            do_flip = False

        if do_flip:
            img, target = flip_image(img), flip_target(target)

        if CCTDataset.stage3:
            memory_long = self._get_memory_long(image_info)
            if do_flip:
                memory_long = flip_memory_long(memory_long)
            return img, target, memory_long
        else:
            return img, target

    def _get_target(self, annot_list, image_info):
        w = image_info["width"]
        h = image_info["height"]
        sw = self.out_width / w
        sh = self.out_height / h

        boxes, labels = [], []
        for annot in annot_list:
            bbox = annot["bbox"]
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = bbox[0] + bbox[2], bbox[1] + bbox[3]
            bbox = [sw * x1, sh * y1, sw * x2, sh * y2]
            cid = self.handler.cat_trans[self.split][annot["category_id"]]
            boxes.append(bbox)
            labels.append(cid)

        return {"boxes": torch.tensor(boxes, dtype=torch.float32), "labels": torch.tensor(labels, dtype=torch.int64)}

    def _get_memory_long(self, image_info):
        def clip_memory_long(memory_long, memory_long_date, date_captured):
            begin, end = [date_captured +
                          x for x in globals.memory_long_interval]
            idx = (begin < memory_long_date) * (memory_long_date < end)
            return memory_long[idx]

        loc = image_info["location"]
        date_captured = datetime.fromisoformat(
            image_info["date_captured"]).timestamp()
        memory_long = self._memory_long_table[self.split][loc]
        memory_long_date = self._memory_long_date_table[self.split][loc]
        memory_long = clip_memory_long(
            memory_long, memory_long_date, date_captured)

        if len(memory_long) == 0:
            memory_long = torch.zeros(1, memory_long.shape[1])

        return memory_long


if __name__ == '__main__':
    CCTDataset.stage3 = True
    cct = CCTDataset()
    total_size = len(cct)
    print(f"total: {total_size}")
    lens = [cct[i][2].size()[0] for i in range(100)]
    max_len = max(lens)
    print(f"max: {max_len}")
