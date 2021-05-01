import itertools
from datetime import datetime

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms.transforms import (Compose, Normalize, Resize,
                                               ToTensor)
from tqdm import tqdm

import globals
from cct_annotation_handler import CCTAnnotationHandler
from cct_datamodule import collate_fn
from faster_rcnn import MyFasterRCNN


class CCTDatasetByLocation(Dataset):
    def __init__(self, handler, split, location):
        super().__init__()
        self.handler = handler
        self.split = split
        self.location = location
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

    def __len__(self):
        return len(self.handler._images_by_location[self.split][self.location])

    def __getitem__(self, index):
        images = self.handler._images_by_location[self.split][self.location]
        image_info = images[index]

        image_path = self.handler.get_image_path(image_info["file_name"])
        img = Image.open(image_path).convert("RGB")
        date_captured = datetime.fromisoformat(image_info["date_captured"])

        return self.transform(img), date_captured


def collate_fn(batch):
    return list(zip(*batch))


def prepare_faster_rcnn(module, tmp):
    def hook(module, input):
        tmp["features"] = input[0]
        tmp["image_sizes"] = input[2]

        return input

    module.net.roi_heads.box_roi_pool.register_forward_pre_hook(hook)


def get_spatiotemporal(images, boxes, date_captured):
    device = images[0].device

    year = [((d.year - 1990) / (2030 - 1990)) for d in date_captured]
    month = [(d.month / 12) for d in date_captured]
    day = [(d.day / 31) for d in date_captured]
    hour = [(d.hour / 24) for d in date_captured]
    minute = [(d.minute / 60) for d in date_captured]
    temporal = [torch.tensor(x, device=device)
                for x in zip(year, month, day, hour, minute)]

    width = [x.shape[-1] for x in images]
    height = [x.shape[-2] for x in images]

    x_center = [(x[:, 0] + x[:, 2]) / 2 for x in boxes]
    y_center = [(x[:, 1] + x[:, 3]) / 2 for x in boxes]
    obj_width = [x[:, 2] - x[:, 0] for x in boxes]
    obj_height = [x[:, 3] - x[:, 1] for x in boxes]
    x_center = [xc / w for w, xc in zip(width, x_center)]
    y_center = [yc / h for h, yc in zip(height, y_center)]
    obj_width = [ow / w for w, ow in zip(width, obj_width)]
    obj_height = [oh / h for h, oh in zip(height, obj_height)]

    spatio = [torch.hstack(x).reshape(-1, 4)
              for x in zip(x_center, y_center, obj_width, obj_height)]

    r = []
    for t, s in zip(temporal, spatio):
        n_boxes = s.shape[0]
        if n_boxes > 0:
            x = torch.vstack([t] * n_boxes)
        else:
            x = torch.tensor([], device=device).reshape(0, 5)
        y = torch.hstack([x, s])
        r.append(y)

    return r


def get_boxes(preds):
    max_size = globals.memory_long_max_features_per_image  # None is acceptable

    all_boxes = []
    for pred in preds:
        idx = pred["scores"] > globals.memory_long_score_threshold
        scores = pred["scores"][idx]
        boxes = pred["boxes"][idx]
        idx = torch.argsort(scores, descending=True)
        boxes = boxes[idx][:max_size]
        all_boxes.append(boxes)

    return all_boxes, [len(x) for x in all_boxes]


def get_date(n_images, date_captured, boxes_per_image):
    date = [
        torch.tensor([date_captured[j].timestamp()]
                     * boxes_per_image[j], dtype=torch.float64)
        for j in range(n_images)
    ]

    return date


def concat(feat, spatiotemporal):
    x = []
    for f, st in zip(feat, spatiotemporal):
        x.append(torch.hstack([f, st]))

    return x


if __name__ == '__main__':
    ckpt_path = globals.faster_rcnn_ckpt_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    module = MyFasterRCNN.load_from_checkpoint(ckpt_path).to(device).eval()
    tmp = {}
    prepare_faster_rcnn(module, tmp)

    handler = CCTAnnotationHandler()
    split_types = ["train", "val"]
    batch_size = globals.training_stage2_batch_size
    num_workers = globals.training_stage2_num_workers
    per_box_features = {}
    date_captured_archive = {}

    for split in split_types:
        per_box_features[split] = {}
        date_captured_archive[split] = {}

        for loc in handler._locs[split]:
            pbf = []  # per-box features
            date_ex = []

            dataset = CCTDatasetByLocation(handler, split, loc)
            dataloader = DataLoader(
                dataset, batch_size=batch_size,
                shuffle=False, num_workers=num_workers,
                collate_fn=collate_fn)

            for batch in tqdm(dataloader, desc=f"{split}/loc[{loc}]"):
                images, date_captured = batch
                images = [x.to(device) for x in images]

                with torch.no_grad():
                    preds = module.net(images)

                    boxes, boxes_per_image = get_boxes(preds)
                    date = get_date(len(images), date_captured,
                                    boxes_per_image)

                    feat = module.net.roi_heads.box_roi_pool(
                        tmp["features"], boxes, tmp["image_sizes"])

                    #feat = module.net.roi_heads.box_head(feat)

                    if not len(feat) > 0:
                        continue

                    feat = F.avg_pool2d(
                        feat, kernel_size=7).squeeze(-1).squeeze(-1)
                    feat = feat.split(boxes_per_image)

                    spatiotemporal = get_spatiotemporal(
                        images, boxes, date_captured)

                    feat_with_st = concat(feat, spatiotemporal)

                    assert len(feat_with_st) == len(date)

                    pbf.append(feat_with_st)
                    date_ex.append(date)

            pbf = list(itertools.chain(*pbf))
            date_ex = list(itertools.chain(*date_ex))

            print(f"memory_long size:{len(pbf)}")
            if len(pbf) > 0:
                pbf = torch.cat(pbf)
            else:
                pbf = torch.zeros(0, globals.feature_size + 9)

            if len(date_ex) > 0:
                date_ex = torch.cat(date_ex)
            else:
                date_ex = torch.zeros(0, dtype=torch.float64)

            per_box_features[split][loc] = pbf.cpu()
            date_captured_archive[split][loc] = date_ex

    torch.save(per_box_features, globals.memory_long_path)
    torch.save(date_captured_archive, globals.memory_long_date_path)
