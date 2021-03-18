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

from cct_annotation_handler import CCTAnnotationHandler
from cct_datamodule import collate_fn
from faster_rcnn import MyFasterRCNN


class CCTDatasetByLocation(Dataset):
    def __init__(self, handler, split, location):
        super().__init__()
        self.handler = handler
        self.split = split
        self.location = location
        self.out_width = 640
        self.out_height = 640
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.25, 0.25, 0.25)
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


if __name__ == '__main__':
    ckpt_path = "best_faster_rcnn.ckpt"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    module = MyFasterRCNN.load_from_checkpoint(ckpt_path).to(device).eval()
    tmp = {}
    prepare_faster_rcnn(module, tmp)

    handler = CCTAnnotationHandler(root_dir="./dataset/cct")
    split_types = ["train", "val"]
    batch_size = 32
    num_workers = 4
    per_box_features = {}

    for split in split_types:
        per_box_features[split] = {}

        for loc in handler._locs[split]:
            pbf = []  # per-box features
            per_box_features[loc] = pbf

            dataset = CCTDatasetByLocation(handler, "train", loc)
            dataloader = DataLoader(
                dataset, batch_size=batch_size,
                shuffle=False, num_workers=num_workers,
                collate_fn=collate_fn)

            for i, batch in enumerate(tqdm(dataloader, desc=f"{split}/loc[{loc}]")):
                images, date_captured = batch
                images = [x.to(device) for x in images]

                with torch.no_grad():
                    preds = module.net(images)
                    boxes = [x["boxes"] for x in preds]
                    boxes_per_image = [len(x["boxes"]) for x in preds]
                    feat = module.net.roi_heads.box_roi_pool(
                        tmp["features"], boxes, tmp["image_sizes"])
                    feat = F.max_pool2d(feat, kernel_size=7).squeeze(-1).squeeze(-1)
                    feat = feat.split(boxes_per_image)
                    spatiotemporal = get_spatiotemporal(
                        images, boxes, date_captured)
                    feat_with_st = []
                    for f, st in zip(feat, spatiotemporal):
                        feat_with_st.append(torch.hstack([f, st]))
                    pbf.append(feat_with_st)

            pbf = list(itertools.chain(*pbf))
            pbf = torch.cat(pbf)
            per_box_features[split][loc] = pbf.cpu()

    torch.save(per_box_features, "memory_long.pt")
