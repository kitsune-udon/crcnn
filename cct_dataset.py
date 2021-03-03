import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class CCTDataset(Dataset):
    def __init__(self, dataset_root="./dataset/cct", split="train", small_image=True, transform=None):
        super().__init__()
        self.dataset_root = dataset_root
        self.split = split
        self.small_image = small_image
        self.transform = transform
        self._set_path()
        self._process()

    def _set_path(self):
        if self.split == "train":
            self.filepath = os.path.join(self.dataset_root, "eccv_18_annotation_files",
                                         "train_annotations.json")
        elif self.split == "val":
            self.filepath = os.path.join(self.dataset_root, "eccv_18_annotation_files",
                                         "cis_val_annotations.json")
        else:
            raise ValueError("unknown split mode")

    def _process(self):
        def annotated_images():
            imgs = {}  # key: id of an annotated image, value: annotation info struct
            for annot in self.annotations:
                # why no-bbox annotations exist?
                if not "bbox" in annot:
                    continue
                image_id = annot["image_id"]
                imgs.setdefault(image_id, [])
                imgs[image_id].append(annot)

            image_table = {}  # key: id of image, value: image info struct
            for img_info in self.images:
                image_table[img_info["id"]] = img_info

            # list of tuples : (image info struct, list of annotation info struct)
            r = []
            for i in list(imgs.items()):
                image_id, annot_list = i
                r.append((image_table[image_id], annot_list))

            return r
        
        def cat_trans():
            sorted_cats = sorted(map(lambda x: x["id"], self.categories))

            r = {}
            for i, e in enumerate(sorted_cats):
                r[e] = i

            return r


        with open(self.filepath, "r") as f:
            d = json.load(f)

        self.info = d["info"]
        self.categories = d["categories"]
        self.images = d["images"]
        self.annotations = d["annotations"]
        self.annotated_images = annotated_images()
        self.cat_trans = cat_trans()

    def __len__(self):
        return len(self.annotated_images)

    def __getitem__(self, index):
        def get_target(annot_list, image_info):
            w = image_info["width"]
            scale = 1

            if self.small_image:
                scale = 1024 / w

            boxes, labels = [], []
            for annot in annot_list:
                bbox = annot["bbox"]
                x1, y1 = bbox[0], bbox[1]
                x2, y2 = bbox[0] + bbox[2], bbox[1] + bbox[3]
                bbox = list(map(lambda x: scale * x, [x1, y1, x2, y2]))
                cid = self.cat_trans[annot["category_id"]]
                boxes.append(bbox)
                labels.append(cid)

            return {"boxes": torch.tensor(boxes, dtype=torch.float32), "labels": torch.tensor(labels, dtype=torch.int64)}

        image_info, annot_list = self.annotated_images[index]
        image_path = os.path.join(
            self.dataset_root, "eccv_18_all_images_sm", image_info["file_name"])
        img = Image.open(image_path).convert("RGB")

        target = get_target(annot_list, image_info)

        if self.transform:
            return self.transform(img), target
        else:
            return img, target


if __name__ == '__main__':
    cct = CCTDataset()
    img, target = cct[0]
    print(img)
    print(target)
    print(cct.categories)
    print(len(cct.categories))
