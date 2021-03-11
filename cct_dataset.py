import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class CCTDataset(Dataset):
    def __init__(self, dataset_root="./dataset/cct", split="train", resize=True, transform=None, out_width=None, out_height=None):
        super().__init__()
        self.dataset_root = dataset_root
        self.split = split
        self.resize = resize
        self.transform = transform
        self.out_width = out_width
        self.out_height = out_height
        if resize:
            assert out_width is not None and out_height is not None
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
            id_to_name = {x["id"]: x for x in self.categories}

            r = {}
            r_inv = {} 
            for i, e in enumerate(sorted_cats):
                r[e] = i + 1 # NOTE: backgrounds are classified to 0
                r_inv[i + 1] = id_to_name[e]

            return r, r_inv


        with open(self.filepath, "r") as f:
            d = json.load(f)

        self.info = d["info"]
        self.categories = d["categories"]
        self.images = d["images"]
        self.annotations = d["annotations"]
        self.annotated_images = annotated_images()
        self.cat_trans, self.cat_trans_inv = cat_trans()

    def __len__(self):
        return len(self.annotated_images)

    def __getitem__(self, index):
        def get_target(annot_list, image_info):
            w = image_info["width"]
            h = image_info["height"]
            if self.resize:
                sw = self.out_width / w
                sh = self.out_height / h

            boxes, labels = [], []
            for annot in annot_list:
                bbox = annot["bbox"]
                x1, y1 = bbox[0], bbox[1]
                x2, y2 = bbox[0] + bbox[2], bbox[1] + bbox[3]
                if self.resize:
                    bbox = [sw * x1, sh * y1, sw * x2, sh * y2]
                else:
                    bbox = [x1, y1, x2, y2]
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
    cct = CCTDataset(out_width=320, out_height=320)
    img, target = cct[0]
    print(img)
    print(target)
    print(cct.categories)
    print(len(cct.categories))
    print(cct.cat_trans_inv)
