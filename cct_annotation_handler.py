import json
import os
from datetime import datetime

import globals


class CCTAnnotationHandler():
    def __init__(self, adjust_to_faster_rcnn=True):
        self.root_dir = globals.dataset_root
        self.split_types = ["train", "val"]
        self.adjust_to_faster_rcnn = adjust_to_faster_rcnn
        self._read_annotation_files()
        self._generate_categories_table()
        self._generate_annotated_images()
        self._generate_locations()
        self._generate_images_by_location()

    def _read_annotation_files(self):
        self._images = {}
        self._annots = {}
        self._cats = {}

        if globals.dataset_name == "cct_small":
            for split in self.split_types:
                if split == "train":
                    fp = os.path.join(self.root_dir, "eccv_18_annotation_files",
                                      "train_annotations.json")
                elif split == "val":
                    fp = os.path.join(self.root_dir, "eccv_18_annotation_files",
                                      "cis_val_annotations.json")
                else:
                    raise ValueError("unknown split mode")

                with open(fp, "r") as f:
                    d = json.load(f)

                self._images[split] = d["images"]
                self._annots[split] = d["annotations"]
                self._cats[split] = d["categories"]
        elif globals.dataset_name == "cct_large":
            def extract_images(all_images, locs):
                def to_int(x):
                    x["location"] = int(x["location"])
                    return x
                return [to_int(x) for x in all_images if int(x["location"]) in locs and x["date_captured"] != "11 11"]

            def extract_annots(all_annots, images):
                all_image_ids = [x["id"] for x in images]
                flags = {k: True for k in all_image_ids}
                return [annot for annot in all_annots if flags.get(annot["image_id"], False)]

            fp = os.path.join(
                self.root_dir, "CaltechCameraTrapsSplits_v0.json")
            with open(fp, "r") as f:
                d = json.load(f)

            locs = {}
            for split in self.split_types:
                locs[split] = [int(x) for x in d["splits"][split]]

            fp = os.path.join(self.root_dir, "caltech_images_20210113.json")
            with open(fp, "r") as f:
                d = json.load(f)

            for split in self.split_types:
                self._images[split] = extract_images(d["images"], locs[split])

            fp = os.path.join(self.root_dir, "caltech_bboxes_20200316.json")
            with open(fp, "r") as f:
                d = json.load(f)

            for split in self.split_types:
                self._cats[split] = d["categories"]
                self._annots[split] = extract_annots(
                    d["annotations"], self._images[split])
        else:
            raise ValueError(f"{globals.dataset_name} is unknown.")

    def _generate_categories_table(self):
        self.cat_trans = {}
        self.cat_trans_inv = {}

        for split in self.split_types:
            sorted_cats = sorted([x["id"] for x in self._cats[split]])
            id_to_name = {x["id"]: x for x in self._cats[split]}

            t = {}
            t_inv = {}
            for i, e in enumerate(sorted_cats):
                k = i + 1 if self.adjust_to_faster_rcnn else i
                t[e] = k
                t_inv[k] = id_to_name[e]

            self.cat_trans[split] = t
            self.cat_trans_inv[split] = t_inv

    def _generate_annotated_images(self):
        self.annotated_images = {}

        for split in self.split_types:
            imgs = {}  # key: id of an annotated image, value: annotation info struct
            for annot in self._annots[split]:
                # why no-bbox annotations exist?
                if not "bbox" in annot:
                    continue
                image_id = annot["image_id"]
                imgs.setdefault(image_id, [])
                imgs[image_id].append(annot)

            image_table = {}  # key: id of image, value: image info struct
            for img_info in self._images[split]:
                image_table[img_info["id"]] = img_info

            # list of tuples : (image info struct, list of annotation info struct)
            r = []
            for i in list(imgs.items()):
                image_id, annot_list = i
                r.append((image_table[image_id], annot_list))

            self.annotated_images[split] = r

    def _generate_locations(self):
        self._locs = {}

        for split in self.split_types:
            locs = sorted(set([x["location"] for x in self._images[split]]))
            self._locs[split] = locs

    def _generate_images_by_location(self):
        self._images_by_location = {}

        for split in self.split_types:
            self._images_by_location[split] = {}
            d = self._images_by_location[split]

            for img in self._images[split]:
                loc = img["location"]
                d.setdefault(loc, [])
                d[loc].append(img)

            for loc in self._locs[split]:
                d[loc] = sorted(
                    d[loc], key=lambda x: datetime.fromisoformat(x["date_captured"]))

    def get_image_path(self, filename):
        if globals.dataset_name == "cct_small":
            image_path = os.path.join(
                self.root_dir, "eccv_18_all_images_sm", filename)
        elif globals.dataset_name == "cct_large":
            image_path = os.path.join(
                self.root_dir, "cct_images", filename)
        else:
            raise ValueError(f"{globals.dataset_name} is unknown.")

        return image_path


if __name__ == '__main__':
    h = CCTAnnotationHandler()
    loc = h._locs["train"][0]
    print(h._locs)
    print(len(h._images_by_location["val"][0]))
