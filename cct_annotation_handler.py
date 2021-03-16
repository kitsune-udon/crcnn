import json
import os


class CCTAnnotationHandler():
    def __init__(self, root_dir="./dataset/cct", adjust_to_faster_rcnn=True):
        self.root_dir = root_dir
        self.split_types = ["train", "val"]
        self.adjust_to_faster_rcnn = adjust_to_faster_rcnn
        self._read_annotation_files()
        self._generate_categories_table()
        self._generate_annotated_images()

    def _read_annotation_files(self):
        self._images = {}
        self._annots = {}
        self._cats = {}

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

    def _generate_categories_table(self):
        self.cat_trans = {}
        self.cat_trans_inv = {}

        for split in self.split_types:
            sorted_cats = sorted(map(lambda x: x["id"], self._cats[split]))
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

    def get_image_path(self, filename):
        image_path = os.path.join(
            self.root_dir, "eccv_18_all_images_sm", filename)
        return image_path


if __name__ == '__main__':
    h = CCTAnnotationHandler()
    print(h.annotated_images["train"][0])
