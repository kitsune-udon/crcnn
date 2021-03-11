
import argparse
import itertools

import torch
from PIL import Image, ImageDraw, ImageFont
from pytorch_lightning.utilities.seed import seed_everything
from torchvision.transforms import Compose, Normalize, ToPILImage
from torchvision.transforms.transforms import Normalize

from crcnn import ContextRCNN
from simple_datamodule import SimpleDataModule
from cct_datamodule import CCTDataModule
from cct_dataset import CCTDataset

class InverseTransform():
    def __init__(self):
        self._transform = Compose([
            Normalize(mean=(0., 0., 0.), std=(4., 4., 4.)),
            Normalize(mean=(-0.5, -0.5, -0.5), std=(1., 1., 1.)),
            ToPILImage()])

    def transform(self, x):
        return self._transform(x)


def draw_prediction(img, pred, threshold, label_to_name=None):
    img = img.convert("RGBA")
    font = ImageFont.truetype("futura.ttf", 15)
    for i in range(len(pred["boxes"])):
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        box = pred["boxes"][i].tolist()
        label = pred["labels"][i].item()
        score = pred["scores"][i].item()
        color = (255, 0, 0, int(255 * score))
        if score > threshold:
            draw.rectangle(box, outline=color, width=1)
            if label_to_name:
                text = label_to_name[label]
            else:
                text = f"{label}"
            draw.text(box[0:2], text, fill=color, font=font)
        img = Image.alpha_composite(img, overlay)
    return img.convert("RGB")


def generate_tiles(images, n_col=3):
    assert n_col > 0
    n_images = len(images)
    assert n_images > 0

    n_row = (n_images + n_col - 1) // n_col
    tile_width = images[0].width
    tile_height = images[0].height
    width, height = tile_width * n_col + n_col - 1, tile_height * n_row + n_row - 1

    out_img = Image.new("RGB", (width, height), color=(0, 0, 0))

    for k, img in enumerate(images):
        i, j = k % n_col, k // n_col
        out_img.paste(img, (i * tile_width + i, j * tile_height + j))

    return out_img


def extract_images_and_preds(ckpt_path, max_images=30):
    dataloader = CCTDataModule().val_dataloader()
    inv = InverseTransform()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = ContextRCNN.load_from_checkpoint(ckpt_path)
    model.to(device)
    model.eval()

    count = 0
    images0, preds0 = [], []
    for batch in dataloader:
        images, _ = batch
        n_images = len(images)
        if count + n_images > max_images:
            break
        images = list(map(lambda x: x.to(device), images))
        with torch.no_grad():
            preds = model.net(images)
        images0.append(images)
        preds0.append(preds)
        count += n_images

    images1 = list(itertools.chain(*images0))
    preds1 = list(itertools.chain(*preds0))

    images2 = []
    for i in range(len(images1)):
        img = inv.transform(images1[i])
        images2.append(img)

    return images2, preds1


parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default=None,
                    help="path of the checkpoint")
parser.add_argument("--threshold", type=float, default=0.95,
                    help="threshold of detection score")
parser.add_argument("--max_images", type=int, default=30,
                    help="number of output images")
args = parser.parse_args()
if not args.ckpt:
    raise ValueError("--ckpt option is not specified")

seed_everything(0)

images, preds = extract_images_and_preds(args.ckpt, args.max_images)
dataset = CCTDataset(out_width=320, out_height=320)
label_to_name = {k: v["name"] for k, v in dataset.cat_trans_inv.items()}

rendered = []
for i in range(len(images)):
    img = draw_prediction(images[i], preds[i], args.threshold, label_to_name=label_to_name)
    rendered.append(img)

out_img = generate_tiles(rendered)
out_img.save("prediction_result.png")