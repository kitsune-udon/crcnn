from collections import Counter

import pytorch_lightning as pl
import torch
import torchvision
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.ops import box_iou

from argparse_utils import from_argparse_args


def _evaluate_iou(target, pred):
    if pred["boxes"].shape[0] == 0:
        return torch.tensor(0., device=pred["boxes"].device)

    return box_iou(target["boxes"], pred["boxes"]).diag().mean()


def _mean_average_precision_sub(
    pred_image_indices,
    pred_probs,
    pred_labels,
    pred_bboxes,
    target_image_indices,
    target_labels,
    target_bboxes,
    iou_threshold,
):
    classes = torch.cat([pred_labels, target_labels]).unique()
    average_precisions = torch.zeros(len(classes))

    for class_idx, c in enumerate(classes):
        desc_indices = torch.argsort(pred_probs, descending=True)[
            pred_labels == c]

        if len(desc_indices) == 0:
            continue

        targets_per_images = Counter(
            [idx.item() for idx in target_image_indices[target_labels == c]])

        targets_assigned = {
            image_idx: torch.zeros(count, dtype=torch.bool) for image_idx, count in targets_per_images.items()
        }

        tps = torch.zeros(len(desc_indices))
        fps = torch.zeros(len(desc_indices))

        for i, pred_idx in enumerate(desc_indices):
            image_idx = pred_image_indices[pred_idx].item()
            gt_bboxes = target_bboxes[(
                target_image_indices == image_idx) & (target_labels == c)]
            ious = box_iou(torch.unsqueeze(pred_bboxes[pred_idx], dim=0), gt_bboxes)
            best_iou, best_target_idx = ious.squeeze(0).max(0) if len(gt_bboxes) > 0 else (0, -1)
            if best_iou > iou_threshold and not targets_assigned[image_idx][best_target_idx]:
                targets_assigned[image_idx][best_target_idx] = True
                tps[i] = 1
            else:
                fps[i] = 1

        tps_cum, fps_cum = torch.cumsum(tps, dim=0), torch.cumsum(fps, dim=0)
        precision = tps_cum / (tps_cum + fps_cum)
        num_targets = len(target_labels[target_labels == c])
        recall = tps_cum / num_targets if num_targets else tps_cum
        precision = torch.cat([reversed(precision), torch.tensor([1.])])
        recall = torch.cat([reversed(recall), torch.tensor([0.])])
        average_precision = - \
            torch.sum((recall[1:] - recall[:-1]) * precision[:-1])
        average_precisions[class_idx] = average_precision

    mean_average_precision = torch.mean(average_precisions)

    return mean_average_precision


def _mean_average_precision(preds, targets, iou_threshold):
    def op(x):
        return torch.cat(x).cpu()

    p_probs, p_labels, p_bboxes, p_image_indices = [], [], [], []
    for i, pred in enumerate(preds):
        p_probs.append(pred["scores"])
        p_labels.append(pred["labels"])
        p_bboxes.append(pred["boxes"])
        p_image_indices.append(torch.tensor([i] * len(pred["boxes"])))

    t_labels, t_bboxes, t_image_indices = [], [], []
    for i, target in enumerate(targets):
        t_labels.append(target["labels"])
        t_bboxes.append(target["boxes"])
        t_image_indices.append(torch.tensor([i] * len(target["boxes"])))

    p_probs, p_labels, p_bboxes, p_image_indices = list(
        map(op, [p_probs, p_labels, p_bboxes, p_image_indices]))

    t_labels, t_bboxes, t_image_indices = list(
        map(op, [t_labels, t_bboxes, t_image_indices]))

    #print(f"t_labels:{t_labels}")
    #print(f"t_bboxes:{t_bboxes}")
    #print(f"t_image_indices:{t_image_indices}")

    return _mean_average_precision_sub(p_image_indices, p_probs, p_labels, p_bboxes, t_image_indices, t_labels, t_bboxes, iou_threshold)


def _movilenet_v2_backbone():
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512), ), aspect_ratios=((0.5, 1., 2.), ))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
    model = FasterRCNN(backbone, num_classes=16,
                       rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)

    return model


class ContextRCNN(pl.LightningModule):
    def __init__(self,
                 *args,
                 learning_rate=None,
                 weight_decay=None,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.net = _movilenet_v2_backbone()

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.net(images, targets)
        loss = sum(list(loss_dict.values()))

        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.net(images)
        iou = torch.stack([_evaluate_iou(t, p)
                           for t, p in zip(targets, preds)])
        mean_average_precision = _mean_average_precision(preds, targets, 0.5)

        return {'val_iou': iou, 'val_map': mean_average_precision}

    def validation_epoch_end(self, outputs):
        avg_iou = torch.cat([o["val_iou"] for o in outputs]).mean()
        avg_map = torch.cat([o["val_map"].unsqueeze(0) for o in outputs]).mean()
        self.log("val_iou", avg_iou, on_epoch=True, logger=True)
        self.log("val_map", avg_map, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                          lr=self.hparams.learning_rate,
                          weight_decay=self.hparams.weight_decay)

        scheduler = StepLR(optimizer, 1, gamma=0.7)

        return [optimizer], [scheduler]

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return from_argparse_args(cls, args, **kwargs)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=0.01)

        return parser
