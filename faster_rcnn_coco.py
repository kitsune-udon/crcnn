import itertools

import pytorch_lightning as pl
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torchvision.models.detection.faster_rcnn import (FasterRCNN,
                                                      FastRCNNPredictor)

import globals
from argparse_utils import from_argparse_args
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from metrics import mean_average_precision


def generate_resnet_model(model_id):
    backbone = resnet_fpn_backbone(model_id, True, trainable_layers=5)
    n_classes = 91
    wh = [globals.image_height, globals.image_width]
    min_size, max_size = min(wh), max(wh)
    model = FasterRCNN(
        backbone,
        n_classes,
        min_size=min_size, max_size=max_size,
        image_mean=globals.image_mean,
        image_std=globals.image_std)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, n_classes + 1)

    return model


def generate_my_model():
    if globals.faster_rcnn_backbone == "resnet101":
        return generate_resnet_model("resnet101")
    elif globals.faster_rcnn_backbone == "resnet50":
        return generate_resnet_model("resnet50")
    else:
        raise ValueError(f"{globals.faster_rcnn_backbone} is not supported.")


class MyFasterRCNNCoco(pl.LightningModule):
    def __init__(self,
                 *args,
                 learning_rate=None,
                 weight_decay=None,
                 max_epochs=None,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.net = generate_my_model()

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.net(images, targets)
        loss = sum(list(loss_dict.values()))

        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        with torch.no_grad():
            preds = self.net(images)

        return {'preds': preds, 'targets': targets}

    def validation_epoch_end(self, outputs):
        preds = itertools.chain(*[o["preds"] for o in outputs])
        targets = itertools.chain(*[o["targets"] for o in outputs])
        mAP = mean_average_precision(
            preds, targets, 0.5, self.device, score_threshold=globals.mAP_score_threshold)

        self.log("val_map", mAP, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(),
                        lr=self.hparams.learning_rate,
                        momentum=0.9,
                        weight_decay=self.hparams.weight_decay)

        gamma = 10 ** -(2 / self.hparams.max_epochs)
        scheduler = StepLR(optimizer, 1, gamma=gamma)

        return [optimizer], [scheduler]

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return from_argparse_args(cls, args, **kwargs)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--learning_rate', type=float, default=0.003)
        parser.add_argument('--weight_decay', type=float, default=5e-4)

        return parser
