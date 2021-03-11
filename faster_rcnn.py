import itertools

import pytorch_lightning as pl
from torch.optim import AdamW
from torchvision.models.detection.faster_rcnn import (FastRCNNPredictor,
                                                      fasterrcnn_resnet50_fpn)

from argparse_utils import from_argparse_args
from metrics import mean_average_precision


class MyFasterRCNN(pl.LightningModule):
    def __init__(self,
                 *args,
                 learning_rate=None,
                 weight_decay=None,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        n_classes = 16
        self.net = fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = self.net.roi_heads.box_predictor.cls_score.in_features
        self.net.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, n_classes + 1)

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

        return {'preds': preds, 'targets': targets}

    def validation_epoch_end(self, outputs):
        preds = itertools.chain(*[o["preds"] for o in outputs])
        targets = itertools.chain(*[o["targets"] for o in outputs])
        mAP = mean_average_precision(preds, targets, 0.5)

        self.log("val_map", mAP, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                          lr=self.hparams.learning_rate,
                          weight_decay=self.hparams.weight_decay)

        return [optimizer]

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return from_argparse_args(cls, args, **kwargs)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--weight_decay', type=float, default=0.01)

        return parser
