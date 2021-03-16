import itertools

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torchvision.models.detection.faster_rcnn import (FastRCNNPredictor,
                                                      fasterrcnn_resnet50_fpn)

from argparse_utils import from_argparse_args
from metrics import mean_average_precision
import torch.nn.functional as F

class ContextRCNN(pl.LightningModule):
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
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad_(False)

        def hook(module, input):
            pooled = F.max_pool2d(input[0], kernel_size=7).squeeze(-1).squeeze(-1)
            print(pooled.size())
            return input
        
        def hook2(module, input):
            features = input[0]
            print(features.size())
            return input
        self.net.roi_heads.box_head.register_forward_pre_hook(hook)
        self.net.box_roi_pool.register_forward_pre_hook(hook2)

    def training_step(self, batch, batch_idx):
        loss = torch.tensor(0.)

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
        mAP = mean_average_precision(preds, targets, 0.5, self.device)

        self.log("val_map", mAP, on_epoch=True, logger=True)

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
