import pytorch_lightning as pl
import torch
import torchvision
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.ops import box_iou

from argparse_utils import from_argparse_args


def _evaluate_iou(target, pred):
    if pred["boxes"].shape[0] == 0:
        return torch.tensor(0., device=pred["boxes"].device)
    
    return box_iou(target["boxes"], pred["boxes"]).diag().mean()

def _movilenet_v2_backbone():
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512), ), aspect_ratios=((0.5, 1., 2.), ))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
    model = FasterRCNN(backbone, num_classes=16, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)

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

        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outs = self.net(images)
        iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, outs)])

        return {'val_iou': iou}

    def validation_epoch_end(self, outputs):
        avg_iou = torch.stack([o["val_iou"] for o in outputs]).mean()

        logs = {
            'val_iou': avg_iou,
        }

        results = {'log': logs}

        return results

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
