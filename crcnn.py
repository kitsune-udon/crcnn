import itertools
import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from argparse_utils import from_argparse_args
from faster_rcnn import MyFasterRCNN
from metrics import mean_average_precision


class AttentionBlock(nn.Module):
    def __init__(self, feature_depth=256, spatiotemporal_size=9):
        super().__init__()
        self.feature_depth = feature_depth
        q_in_dim = self.feature_depth
        qk_out_dim = self.feature_depth
        kv_in_dim = self.feature_depth + spatiotemporal_size
        v_out_dim = self.feature_depth
        f_out_dim = self.feature_depth
        self.q = nn.Linear(q_in_dim, qk_out_dim)
        self.k = nn.Linear(kv_in_dim, qk_out_dim)
        self.v = nn.Linear(kv_in_dim, v_out_dim)
        self.f = nn.Linear(v_out_dim, f_out_dim)

    def forward(self, A, B):
        q = self.q(F.max_pool2d(A, kernel_size=7).squeeze(-1).squeeze(-1))
        k = self.k(B)
        v = self.v(B)
        # normalize is necessary?
        temparature = 0.1
        qk = q.matmul(k.transpose(1, 0))
        w = F.softmax(
            qk / (temparature * math.sqrt(self.feature_depth)), dim=0)
        f = self.f(w.matmul(v))

        return f


def prepare_faster_rcnn(crcnn_module, tmp):
    def hook(module, input):
        tmp["proposals"] = input[1]

        return input

    def hook2(module, input):
        boxes_per_images = [x.shape[0] for x in tmp["proposals"]]
        features = input[0].split(boxes_per_images)

        r = []
        for feature, memory_long in zip(features, tmp["memory_long"]):
            context = crcnn_module.attention(feature, memory_long)
            context = context.unsqueeze(-1).unsqueeze(-1)
            feature_with_context = feature + context
            r.append(feature_with_context)

        return torch.cat(r)

    crcnn_module.net.roi_heads.box_roi_pool.register_forward_pre_hook(hook)
    crcnn_module.net.roi_heads.box_head.register_forward_pre_hook(hook2)


class ContextRCNN(pl.LightningModule):
    def __init__(self,
                 *args,
                 learning_rate=None,
                 weight_decay=None,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.net = MyFasterRCNN.load_from_checkpoint(
            "best_faster_rcnn.ckpt").net
        self.net.eval()
        self.attention = AttentionBlock()
        self._tmp = {}
        prepare_faster_rcnn(self, self._tmp)

    def training_step(self, batch, batch_idx):
        images, targets, memory_long = batch
        self._tmp["memory_long"] = [x.to(self.device) for x in memory_long]
        loss_dict = self.net(images, targets)
        loss = sum([loss for loss in loss_dict.values()])

        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets, memory_long = batch
        self._tmp["memory_long"] = [x.to(self.device) for x in memory_long]
        with torch.no_grad():
            preds = self.net(images)

        return {'preds': preds, 'targets': targets}

    def validation_epoch_end(self, outputs):
        preds = itertools.chain(*[o["preds"] for o in outputs])
        targets = itertools.chain(*[o["targets"] for o in outputs])
        mAP = mean_average_precision(preds, targets, 0.5, self.device)

        self.log("val_map", mAP, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.attention.parameters(),
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
