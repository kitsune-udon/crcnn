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

# Current memory bank of my context r-cnn implementation is variable length.
# So, Batch-Normalization is not for my context r-cnn implementation,
# it could be applied to a fix length memory bank version.
class ContextProjection(nn.Module):
    def __init__(self, in_dim, out_dim, normalize_output=True, use_batchnorm=False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        if use_batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.normalize_output = normalize_output
        self.use_batchnorm = use_batchnorm

    def forward(self, x):
        if self.use_batchnorm:
            x = self.relu(self.bn(self.linear(x)))
        else:
            x = self.relu(self.linear(x))
        if self.normalize_output:
            return F.normalize(x, p=2, dim=-1)
        else:
            return x

# should use dropouts for generalization?
class AttentionBlock(nn.Module):
    def __init__(self,
                 q_in_dim=256,
                 qk_out_dim=512,
                 v_out_dim=512,
                 spatiotemporal_size=9,
                 temparature=0.1):
        super().__init__()
        self.temparature = temparature
        self.q_in_dim = q_in_dim
        kv_in_dim = q_in_dim + spatiotemporal_size
        self.q = ContextProjection(q_in_dim, qk_out_dim)
        self.k = ContextProjection(kv_in_dim, qk_out_dim)
        self.v = ContextProjection(kv_in_dim, v_out_dim)
        self.f = ContextProjection(v_out_dim, q_in_dim, normalize_output=False)

    def forward(self, A, B):
        scaler = 1. / (self.temparature * math.sqrt(self.q_in_dim))
        q, k, v = self.q(A), self.k(B), self.v(B)
        qk = torch.einsum("ij,kj->ik", q, k)
        w = F.softmax(qk * scaler, dim=-1)
        f = self.f(w.matmul(v))

        return f


def prepare_faster_rcnn(crcnn_module, tmp):
    def hook(module, input):
        tmp["proposals"] = input[1]

        return input

    def hook2(module, input):
        boxes_per_images = [x.shape[0] for x in tmp["proposals"]]
        features = input[0]
        features = features.split(boxes_per_images)

        r = []
        for feature, memory_long in zip(features, tmp["memory_long"]):
            feature_pooled = F.max_pool2d(
                feature, kernel_size=7).squeeze(-1).squeeze(-1)
            context = crcnn_module.attention(feature_pooled, memory_long)
            feature_with_context = feature + \
                context.unsqueeze(-1).unsqueeze(-1)
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
        params = [{'params': self.attention.parameters()},
                  {'params': self.net.roi_heads.box_head.parameters()}] # need?
        optimizer = AdamW(params,
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
