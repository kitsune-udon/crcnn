import itertools
import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

import globals
from argparse_utils import from_argparse_args
from faster_rcnn import MyFasterRCNN
from metrics import mean_average_precision


class ContextProjection(nn.Module):
    def __init__(self, in_dim, out_dim, normalize_output=True, use_batchnorm=True, use_relu=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU(inplace=True) if use_relu else nn.Identity()
        self.normalize_output = normalize_output

    def forward(self, x):
        x = self.relu(self.bn(self.linear(x)))
        if self.normalize_output:
            return F.normalize(x, p=2, dim=-1)
        else:
            return x


def pack_variable_tensors(tensors):
    lens = [x.shape[0] for x in tensors]
    return torch.cat(tensors), lens


class AttentionBlock(nn.Module):  # should use dropouts for generalization?
    def __init__(self,
                 q_in_dim=256,
                 qk_out_dim=512,
                 v_out_dim=512,
                 spatiotemporal_size=9,
                 temparature=0.01):
        super().__init__()
        self.temparature = temparature
        self.q_in_dim = q_in_dim
        kv_in_dim = q_in_dim + spatiotemporal_size
        self.q = ContextProjection(q_in_dim, qk_out_dim)
        self.k = ContextProjection(kv_in_dim, qk_out_dim)
        self.v = ContextProjection(kv_in_dim, v_out_dim)
        self.f = ContextProjection(v_out_dim, q_in_dim, normalize_output=False)

    def forward(self, A, B):
        # A: List[Tensor(n_features, feature_size)]
        # B: List[Tensor(n_features_of_memory, feature_size_of_memory)]
        A0, A_lens = pack_variable_tensors(A)
        B0, B_lens = pack_variable_tensors(B)
        scaler = 1. / (self.temparature * math.sqrt(self.q_in_dim))
        q, k, v = self.q(A0), self.k(B0), self.v(B0)
        q, k, v = q.split(A_lens), k.split(B_lens), v.split(B_lens)
        qk = [torch.einsum("ij,kj->ik", q_, k_) for q_, k_ in zip(q, k)]
        w = [F.softmax(qk_ * scaler, dim=-1) for qk_ in qk]
        wv = [torch.einsum("ik,kj->ij", w_, v_) for w_, v_ in zip(w, v)]
        wv_packed, wv_lens = pack_variable_tensors(wv)
        f = self.f(wv_packed).split(wv_lens)

        return f  # List[Tensor(n_features, feature_size)]


def prepare_faster_rcnn(crcnn_module, tmp):
    def hook(module, input):
        tmp["proposals"] = input[1]

        return input

    def hook2(module, input):
        def pool(x):
            return F.avg_pool2d(x, kernel_size=7).squeeze(-1).squeeze(-1)

        def features_with_context(features, contexts):
            return [f + c.unsqueeze(-1).unsqueeze(-1) for f, c in zip(features, contexts)]

        boxes_per_images = [x.shape[0] for x in tmp["proposals"]]
        features = input[0]
        assert features.shape[1] == globals.feature_size
        features = features.split(boxes_per_images)
        features_pooled = [pool(f) for f in features]
        contexts = crcnn_module.attention(features_pooled, tmp["memory_long"])

        return torch.cat(features_with_context(features, contexts))

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
            globals.faster_rcnn_ckpt_path).net
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
        params = [{'params': self.attention.parameters()}]

        if globals.update_box_head_params:
            params.append({'params': self.net.roi_heads.box_head.parameters()})

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
        parser.add_argument('--weight_decay', type=float, default=0.05)

        return parser


if __name__ == '__main__':
    x = [torch.rand(3, 5), torch.rand(2, 5), torch.rand(1, 5)]
    m = [torch.rand(1, 14), torch.rand(2, 14), torch.rand(3, 14)]
    block = AttentionBlock(q_in_dim=5, qk_out_dim=7, v_out_dim=7)
    c = block(x, m)
    print(c)
