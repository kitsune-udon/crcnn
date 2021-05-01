import itertools
import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

import globals
from argparse_utils import from_argparse_args
from faster_rcnn import MyFasterRCNN
from metrics import mean_average_precision


def pack_variable_tensors(tensors):
    lens = [x.shape[0] for x in tensors]
    return torch.cat(tensors), lens


class AttentionBlock(nn.Module):  # should use dropouts for generalization?
    def __init__(self,
                 q_in_dim,
                 qk_out_dim,
                 v_out_dim,
                 n_heads,
                 temparature,
                 spatiotemporal_size=9):
        super().__init__()
        assert qk_out_dim % n_heads == 0
        assert v_out_dim % n_heads == 0
        self.n_heads = n_heads
        self.scaler = 1. / (temparature * math.sqrt(qk_out_dim // n_heads))
        kv_in_dim = q_in_dim + spatiotemporal_size
        self.q = nn.Linear(q_in_dim, qk_out_dim)
        self.k = nn.Linear(kv_in_dim, qk_out_dim)
        self.v = nn.Linear(kv_in_dim, v_out_dim)
        self.f = nn.Linear(v_out_dim, q_in_dim)

    def forward(self, A, B, n_boxes_per_images):
        # A: Tensor(n_features, feature_size)
        # B: List[Tensor(n_features_of_memory, feature_size_of_memory)]
        def split_by_head(x):
            return rearrange(x, "i (h j) -> i h j", h=self.n_heads)

        A_lens = n_boxes_per_images
        B0, B_lens = pack_variable_tensors(B)
        q, k, v = self.q(A), self.k(B0), self.v(B0)
        q, k, v = split_by_head(q), split_by_head(k), split_by_head(v)
        q, k, v = q.split(A_lens), k.split(B_lens), v.split(B_lens)
        qk = [torch.einsum("ihj,khj->ihk", q_, k_) for q_, k_ in zip(q, k)]
        w = [F.softmax(qk_ * self.scaler, dim=-1) for qk_ in qk]
        wv = [torch.einsum("ihk,khj->ihj", w_, v_) for w_, v_ in zip(w, v)]
        wv = [rearrange(wv_, "i h j -> i (h j)") for wv_ in wv]
        f = self.f(torch.cat(wv))

        return f  # Tensor(n_features, feature_size)


class FeedForward(nn.Module):
    def __init__(self, in_dim, n_hidden):
        super().__init__()
        self.l1 = nn.Linear(in_dim, n_hidden)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(n_hidden, in_dim)

    def forward(self, x):
        return self.l2(self.relu(self.l1(x)))


class ContextBlender(nn.Module):
    def __init__(self,
                 n_attention_blocks,
                 n_attention_heads,
                 q_in_dim,
                 qk_out_dim,
                 v_out_dim,
                 ff_n_hidden,
                 temparature,
                 ):
        super().__init__()
        self.n_attention_blocks = n_attention_blocks
        self.attention_blocks = nn.ModuleList([AttentionBlock(
            q_in_dim, qk_out_dim, v_out_dim, n_attention_heads, temparature) for _ in range(n_attention_blocks)])
        self.ffs = nn.ModuleList([FeedForward(q_in_dim, ff_n_hidden)
                                 for _ in range(n_attention_blocks)])
        self.norms1 = nn.ModuleList([nn.LayerNorm(q_in_dim)
                                     for _ in range(n_attention_blocks)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(q_in_dim)
                                     for _ in range(n_attention_blocks)])
        self.ff_pre = FeedForward(q_in_dim, ff_n_hidden)
        self.ff_post = FeedForward(q_in_dim, ff_n_hidden)

    def forward(self, features, memory_long, n_boxes_per_images):
        z = features
        z = F.avg_pool2d(z, kernel_size=7).squeeze(-1).squeeze(-1)
        for i in range(self.n_attention_blocks):
            z = self.norms1[i](z + self.attention_blocks[i]
                               (z, memory_long, n_boxes_per_images))
            z = self.norms2[i](z + self.ffs[i](z))
        z = self.ff_post(z)
        z = z.unsqueeze(-1).unsqueeze(-1)

        return features + z


def prepare_faster_rcnn(crcnn_module, tmp):
    def hook(module, input):
        tmp["proposals"] = input[1]

        return input

    def hook2(module, input):
        features = input[0]
        assert features.shape[1] == globals.feature_size

        n_boxes_per_images = [x.shape[0] for x in tmp["proposals"]]

        return crcnn_module.context_blender(features, tmp["memory_long"], n_boxes_per_images)

    crcnn_module.net.roi_heads.box_roi_pool.register_forward_pre_hook(hook)
    crcnn_module.net.roi_heads.box_head.register_forward_pre_hook(hook2)


class ContextRCNN(pl.LightningModule):
    def __init__(self,
                 *args,
                 learning_rate=None,
                 weight_decay=None,
                 max_epochs=None,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.net = MyFasterRCNN.load_from_checkpoint(
            globals.faster_rcnn_ckpt_path).net
        self.context_blender = ContextBlender(
            n_attention_blocks=globals.n_attention_blocks,
            n_attention_heads=globals.n_attention_heads,
            q_in_dim=globals.feature_size,
            qk_out_dim=globals.qk_out_dim,
            v_out_dim=globals.v_out_dim,
            ff_n_hidden=globals.ff_n_hidden,
            temparature=globals.attention_softmax_temparature
        )
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
        mAP = mean_average_precision(
            preds, targets, 0.5, self.device, score_threshold=globals.mAP_score_threshold)

        self.log("val_map", mAP, on_epoch=True, logger=True)

    def configure_optimizers(self):
        params = [
            {'params': self.context_blender.parameters()},
        ]
        if False:
            params.append({'params': self.net.roi_heads.box_head.parameters()})

        optimizer = SGD(
            params,
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


if __name__ == '__main__':
    x = [torch.rand(3, 5), torch.rand(2, 5), torch.rand(1, 5)]
    m = [torch.rand(1, 14), torch.rand(2, 14), torch.rand(3, 14)]
    block = AttentionBlock(q_in_dim=5, qk_out_dim=7, v_out_dim=7)
    c = block(x, m)
    print(c)
