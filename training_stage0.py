from coco_datamodule import CocoDataModule
from pytorch_lightning.callbacks import ModelCheckpoint

import globals
from coco_datamodule import CocoDataModule
from faster_rcnn_coco import MyFasterRCNNCoco
from train_utils import CopyBestCheckpoint, main

if __name__ == '__main__':
    checkpoint_callback = ModelCheckpoint(
        "./checkpoints",
        monitor='val_map', save_top_k=1, mode="max")
    dst = globals.pretrained_faster_rcnn_ckpt_path
    copy_best_checkpoint_callback = CopyBestCheckpoint(
        dst, checkpoint_callback)
    callbacks = [checkpoint_callback, copy_best_checkpoint_callback]

    main(CocoDataModule, MyFasterRCNNCoco, "training_stage0", callbacks=callbacks)
