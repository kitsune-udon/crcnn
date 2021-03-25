from pytorch_lightning.callbacks import ModelCheckpoint

import globals
from cct_datamodule import CCTDataModule
from cct_dataset import CCTDataset
from crcnn import ContextRCNN
from train_utils import CopyBestCheckpoint, main

if __name__ == '__main__':
    checkpoint_callback = ModelCheckpoint(
        "./checkpoints",
        monitor='val_map', save_top_k=1, mode="max")
    dst = globals.crcnn_ckpt_path
    copy_best_checkpoint_callback = CopyBestCheckpoint(
        dst, checkpoint_callback)
    callbacks = [checkpoint_callback, copy_best_checkpoint_callback]

    CCTDataset.stage3 = True

    main(CCTDataModule, ContextRCNN, "training_stage3", callbacks=callbacks)
