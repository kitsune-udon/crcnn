from pytorch_lightning.callbacks import ModelCheckpoint

from cct_datamodule import CCTDataModule
from faster_rcnn import MyFasterRCNN
from train_utils import CopyBestCheckpoint, main

if __name__ == '__main__':
    checkpoint_callback = ModelCheckpoint(
        "./checkpoints",
        monitor='val_map', save_top_k=1, mode="max")
    dst = "./best_faster_rcnn.ckpt"
    copy_best_checkpoint_callback = CopyBestCheckpoint(
        dst, checkpoint_callback)
    callbacks = [checkpoint_callback, copy_best_checkpoint_callback]

    main(CCTDataModule, MyFasterRCNN, "training_stage1", callbacks=callbacks)
