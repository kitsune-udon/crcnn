import shutil

from pytorch_lightning.callbacks import Callback, ModelCheckpoint

from cct_datamodule import CCTDataModule
from faster_rcnn import MyFasterRCNN
from train_utils import main


class CopyBestCheckpoint(Callback):
    def __init__(self, checkpoint_callback):
        self.checkpoint_callback = checkpoint_callback

    def on_fit_end(self, trainer, pl_module):
        src = self.checkpoint_callback.best_model_path
        dst = "./best_faster_rcnn.ckpt"
        shutil.copy(src, dst)


if __name__ == '__main__':
    checkpoint_callback = ModelCheckpoint(
        "./checkpoints",
        monitor='val_map', save_top_k=1, mode="max")
    copy_best_checkpoint_callback = CopyBestCheckpoint(checkpoint_callback)
    callbacks = [checkpoint_callback, copy_best_checkpoint_callback]

    main(CCTDataModule, MyFasterRCNN, "training_stage1", callbacks=callbacks)
