from cct_datamodule import CCTDataModule
from train_utils import main
from crcnn import ContextRCNN
from faster_rcnn import MyFasterRCNN
from simple_datamodule import SimpleDataModule


if __name__ == '__main__':
    main(CCTDataModule, ContextRCNN, "crcnn")
