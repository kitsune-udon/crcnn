from cct_datamodule import CCTDataModule
from train_utils import main
from crcnn import ContextRCNN


if __name__ == '__main__':
    main(CCTDataModule, ConContextRCNN, "crcnn")
