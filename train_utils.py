import shutil
from argparse import ArgumentParser
from pprint import pprint
from warnings import warn

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything


class CopyBestCheckpoint(Callback):
    def __init__(self, dest_path, checkpoint_callback):
        self.dst = dest_path
        self.checkpoint_callback = checkpoint_callback

    def on_fit_end(self, trainer, pl_module):
        src = self.checkpoint_callback.best_model_path
        shutil.copy(src, self.dst)


def validate_args(args):
    distributed = (args.num_nodes > 1) or \
        (args.distributed_backend is not None)

    if args.seed is None and distributed:
        warn("In a distributed running, '--seed' option must be specified.")


def main(dm_cls, model_cls, logger_name, callbacks=None):
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=None, help="random seed")

    parser = pl.Trainer.add_argparse_args(parser)
    parser = dm_cls.add_argparse_args(parser)
    parser = model_cls.add_argparse_args(parser)

    args = parser.parse_args()

    print(f"received command line arguments:")
    pprint(vars(args), depth=1)

    validate_args(args)
    seed_everything(args.seed)

    logger = TensorBoardLogger('tb_logs', name=logger_name)

    trainer = pl.Trainer.from_argparse_args(args,
                                            deterministic=True,
                                            callbacks=callbacks,
                                            logger=logger
                                            )

    dm = dm_cls.from_argparse_args(args)
    model = model_cls.from_argparse_args(args)

    trainer.fit(model, dm)
