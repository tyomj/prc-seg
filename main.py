from typing import Dict

import hydra
import pytorch_lightning as pl

from src.pl_modules import PLModel
from src.utils import load_obj

pl.seed_everything(42)


@hydra.main(config_path='config.yaml')
def run(cfg: Dict) -> None:
    model = PLModel(prms=cfg)
    print(model)

    model_checkpoint = pl.callbacks.ModelCheckpoint(
        **cfg.callbacks.model_checkpoint.params)

    loggers = []
    if cfg.logging.log:
        for name, logger in cfg.logging.loggers.items():
            if logger.params is not None:
                loggers.append(
                    load_obj(logger.class_name)(**dict(logger.params)))
            else:
                loggers.append(load_obj(logger.class_name))
    trainer = pl.Trainer(
        logger=loggers,
        checkpoint_callback=model_checkpoint,
        **cfg.trainer,
    )
    trainer.fit(model)


if __name__ == '__main__':
    run()
