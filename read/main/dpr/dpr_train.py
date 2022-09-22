import os
import warnings
from typing import List, Tuple

import hydra
import pyrootutils
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from read.dpr.module import DPRModule
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

from read.utils.hydra import instantiate_loggers, instantiate_callbacks
from loguru import logger

@hydra.main(version_base="1.2", config_path=root / "config" / "train", config_name="config_dpr.yaml")
def main(cfg: DictConfig) -> float:

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    logger.info(f"Instantiating data module")
    datamodule= hydra.utils.instantiate(cfg.datamodule)

    logger.info(f"Instantiating model")
    model = hydra.utils.instantiate(cfg.model)

    logger.info("Instantiating loggers...")
    loggers: List[LightningLoggerBase] = instantiate_loggers(cfg.get("logger"))

    logger.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=loggers
    )

    if cfg.get("train"):
        logger.info("*** Running Training ***")
        trainer.fit(model=model, datamodule=datamodule)

    train_metrics = trainer.callback_metrics

    ckpt_path = trainer.checkpoint_callback.best_model_path


    if cfg.get("dev"):
        logger.info("*** Running Evaluation ***")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            logger.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        logger.info(f"Best ckpt path: {ckpt_path}")

    if cfg.get("predict"):
        logger.info("*** Running Prediction ***")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            logger.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        logger.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    print("Checkpoint_path", ckpt_path)
    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}
    if ckpt_path != "":
        best_model = DPRModule.load_from_checkpoint(ckpt_path)
        logger.info("*** Saving best model ***")
        best_model.save(Path(cfg.paths.output_dir) / "dpr_model")
        best_model.model.qtx_model.encoder.push_to_hub("clapika2010/qtx_encoder")
        best_model.model.ctx_model.encoder.push_to_hub("clapika2010/ctx_encoder")

    return metric_dict


if __name__ == "__main__":
    main()
