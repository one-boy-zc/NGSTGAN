import torch
import yaml
from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
import warnings
warnings.filterwarnings("ignore")

from data_loader import DataInterface
from models import ModelInterface

RANDOM_SEED = 0
seed_everything(RANDOM_SEED)

def main(config_file: str):
    torch.set_float32_matmul_precision("high")
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), yaml.FullLoader)

    data_interface = DataInterface(**config)
    model_interface = ModelInterface(**config)

    # logger and callbacks
    logger = CSVLogger(
        save_dir=config["logs_dir"],
        name=config["display_name"],
        flush_logs_every_n_steps=config["flush_logs_every_n_steps"],
    )

    callbacks = [
        ModelCheckpoint(
            monitor=config["ckpt_monitor"],
            dirpath=f"{config['checkpoints_dir']}/{config['display_name']}",
            filename="best-{epoch}-{val_PeakSignalNoiseRatio:.4f}",
            save_top_k=1,
            mode=config["monitor_mode"],  # psnr: max, lpips: min
            save_last=True,
        ),
        TQDMProgressBar(refresh_rate=config["flush_logs_every_n_steps"])

    ]

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        accelerator='gpu',
        max_epochs=config["epochs"],
        precision=config["precision"],
        log_every_n_steps=config["flush_logs_every_n_steps"],
        fast_dev_run=False,
    )

    trainer.fit(model_interface, data_interface, ckpt_path="last" if config["resume_from_ckpt"] else None)
    trainer.test(model_interface, data_interface, ckpt_path="best")


if __name__ == "__main__":
    main(r".\configs\swin_ir_NG.yml")
