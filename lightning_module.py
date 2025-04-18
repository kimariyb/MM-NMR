import yaml
import torch

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import l1_loss
from pytorch_lightning import LightningModule


class SpectraLightningModule(LightningModule):
    def __init__(self, config) -> None:
        super(SpectraLightningModule, self).__init__()
        self.save_hyperparameters(config)
        self.model = ...

        self._reset_losses_dict()

    def configure_optimizers(self):
        optimizer = AdamW(
            params=self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            "min",
            factor=self.hparams.lr_factor,
            patience=self.hparams.lr_patience,
            min_lr=self.hparams.lr_min,
            cooldown=20
        )
        
        lr_scheduler = {
            "scheduler": scheduler,
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }

        return [optimizer], [lr_scheduler]
    
    def forward(self, batch):
        return self.model(batch)
    
    def step(self, batch, loss_fn, stage, batch_idx):
        with torch.set_grad_enabled(stage == "train"):
            pred = self(batch)
            label = batch.y[batch.mask]

            # make pred and label to (N)
            pred, label = pred.squeeze(-1), label.squeeze(-1)
            pred = (pred * self.hparams.std) + self.hparams.mean
            
            # calculate loss
            loss = loss_fn(pred, label)
            self.losses[stage].append(loss.detach())
    
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, l1_loss, "train", batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, l1_loss, "val", batch_idx)

    def test_step(self, batch, batch_idx):
        return self.step(batch, l1_loss, "test", batch_idx)

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / float(self.hparams.lr_warmup_steps),
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()
  
    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            result_dict = {
                "epoch": float(self.current_epoch),
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                "train_loss": torch.stack(self.losses["train"]).mean(),
                "val_loss": torch.stack(self.losses["val"]).mean(),
            }

            # add test loss if available
            if len(self.losses["test"]) > 0:
                result_dict["test_loss"] = torch.stack(
                    self.losses["test"]
                ).mean()

            self.log_dict(result_dict, prog_bar=True, sync_dist=True)

        self._reset_losses_dict()

    def on_test_epoch_end(self):
        result_dict = {}
        if len(self.losses["test"]) > 0:
            result_dict["test_loss"] = torch.stack(self.losses["test"]).mean()
        self.log_dict(result_dict, sync_dist=True)
        self._reset_losses_dict()

    def _reset_losses_dict(self):
        self.losses = {
            "train": [],
            "val": [],
            "test": [],
        }

    def load_yml_config(self, config_path):
        r"""
        load config from yml files
        """
        if config_path.endswith(".yml") or config_path.endswith(".yaml"):
            with open(config_path, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise ValueError("config_path should be a yml file")

        return config

