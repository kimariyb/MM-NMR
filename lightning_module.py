import yaml
import torch

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.functional import l1_loss
from pytorch_lightning import LightningModule

from models.heads.regressor import MultiModalFusionRegressor


class SpectraLightningModule(LightningModule):
    def __init__(self, config) -> None:
        super(SpectraLightningModule, self).__init__()
        self.save_hyperparameters(config)
        self.gnn_args, self.sphere_args = self.load_yml_config(
            gnn_config_path=self.hparams.gnn_args,
            sphere_config_path=self.hparams.geom_args,
        )   
        self.model = MultiModalFusionRegressor(
            gnn_args=self.gnn_args, 
            sphere_args=self.sphere_args, 
            mean=self.hparams.mean,
            std=self.hparams.std,
        )
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
    
    def step(self, batch, loss_fn, stage):
        with torch.set_grad_enabled(stage == "train"):
            pred = self(batch)
            label = batch.y[batch.mask]
            
            # unsqueeze the label and pred
            pred = pred.unsqueeze(-1)
            label = label.unsqueeze(-1)
            # calculate loss
            loss = loss_fn(pred, label)
            self.losses[stage].append(loss.detach())
    
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, l1_loss, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, l1_loss, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, l1_loss, "test")

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

    def load_yml_config(self, gnn_config_path, sphere_config_path):
        r"""
        load config from yml files
        """
        if gnn_config_path.endswith(".yml") or gnn_config_path.endswith(".yaml"):
            with open(gnn_config_path, "r") as f:
                gnn_config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise ValueError("gnn_config_path should be a yml file")

        if sphere_config_path.endswith(".yml") or sphere_config_path.endswith(".yaml"):
            with open(sphere_config_path, "r") as f:
                sphere_config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise ValueError("sphere_config_path should be a yml file")

        return gnn_config, sphere_config

