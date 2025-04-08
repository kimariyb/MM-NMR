import yaml
import torch

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning import LightningModule

from models.heads.regressor import MultiModalFusionRegressor


class SpectraLightningModule(LightningModule):
    def __init__(self, config) -> None:
        super(SpectraLightningModule, self).__init__()

        self.save_hyperparameters(config)
        self.gnn_args, self.sphere_args, self.model_args = self.load_yml_config(
            gnn_config_path=self.hparams.gnn_config_path,
            sphere_config_path=self.hparams.sphere_config_path,
            model_config_path=self.hparams.model_config_path,
        )
        
        self.model = MultiModalFusionRegressor(
            gnn_args=self.gnn_args, 
            sphere_args=self.sphere_args, 
            model_args=self.model_args,
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
    
    def step(self, batch, stage):
        with torch.set_grad_enabled(stage == "train"):
            pred, z1, z2 = self(batch)
            losses = self._calculate_loss(batch, pred, z1, z2)

        self.log(f"{stage}_loss", losses["total_loss"], on_step=(stage=="train"), on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"{stage}_label_loss", losses["label_loss"], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log(f"{stage}_contrastive_loss", losses["contrastive_loss"], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        return {"loss": losses["total_loss"], "preds": pred} # 可以选择性返回预测值


    def training_step(self, batch, batch_idx):
        result = self.step(batch, "train")
        return result["loss"]

    def validation_step(self, batch, batch_idx):
        result = self.step(batch, "val")
        return result["loss"]
    
    def test_step(self, batch, batch_idx):
        result = self.step(batch, "test")
        return result["loss"]
    
    def _calc_loss(self, batch, pred, z1, z2):
        if not hasattr(batch, 'y') or not hasattr(batch, 'mask'):
            raise AttributeError("Batch object must have 'y' and 'mask' attributes for loss calculation.")

        label = batch.y[batch.mask]
        
        total_loss, label_loss, contrastive_loss = self.model.calc_loss(
            pred=pred, label=label, z1=z1, z2=z2
        )    
        
        return {
            "total_loss": total_loss,
            "label_loss": label_loss,
            "contrastive_loss": contrastive_loss
        }

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

    def load_yml_config(self, gnn_config_path, sphere_config_path, model_config_path):
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

        if model_config_path.endswith(".yml") or model_config_path.endswith(".yaml"):
            with open(model_config_path, "r") as f:
                model_config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise ValueError("model_config_path should be a yml file")

        return gnn_config, sphere_config, model_config