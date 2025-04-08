import os

from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningDataModule

from data.carbon import CarbonDataset
from lightning_utils import make_splits


class SpectraDataModule(LightningDataModule):
    def __init__(self, hparams):
        super(SpectraDataModule, self).__init__()
        self.hparams.update(hparams.__dict__) if hasattr(
            hparams, "__dict__"
        ) else self.hparams.update(hparams)
        self._saved_dataloaders = dict()
        self.dataset = None

    def prepare_dataset(self):
        if self.hparams["dataset"] == "carbon":
            self.dataset = CarbonDataset(self.hparams["dataset_root"],)
        elif self.hparams["dataset"] == "hydrogen":
            raise NotImplementedError
        
        self.idx_train, self.idx_val, self.idx_test = make_splits(
            dataset_len=len(self.dataset), 
            train_size=self.hparams["train_size"],
            val_size=self.hparams["val_size"],
            test_size=self.hparams["test_size"],
            seed=self.hparams["seed"],
            filename=os.path.join(self.hparams["log_dir"], "splits.npz"),
            splits=self.hparams["splits"],
        )

        print(
            f"train {len(self.idx_train)}, val {len(self.idx_val)}, test {len(self.idx_test)}"
        )

        self.train_dataset = Subset(self.dataset, self.idx_train)
        self.val_dataset = Subset(self.dataset, self.idx_val)
        self.test_dataset = Subset(self.dataset, self.idx_test)

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, "val")

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, "test")

    def _get_dataloader(self, dataset, stage, store_dataloader=True):
        store_dataloader = store_dataloader and not self.hparams["reload"]
        if stage in self._saved_dataloaders and store_dataloader:
            return self._saved_dataloaders[stage]

        if stage == "train":
            batch_size = self.hparams["batch_size"]
            shuffle = True
        elif stage in ["val", "test"]:
            batch_size = self.hparams["inference_batch_size"]
            shuffle = False
            
        dl = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams["num_workers"],
            pin_memory=False,
            drop_last=False,
        )

        if store_dataloader:
            self._saved_dataloaders[stage] = dl
            
        return dl

