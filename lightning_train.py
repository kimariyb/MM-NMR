import os
import re
import argparse

import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.strategies import SingleDeviceStrategy

from lightning_data import SpectraDataModule
from lightning_module import SpectraLightningModule
from lightning_utils import LoadFromFile, number, save_argparse


def get_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        "--load-model",
        default=None,
        type=str,
        help="Restart training using a model checkpoint",
    )  # keep first
    
    parser.add_argument(
        "--conf",
        "-c",
        type=open,
        action=LoadFromFile,
        help="Configuration yaml file",
    )  # keep second
    
    parser.add_argument(
        "--gnn-args",
        type=str,
        default='./configs/graph.yml',
        help="Arguments for the GNN model (in yaml format)",
    )
    
    parser.add_argument(
        "--geom-args",
        type=str,
        default='./configs/geometry.yml',
        help="Arguments for the geometry model (in yaml format)",
    )
    
    parser.add_argument(
        "--model-args",
        type=str,
        default='./configs/model.yml',
        help="Arguments for the model (in yaml format)",
    )

    # training settings
    parser.add_argument(
        "--num-epochs", default=300, type=int, help="number of epochs"
    )
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=0,
        help="How many steps to warm-up over. Defaults to 0 for no warm-up",
    )
    parser.add_argument("--lr", default=2e-4, type=float, help="learning rate")
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=10,
        help="Patience for lr-schedule. Patience per eval-interval of validation",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=1e-6,
        help="Minimum learning rate before early stop",
    )
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.8,
        help="Minimum learning rate before early stop",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-5, help="Weight decay strength"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=30,
        help="Stop training after this many epochs without improvement",
    )

    # dataset specific
    parser.add_argument(
        "--dataset",
        default="carbon",
        type=str,
        choices=["carbon", "hydrogen"],
        help="Name of the torch_geometric dataset. Default is carbon",
    )
    parser.add_argument(
        "--dataset-root", default='./data', type=str, help="Data storage directory"
    )
    parser.add_argument(
        "--mean", default=None, type=float, help="Mean of the dataset"
    )
    parser.add_argument(
        "--std",
        default=None,
        type=float,
        help="Standard deviation of the dataset",
    )
    # dataloader specific
    parser.add_argument(
        "--reload",
        type=int,
        default=1,
        help="Reload dataloaders every n epoch",
    )
    parser.add_argument(
        "--batch-size", default=32, type=int, help="batch size"
    )
    parser.add_argument(
        "--inference-batch-size",
        default=None,
        type=int,
        help="Batchsize for validation and tests.",
    )
    parser.add_argument(
        "--standardize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, multiply prediction by dataset std and add mean",
    )
    parser.add_argument(
        "--splits",
        default=None,
        help="Npz with splits idx_train, idx_val, idx_test",
    )
    parser.add_argument(
        "--train-size",
        type=number,
        default=0.8,
        help="Percentage/number of samples in training set (None to use all remaining samples)",
    )
    parser.add_argument(
        "--val-size",
        type=number,
        default=0.1,
        help="Percentage/number of samples in validation set (None to use all remaining samples)",
    )
    parser.add_argument(
        "--test-size",
        type=number,
        default=0.1,
        help="Percentage/number of samples in test set (None to use all remaining samples)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=6,
        help="Number of workers for data prefetch",
    )

    parser.add_argument(
        "--log-dir", type=str, default="./logs", help="Log directory"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        choices=["train", "inference"],
        help="Train or inference",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--redirect",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Redirect stdout and stderr to log_dir/log",
    )
    parser.add_argument(
        "--accelerator",
        default="gpu",
        help='Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "auto")',
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1,
        help="Save interval, one save per n epochs (default: 10)",
    )

    args = parser.parse_args()

    if args.inference_batch_size is None:
        args.inference_batch_size = args.batch_size

    if args.task == "train":
        save_argparse(
            args, os.path.join(args.log_dir, "input.yaml"), exclude=["conf"]
        )

    return args


def auto_exp(args):
    dir_name = (
        f"bs_{args.batch_size}"
        + f"_L{args.num_layers}_D{args.embedding_dim}_F{args.ffn_embedding_dim}"
        + f"_H{args.num_heads}_rbf_{args.num_rbf}"
        + f"_norm_{args.norm_type}"
        + f"_lr_{args.lr}"
        + f"_cutoff_{args.cutoff}"
        + f"_seed_{args.seed}"
    )

    if args.load_model is None:
        # resume from checkpoint if cluster breaks down
        args.log_dir = os.path.join(args.log_dir, dir_name)
        if os.path.exists(args.log_dir):
            if os.path.exists(
                os.path.join(args.log_dir, "checkpoints", "last.ckpt")
            ):
                args.load_model = os.path.join(
                    args.log_dir, "checkpoints", "last.ckpt"
                )
                print(
                    f"***** model {args.log_dir} exists, resuming from the last checkpoint *****"
                )
            csv_path = os.path.join(args.log_dir, "metrics", "metrics.csv")
            while os.path.exists(csv_path):
                csv_path = csv_path + ".bak"
            if os.path.exists(
                os.path.join(args.log_dir, "metrics", "metrics.csv")
            ):
                os.rename(
                    os.path.join(args.log_dir, "metrics", "metrics.csv"),
                    csv_path,
                )

    return args


def main():
    args = get_args()

    pl.seed_everything(args.seed, workers=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initialize data module
    args = auto_exp(args)

    data = SpectraDataModule(args)
    data.prepare_dataset()
    args.mean, args.std = data.mean, data.std

    model = SpectraLightningModule(args)

    csv_logger = CSVLogger(args.log_dir, name="metrics", version="")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    if args.task == "train":
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(args.log_dir, "checkpoints"),
            monitor="val_loss",
            save_top_k=10,
            save_last=True,
            every_n_epochs=args.save_interval,
            filename="{epoch}-{val_loss:.4f}",
        )

        early_stopping = EarlyStopping(
            "val_loss", patience=args.early_stopping_patience
        )
        tb_logger = TensorBoardLogger(
            args.log_dir,
            name="tensorbord",
            version="",
            default_hp_metric=False,
        )

        strategy = SingleDeviceStrategy(device=device)

        trainer = pl.Trainer(
            max_epochs=args.num_epochs,
            num_nodes=args.num_nodes,
            accelerator=args.accelerator,
            deterministic=True,
            default_root_dir=args.log_dir,
            callbacks=[early_stopping, checkpoint_callback, lr_monitor],
            logger=[tb_logger, csv_logger],
            reload_dataloaders_every_n_epochs=args.reload,
            precision=args.precision,
            strategy=strategy,
            enable_progress_bar=True,
            inference_mode=False,
        )

        trainer.fit(model, datamodule=data, ckpt_path=args.load_model)

    test_trainer = pl.Trainer(
        enable_model_summary=True,
        logger=[csv_logger],
        max_epochs=-1,
        num_nodes=1,
        devices=1,
        default_root_dir=args.log_dir,
        enable_progress_bar=True,
        callbacks=[ModelSummary()],
        accelerator=args.accelerator,
        inference_mode=False,
    )

    if args.task == "train":
        trainer.test(
            model=model,
            ckpt_path=trainer.checkpoint_callback.best_model_path,
            datamodule=data,
        )
    elif args.task == "inference":
        ckpt = torch.load(args.load_model, map_location="cpu")
        model.model.load_state_dict(
            {
                re.sub(r"^model\.", "", k): v
                for k, v in ckpt["state_dict"].items()
            }
        )
        test_trainer.test(model=model, datamodule=data)


if __name__ == "__main__":
    main()
