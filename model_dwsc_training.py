from argparse import ArgumentParser
import os

# FILTER WARNINGS
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn import BCELoss
from torch.optim import Adam
from optimizer.lookahead import Lookahead
from optimizer.ralamb import Ralamb

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything, loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from albumentations import Compose, ShiftScaleRotate, GridDistortion, Cutout

from utils.metrics import accuracy, compute_macro_auprc, compute_micro_auprc, compute_micro_F1, mean_average_precision

import pandas as pd
from prepare_data.urbansound8k import UrbanSound8K_TALNet
from prepare_data.esc50 import ESC50_TALNet
from prepare_data.sonycust import SONYCUST_TALNet
from models.depthwise_separable_conv_block import DESSEDDilatedTag
from losses.DCASEmaskedLoss import *
import config


class DWSCClassifier(LightningModule):
    def __init__(self, hparams, fold):
        super().__init__()

        # Save hparams for later
        self.hparams = hparams
        self.fold = fold

        if self.hparams.dataset == "US8K":
            self.dataset_folder = config.path_to_UrbanSound8K
            self.nb_classes = 10
            self.best_scores = [0] * 5
        elif self.hparams.dataset == "ESC50":
            self.dataset_folder = config.path_to_ESC50
            self.nb_classes = 50
            self.best_scores = [0] * 5
        elif self.hparams.dataset == "SONYCUST":
            self.dataset_folder = config.path_to_SONYCUST
            self.nb_classes = 31
            self.best_scores = [0] * 10
        else:
            None

        #
        # Settings for the SED models
        model_param = {
            "cnn_channels": self.hparams.cnn_channels,
            "cnn_dropout": self.hparams.cnn_dropout,
            "dilated_output_channels": self.hparams.dilated_output_channels,
            "dilated_kernel_size": self.hparams.dilated_kernel_size,  # time, feature
            "dilated_stride": self.hparams.dilated_stride,  # time, feature
            "dilated_padding": self.hparams.dilated_padding,
            "dilation_shape": self.hparams.dilation_shape,
            "dilated_nb_features": 84,
            "nb_classes": self.nb_classes,
            "inner_kernel_size": 3, 
            "inner_padding": 1, 
        }

        self.model = DESSEDDilatedTag(**model_param)
        if self.hparams.dataset != "SONYCUST":
            self.loss = BCELoss(reduction="none")
        else:
            self.loss_c = BCELoss(reduction="none")
            self.loss_f = Masked_loss(BCELoss(reduction="none"))

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--cnn_channels", type=int, default=256)
        parser.add_argument("--cnn_dropout", type=float, default=0.25)
        parser.add_argument("--dilated_output_channels", type=int, default=256)
        parser.add_argument("--dilated_kernel_size", nargs="+", type=int, default=[7, 7])
        parser.add_argument("--dilated_stride", nargs="+", type=int, default=[1, 3])
        parser.add_argument("--dilated_padding", nargs="+", type=int, default=[30, 0])
        parser.add_argument("--dilation_shape", nargs="+", type=int, default=[10, 1])

        parser.add_argument(
            "--pooling", type=str, default="att", choices=["max", "ave", "lin", "exp", "att", "auto"],
        )

        parser.add_argument("--batch_size", type=int, default=24)
        parser.add_argument("--shuffle", type=bool, default=True)
        parser.add_argument("--init_lr", type=float, default=1e-3)

        parser.add_argument("--num_mels", type=int, default=64)
        parser.add_argument("--dataset", type=str, default="SONYCUST", choices=["US8K", "ESC50", "SONYCUST"])

        return parser

    def forward(self, x):
        x = self.model(x)
        return x

    def prepare_data(self):
        # Dataset parameters

        # Creating dataset
        if self.hparams.dataset == "US8K":
            data_param = {
                "dataset_folder": self.dataset_folder,
                "fold": self.fold,
            }
            self.dataset = UrbanSound8K_TALNet(**data_param)
            (self.train_dataset, self.val_dataset) = self.dataset.train_validation_split()

        elif self.hparams.dataset == "ESC50":
            data_param = {
                "dataset_folder": self.dataset_folder,
                "fold": self.fold,
            }
            self.dataset = ESC50_TALNet(**data_param)
            (self.train_dataset, self.val_dataset) = self.dataset.train_validation_split()
        elif self.hparams.dataset == "SONYCUST":
            data_param = {
                "sonycust_folder": self.dataset_folder,
                "mode": "both",
                "cleaning_strat": "DCASE",
            }
            self.dataset = SONYCUST_TALNet(**data_param)
            self.train_dataset, self.val_dataset, _ = self.dataset.train_validation_test_split()
        else:
            None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size, shuffle=self.hparams.shuffle, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=4)

    def configure_optimizers(self):
        """
        optim_param = {
            'lr': self.hparams.init_lr
            }
        optimizer = Adam(self.model.parameters(), **optim_param)
        """
        base_optim_param = {"lr": self.hparams.init_lr}
        base_optim = Ralamb(self.model.parameters(), **base_optim_param)
        optim_param = {"k": 5, "alpha": 0.5}
        optimizer = Lookahead(base_optim, **optim_param)

        return optimizer

    def training_step(self, batch, batch_idx):
        if self.hparams.dataset != "SONYCUST":
            data, target = batch["input_vector"].float(), batch["label"].float()
            output = self.forward(data)
            loss = self.loss(output, target).mean()
        else:
            data, target_c, target_f = (
                batch["input_vector"].float(),
                batch["label"]["coarse"].float(),
                batch["label"]["full_fine"].float(),
            )
            target = torch.cat([target_c, target_f], 1)
            output = self.forward(data)
            outputs_c, outputs_f = torch.split(output, [8, 23], 1)
            loss = torch.cat([self.loss_c(outputs_c, target_c).mean(0), self.loss_f(outputs_f, target_f),], 0,).mean()

        return {"loss": loss, "log": {"1_loss/train_loss": loss}}

    def validation_step(self, batch, batch_idx):
        if self.hparams.dataset != "SONYCUST":
            data, target = batch["input_vector"].float(), batch["label"].float()
            output = self.forward(data)
            # Compute loss of the batch
            loss = self.loss(output, target)
        else:
            data, target_c, target_f = (
                batch["input_vector"].float(),
                batch["label"]["coarse"].float(),
                batch["label"]["full_fine"].float(),
            )
            target = torch.cat([target_c, target_f], 1)
            output = self.forward(data)
            outputs_c, outputs_f = torch.split(output, [8, 23], 1)
            # Compute loss of the batch
            loss = torch.cat([self.loss_c(outputs_c, target_c).mean(0), self.loss_f(outputs_f, target_f),], 0,)

        return {
            "val_loss": loss,
            "output": output,
            "target": target,
        }

    def validation_epoch_end(self, outputs):
        val_loss = torch.cat([o["val_loss"] for o in outputs], 0).mean()
        all_outputs = torch.cat([o["output"] for o in outputs], 0).cpu().numpy()
        all_targets = torch.cat([o["target"] for o in outputs], 0).cpu().numpy()

        if self.hparams.dataset == "SONYCUST":
            # Logic for SONYCUST
            X_mask = ~torch.BoolTensor(
                [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                ]
            )
            outputs_split = np.split(all_outputs, [8, 31], 1)
            all_outputs_coarse, all_outputs_fine = outputs_split[0], outputs_split[1]

            all_targets = all_targets[:, X_mask]
            targets_split = np.split(all_targets, [8, 31], 1)
            all_targets_coarse, all_targets_fine = targets_split[0], targets_split[1]

            accuracy_c = accuracy(all_targets_coarse, all_outputs_coarse)
            f1_micro_c = compute_micro_F1(all_targets_coarse, all_outputs_coarse)
            auprc_micro_c = compute_micro_auprc(all_targets_coarse, all_outputs_coarse)
            auprc_macro_c = compute_macro_auprc(all_targets_coarse, all_outputs_coarse)
            map_coarse = mean_average_precision(all_targets_coarse, all_outputs_coarse)

            accuracy_f = accuracy(all_targets_fine, all_outputs_fine)
            f1_micro_f = compute_micro_F1(all_targets_fine, all_outputs_fine)
            auprc_micro_f = compute_micro_auprc(all_targets_fine, all_outputs_fine)
            auprc_macro_f = compute_macro_auprc(all_targets_fine, all_outputs_fine)
            map_fine = mean_average_precision(all_targets_fine, all_outputs_fine)

            if accuracy_c > self.best_scores[0]:
                self.best_scores[0] = accuracy_c
            if f1_micro_c > self.best_scores[1]:
                self.best_scores[1] = f1_micro_c
            if auprc_micro_c > self.best_scores[2]:
                self.best_scores[2] = auprc_micro_c
            if auprc_macro_c > self.best_scores[3]:
                self.best_scores[3] = auprc_macro_c
            if map_coarse > self.best_scores[4]:
                self.best_scores[4] = map_coarse

            if accuracy_f > self.best_scores[5]:
                self.best_scores[5] = accuracy_f
            if f1_micro_f > self.best_scores[6]:
                self.best_scores[6] = f1_micro_f
            if auprc_micro_f > self.best_scores[7]:
                self.best_scores[7] = auprc_micro_f
            if auprc_macro_f > self.best_scores[8]:
                self.best_scores[8] = auprc_macro_f
            if map_fine > self.best_scores[9]:
                self.best_scores[9] = map_fine

            log_temp = {
                "2_valid_coarse/1_accuracy@0.5": accuracy_c,
                "2_valid_coarse/1_f1_micro@0.5": f1_micro_c,
                "2_valid_coarse/1_auprc_micro": auprc_micro_c,
                "2_valid_coarse/1_auprc_macro": auprc_macro_c,
                "2_valid_coarse/1_map_coarse": map_coarse,
                "3_valid_fine/1_accuracy@0.5": accuracy_f,
                "3_valid_fine/1_f1_micro@0.5": f1_micro_f,
                "3_valid_fine/1_auprc_micro": auprc_micro_f,
                "3_valid_fine/1_auprc_macro": auprc_macro_f,
                "3_valid_fine/1_map_fine": map_fine,
            }

            tqdm_dict = {
                "val_loss": val_loss,
                "m_auprc_c": auprc_macro_c,
            }

        else:
            # Logic for ESC50 and US8K
            accuracy_score = accuracy(all_targets, all_outputs)
            f1_micro = compute_micro_F1(all_targets, all_outputs)
            auprc_micro = compute_micro_auprc(all_targets, all_outputs)
            _, auprc_macro = compute_macro_auprc(all_targets, all_outputs, True)
            map_score = mean_average_precision(all_targets, all_outputs)

            if accuracy_score > self.best_scores[0]:
                self.best_scores[0] = accuracy_score
            if f1_micro > self.best_scores[1]:
                self.best_scores[1] = f1_micro
            if auprc_micro > self.best_scores[2]:
                self.best_scores[2] = auprc_micro
            if auprc_macro > self.best_scores[3]:
                self.best_scores[3] = auprc_macro
            if map_score > self.best_scores[4]:
                self.best_scores[4] = map_score

            log_temp = {
                "2_valid/1_accuracy0.5": accuracy_score,
                "2_valid/1_f1_micro0.5": f1_micro,
                "2_valid/1_auprc_micro": auprc_micro,
                "2_valid/1_auprc_macro": auprc_macro,
                "2_valid/1_map": map_score,
            }

            tqdm_dict = {
                "val_loss": val_loss,
                "acc": accuracy_score,
            }

        log = {
            "step": self.current_epoch,
            "1_loss/val_loss": val_loss,
        }

        log.update(log_temp)

        return {"progress_bar": tqdm_dict, "log": log}


def main(hparams, fold):
    seed_everything(hparams.seed)
    MAIN_DIR = os.path.join(config.path_to_summaries, "DWSCAllDatasets/")

    model = DWSCClassifier(hparams, fold)

    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(MAIN_DIR, "logs"))
    if hparams.dataset != "SONYCUST":
        early_stopping = EarlyStopping("2_valid/1_accuracy0.5", patience=50, mode="max")
    else:
        early_stopping = EarlyStopping("2_valid_coarse/1_auprc_macro", patience=30, mode="max")
    trainer = Trainer.from_argparse_args(
        hparams,
        default_root_dir=MAIN_DIR,
        logger=tb_logger,
        early_stop_callback=early_stopping,
        # fast_dev_run=True,
        checkpoint_callback=None,
        gpus=1,
    )
    trainer.fit(model)
    with open(os.path.join(MAIN_DIR, "logs/report.txt"), "a") as file:
        if hparams.dataset != "SONYCUST":
            file.write(hparams.dataset + " fold : " + str(fold) + "\n")
        else:
            file.write(hparams.dataset + "\n")
        file.write(str(model.best_scores) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser = DWSCClassifier.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    if hparams.dataset == "US8K":
        for i in range(1, 11):
            main(hparams, i)

    elif hparams.dataset == "ESC50":
        for i in range(1, 6):
            main(hparams, i)

    elif hparams.dataset == "SONYCUST":
        main(hparams, 1)
