from abc import ABC
from spaghettini import quick_register

import pytorch_lightning as pl

from src.utils.misc import *


@quick_register
class MimickingClassification(pl.LightningModule, ABC):
    def __init__(self, model, train_loader, valid_loader, optimizer, loss_fn, save_checkpoint=False, **kwargs):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.save_checkpoint = save_checkpoint

    def forward(self, inputs, **kwargs):
        return self.model(inputs, **kwargs)

    def training_step(self, data_batch, batch_nb, *args, **kwargs):
        loss, logs = self.common_step(data_batch, prepend_key="training/")

        # ____ Log metrics. ____
        self.logger.log_metrics(logs, step=self.trainer.total_batch_idx)

        return {"loss": loss}

    def validation_step(self, data_batch, batch_nb):
        _, logs = self.common_step(data_batch , prepend_key="validation/")

        return logs

    def validation_end(self, outputs):
        averaged_metrics = average_values_in_list_of_dicts(outputs)

        # ____ Log the averaged metrics. ____
        self.logger.log_metrics(averaged_metrics, step=self.trainer.total_batch_idx)

        # TODO: Find a way to avoid this solution.
        return {"val_loss": torch.tensor(self.trainer.total_batches)}

    def common_step(self, data_batch, prepend_key=""):
        assert (self.training and (prepend_key == "training/")) or \
               (not self.training and (prepend_key == "validation/"))
        # ____ Unpack the data batch. ____
        xs, ys = self.unpack_data_batch(data_batch)

        # ____ Make predictions. ____
        preds, model_logs = self.forward(xs)

        # ____ Compute loss. ____
        loss = self.loss_fn(ys, preds)

        # ____ Log the metrics computed. ____
        logs = self.log_forward_stats(loss, prepend_key)

        # ____ Return. ____
        return loss, logs

    def log_forward_stats(self, loss, prepend_key):
        logs = dict()

        # ____ Log losses. ____
        logs["{}/loss".format(prepend_key)] = float(loss)

        return logs

    def unpack_data_batch(self, data_batch):
        xs, ys = data_batch
        # The first dimensions is added automatically by the data loaders. In this implementation, we're generating
        # our own batch dimension, so we have to get rid of the first dimension added by the loader.
        xs, ys = xs.squeeze(0), ys.squeeze(0)

        return xs, ys

    def configure_optimizers(self):
        return [self.optimizer(self.model.parameters())]

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader

    def on_save_checkpoint(self, checkpoint):
        if not self.save_checkpoint:
            for key, value in checkpoint.items():
                checkpoint[key] = None
