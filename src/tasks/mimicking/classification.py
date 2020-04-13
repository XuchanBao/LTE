from abc import ABC
from spaghettini import quick_register

import pytorch_lightning as pl

from src.utils.misc import *
from src.dl.metrics.metrics import compute_accuracy
from src.dl.metrics.metrics import compute_distortion_ratios


@quick_register
class MimickingClassification(pl.LightningModule, ABC):
    def __init__(self, model, train_loader, valid_loader, optimizer, loss_fn, benchmark_rules=None,
                 save_checkpoint=False, **kwargs):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.benchmark_rules = benchmark_rules
        self.save_checkpoint = save_checkpoint

    def forward(self, inputs, **kwargs):
        return self.model(inputs, **kwargs)

    def training_step(self, data_batch, batch_nb, *args, **kwargs):
        loss, logs = self.common_step(self.forward, data_batch, prepend_key="training/model")

        # ____ Log metrics. ____
        self.logger.log_metrics(logs, step=self.trainer.total_batch_idx)

        return {"loss": loss}

    def validation_step(self, data_batch, batch_nb):
        logs = dict()

        # Run validation for the trained model.
        _, logs_model = self.common_step(self.forward, data_batch, prepend_key="validation/model")
        logs.update(logs_model)

        # Run the validation for other voting rules for benchmarking.
        for rule_name, rule_fn in self.benchmark_rules.items():
            _, logs_bnchmk = self.common_step(rule_fn, data_batch, prepend_key="validation/{}".format(rule_name))
            logs.update(logs_bnchmk)

        return logs

    def validation_end(self, outputs):
        averaged_metrics = average_values_in_list_of_dicts(outputs)

        # ____ Log the averaged metrics. ____
        self.logger.log_metrics(averaged_metrics, step=self.trainer.total_batch_idx)

        # TODO: Find a way to avoid this solution.
        return {"val_loss": torch.tensor(self.trainer.total_batches)}

    def common_step(self, forward_fn, data_batch, prepend_key=""):
        # ____ Unpack the data batch. ____
        xs, ys, utilities = self.unpack_data_batch(data_batch)

        # ____ Make predictions. ____
        preds = forward_fn(xs)

        # Since we have a logit per candidate in the end, we have to remove the last dimension.
        preds = preds.squeeze(2) if (len(preds.shape) == 3) else preds

        # ____ Compute loss. ____
        loss = self.loss_fn(preds, ys)

        # ____ Log the metrics computed. ____
        logs = self.log_forward_stats(xs, ys, preds, utilities, loss, prepend_key)

        # ____ Return. ____
        return loss, logs

    def log_forward_stats(self, xs, ys, preds, utilities, loss, prepend_key):
        logs = dict()

        # ____ Log losses. ____
        logs["{}/loss".format(prepend_key)] = float(loss)

        # ____ Log the accuracies. ____
        acc = compute_accuracy(logits=preds, scalar_targets=ys)
        logs["{}/acc".format(prepend_key)] = float(acc)

        # ____ Log the distortion ratios. ____
        inv_distortion_ratios = compute_distortion_ratios(logits=preds, utilities=utilities)
        self.logger.experiment.add_histogram(tag="inv_dist_ratios", values=inv_distortion_ratios)

        return logs

    def unpack_data_batch(self, data_batch):
        xs, ys, utilities = data_batch
        # The first dimensions is added automatically by the data loaders. In this implementation, we're generating
        # our own batch dimension, so we have to get rid of the first dimension added by the loader.
        xs, ys, utilities = xs.squeeze(0), ys.squeeze(0), utilities.squeeze(0)

        return xs, ys, utilities

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
