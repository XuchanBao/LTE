from abc import ABC
from spaghettini import quick_register

import pytorch_lightning as pl

from src.utils.misc import *
from src.dl.metrics.metrics import compute_accuracy
from src.dl.metrics.metrics import compute_distortion_ratios
from src.visualizations import histogram_overlayer


INV_DISTORTION_KEY = "inv_dist_ratios"


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

        if benchmark_rules is None:
            self.benchmark_rules = dict()
        else:
            self.benchmark_rules = benchmark_rules

        self.save_checkpoint = save_checkpoint

    def forward(self, inputs, **kwargs):
        return self.model(inputs, **kwargs)

    def training_step(self, data_batch, batch_nb, *args, **kwargs):
        loss, metric_logs, hist_logs = self.common_step(self.forward, data_batch, prepend_key="training/model")

        # ____ Log metrics. ____
        self.logger.log_metrics(metric_logs, step=self.trainer.total_batch_idx)

        return {"loss": loss}

    def validation_step(self, data_batch, batch_nb):
        metric_logs = dict()
        hist_logs = dict()

        # Run validation for the trained model.
        _, logs_model, hist_logs_model = self.common_step(self.forward, data_batch, prepend_key="validation/model")
        metric_logs.update(logs_model)
        hist_logs.update(hist_logs_model)

        # Run the validation for other voting rules for benchmarking.
        for rule_name, rule_fn in self.benchmark_rules.items():
            _, logs_bnchmk, hist_logs_bnchmk = self.common_step(rule_fn, data_batch,
                                                                prepend_key="validation/{}".format(rule_name))
            metric_logs.update(logs_bnchmk)
            hist_logs.update(hist_logs_bnchmk)

        return metric_logs, hist_logs

    def validation_end(self, outputs):
        # ____ Break down the outputs collected during validation. ____)
        # First argument returned by validation_step is the dict of metrics.
        metrics_dicts_list = [collected_tuple[0] for collected_tuple in outputs]

        # Second argument returned by validation_step is the dict of histogram metrics.
        hist_dicts_list = [collected_tuple[1] for collected_tuple in outputs]

        # ____ Average metrics and concatenate histogram metrics. ____
        averaged_metrics = average_values_in_list_of_dicts(metrics_dicts_list)
        concat_hist_metrics = concatenate_values_in_list_of_dicts(hist_dicts_list)

        # ____ Log the averaged metrics and histogram metrics. ____
        self.logger.log_metrics(averaged_metrics, step=self.trainer.total_batch_idx)
        for k, v in concat_hist_metrics.items():
            self.logger.experiment.add_histogram(tag=k, values=v, global_step=self.current_epoch)

        # ___ Run manual histogram plotter. ____
        inv_distortion_values_dict = {k: v for k, v in concat_hist_metrics.items() if INV_DISTORTION_KEY in k}
        save_dir = os.path.join(self.logger.save_dir, self.logger.name, "version_" + str(self.logger.version),
                                "inv_dist_histograms", f"epoch_{self.current_epoch}")
        histogram_overlayer(inv_distortion_values_dict, save_dir=save_dir)

        # TODO: Find a way to avoid this solution.
        return {"val_loss": torch.tensor(self.trainer.total_batches)}

    def common_step(self, forward_fn, data_batch, prepend_key=""):
        # ____ Unpack the data batch. ____
        xs, ys, utilities = self.unpack_data_batch(data_batch)

        if forward_fn == self.forward:
            # ____ Make predictions. ____
            preds = forward_fn(xs)
        else:
            # ____ Evaluate baseline. ____
            rankings = torch.argsort(utilities, axis=2, descending=True)
            preds, _ = forward_fn(rankings)
            preds = preds.type_as(xs)

        # Since we have a logit per candidate in the end, we have to remove the last dimension.
        preds = preds.squeeze(2) if (len(preds.shape) == 3) else preds

        # ____ Compute loss. ____
        loss = self.loss_fn(preds, ys)

        # ____ Log the metrics computed. ____
        metric_logs, histogram_logs = self.log_forward_stats(xs, ys, preds, utilities, loss, prepend_key)

        # ____ Return. ____
        return loss, metric_logs, histogram_logs

    def log_forward_stats(self, xs, ys, preds, utilities, loss, prepend_key):
        metric_logs = dict()
        hist_logs = dict()

        # ____ Log losses. ____
        metric_logs["{}/loss".format(prepend_key)] = float(loss)

        # ____ Log the accuracies. ____
        acc = compute_accuracy(logits=preds, scalar_targets=ys)
        metric_logs["{}/acc".format(prepend_key)] = float(acc)

        # ____ Log the distortion ratios. ____
        inv_distortion_ratios = compute_distortion_ratios(logits=preds, utilities=utilities)
        hist_logs[f"{prepend_key}/{INV_DISTORTION_KEY}"] = inv_distortion_ratios

        return metric_logs, hist_logs

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
