from typing import Optional, List

from overrides import overrides
import torch

from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric


@Metric.register("multiset_fscore")
class MultisetFScore(Metric):
    """
    FScore metric for a multiset of labels.
    """
    def __init__(self, null_label_id) -> None:
        self.correct_count = 0.
        self.total_gold = 0.
        self.total_predicted = 0.
        self.null_label_id = null_label_id

    def __call__(self,
                 predictions: List[torch.Tensor],
                 gold_labels: List[torch.Tensor],
                 mask: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A list of tensors of predictions, each of shape (batch_size, sent_length, num_classes).
            It is a list because we may have several layers of label prediction (more than one label per word).
        gold_labels : ``torch.Tensor``, required.
            A list of tensors, each of integer class label of shape (batch_size, sent_length). They must be the same
            shape as the ``predictions`` tensors without the ``num_classes`` dimension.
            It is a list because we may have several layers of label prediction (more than one label per word).
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as the tensors in ``gold_labels``.
        """
        # remove from GPU and remove gradient
        predictions = [next(self.unwrap_to_tensors(p)) for p in predictions]
        gold_labels = [next(self.unwrap_to_tensors(g)) for g in gold_labels]
        mask = next(self.unwrap_to_tensors(mask))

        # Some sanity checks.
        if predictions[0].dim() != 3:
            raise ConfigurationError("Incorrect prediction dimension in MultisetFScore!")
        if len(predictions) != len(gold_labels):
            raise ConfigurationError("Different number of predicted and gold label layers in MultisetFScore!")
        batch_size = predictions[0].size(0)
        sent_length = predictions[0].size(1)
        num_classes = predictions[0].size(2)
        if mask is not None:
            if mask.dim() != 2 or mask.size()[0] != batch_size or mask.size()[1] != sent_length:
                raise ConfigurationError("Incorrect mask dimension in MultisetFScore!")
        for p, g in zip(predictions, gold_labels):
            psize = p.size()
            gsize = g.size()
            if p.dim() != 3 or g.dim() != 2 or psize[0] != batch_size or gsize[0] != batch_size\
                or gsize[1] != sent_length or psize[1] != sent_length:
                raise ConfigurationError("Incorrect dimensions in MultisetFScore!")
        for p in predictions:
            if p.size(-1) != num_classes:
                raise ConfigurationError("Label prediction layers passed to MultisetFScore have unequal number of"
                                         "categories!.")
        for g in gold_labels:
            if (g >= num_classes).any():
                raise ConfigurationError("A gold label passed to MultisetFScore contains an id >= {}, "
                                         "the number of classes.".format(num_classes))

        for k in range(batch_size):
            prediction_counts = dict()
            gold_counts = dict()
            for p, g in zip(predictions, gold_labels):
                for i in range(sent_length):
                    if mask is None or mask[k][i] > 1e-12:
                            _, pred_here = p[k][i].max(0)
                            pred_here = pred_here.item()
                            gold_here = g[k][i].item()
                            if not pred_here == self.null_label_id:
                                prediction_counts[pred_here] = prediction_counts.get(pred_here, 0) + 1
                            if not gold_here == self.null_label_id:
                                gold_counts[gold_here] = gold_counts.get(gold_here, 0) + 1
            self.total_predicted += sum(prediction_counts.values())
            self.total_gold += sum(gold_counts.values())
            for label in gold_counts.keys():
                self.correct_count += min(gold_counts.get(label, 0), prediction_counts.get(label, 0))


    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        The accumulated f-score.
        """
        if self.total_gold > 1e-12:
            recall = float(self.correct_count) / float(self.total_gold)
        else:
            recall = 0.0
        if self.total_predicted > 1e-12:
            precision = float(self.correct_count) / float(self.total_predicted)
        else:
            precision = 0.0
        if recall > 1e-12 or precision > 1e-12:
            f = 2*recall*precision/(recall + precision)
        else:
            f = 0.0
        if reset:
            self.reset()
        return {"fscore": f, "recall": recall, "precision": precision}

    @overrides
    def reset(self):
        self.correct_count = 0.0
        self.total_gold = 0.0
        self.total_predicted = 0.0