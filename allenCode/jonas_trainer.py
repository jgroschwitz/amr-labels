from comet_ml import Experiment

import logging
import numpy as np
import os
import time
import datetime
import traceback
from typing import Dict, Optional, List, Tuple, Union, Iterable, Any

import torch
import torch.optim.lr_scheduler


from allennlp.common import Params
from allennlp.common.checks import ConfigurationError, parse_cuda_device
from allennlp.common.util import (dump_metrics, gpu_memory_mb, peak_memory_mb,
                                  lazy_groups_of)
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.models.model import Model
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.predictors import Predictor, SentenceTaggerPredictor
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer_base import TrainerBase
from allennlp.training.trainer import Trainer, TrainerPieces
from allennlp.training import util as training_util
from allennlp.training.moving_average import MovingAverage

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


### mostly copied from the original Trainer class
### author: Jonas Groschwitz


@TrainerBase.register("jonas")
class MyTrainer(Trainer):
    def __init__(self,
                 model: Model,
                 optimizer: torch.optim.Optimizer,
                 iterator: DataIterator,
                 train_dataset: Iterable[Instance],
                 validation_dataset: Optional[Iterable[Instance]] = None,
                 patience: Optional[int] = None,
                 validation_metric: str = "-loss",
                 validation_iterator: DataIterator = None,
                 shuffle: bool = True,
                 num_epochs: int = 20,
                 serialization_dir: Optional[str] = None,
                 num_serialized_models_to_keep: int = 20,
                 keep_serialized_model_every_num_seconds: int = None,
                 checkpointer: Checkpointer = None,
                 model_save_interval: float = None,
                 cuda_device: Union[int, List] = -1,
                 grad_norm: Optional[float] = None,
                 grad_clipping: Optional[float] = None,
                 learning_rate_scheduler: Optional[LearningRateScheduler] = None,
                 momentum_scheduler: Optional[MomentumScheduler] = None,
                 summary_interval: int = 100,
                 histogram_interval: int = None,
                 should_log_parameter_statistics: bool = True,
                 should_log_learning_rate: bool = False,
                 log_batch_size_period: Optional[int] = None,
                 moving_average: Optional[MovingAverage] = None,
                 comet_experiment: Experiment = None,
                 predictor: [Predictor] = None,
                 prediction_log_file: str = None) -> None:
        """
        calls the Trainer initializer and also remembers where to write the prediction log
        """
        # THISISNEW: the whole function rewritten (calls super though, so does mostly the same thing)
        super().__init__(model,
                         optimizer,
                         iterator,
                         train_dataset,
                         validation_dataset=validation_dataset,
                         patience=patience,
                         validation_metric=validation_metric,
                         validation_iterator=validation_iterator,
                         shuffle=shuffle,
                         num_epochs=num_epochs,
                         serialization_dir=serialization_dir,
                         num_serialized_models_to_keep=num_serialized_models_to_keep,
                         keep_serialized_model_every_num_seconds=keep_serialized_model_every_num_seconds,
                         checkpointer=checkpointer,
                         model_save_interval=model_save_interval,
                         cuda_device=cuda_device,
                         grad_norm=grad_norm,
                         grad_clipping=grad_clipping,
                         learning_rate_scheduler=learning_rate_scheduler,
                         momentum_scheduler=momentum_scheduler,
                         summary_interval=summary_interval,
                         histogram_interval=histogram_interval,
                         should_log_parameter_statistics=should_log_parameter_statistics,
                         should_log_learning_rate=should_log_learning_rate,
                         log_batch_size_period=log_batch_size_period,
                         moving_average=moving_average)

        self.comet_experiment = comet_experiment
        self.predictor = predictor
        if prediction_log_file:
            self.prediction_log_file = open(prediction_log_file, "w")
        else:
            self.prediction_log_file = None



    def train(self) -> Dict[str, Any]:
        """
        Trains the supplied model with the supplied parameters. Mostly identical with the original function from Trainer.
        """
        try:
            epoch_counter = self._restore_checkpoint()
        except RuntimeError:
            traceback.print_exc()
            raise ConfigurationError("Could not recover training from the checkpoint.  Did you mean to output to "
                                     "a different serialization directory or delete the existing serialization "
                                     "directory?")

        training_util.enable_gradient_clipping(self.model, self._grad_clipping)

        logger.info("Beginning training.")

        train_metrics: Dict[str, float] = {}
        val_metrics: Dict[str, float] = {}
        this_epoch_val_metric: float = None
        metrics: Dict[str, Any] = {}
        epochs_trained = 0
        training_start_time = time.time()

        metrics['best_epoch'] = self._metric_tracker.best_epoch
        for key, value in self._metric_tracker.best_epoch_metrics.items():
            metrics["best_validation_" + key] = value

        for epoch in range(epoch_counter, self._num_epochs):

            # if self._validation_data is not None:
            #     with torch.no_grad():
            #
            #         #THISISNEW: writing to the prediction log here -- printing to terminal here for debugging
            #         if self.prediction_log_file and self.predictor:
            #             for instance in self._validation_data:
            #                 logits_sentence = self.predictor.predict_instance(instance)['tag_logits']
            #                 for word, gold, predictions in zip(instance["sentence"], instance["labels"], logits_sentence):
            #                     prediction = str(word)+" ("+str(gold)+")"
            #                     top_five = np.array(predictions).argsort()[-5:][::-1]
            #                     top_five = [self.model.vocab.get_token_from_index(i, 'labels')+ " " + str(predictions[i]) for i in top_five]
            #                     prediction += str(top_five)
            #                     print(str(prediction))
            #                 print()

            epoch_start_time = time.time()
            train_metrics = self._train_epoch(epoch)

            # get peak of memory usage
            if 'cpu_memory_MB' in train_metrics:
                metrics['peak_cpu_memory_MB'] = max(metrics.get('peak_cpu_memory_MB', 0),
                                                    train_metrics['cpu_memory_MB'])
            for key, value in train_metrics.items():
                if key.startswith('gpu_'):
                    metrics["peak_"+key] = max(metrics.get("peak_"+key, 0), value)

            if self._validation_data is not None:
                with torch.no_grad():

                    # THISISNEW: writing to the prediction log here
                    # TODO: maybe remove the printing again and do the whole dev set (or larger segment), but the current version is useful right now
                    if self.prediction_log_file and self.predictor:
                        for instance in self._validation_data[:5]:
                            logits_sentence = self.predictor.predict_instance(instance)['tag_logits']
                            # print("---------------")
                            for word, gold, predictions in zip(instance["sentence"], instance["labels"], logits_sentence):
                                prediction = str(word)+"("+str(gold)+"):\t"
                                top_five = np.array(predictions).argsort()[-5:][::-1]
                                top_five = [self.model.vocab.get_token_from_index(i, 'labels')+ " " + str(predictions[i]) for i in top_five]
                                prediction += str(top_five)
                                self.prediction_log_file.write(str(prediction)+"\n")
                                # print(prediction)
                            # print("---------------")
                            self.prediction_log_file.write("\n")
                        self.prediction_log_file.write("------------\n\n")
                        self.prediction_log_file.flush()



                    # We have a validation set, so compute all the metrics on it.
                    val_loss, num_batches = self._validation_loss()
                    val_metrics = training_util.get_metrics(self.model, val_loss, num_batches, reset=True)

                    # Check validation metric for early stopping
                    this_epoch_val_metric = val_metrics[self._validation_metric]
                    self.comet_experiment.log_metric("validation_metric", this_epoch_val_metric)
                    self._metric_tracker.add_metric(this_epoch_val_metric)

                    if self._metric_tracker.should_stop_early():
                        logger.info("Ran out of patience.  Stopping training.")
                        break

            self._tensorboard.log_metrics(train_metrics,
                                          val_metrics=val_metrics,
                                          log_to_console=True,
                                          epoch=epoch + 1)  # +1 because tensorboard doesn't like 0

            # Create overall metrics dict
            training_elapsed_time = time.time() - training_start_time
            metrics["training_duration"] = str(datetime.timedelta(seconds=training_elapsed_time))
            metrics["training_start_epoch"] = epoch_counter
            metrics["training_epochs"] = epochs_trained
            metrics["epoch"] = epoch

            for key, value in train_metrics.items():
                metrics["training_" + key] = value
            for key, value in val_metrics.items():
                metrics["validation_" + key] = value

            if self._metric_tracker.is_best_so_far():
                # Update all the best_ metrics.
                # (Otherwise they just stay the same as they were.)
                metrics['best_epoch'] = epoch
                for key, value in val_metrics.items():
                    metrics["best_validation_" + key] = value

                self._metric_tracker.best_epoch_metrics = val_metrics

            if self._serialization_dir:
                dump_metrics(os.path.join(self._serialization_dir, f'metrics_epoch_{epoch}.json'), metrics)

            # The Scheduler API is agnostic to whether your schedule requires a validation metric -
            # if it doesn't, the validation metric passed here is ignored.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step(this_epoch_val_metric, epoch)
            if self._momentum_scheduler:
                self._momentum_scheduler.step(this_epoch_val_metric, epoch)

            self._save_checkpoint(epoch)

            epoch_elapsed_time = time.time() - epoch_start_time
            logger.info("Epoch duration: %s", datetime.timedelta(seconds=epoch_elapsed_time))

            if epoch < self._num_epochs - 1:
                training_elapsed_time = time.time() - training_start_time
                estimated_time_remaining = training_elapsed_time * \
                    ((self._num_epochs - epoch_counter) / float(epoch - epoch_counter + 1) - 1)
                formatted_time = str(datetime.timedelta(seconds=int(estimated_time_remaining)))
                logger.info("Estimated training time remaining: %s", formatted_time)

            epochs_trained += 1

        # make sure pending events are flushed to disk and files are closed properly
        # self._tensorboard.close() # THISISNEW caused some sort of error

        # Load the best model state before returning
        best_model_state = self._checkpointer.best_model_state()
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        self.comet_experiment.end()

        return metrics


    # Requires custom from_params. But mostly identical to original from Trainer class.
    @classmethod
    def from_params(cls, params, serialization_dir, recover):

        pieces = TrainerPieces.from_params(params, serialization_dir, recover)  # pylint: disable=no-member
        return MyTrainer.from_params_old_interface(model=pieces.model,
                                       serialization_dir=serialization_dir,
                                       iterator=pieces.iterator,
                                       train_data=pieces.train_dataset,
                                       validation_data=pieces.validation_dataset,
                                       params=pieces.params,
                                       validation_iterator=pieces.validation_iterator)


    @classmethod
    def from_params_old_interface(cls,  # type: ignore
                    model: Model,
                    serialization_dir: str,
                    iterator: DataIterator,
                    train_data: Iterable[Instance],
                    validation_data: Optional[Iterable[Instance]],
                    params: Params,
                    validation_iterator: DataIterator = None) -> 'Trainer':
        # pylint: disable=arguments-differ
        patience = params.pop_int("patience", None)
        validation_metric = params.pop("validation_metric", "-loss")
        shuffle = params.pop_bool("shuffle", True)
        num_epochs = params.pop_int("num_epochs", 20)
        cuda_device = parse_cuda_device(params.pop("cuda_device", -1))
        grad_norm = params.pop_float("grad_norm", None)
        grad_clipping = params.pop_float("grad_clipping", None)
        lr_scheduler_params = params.pop("learning_rate_scheduler", None)
        momentum_scheduler_params = params.pop("momentum_scheduler", None)
        prediction_log_file = params.pop("prediction_log_file", None) # THISISNEW: storing the prediction log here

        if isinstance(cuda_device, list):
            model_device = cuda_device[0]
        else:
            model_device = cuda_device
        if model_device >= 0:
            # Moving model to GPU here so that the optimizer state gets constructed on
            # the right device.
            print("running on GPU: "+str(model_device))
            model = model.cuda(model_device)
        else:
            print("running on CPU")

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))

        # THISISNEW:
        if "dataset_reader" in params:
            predictor = SentenceTaggerPredictor(model, dataset_reader=DatasetReader.from_params(params.pop("dataset_reader"))) # TODO: give predictor as parameter

        if "moving_average" in params:
            moving_average = MovingAverage.from_params(params.pop("moving_average"), parameters=parameters)
        else:
            moving_average = None

        if lr_scheduler_params:
            lr_scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params)
        else:
            lr_scheduler = None
        if momentum_scheduler_params:
            momentum_scheduler = MomentumScheduler.from_params(optimizer, momentum_scheduler_params)
        else:
            momentum_scheduler = None

        if 'checkpointer' in params:
            if 'keep_serialized_model_every_num_seconds' in params or \
                    'num_serialized_models_to_keep' in params:
                raise ConfigurationError(
                        "Checkpointer may be initialized either from the 'checkpointer' key or from the "
                        "keys 'num_serialized_models_to_keep' and 'keep_serialized_model_every_num_seconds'"
                        " but the passed config uses both methods.")
            checkpointer = Checkpointer.from_params(params.pop("checkpointer"))
        else:
            num_serialized_models_to_keep = params.pop_int("num_serialized_models_to_keep", 20)
            keep_serialized_model_every_num_seconds = params.pop_int(
                    "keep_serialized_model_every_num_seconds", None)
            checkpointer = Checkpointer(
                    serialization_dir=serialization_dir,
                    num_serialized_models_to_keep=num_serialized_models_to_keep,
                    keep_serialized_model_every_num_seconds=keep_serialized_model_every_num_seconds)
        model_save_interval = params.pop_float("model_save_interval", None)
        summary_interval = params.pop_int("summary_interval", 100)
        histogram_interval = params.pop_int("histogram_interval", None)
        should_log_parameter_statistics = params.pop_bool("should_log_parameter_statistics", True)
        should_log_learning_rate = params.pop_bool("should_log_learning_rate", False)
        log_batch_size_period = params.pop_int("log_batch_size_period", None)

        #THISISNEW: comet.ml setup
        comet_experiment_name= params.pop("comet_experiment_name", "amr-labels-experimentation")
        comet_experiment = Experiment(api_key="Yt3xk2gaFeevDwlxSNzN2VUKh",
                                project_name=comet_experiment_name, workspace="jgroschwitz", auto_metric_logging=False)

        params.assert_empty(cls.__name__)
        return cls(model, optimizer, iterator,
                   train_data, validation_data,
                   patience=patience,
                   validation_metric=validation_metric,
                   validation_iterator=validation_iterator,
                   shuffle=shuffle,
                   num_epochs=num_epochs,
                   serialization_dir=serialization_dir,
                   cuda_device=cuda_device,
                   grad_norm=grad_norm,
                   grad_clipping=grad_clipping,
                   learning_rate_scheduler=lr_scheduler,
                   momentum_scheduler=momentum_scheduler,
                   checkpointer=checkpointer,
                   model_save_interval=model_save_interval,
                   summary_interval=summary_interval,
                   histogram_interval=histogram_interval,
                   should_log_parameter_statistics=should_log_parameter_statistics,
                   should_log_learning_rate=should_log_learning_rate,
                   log_batch_size_period=log_batch_size_period,
                   moving_average=moving_average,
                   comet_experiment = comet_experiment,
                   predictor = predictor,# THISISNEW
                   prediction_log_file=prediction_log_file) # THISISNEW: storing the prediction log filepath here