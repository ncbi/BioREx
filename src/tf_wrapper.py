# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:01:26 2021

@author: laip2
"""

import datetime
import math
import os
from typing import Callable, Dict, Optional, Tuple

from transformers.file_utils import ENV_VARS_TRUE_VALUES


# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    is_comet_available,
    is_wandb_available,
)

import numpy as np
import tensorflow as tf
from tensorflow.python.distribute.values import PerReplica

from transformers.modeling_tf_utils import TFPreTrainedModel
from transformers.optimization_tf import GradientAccumulator, create_optimizer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction, IntervalStrategy, PredictionOutput, set_seed
from transformers.training_args_tf import TFTrainingArguments
from transformers.utils import logging
from transformers import TFTrainer

if is_wandb_available():
    import wandb

if is_comet_available():
    import comet_ml

logger = logging.get_logger(__name__)

'''
    refers to https://github.com/huggingface/transformers/blob/1c06240e1b3477728129bb58e7b6c7734bb5074e/src/transformers/trainer_tf.py
'''
class TFTrainerWrapper(TFTrainer):
    
    def __init__(
            self,
            model: TFPreTrainedModel,
            args: TFTrainingArguments,
            train_dataset: tf.data.Dataset,
            eval_dataset: tf.data.Dataset,
            compute_metrics: Callable[[EvalPrediction], Dict],
            main_metric_name: str,
            tb_writer: Optional[tf.summary.SummaryWriter] = None,
            optimizers: Tuple[tf.keras.optimizers.Optimizer, tf.keras.optimizers.schedules.LearningRateSchedule] = (
                None,
                None,
            ),
        ):
        super().__init__(
            model                = model,
            args                 = args,
            train_dataset        = train_dataset,
            eval_dataset         = eval_dataset,
            compute_metrics      = compute_metrics,
            tb_writer            = tb_writer,
            optimizers           = optimizers)
        
        # below refers to line 366 of https://github.com/huggingface/transformers/blob/1c06240e1b3477728129bb58e7b6c7734bb5074e/src/transformers/trainer_tf.py#L287
        self.main_metric_name = main_metric_name if main_metric_name.startswith("eval_") else "eval_" + main_metric_name


    '''def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        TFTrainer's init through :obj:`optimizers`, or subclass and override this method.
        """
        if not self.optimizer and not self.lr_scheduler:
            warmup_steps = (
                self.args.warmup_steps
                if self.args.warmup_steps > 0
                else math.ceil(num_training_steps * self.args.warmup_ratio)
            )
    
            self.optimizer, self.lr_scheduler = create_optimizer(
                self.args.learning_rate,
                num_training_steps,
                warmup_steps,
                adam_beta1=self.args.adam_beta1,
                adam_beta2=self.args.adam_beta2,
                adam_epsilon=self.args.adam_epsilon,
                weight_decay_rate=self.args.weight_decay,
                power=self.args.poly_power,
            )'''
            
    def train(self, output_dir: Optional[str] = None) -> None:
        """
        Train method to train the model.
        """
        
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        
        train_ds = self.get_train_tfdataset()

        if self.args.debug:
            tf.summary.trace_on(graph=True, profiler=True)

        self.gradient_accumulator.reset()

        num_update_steps_per_epoch = self.num_train_examples / self.total_train_batch_size

        # In fact, ``self.args.dataloader_drop_last`` has no effect in `trainer_tf.py`, because
        # the dataset is repeated before being batched.
        # It has the effect only when TPU is used which requires explicit tensor shape in order to make
        # the gradient accumulation implementation work.
        approx = math.floor if self.args.dataloader_drop_last else math.ceil
        num_update_steps_per_epoch = approx(num_update_steps_per_epoch)

        # At least one update for each epoch.
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        self.steps_per_epoch = num_update_steps_per_epoch

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            epochs = (self.args.max_steps // self.steps_per_epoch) + int(
                self.args.max_steps % self.steps_per_epoch > 0
            )
        else:
            t_total = self.steps_per_epoch * self.args.num_train_epochs
            epochs = self.args.num_train_epochs

        # Since ``self.args.num_train_epochs`` can be `float`, we make ``epochs`` be a `float` always.
        epochs = float(epochs)

        with self.args.strategy.scope():
            self.create_optimizer_and_scheduler(num_training_steps=t_total)
            folder = os.path.join(self.args.output_dir, PREFIX_CHECKPOINT_DIR)
            ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
            self.model.ckpt_manager = tf.train.CheckpointManager(ckpt, folder, max_to_keep=self.args.save_total_limit)

            iterations = self.optimizer.iterations
            epochs_trained = 0
            steps_trained_in_current_epoch = 0
            if self.model.ckpt_manager.latest_checkpoint:

                logger.info(
                    "Checkpoint file %s found and restoring from checkpoint", self.model.ckpt_manager.latest_checkpoint
                )
                ckpt.restore(self.model.ckpt_manager.latest_checkpoint).expect_partial()

                self.global_step = iterations.numpy()

                epochs_trained = self.global_step // self.steps_per_epoch
                steps_trained_in_current_epoch = self.global_step % self.steps_per_epoch

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

            tf.summary.experimental.set_step(self.global_step)

            with self.tb_writer.as_default():
                tf.summary.text("args", self.args.to_json_string())

            self.tb_writer.flush()

            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", self.num_train_examples)
            # TODO: We might want to print a more precise ``epochs`` if self.args.max_steps > 0 ?
            logger.info("  Num Epochs = %d", epochs)
            logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
            logger.info(
                "  Total train batch size (w. parallel, distributed & accumulation) = %d", self.total_train_batch_size
            )
            logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
            logger.info("  Steps per epoch = %d", self.steps_per_epoch)
            logger.info("  Total optimization steps = %d", t_total)

            self.train_loss = tf.keras.metrics.Sum()
            start_time = datetime.datetime.now()


            
            best_result = -1.
            for epoch_iter in range(epochs_trained, int(epochs)):
                # Reset the past mems state at the beginning of each epoch if necessary.
                if self.args.past_index >= 0:
                    self._past = None

                for step, batch in enumerate(train_ds):

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        continue

                    self.distributed_training_steps(batch)

                    self.global_step = iterations.numpy()
                    self.epoch_logging = epoch_iter + (step + 1) / self.steps_per_epoch

                    training_loss = self.train_loss.result() / (step + 1)

                    if self.args.debug:
                        logs = {}
                        logs["loss"] = training_loss.numpy()
                        logs["epoch"] = self.epoch_logging

                        self.log(logs)

                    if self.global_step == 1 and self.args.debug:
                        with self.tb_writer.as_default():
                            tf.summary.trace_export(
                                name="training", step=self.global_step, profiler_outdir=self.args.logging_dir
                            )

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs = {}
                        logs["loss"] = training_loss.numpy()
                        logs["learning_rate"] = self.lr_scheduler(self.global_step).numpy()
                        logs["epoch"] = self.epoch_logging

                        self.log(logs)
                    '''
                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        ckpt_save_path = self.model.ckpt_manager.save()

                        logger.info("Saving checkpoint for step {} at {}".format(self.global_step, ckpt_save_path))
                    '''

                    if self.args.max_steps > 0 and self.global_step >= t_total:
                        break

                    if self.global_step % self.steps_per_epoch == 0:
                        break

                result = self.evaluate()
                if best_result < result[self.main_metric_name]:#save_model
                    best_result = result[self.main_metric_name]
                    #self.save_model()
                    #ckpt_save_path = self.model.ckpt_manager.save()

                    logger.info("Saving model in {}".format(output_dir))
            
                    if not isinstance(self.model, TFPreTrainedModel):
                        raise ValueError("Trainer.model appears to not be a PreTrainedModel")
            
                    self.model.save_pretrained(output_dir)
                    

                    #logger.info("Saving checkpoint for step {} at {}".format(self.global_step, ckpt_save_path))

                self.train_loss.reset_states()

                if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                    break

            end_time = datetime.datetime.now()

            logger.info("Training took: {}".format(str(end_time - start_time)))

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")