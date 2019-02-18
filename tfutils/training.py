"""

"""
import warnings
import numpy as np
import time
import datetime
import math
import os
import tensorflow as tf
from collections import defaultdict

from tfutils.modules import Module
from tfutils.utils import get_logger, get_dict_dataloader, \
    get_statistics, StatisticsKeyComparator, save_dict, is_trainable, restore_latest_checkpoint

# global session
_SESSION = None


def set_session(session):
    """Sets the global TensorFlow session.

    # Arguments
        session: A TF Session.
    """
    global _SESSION
    _SESSION = session


def get_session(gpu_options=None, initialize_vars=False):
    """Returns the TF session to be used by the backend.

    If a default TensorFlow session is available, we will return it.

    Else, we will return the global session.

    If no global session exists at this point:
    we will create a new global session.

    # Returns
        A TensorFlow session.
    """
    global _SESSION

    default_session = tf.get_default_session()

    if default_session is not None:
        session = default_session
    else:
        if _SESSION is None:
            gpu_options = gpu_options or tf.GPUOptions(allow_growth=True)
            config = tf.ConfigProto(gpu_options=gpu_options)
            _SESSION = tf.Session(config=config)
        session = _SESSION
    if initialize_vars:
        with session.graph.as_default():
            variables = tf.global_variables()
            candidate_vars = []
            for v in variables:
                if not getattr(v, '_keras_initialized', False):
                    candidate_vars.append(v)
            if candidate_vars:
                # This step is expensive, so we only run it on variables
                # not already marked as initialized.
                is_initialized = session.run(
                    [tf.is_variable_initialized(v) for v in candidate_vars])
                uninitialized_vars = []
                for flag, v in zip(is_initialized, candidate_vars):
                    if not flag:
                        uninitialized_vars.append(v)
                    v._keras_initialized = True
                if uninitialized_vars:
                    session.run(tf.variables_initializer(uninitialized_vars))
    # hack for list_devices() function.
    # list_devices() function is not available under tensorflow r1.3.
    return session


def format_time(seconds):
    return str(datetime.timedelta(seconds=seconds))


class Trainer(Module):

    def compile(self, optimizer, clip_by_value=None, l2=None):
        """Add training operations."""
        self.optimizer = optimizer
        self.clip_by_value = clip_by_value
        self.l2 = l2
        # Add l2 loss if needed
        if l2:
            vars = self.collect_dict(Module.TRAINABLE_WEIGHTS, dict_values=True)
            # Filter bias
            vars = [v for v in vars if len(v.shape) > 1]
            l2loss = tf.multiply(tf.add_n([tf.nn.l2_loss(v) for v in vars]), l2, name='{}/l2loss'.format(self.name))
            self.add_to_dict(self.LOSSES, l2loss)

        return self

    def get_feed_fetch_dict(self):
        """Get additional feed dict, fetch dic(outputs and updates).
        Buffer operations to avoid creating them multiple times.
        """
        if self.mode in self._feed_fetches:
            return self._feed_fetches[self.mode]

        fetches = {}
        outputs = self.collect_dict(Module.OUTPUTS)
        fetches.update(outputs=outputs)
        if self.mode in (Module.TRAIN, Module.EVALUATE):
            if self._loss is not None:
                loss = self._loss
            else:
                losses = tf.stack(self.collect_dict(Module.LOSSES, dict_values=True), axis=0, name='losses')
                loss = tf.reduce_sum(losses, axis=0, keepdims=False, name='loss')
                assert 'loss' not in outputs, 'loss should not in outputs'
                self._loss = loss
            fetches['outputs'].update(loss=loss)

            if self.mode == Module.TRAIN:
                updates = self.collect_dict(Module.UPDATES, dict_values=True)
                if not hasattr(self, 'optimizer'):
                    warnings.warn('optimizer is not set')
                else:
                    vars = self.collect_dict(Module.TRAINABLE_WEIGHTS, dict_values=True)
                    grads_vars = self.optimizer.compute_gradients(loss, var_list=vars)
                    if self.clip_by_value:
                        grads_vars = [[tf.clip_by_value(g, -self.clip_by_value, self.clip_by_value), v]
                                      for g, v in grads_vars]
                    op = self.optimizer.apply_gradients(grads_vars, name='apply_gradients')
                    updates.append(op)
                fetches.update(updates=updates)
        feed_dict = self.collect_dict(Module.ADDITIONAL_FEED_DICT)

        self._feed_fetches[self.mode] = [feed_dict, fetches]
        return feed_dict, fetches

    def run_batch(self, batch, additional_feed_dict, fetch_dict, sess):
        # additional feed_dict
        feed_dict = {}
        feed_dict.update(additional_feed_dict)
        feed_dict.update(batch)
        return sess.run(fetch_dict, feed_dict)

    def fit(self, data,
            batch_size,
            epochs,
            verbose=1,
            callbacks=None,
            validation_split=0,
            validation_data=None,
            validation_every_n_epochs=1,
            shuffle=True):
        """
        Consider dict of key: numpy array as input.
        """

        # Check length
        items = list(data.items())
        length = len(items[0][1])
        for k, v in items:
            if len(v) != length:
                raise RuntimeError('length of {} is not equal to {}'.format(k, length))

        # split data
        if validation_split > 0:
            validation_size = validation_split if validation_split > 1 else int(length * validation_split)
            assert 0 < validation_size < length, 'validation_split should be (0, 1) or a integer size.'
            assert validation_data is None, 'validation_split should not be specified when validation_data exists.'
            idx = np.arange(0, length)
            if shuffle:
                np.random.shuffle(idx)
            data = {k: v[idx[:-validation_size]] for k, v in items}
            validation_data = {k: v[idx[-validation_size:]] for k, v in items}
            if verbose:
                k = items[0][0]
                print('training size', len(data[k]), 'evaluation size', len(validation_data[k]))

        data = get_dict_dataloader(data, batch_size, shuffle=shuffle)
        if validation_data:
            validation_data = get_dict_dataloader(validation_data, batch_size, shuffle=False)
        self.fit_generator(data, epochs, verbose, callbacks, validation_data,
                           validation_every_n_epochs)

    def fit_generator(self, data,
                      epochs,
                      verbose=1,
                      callbacks=None,
                      validation_data=None,
                      validation_every_n_epochs=1):
        """
        data: a DataLoader has length and generate batches of data.
        """
        callbacks = callbacks or []
        # setup callbacks and its parameters
        callbacks = CallbackList(callbacks).set_model(self).set_params({'verbose': verbose,
                                                                        'epochs': epochs})

        train_add_feed, train_fetches = self.set_mode(train=True).get_feed_fetch_dict()

        if validation_data:
            if validation_every_n_epochs is None:
                raise RuntimeError('validation_every_n_epochs')
            validation_add_feed, validation_fetches = self.set_mode(evaluate=True).get_feed_fetch_dict()

        # training begin
        self.stop_training = False
        sess = get_session(initialize_vars=True)
        # initializing operations
        sess.run(self.collect_dict(Module.INITIALIZERS))
        callbacks.on_train_begin()

        for e in range(1, 1 + epochs):
            # epoch begin
            self.set_mode(train=True)
            callbacks.on_epoch_begin(e)
            if self.stop_training:
                break
            for batch in data:
                # train batch begin
                callbacks.on_batch_begin(batch)
                fetches = self.run_batch(batch, train_add_feed, train_fetches, sess)
                # train batch end
                callbacks.on_batch_end(batch, fetches['outputs'])
                if self.stop_training:
                    break
            callbacks.on_epoch_end(e, locals())
            # Try evaluation
            if e % validation_every_n_epochs == 0 and validation_data:
                # validation epoch begin
                self.set_mode(evaluate=True)
                callbacks.on_epoch_begin(e)
                for batch in validation_data:
                    # validation batch begin
                    callbacks.on_batch_begin(batch)
                    fetches = self.run_batch(batch, validation_add_feed, validation_fetches, sess)
                    # validation batch end
                    callbacks.on_batch_end(batch, fetches['outputs'])
                # validation epoch end
                callbacks.on_epoch_end(e)
        # training end
        callbacks.on_train_end()

    def evaluate(self, data, batch_size=32, callbacks=None, verbose=1):
        self.set_mode(evaluate=True)
        assert isinstance(data, dict), 'Dict input data or single input data is needed'
        data = get_dict_dataloader(data, batch_size=batch_size, shuffle=False, drop_last=False)
        self.evaluate_generator(data, callbacks, verbose)

    def evaluate_generator(self, data, callbacks=None, verbose=1):
        sess = get_session(initialize_vars=False)
        assert callbacks is not None, 'callbacks should be not empty.'
        callbacks = CallbackList(callbacks) if isinstance(callbacks, (list, tuple)) else callbacks
        callbacks.set_params({'epochs': 1, 'verbose': verbose})
        callbacks.set_model(self)
        validation_add_feed, validation_fetches = self.set_mode(evaluate=True).get_feed_fetch_dict()

        callbacks.on_epoch_begin(1)
        for batch in data:
            callbacks.on_epoch_begin(batch)
            fetches = self.run_batch(batch, validation_add_feed, validation_fetches, sess)
            callbacks.on_batch_end(batch, fetches['outputs'])
        callbacks.on_epoch_end(1)

    def predict(self, data, batch_size=32, out_keys=None, concatenate=None, callbacks=None, verbose=1):
        """As predict_generator."""
        self.set_mode(predict=True)
        assert isinstance(data, dict), 'Dict input data or single input data is needed'
        data = get_dict_dataloader(data, batch_size=batch_size, shuffle=False, drop_last=False)
        return self.predict_generator(data, out_keys, concatenate, callbacks, verbose)

    def predict_generator(self, data, out_keys=None, concatenate=None, callbacks=None, verbose=1):
        """Collate the output keys or use a callback.
        if out_keys is not None, filter all outputs who are not in out_keys.
        concatenate: if not None, concatenate outputs specified in out_keys
        """
        callbacks = callbacks or []
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.set_params({'epochs': 1, 'verbose': verbose})

        add_feed, fetches = self.set_mode(predict=True).get_feed_fetch_dict()
        outputs = fetches['outputs']
        if out_keys is not None:
            filter_keys = set(outputs.keys()) - set(out_keys)
            for k in filter_keys:
                outputs.pop(k)
        results = []
        sess = get_session(initialize_vars=False)

        callbacks.on_epoch_begin(1)
        for batch in data:
            callbacks.on_batch_begin(batch)
            f = self.run_batch(batch, add_feed, fetches, sess)
            if concatenate:
                results.append(f['outputs'])
            callbacks.on_batch_end(batch, f['outputs'])
        callbacks.on_epoch_end(1)

        if results:
            results = concatenate(results)
            results = [results[k] for k in out_keys]
            if len(results) == 1:
                results = results[0]
        return results


class Callback(object):
    """Abstract base class used to build new callbacks.

    # Properties
        params: dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: Reference of the model being trained.

    """

    def __init__(self):
        self.params = None
        self.model = None
        self.epoch_counter = defaultdict(int)
        self.batch_counter = defaultdict(int)

    def set_params(self, params):
        self.params = params
        return self

    def set_model(self, model):
        self.model = model
        return self

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        mode = self.model.mode
        self.epoch_counter[mode] += 1
        if mode != 'train':
            self.epoch_counter[mode] = 0

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        mode = self.model.mode
        self.batch_counter[mode] += 1

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class BaseCallback(Callback):

    def __init__(self, base_dir,
                 logger=None,
                 log_every_n_epochs=None,
                 log_every_n_batches=None,
                 train_statistic=None,
                 evaluate_statistic=None):
        super(BaseCallback, self).__init__()
        os.makedirs(base_dir, exist_ok=True)
        self.base_dir = base_dir
        self.logger = logger or get_logger(os.path.join(base_dir, 'log'))
        if log_every_n_epochs is None and log_every_n_batches is None:
            log_every_n_epochs = 1
        self.log_every_n_batches = log_every_n_batches or math.inf
        self.log_every_n_epochs = log_every_n_epochs or math.inf
        self.train_statistic = get_statistics(train_statistic)
        self.evaluate_statistic = get_statistics(evaluate_statistic)
        self.final_train_statistic = {}
        self.final_evaluate_statistic = {}

        self.train_begin_time = None
        self.evaluate_begin_time = None

    def on_train_begin(self, logs=None):
        self.train_begin_time = time.time()

    def on_train_end(self, logs=None):
        last = time.time() - self.train_begin_time
        if not self.train_statistic.empty:
            self.final_train_statistic = self.train_statistic.pop()
        self.logger.info('Training statistics {}'.format(str(self.final_train_statistic)))
        self.logger.info('Evaluation statistics {}'.format(str(self.final_evaluate_statistic)))
        self.logger.info('Training time {}'.format(format_time(last)))

    def on_epoch_begin(self, epoch, logs=None):
        mode = self.model.mode
        if mode == Module.EVALUATE and self.evaluate_statistic:
            self.evaluate_begin_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        super(BaseCallback, self).on_epoch_end(epoch, logs)
        mode = self.model.mode
        if mode == Module.EVALUATE and self.evaluate_statistic:
            last = time.time() - self.evaluate_begin_time
            self.logger.info('Evaluation time {}, evaluation statistic {}'
                             .format(last, self.evaluate_statistic.description()))
            self.final_evaluate_statistic = self.evaluate_statistic.pop()

        else:
            if self.epoch_counter[mode] % self.log_every_n_epochs == 0:
                last = time.time() - self.train_begin_time
                self.logger.info('Time used {}, epoch {}/{}, training statistics {}'
                                 .format(format_time(last), epoch,  self.params['epochs'],
                                         self.train_statistic.description()))
                self.final_train_statistic = self.train_statistic.pop()

    def on_batch_begin(self, batch, logs=None):
        super(BaseCallback, self).on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        super(BaseCallback, self).on_batch_end(batch, logs)
        mode = self.model.mode
        outputs = logs or {}
        if mode == Module.TRAIN:
            self.train_statistic.add(batch, outputs)
            if self.batch_counter[mode] % self.log_every_n_batches == 0:
                self.logger.info('Epoch {}/{}, batch {}, training: {}'
                                 .format(self.epoch_counter[mode] + 1, self.params['epochs'],
                                         self.batch_counter[mode],
                                         self.train_statistic.description()))
                self.final_train_statistic = self.train_statistic.pop()

        elif mode == Module.EVALUATE:
            self.evaluate_statistic.add(batch, outputs)


class Checkpoint(Callback):

    def __init__(self, base_dir,
                 logger=None,
                 max_to_keep=1,
                 checkpoint_every_n_epochs=None,
                 evaluate_statistics=None,  # evaluate statistics
                 cmp_metric=None,  # default compare loss
                 early_stop=None, ):
        """"""
        super(Checkpoint, self).__init__()
        os.makedirs(base_dir, exist_ok=True)
        self.ckpts_dir = os.path.join(base_dir, 'checkpoints')
        self.ckpts_filename = os.path.join(self.ckpts_dir, 'checkpoint')
        self.best_ckpt_dir = os.path.join(base_dir, 'best_checkpoint')
        self.best_ckpt_filename = os.path.join(self.best_ckpt_dir, 'checkpoint')
        os.makedirs(self.ckpts_dir, exist_ok=True)
        os.makedirs(self.best_ckpt_dir, exist_ok=True)

        self.base_dir = base_dir
        self.logger = logger or get_logger(os.path.join(base_dir, 'log'))
        self.max_to_keep = max_to_keep
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs or math.inf
        self.evaluate_statistics = get_statistics(evaluate_statistics)
        self.cmp_metric = cmp_metric or StatisticsKeyComparator()
        self.early_stop = early_stop or math.inf
        self._stop_counter = 0
        self.best_evaluate_statistics = None
        self.ckpts_saver = tf.train.Saver(max_to_keep=max_to_keep)
        self.best_ckpt_saver = tf.train.Saver(max_to_keep=1)

    def on_epoch_end(self, epoch, logs=None):
        super(Checkpoint, self).on_epoch_end(epoch, logs)
        mode = self.model.mode
        # save checkpoints
        if mode == Module.TRAIN and self.epoch_counter[mode] % self.checkpoint_every_n_epochs == 0:
            prefix = self.ckpts_saver.save(get_session(initialize_vars=False),
                                           save_path=self.ckpts_filename,
                                           global_step=epoch)
        # evaluate
        if mode == Module.EVALUATE and self.evaluate_statistics:
            r = self.evaluate_statistics.pop()
            r.update(epoch=epoch)
            if self.best_evaluate_statistics is None or self.cmp_metric(new=r,
                                                                        old=self.best_evaluate_statistics):
                self.best_evaluate_statistics = r
                self._stop_counter = 0
                self.logger.info('Better evaluation result {}'.format(str(r)))
                # save checkpoint
                prefix = self.best_ckpt_saver.save(get_session(initialize_vars=False),
                                                   save_path=self.best_ckpt_filename,
                                                   global_step=epoch)
                save_dict(r, prefix + '.yaml')
            else:
                self._stop_counter += 1
                self.logger.info('Evaluation counter {}'.format(self._stop_counter))
            if self._stop_counter == self.early_stop:
                self.logger.info('Early stop triggered.')
                self.model.stop_training = True

    def on_batch_end(self, batch, logs=None):
        super(Checkpoint, self).on_batch_end(batch, logs)
        mode = self.model.mode
        logs = logs or {}
        if mode == Module.EVALUATE and self.evaluate_statistics:
            self.evaluate_statistics.add(batch, logs)

    def on_train_end(self, logs=None):
        super(Checkpoint, self).on_train_end(logs)
        self.logger.info('Best evaluation statistics {}'.format(str(self.best_evaluate_statistics)))

    def restore(self, best=False, last=False, save_dir=None):
        """Restore checkpoint."""
        if best:
            d = save_dir or self.best_ckpt_dir
            if os.path.exists(d):
                self.logger.info('Found best checkpoint dir {}'.format(d))
                restore_latest_checkpoint(self.best_ckpt_saver, get_session(initialize_vars=False), d)
                return
            self.logger.info('Best checkpoint dir does not exist.')
        if last:
            d = save_dir or self.ckpts_filename
            if os.path.exists(d):
                self.logger.info('Found checkpoint dir {}'.format(d))
                restore_latest_checkpoint(self.ckpts_saver, get_session(initialize_vars=False), d)
                return
            self.logger.info('Checkpoint dir does not exist.')
        self.logger.warning('No checkpoint found')


class CallbackList(object):
    """Container abstracting a list of callbacks.

    # Arguments
        callbacks: List of `Callback` instances.
    """

    def __init__(self, callbacks=None):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)
        return self

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)
        return self

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.

        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.

        # Arguments
            epoch: integer, index of epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        """Called right before processing a batch.

        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        """Called at the end of a batch.

        # Arguments
            batch: integer, index of batch within the current epoch.
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        """Called at the beginning of training.

        # Arguments
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        """Called at the end of training.

        # Arguments
            logs: dictionary of logs.
        """
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def __iter__(self):
        return iter(self.callbacks)
