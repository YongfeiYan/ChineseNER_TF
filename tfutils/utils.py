"""

"""
import tensorflow as tf
import logging
import warnings
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
import os
import sys
from contextlib import contextmanager
from traceback import print_tb
from collections import abc
import yaml
from itertools import chain
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import rnn

##############################################################
# Experiments
##############################################################


@contextmanager
def capture_all_exception(_run):
    """Capture all Errors and Exceptions, print traceback and flush stdout stderr."""
    try:
        yield None

    except Exception:
        exc_type, exc_value, trace = sys.exc_info()
        print(exc_type, exc_value, trace)
        print_tb(trace)

    finally:
        sys.stdout.flush()
        sys.stderr.flush()


_LOGGER = None


def get_logger(log_file, use_global=True):
    """Set global _LOGGER if use_global."""
    global _LOGGER

    if use_global and _LOGGER:
        return _LOGGER

    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)

    if use_global:
        _LOGGER = logger
    return logger


##############################################################
# Statistics
##############################################################


class WeightedStatistic:
    def __init__(self, name, init, postprocessing=None):
        self.name = name
        self.init = init
        self.v = init
        self.w = 0
        self.postprocessing = postprocessing

    def add(self, v, w):
        self.v = self.v + v * w
        self.w = self.w + w

    def get(self):
        # No accumulated value
        if self.w == 0:
            return 0
        v = self.v / self.w
        if self.postprocessing is not None:
            v = self.postprocessing(v)
        return v

    def clear(self):
        self.v = self.init
        self.w = 0

    def __repr__(self):
        return '{} {:.5f}'.format(self.name, self.get())


class BatchSizeWeightedStatistics:
    def __init__(self, keys):
        self.keys = set(keys)
        self.statistics = None
        self._empty = True
        self.clear()

    @property
    def empty(self):
        return self._empty

    def clear(self):
        self._empty = True
        self.statistics = {k: WeightedStatistic(k, 0, None) for k in self.keys}

    def add(self, data, outputs):
        """
        data: a dict of batch of examples.
        outputs: a dict of keys and retrieved values.
        """
        self._empty = False
        w = 1
        keys = list(data.keys())
        if len(keys) == 0:
            warnings.warn('Empty data and set w = 1')
        else:
            w = len(data[keys[0]])
        for k, v in self.statistics.items():
            if k in outputs:
                v.add(outputs[k], w)

    def pop(self):
        """Return the dict and reset."""
        d = self.get_dict()
        self.clear()
        return d

    def get_dict(self):
        statistics = self.statistics or {}
        return {k: v.get() for k, v in statistics.items()}

    def get(self, key):
        return self.statistics[key].get()

    def description(self, prefix='', digits=3):
        format_str = '{} {:.' + str(digits) + 'f}'
        return ', '.join([format_str.format(prefix + k, v) for k, v in self.get_dict().items()])


class StatisticsKeyComparator(object):
    def __init__(self, key='loss', cmp='less'):
        if isinstance(cmp, str):
            cmp = np.less if cmp == 'less' else np.greater
        else:
            assert callable(cmp), 'cmp should be less, greater or callable'
        self.cmp = cmp
        self.key = key

    def __call__(self, new, old):
        """Return true if new is better than old.
        new, old: dict or statistics object which supports get method.
        """
        return self.cmp(new.get(self.key), old.get(self.key))


def get_statistics(stat):
    """
    stat: [k1, k2], then use BatchSize
    """
    if isinstance(stat, str):
        stat = [stat]
    if isinstance(stat, (list, tuple)):
        stat = BatchSizeWeightedStatistics(stat)
    return stat


##############################################################
# Dataset and DataLoader
##############################################################


def concatenate(batch):
    r"""Concatenate the values in a batch to a sample."""
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        return np.concatenate(batch, 0)
    elif isinstance(batch[0], int):
        return np.array(batch, dtype=np.int64)
    elif isinstance(batch[0], float):
        return np.array(batch, dtype=np.float32)
    elif isinstance(batch[0], (str, bytes)):
        return batch
    elif isinstance(batch[0], abc.Mapping):
        return {key: concatenate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], abc.Sequence):
        return list(chain(*batch))

    raise TypeError((error_msg.format(type(batch[0]))))


def collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        return np.stack(batch, 0)
    elif isinstance(batch[0], int):
        return np.array(batch, dtype=np.int64)
    elif isinstance(batch[0], float):
        return np.array(batch, dtype=np.float32)
    elif isinstance(batch[0], (str, bytes)):
        return batch
    elif isinstance(batch[0], abc.Mapping):
        return {key: collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], abc.Sequence):
        return batch

    raise TypeError((error_msg.format(type(batch[0]))))


class DictDataset(Dataset):

    def __init__(self, data):
        super(DictDataset, self).__init__()
        self.data = data
        items = list(data.items())
        length = len(items[0][1])
        for k, v in items:
            assert len(v) == length, 'length of {} is not equal to {}.'.format(k, length)
        self.length = length

    def __getitem__(self, item):
        return {k: v[item] for k, v in self.data.items()}

    def __len__(self):
        return self.length


def get_dict_dataloader(data, batch_size=1, shuffle=False, drop_last=False):
    assert isinstance(data, dict), 'data should be dict, got {}.'.format(type(data))
    dataset = DictDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            collate_fn=collate, drop_last=drop_last)
    return dataloader


class LoaderWrapper:
    def __init__(self, loader, fields=None, transform=None):
        """
        fields: dict of attrs
        if transform is not None, then transform will be used.
        """
        self.loader = loader
        self.fields = fields
        self.transform = transform
        assert fields or transform, 'fields or transform should not be both None.'

    def __iter__(self):
        for batch in self.loader:
            if callable(self.transform):
                yield self.transform(batch)
            else:
                yield {k: getattr(batch, v) for k, v in self.fields.items()}

#####################################################################
# Metrics
#####################################################################


def ndim(x):
    """Returns the number of axes in a tensor, as an integer.
    """
    return len(x.get_shape())


def binary_accuracy(y_true, y_pred):
    return tf.reduce_mean(tf.equal(y_true, tf.round(y_pred)), axis=-1)


def categorical_accuracy(y_true, y_pred, mean=True):
    acc = tf.cast(tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1)),
                  tf.float32)
    if mean:
        acc = tf.reduce_mean(acc, axis=-1)

    return acc


def sparse_categorical_accuracy(y_true, y_pred, mean=True):
    # reshape in case it's in shape (num_samples, 1) instead of (num_samples,)
    if ndim(y_true) == ndim(y_pred):
        y_true = tf.squeeze(y_true, -1)
    # convert dense predictions to labels
    y_pred_labels = tf.argmax(y_pred, axis=-1)
    y_pred_labels = tf.cast(y_pred_labels, y_true.dtype)
    acc = tf.cast(tf.equal(y_true, y_pred_labels), tf.float32)
    if mean:
        acc = tf.reduce_mean(acc, axis=-1)
    return acc


def top_k_categorical_accuracy(y_true, y_pred, k=5):
    return tf.reduce_mean(tf.nn.in_top_k(y_pred, tf.argmax(y_true, axis=-1), k), axis=-1)


def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
    # If the shape of y_true is (num_samples, 1), flatten to (num_samples,)
    return tf.reduce_mean(tf.nn.in_top_k(y_pred, tf.cast(tf.reshape(y_true, [-1]), 'int32'), k), axis=-1)


#####################################################################
# saving
#####################################################################


def yaml_load(data_path):
    with open(data_path) as f:
        return yaml.load(f)


def yaml_dump(data, data_path):
    with open(data_path, 'w') as f:
        yaml.dump(data, f)


def save_dict(d, filename):
    """Save dict as yaml."""

    def _map(v):
        if type(v).__module__ == 'numpy':
            return v.tolist()
        else:
            return v

    yaml_dump({k: _map(v) for k, v in d.items()}, filename)


#####################################################################
# tf operations
#####################################################################


def is_trainable(var):
    """Whether var is in TRAINABLE_VARIABLES."""
    return var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


def restore_latest_checkpoint(saver, sess, ckpts_dir):
    p = yaml_load(os.path.join(ckpts_dir, 'checkpoint'))['model_checkpoint_path']
    if not p.startswith('/'):
        p = os.path.join(ckpts_dir, p)
    saver.restore(sess, p)
