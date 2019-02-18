# encoding=utf8
import os
from os import path

import tensorflow as tf
from sacred import Experiment

from data import get_iterator
from data import read_lines, accuracy_report
from model import LSTMCRF
from tfutils.training import BaseCallback, Checkpoint, get_session
from tfutils.utils import StatisticsKeyComparator, yaml_dump
from tfutils.utils import capture_all_exception
from tfutils.preprocessing import read_pre_embedding


class EvaluateStatistics:
    def __init__(self, model, base_dir, gold_true_path, id_to_tag):
        self.model = model
        self.base_dir = base_dir
        self.pred_path = os.path.join(base_dir, 'pred')
        self.gold_true_path = gold_true_path
        self.gold_true = read_lines(gold_true_path)
        self.id_to_tag = id_to_tag
        self.clear()

    @property
    def empty(self):
        return (not hasattr(self, 'statistics')) or not bool(self.statistics)

    def clear(self):
        self.prediction = []
        self.statistics = None

    def add(self, data, outputs):
        """
        data: a dict of batch of examples.
        outputs: a dict of keys and retrieved values.
        """
        logits, lengths = outputs['logits'], outputs['lengths']
        paths = self.model.decode(logits, lengths)
        self.prediction.extend([[self.id_to_tag[t] for t in p[:l]] for p, l in zip(paths, lengths)])

    def pop(self):
        """Return the dict and reset."""
        d = self.get_dict()
        self.clear()
        return d

    def get_dict(self):
        if self.statistics is None:
            assert len(self.gold_true) == len(self.prediction), 'prediction size is not equal to gold size'
            with open(self.pred_path, 'w') as f:
                for d, p in zip(self.gold_true, self.prediction):
                    d = d[0]
                    f.write('\n'.join([w + '\t' + t for w, t in zip(d, p)]))
                    f.write('\n\n')
            self.statistics = accuracy_report(self.pred_path, self.gold_true_path)
        return self.statistics

    def get(self, key):
        return self.get_dict()[key]

    def description(self, prefix='', digits=3):
        format_str = '{} {:.' + str(digits) + 'f}'
        return ', '.join([format_str.format(prefix + k, v) for k, v in self.get_dict().items()])


exp = Experiment('NER')


@exp.config
def config():
    model = {
        'char_dim': 100,
        'num_chars': 0,
        'seg_dim': 20,
        'num_segs': 0,
        'lstm_dim': 100,
        'num_tags': 0,
        'dropout_keep': 0.5,
    }
    emb_path = 'wiki_100.utf8'
    optimizer = 'adam'
    lr = 0.001
    clip_by_value = 5
    epochs = 100
    data = {
        'data_dir': 'data/example',
        'batch_size': 20,
        'vocab_size': 10000,
    }


@exp.main
def main(_config, _log, _run):
    with capture_all_exception(_run):
        base_dir = _run.observers[0].dir

        # data
        train, fields = get_iterator(path.join(_config['data']['data_dir'], 'train'),
                                     batch_size=_config['data']['batch_size'],
                                     vocab_size=_config['data']['vocab_size'],
                                     shuffle=True)
        dev_path = path.join(_config['data']['data_dir'], 'dev')
        dev, _ = get_iterator(dev_path,
                              batch_size=_config['data']['batch_size'],
                              shuffle=False, fields=fields)
        test_path = path.join(_config['data']['data_dir'], 'test')
        test, _ = get_iterator(test_path,
                               batch_size=_config['data']['batch_size'],
                               shuffle=False, fields=fields)

        # create model
        model_config = _config['model']
        model_config['num_chars'] = len(fields[0][1].vocab.itos)
        model_config['num_tags'] = len(fields[1][1].vocab.itos)
        model_config['num_segs'] = len(fields[2][1].vocab.itos)
        _log.info('model_config {}'.format(model_config))
        yaml_dump(_config, os.path.join(base_dir, 'config.yaml'))

        model = LSTMCRF(**model_config)
        _log.info('model repr: \n{}'.format(model))

        if _config['optimizer'] == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=_config['lr'])
        else:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=_config['lr'])
        model.compile(optimizer, _config['clip_by_value'])

        sess = get_session(initialize_vars=True)
        with sess.as_default():
            # load pretrained embedding
            emb_path = _config['emb_path']
            if path.exists(emb_path):
                _log.info("Try to load embedding from {}".format(emb_path))
                emb = read_pre_embedding(emb_path, fields[0][1].vocab.stoi)
                tf.assign(model.char_embedding.embedding, emb).eval()

            # callbacks, fit model
            checkpoint = Checkpoint(base_dir, evaluate_statistics=EvaluateStatistics(model, base_dir, dev_path,
                                                                                     fields[1][1].vocab.itos),
                                    cmp_metric=StatisticsKeyComparator('f1', 'greater'),
                                    logger=_log, checkpoint_every_n_epochs=1)
            base_callback = BaseCallback(base_dir, log_every_n_batches=200, train_statistic=['loss'],
                                         evaluate_statistic=['loss'], logger=_log)
            model.fit_generator(train, _config['epochs'], callbacks=[base_callback, checkpoint],
                                validation_data=dev)

            # test model performance
            checkpoint.restore(best=True)
            stat = EvaluateStatistics(model, base_dir, test_path, fields[1][1].vocab.itos)
            metric = BaseCallback(base_dir, evaluate_statistic=stat, logger=_log)
            _log.info('Testing on test dataset')
            model.evaluate_generator(test, callbacks=[metric])


if __name__ == "__main__":
    exp.run_commandline()
