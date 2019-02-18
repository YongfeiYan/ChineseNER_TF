import tensorflow as tf
from tfutils.modules import Sequential, get_embedding, LSTM, Dense, Dropout
from tfutils.training import Trainer
from tfutils.modules import Module
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.crf import *
import numpy as np


class LSTMCRF(Trainer):

    def __init__(self, char_dim, num_chars, seg_dim, num_segs, dropout_keep,
                 lstm_dim, num_tags):
        super(LSTMCRF, self).__init__(name='LSTMCRF')
        self.char_dim = char_dim
        self.num_chars = num_chars
        self.seg_dim = seg_dim
        self.num_segs = num_segs
        self.lstm_dim = lstm_dim
        self.num_tags = num_tags
        self.dropout_keep = dropout_keep
        self.inputs_feed_dict = {}

        initializer = initializers.xavier_initializer()
        self.initializer = initializer

        # Add placeholders for the model
        self.char_inputs = tf.placeholder(dtype=tf.int32,
                                          shape=[None, None],
                                          name="ChatInputs")
        self.inputs_feed_dict['sent'] = self.char_inputs
        self.char_embedding = get_embedding(num_embeddings=self.num_chars, embedding_dim=self.char_dim,
                                            trainable=True, name='char_embedding', initializer=initializer)

        embed = [self.char_embedding(self.char_inputs)]

        if self.seg_dim:
            self.seg_inputs = tf.placeholder(dtype=tf.int32,
                                             shape=[None, None],
                                             name="SegInputs")
            self.inputs_feed_dict['seg'] = self.seg_inputs
            self.seg_embedding = get_embedding(num_embeddings=self.num_segs, embedding_dim=self.seg_dim,
                                               trainable=True, name='seg_embedding', initializer=initializer)
            embed.append(self.seg_embedding(self.seg_inputs))

        embed = tf.concat(embed, axis=-1)
        self.dropout = Dropout(dropout_keep)

        lstm_inputs = self.dropout(embed)
        self.lstm = LSTM(self.lstm_dim, use_peepholes=True, initializer=initializer, bidirectional=True)
        used = tf.sign(tf.abs(self.char_inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32, name='length')
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]
        lstm_outputs, (_, _) = self.lstm(lstm_inputs, sequence_length=self.lengths)

        self.projection = Sequential([Dense(2 * self.lstm_dim, self.lstm_dim,
                                            activation='tanh', kernel_initializer=initializer),
                                      Dense(self.lstm_dim, self.num_tags,
                                            kernel_initializer=initializer)])
        self.logits = self.projection(lstm_outputs)

        self.targets = tf.placeholder(dtype=tf.int32,
                                      shape=[None, None],
                                      name="Targets")
        self.inputs_feed_dict['tag'] = self.targets
        self.loss = self.loss_layer(self.logits, self.lengths)

        self.add_to_dict(Module.OUTPUTS, {'logits': self.logits, 'lengths': self.lengths})
        self.add_to_dict(Module.LOSSES, self.loss)

    def run_batch(self, batch, additional_feed_dict, fetch_dict, sess):
        """
        batch: dict, each key contains data used to feed a placeholder.
        additional_feed_dict: feed_dict of dropout
        fetch_dict: fetch operations during training and evaluation.
        """

        feed_dict = {}
        feed_dict.update(additional_feed_dict)
        for k in self.inputs_feed_dict:
            feed_dict[self.inputs_feed_dict[k]] = batch[k]
        return sess.run(fetch_dict, feed_dict)

    def loss_layer(self, project_logits, lengths, name=None):
        """
        calculate crf loss
        :param project_logits: [batch_size, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss" if not name else name):
            small = -1000.0
            # pad logits for crf loss
            start_logits = tf.concat(
                [small * tf.ones(shape=[self.batch_size, 1, self.num_tags]), tf.zeros(shape=[self.batch_size, 1, 1])],
                axis=-1)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.num_tags * tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)

            trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags + 1, self.num_tags + 1],
                initializer=self.initializer)
            log_likelihood, self.trans = crf_log_likelihood(
                inputs=logits,
                tag_indices=targets,
                transition_params=trans,
                sequence_lengths=lengths + 1)
            return tf.reduce_mean(-log_likelihood, name='loss')

    def decode(self, logits, lengths):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :return:
        """
        # inference final labels use Viterbi Algorithm
        paths = []
        matrix = self.trans.eval()
        small = -1000.0
        start = np.asarray([[small] * self.num_tags + [0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = viterbi_decode(logits, matrix)

            paths.append(path[1:])
        return paths
