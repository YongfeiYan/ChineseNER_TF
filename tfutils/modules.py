import warnings
import tensorflow as tf
from collections import namedtuple, defaultdict, OrderedDict, abc
from itertools import chain
from tensorflow import keras
from tfutils.utils import is_trainable
import numpy as np
from os import path


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


class Module(object):

    _UID = defaultdict(int)
    # mode
    TRAIN = 'train'
    EVALUATE = 'evaluate'
    PREDICT = 'predict'
    # dictionaries
    INPUTS = '_inputs'
    ADDITIONAL_FEED_DICT = '_additional_feed_dict'
    OUTPUTS = '_outputs'
    UPDATES = '_updates'
    INITIALIZERS = '_initializers'
    MODULES = '_modules'
    TENSORS = '_tensors'
    LOSSES = '_losses'
    TRAINABLE_WEIGHTS = '_trainable_weights'
    NON_TRAINABLE_WEIGHTS = '_non_trainable_weights'
    DICTS = {INPUTS, ADDITIONAL_FEED_DICT, OUTPUTS, UPDATES, INITIALIZERS, MODULES, TENSORS,
             LOSSES, TRAINABLE_WEIGHTS,
             NON_TRAINABLE_WEIGHTS}

    @staticmethod
    def get_uid(prefix):
        Module._UID[prefix] += 1
        return Module._UID[prefix]

    def __init__(self, **kargs):
        allowed_kargs = {
            'name', 'batch_input_shapes', 'dtype',
            'trainable', 'mode',
        }
        for kwarg in kargs:
            if kwarg not in allowed_kargs:
                raise TypeError('Keyword argument not understood:', kwarg)
        name = kargs.get('name')
        if not name:
            prefix = self.__class__.__name__
            name = prefix + '_' + str(Module.get_uid(prefix))
        self.name = name

        self.batch_input_shapes = kargs.get('batch_input_shapes', None)
        self.trainable = kargs.get('trainable', True)
        self._mode = kargs.get('mode', 'train')
        self._feed_fetches = {}
        self._loss = None
        self.dtype = kargs.get('dtype', None)

        # dicts
        for k in Module.DICTS:
            setattr(self, k, {})

    @property
    def mode(self):
        return self._mode

    def set_mode(self, train=False, evaluate=False, predict=False):
        """Set mode recursively for all children."""
        assert int(train) + int(evaluate) + int(predict) == 1, 'Only one mode can be specified, got {}/{}/{}'.format(
            train, evaluate, predict
        )
        if train:
            mode = Module.TRAIN
        if evaluate:
            mode = Module.EVALUATE
        if predict:
            mode = Module.PREDICT

        def _fn(m):
            if isinstance(m, Module):
                m._mode = mode
        self.apply(_fn)
        return self

    def add_to_dict(self, key, values, exists_ok=False):
        """values: dict, list or single element of tf type(with name attributed)."""
        d = getattr(self, key)
        if not isinstance(values, (list, dict, tuple)):
            values = [values]
        if not isinstance(values, dict):
            values = {v.name: v for v in values}
        if not exists_ok:
            for k in values.keys():
                assert k not in d.keys(), '{} already added {}'.format(k, key)
        d.update(values)
        return self

    def get_dict(self, key):
        return getattr(self, key)

    def named_children_recursive(self, memo=None):
        """memo: a set of names to filter."""
        memo = memo or set()
        for k, v in chain(self.get_dict(Module.MODULES).items(),
                          self.get_dict(Module.INPUTS).items(),
                          self.get_dict(Module.TENSORS).items()
                          ):
            if k not in memo:
                memo.add(k)
                if isinstance(v, Module):
                    for _k, _v in v.named_children_recursive(memo):
                        yield _k, _v
                yield k, v

    def named_children(self):
        memo = set()
        for k, v in chain(self.get_dict(Module.MODULES).items(),
                          self.get_dict(Module.INPUTS).items(),
                          self.get_dict(Module.TENSORS).items()
                          ):
            if k not in memo:
                memo.add(k)
                yield k, v

    def children(self):
        for _, v in self.named_children():
            yield v

    def __setattr__(self, key, value):
        if isinstance(value, (tf.Tensor, tf.Variable, Module)):
            if isinstance(value, tf.Tensor):
                if value.op.type == 'Placeholder':
                    self.add_to_dict(Module.INPUTS, value)
                else:
                    self.add_to_dict(Module.TENSORS, value)
            elif isinstance(value, tf.Variable):
                if value.name not in self.get_dict(Module.TRAINABLE_WEIGHTS) and \
                        value.name not in self.get_dict(Module.NON_TRAINABLE_WEIGHTS):
                    if is_trainable(value):
                        self.add_to_dict(Module.TRAINABLE_WEIGHTS, value)
                    else:
                        self.add_to_dict(Module.NON_TRAINABLE_WEIGHTS, value)
            elif isinstance(value, Module):
                self.add_to_dict(Module.MODULES, value)

        self.__dict__[key] = value

    def to_yaml(self):
        raise NotImplementedError()

    def from_config(self):
        raise NotImplementedError()

    def add_weight(self,
                   name,
                   shape,
                   dtype=None,
                   initializer=None,
                   regularizer=None,
                   trainable=True):
        """Adds a weight variable to the layer.

        # Arguments
            name: String, the name for the weight variable.
            shape: The shape tuple of the weight.
            dtype: The dtype of the weight.
            initializer: An Initializer instance (callable). Or a value
            regularizer: An optional Regularizer instance.
            trainable: A boolean, whether the weight should
                be trained via backprop or not (assuming
                that the layer itself is also trainable).

        # Returns
            The created weight variable.
        """
        if isinstance(initializer, (str, abc.Callable)):
            initializer = keras.initializers.get(initializer)(shape)
        dtype = dtype or tf.float32

        weight = tf.Variable(initializer, dtype=tf.as_dtype(dtype), name=name, trainable=trainable)
        if regularizer is not None:
            with tf.name_scope('{}/weight_regularizer'.format(name)):
                self.add_to_dict(Module.LOSSES, regularizer(weight))
        if trainable:
            self.add_to_dict(Module.TRAINABLE_WEIGHTS, weight, exists_ok=False)
        else:
            self.add_to_dict(Module.NON_TRAINABLE_WEIGHTS, weight, exists_ok=False)

        return weight

    def collect_dict(self, key, init_value=None, dict_values=False):
        """Gather values of all modules which has the attribute attr."""
        init_value = init_value or {}
        assert isinstance(init_value, dict), 'init_value should be a dict'

        def _update(v):
            assert isinstance(v, dict), 'value {} is not dict'.format(v)
            for k in v.keys():
                if k in init_value.keys():
                    warnings.warn('{}: {} already exists and will be overwrote.'.format(k, init_value[k]))
            init_value.update(v)

        def _fn(m):
            if isinstance(m, Module):
                d = m.get_dict(key)
                _update(d)
        self.apply(_fn)

        if isinstance(init_value, dict) and dict_values:
            init_value = list(init_value.values())

        return init_value

    def apply(self, fn):
        """Apply a function recursively for all children uniquely.
        If a node has more than one parents, than this node will only be applied once.
        """
        for k, v in self.named_children_recursive():
            fn(v)
        fn(self)
        return self

    def __call__(self, *arg, **kargs):
        """This layer's operation logic."""
        raise NotImplementedError()

    def extra_repr(self):
        r"""Set the extra representation of the module

        To print customized extra information, you should reimplement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return ''

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        children = set(self.children())
        for k in self.__dict__:
            module = self.__dict__[k]
            if isinstance(module, (Module, tf.Tensor, tf.Variable)) and module in children:
                mod_str = repr(module)
                mod_str = _addindent(mod_str, 2)
                child_lines.append('(' + k + '): ' + mod_str)
                children.remove(module)
        for module in children:
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append(mod_str)
        lines = extra_lines + child_lines

        main_str = self.__class__.__name__ + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str


class Dense(Module):

    def __init__(self,
                 in_features,
                 out_features,
                 use_bias=True,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kargs):
        super(Dense, self).__init__(**kargs)
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.weight = self.add_weight('{}/weight'.format(self.name),
                                     (in_features, out_features),
                                     dtype=self.dtype,
                                     initializer=kernel_initializer,
                                     regularizer=kernel_regularizer)
        if use_bias:
            self.bias = self.add_weight('{}/bias'.format(self.name),
                                        (out_features,),
                                        dtype=self.dtype,
                                        initializer=bias_initializer,
                                        regularizer=bias_regularizer)
        self.activation = keras.activations.get(activation)

    def __call__(self, x):
        """Apply linear operation on the last dimension of x.
        x: (..., in_features)
        return:
            (..., out_features)
        """
        r = keras.backend.dot(x, self.weight)

        if self.use_bias:
            r = r + self.bias
        return self.activation(r)

    def extra_repr(self):
        s = '{}: {}'.format(self.weight.name, self.weight)
        if self.use_bias:
            s = s + '\n{}: {}'.format(self.bias.name, self.bias)
        s = s + '\n{}'.format(self.activation)
        return s


class Sequential(Module):

    def __init__(self, modules, **kargs):
        super(Sequential, self).__init__(**kargs)
        self._modules = OrderedDict()
        for m in modules:
            self.add(m)

    @property
    def modules(self):
        return list(self._modules.values())

    def add(self, m):
        self.add_to_dict(Module.MODULES, m)

    def pop(self, name=None):
        """If name is None, pop the last.
        Else pop the specified named module.
        """
        if name is None:
            name = list(self._modules.keys())[-1]
        if name not in self._modules:
            raise RuntimeError('{} does not exists'.format(name))
        self._modules.pop(name)

    def __call__(self, x):
        for m in self._modules.values():
            x = m(x)
        return x


class Dropout(Module):

    def __init__(self, keep_prob, noise_shape=None, seed=None, **kargs):
        super(Dropout, self).__init__(**kargs)
        assert not isinstance(keep_prob, tf.Tensor), 'keep_prob should be a float.'
        self._keep_prob = keep_prob
        self.keep_prob = tf.placeholder(shape=(), name='{}/keep_prob'.format(self.name), dtype=tf.float32)
        self.noise_shape = noise_shape
        self.seed = seed

    def get_dict(self, key):
        if key != Module.ADDITIONAL_FEED_DICT:
            return super(Dropout, self).get_dict(key)
        if not hasattr(self, '_keep_prob'):
            return {}

        return {self.keep_prob: self._keep_prob if self.mode == Module.TRAIN else 1}

    def __call__(self, x):
        return tf.nn.dropout(x, self.keep_prob, self.noise_shape, self.seed, self.name)

    def extra_repr(self):
        return 'keep_prob: {}'.format(self._keep_prob)


class Flatten(Module):

    def __call__(self, x):
        batch_size = tf.shape(x)[0]
        # return tf.reshape(x, [-1, tf.reduce_prod(x.shape[1:])])
        return tf.reshape(x, [batch_size, -1])


def get_embedding(vocab_path=None,
                  embedding_np=None,
                  num_embeddings=0, embedding_dim=0, **kargs):
    """Create embedding from:
    1. saved numpy vocab array, vocab_path, freeze
    2. numpy embedding array, embedding_np, freeze
    3. raw embedding n_vocab, embedding_dim
    """
    if isinstance(vocab_path, str) and path.exists(vocab_path):
        embedding_np = np.load(vocab_path)
    if embedding_np is not None:
        return Embedding.from_pretrained(tf.constant(embedding_np, dtype=tf.float32), **kargs)
    return Embedding(num_embeddings, embedding_dim, **kargs)


class Embedding(Module):

    def __init__(self, num_embeddings, embedding_dim, initializer='uniform', regularizer=None,
                 max_norm=None, **kargs):
        super(Embedding, self).__init__(**kargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = self.add_weight(self.name, shape=(num_embeddings, embedding_dim),
                                         dtype=self.dtype, initializer=initializer, regularizer=regularizer,
                                         trainable=self.trainable)
        self.max_norm = max_norm

    @classmethod
    def from_pretrained(cls, array, regularizer=None, max_norm=None, trainable=False, **kargs):
        """Array is a numpy array or a Tensor."""
        assert isinstance(array, (np.ndarray, tf.Tensor))
        assert len(array.shape) == 2, 'Embedding should be 2 dimensional.'
        return cls(array.shape[0], array.shape[1], initializer=array, regularizer=regularizer,
                   max_norm=max_norm, trainable=trainable, **kargs)

    def __call__(self, ids):
        """"""
        return tf.nn.embedding_lookup(self.embedding, ids, max_norm=self.max_norm)

    def extra_repr(self):
        return 'trainable: {}'.format(len(self.get_dict(self.TRAINABLE_WEIGHTS)) > 0) + \
               '\nnum_embeddings: {}'.format(self.num_embeddings) + \
               '\nembedding_dim: {}'.format(self.embedding_dim) + \
               '\nmax_norm: {}'.format(self.max_norm)


class LSTM(Module):
    def __init__(self, num_units,
                 use_peepholes=False,
                 cell_clip=None,
                 initializer=None,
                 num_layers=1,
                 keep_prob=None,
                 bidirectional=False,
                 **kargs):
        """keep_prob: dropout at after each layer."""
        super(LSTM, self).__init__(**kargs)
        self.num_units = num_units
        self.num_layers = num_layers
        self.keep_prob = keep_prob
        if keep_prob:
            self.dropout = Dropout(keep_prob, name='{}/dropout'.format(self.name))
        self.bidirectional = bidirectional
        self.forward_cells = []
        self.backward_cells = []
        self.variable_scope_names = []
        for i in range(num_layers):
            scope_name = '{}_layer_{}'.format(self.name, i)
            self.variable_scope_names.append(scope_name)
            with tf.variable_scope(scope_name) as vs:
                self.forward_cells.append(tf.nn.rnn_cell.LSTMCell(num_units, use_peepholes, cell_clip, initializer,
                                                                  state_is_tuple=True,
                                                                  name='forward'))
                if self.bidirectional:
                    self.backward_cells.append(tf.nn.rnn_cell.LSTMCell(num_units, use_peepholes, cell_clip, initializer,
                                                                       state_is_tuple=True,
                                                                       name='backward'))
                # variables are not created yet.

    def zero_state(self, batch_size, dtype=None):
        """Forward states, backward states.
        states: layers x (h, c)
        """
        dtype = dtype or tf.float32
        forward_states = tuple([c.zero_state(batch_size, dtype) for c in self.forward_cells])
        if not self.bidirectional:
            return forward_states
        backward_states = tuple([c.zero_state(batch_size, dtype) for c in self.backward_cells])
        return forward_states, backward_states

    def __call__(self, inputs, sequence_length=None, initial_states=None, dtype=None):
        if initial_states is None:
            batch_size = tf.shape(inputs)[0]
            initial_states = self.zero_state(batch_size, dtype)
        forward_states = [None] * self.num_layers
        backward_states = [None] * self.num_layers

        for i, n in enumerate(self.variable_scope_names):
            with tf.variable_scope(n) as vs:
                if not self.bidirectional:
                    inputs, f = tf.nn.dynamic_rnn(self.forward_cells[i], inputs, sequence_length,
                                                  initial_state=initial_states[0][i], time_major=False)
                    forward_states[i] = f
                else:
                    inputs, (f, b) = tf.nn.bidirectional_dynamic_rnn(self.forward_cells[i], self.backward_cells[i],
                                                                     inputs, sequence_length, initial_states[0][i],
                                                                     initial_states[1][i], time_major=False)
                    forward_states[i] = f
                    backward_states[i] = b
                    inputs = tf.concat(inputs, axis=2)
                if self.keep_prob:
                    inputs = self.dropout(inputs)

                # Add trainable variables
                variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, vs.name)
                self.add_to_dict(self.TRAINABLE_WEIGHTS, variables)

        return (inputs, (forward_states, backward_states)) if self.bidirectional else (inputs, forward_states)

    def extra_repr(self):
        return 'num_units: {}'.format(self.num_units) + '\nnum_layers: {}'.format(self.num_layers) + \
               '\nbidirectional: {}'.format(self.bidirectional) + '\nkeep_prob: {}'.format(self.keep_prob)


# import torch
# torch.nn.LSTM
# # input_size, hidden_size,
# #                  num_layers=1, bias=True, batch_first=False,
# #                  dropout=0, bidirectional=False
# tf.nn.bidirectional_dynamic_rnn()
# tf.nn.rnn_cell.LSTMCell
# tf.nn.rnn_cell.MultiRNNCell
# tf.nn.rnn_cell.DropoutWrapper
# tf.nn.rnn_cell.BasicLSTMCell
# tf.nn.dynamic_rnn()

