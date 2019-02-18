import numpy as np
from collections import defaultdict, Counter
import torch
import torchtext
from torchtext.data import batch
import six
import random


def read_pre_embedding(emb_path, stoi, num_embedding=None, dim=None, sep=' ', skip_lines=0, unk_init=np.zeros_like):
    """Read pre trained embedding from files.
    emb_path: first line contains total lines and embedding dimension. other should be specified in dim.
    file format:
        total_lines dim , optimal
        w embedding
        ...

    num_embedding: optional, default to len(stoi).
    """
    if num_embedding is None:
        num_embedding = len(stoi)
    with open(emb_path) as f:
        if dim is None:
            assert not skip_lines, 'dim should be specified in the first line'
            rows, dim = f.readline().split()
            dim = int(dim)
            print('total rows', rows, 'dim of embedding', dim)
        else:
            print('dim of embedding', dim)
            for _ in range(skip_lines):
                f.readline()

        vectors = np.zeros((num_embedding, dim), dtype=np.float32)
        assigned = [0] * num_embedding

        for line in f:
            line = line.strip('\n').split(sep)
            w = sep.join(line[:-dim])
            idx = stoi.get(w, None)
            if idx is None:
                continue
            vec = np.asarray(list(map(float, line[-dim:])), dtype=np.float32)
            idx = stoi[w]
            assigned[idx] = 1
            vectors[idx] = vec

        tot = sum(assigned)
        print(tot, 'of', len(stoi), ' vectors read, rate:', tot / len(stoi))

        for i, b in enumerate(assigned):
            if not b:
                vectors[i] = unk_init(vectors[i])

    return vectors


class Vocab(object):
    """Defines a vocabulary object that will be used to numericalize a field.

    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
    """

    def __init__(self, counter, max_size=None, min_freq=1, specials=None, specials_first=True, unk_idx=0):
        """Create a Vocab object from a collections.Counter.

        Arguments:
            counter: collections.Counter object holding the frequencies of
                each value found in the data.
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary in addition to an <unk>
                token. Default: ['<pad>']
            specials_first: Whether to add special tokens into the vocabulary at first.
                If it is False, they are added into the vocabulary at last.
                Default: True.
            unk_idx: The default index for UNK token.
        """
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list()
        specials = specials or []
        if specials_first:
            self.itos = list(specials)

        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in specials:
            del counter[tok]

        max_size = None if max_size is None else max_size + len(self.itos)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        if not specials_first:
            self.itos.extend(list(specials))

        self.stoi = defaultdict(lambda: unk_idx)
        # stoi is simply a reverse dict for itos
        self.stoi.update({tok: i for i, tok in enumerate(self.itos)})

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1


class Field(torchtext.data.Field):
    """Add numpy type, reverse function to Field."""

    dtypes = {
        torch.float32: float,
        torch.float: float,
        torch.float64: float,
        torch.double: float,
        torch.float16: float,
        torch.half: float,

        torch.uint8: int,
        torch.int8: int,
        torch.int16: int,
        torch.short: int,
        torch.int32: int,
        torch.int: int,
        torch.int64: int,
        torch.long: int,
        np.uint: int, np.uint8: int, np.uint16: int, np.uint32: int, np.uint64: int,
        np.int8: int, np.int: int, np.int32: int, np.int64: int,
        np.float32: float, np.float64: float
    }

    def __init__(self, **kwargs):
        super(Field, self).__init__(**kwargs)

    def reverse(self, batch):
        """Convert batch to text."""
        if isinstance(self.dtype, torch.dtype):
            if not self.batch_first:
                batch = batch.t()
            with torch.cuda.device_of(batch):
                batch = batch.tolist()
        else:
            # Numpy batch
            batch = batch.tolist()
        batch = [[self.vocab.itos[ind] for ind in ex] for ex in batch]  # denumericalize

        def trim(s, t):
            sentence = []
            for w in s:
                if w == t:
                    break
                sentence.append(w)
            return sentence

        batch = [trim(ex, self.eos_token) for ex in batch]  # trim pad first eos

        def filter_special(tok):
            return tok not in (self.init_token, self.pad_token)

        batch = [filter(filter_special, ex) for ex in batch]
        return batch

    def _convert_to_dtype(self, data, dtype=None, device=None):
        dtype = dtype or self.dtype
        if isinstance(dtype, torch.dtype):
            return torch.tensor(data, dtype=dtype, device=device)
        # numpy data
        return np.array(data, dtype=dtype)

    def numericalize(self, arr, device=None):
        """Turn a batch of examples that use this field into a Variable.

        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.

        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (str or torch.device): A string or instance of `torch.device`
                specifying which device the Variables are going to be created on.
                If left as default, the tensors will be created on cpu. Default: None.
        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths = arr
            lengths = self._convert_to_dtype(lengths, device=device)

        if self.use_vocab:
            if self.sequential:
                arr = [[self.vocab.stoi[x] for x in ex] for ex in arr]
            else:
                arr = [self.vocab.stoi[x] for x in arr]

            if self.postprocessing is not None:
                arr = self.postprocessing(arr, self.vocab)
        else:
            if self.dtype not in self.dtypes:
                raise ValueError(
                    "Specified Field dtype {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.dtype))
            numericalization_func = self.dtypes[self.dtype]
            # It doesn't make sense to explicitly coerce to a numeric type if
            # the data is sequential, since it's unclear how to coerce padding tokens
            # to a numeric type.
            if not self.sequential:
                arr = [numericalization_func(x) if isinstance(x, six.string_types)
                       else x for x in arr]
            if self.postprocessing is not None:
                arr = self.postprocessing(arr, None)

        var = self._convert_to_dtype(arr, device=device)

        if isinstance(self.dtype, torch.dtype):
            if self.sequential and not self.batch_first:
                var.t_()
            if self.sequential:
                var = var.contiguous()
        else:
            if self.sequential and not self.batch_first:
                var = var.T
        if self.include_lengths:
            return var, lengths
        return var


def pool(data, batch_size, key, batch_size_fn=lambda new, count, sofar: count,
         random_shuffler=None, shuffle=False, shuffle_batch_no=100000, sort_within_batch=False):
    """Sort within buckets, then batch, then shuffle batches.

    Partitions data into chunks of size 100*batch_size, sorts examples within
    each chunk using sort_key, then batch these examples and shuffle the
    batches.
    """

    if random_shuffler is None:
        random_shuffler = random.shuffle
    for p in batch(data, batch_size * shuffle_batch_no, batch_size_fn):
        p_batch = batch(sorted(p, key=key), batch_size, batch_size_fn) \
            if sort_within_batch \
            else batch(p, batch_size, batch_size_fn)
        if shuffle:
            for b in random_shuffler(list(p_batch)):
                yield b
        else:
            for b in list(p_batch):
                yield b


class BucketIterator(torchtext.data.Iterator):
    """Defines an iterator that batches examples of similar lengths together.

    Minimizes amount of padding needed while producing freshly shuffled
    batches for each new epoch. See pool for the bucketing procedure used.
    """

    def __init__(self, dataset, batch_size, sort_key=None, device=None,
                 batch_size_fn=None, train=True,
                 repeat=False, shuffle=None, sort=None,
                 sort_within_batch=None, shuffle_batch_no=100000):
        super(BucketIterator, self).__init__(dataset, batch_size, sort_key, device, batch_size_fn,
                                             train, repeat, shuffle, sort, sort_within_batch)
        self.shuffle_batch_no = shuffle_batch_no

    def create_batches(self):
        self.batches = pool(self.data(), self.batch_size,
                            self.sort_key, self.batch_size_fn,
                            random_shuffler=self.random_shuffler,
                            shuffle=self.shuffle,
                            shuffle_batch_no=self.shuffle_batch_no,
                            sort_within_batch=self.sort_within_batch)
