# coding=utf-8
"""Examples of feature extractors."""
from __future__ import division, print_function
from future.builtins import range

from collections import defaultdict
from math import factorial

import numpy as np
import pandas as pd

from whatami import whatable, what2id, whatadd


@whatable
class FeatureExtractor(object):

    def compute(self, x):
        raise NotImplementedError()

    def _outnames(self):
        return ()

    def _infonames(self):
        return ()

    def fnames(self):
        outnames = self._outnames()
        intronames = self._infonames()
        if not outnames:
            outnames = [what2id(self)]
        else:
            outnames = whatadd(self.what(), 'out', outnames)
        return outnames + whatadd(self.what(), 'info', intronames)


# --- Permutation entropy

class PermEn(FeatureExtractor):
    """
    Computes Bandt and Pompe permutation entropy.

    Parameters
    ----------
    column : string, default dtheta
      The time series columns

    order : int [2...], default 2
      The embedding dimension
      (= length of alphabet symbols)

    normalize : boolean, default True
      Make the result to be in [0, 1]

    return_counts : boolean, default False
      Return the symbol distribution too?

    Examples  (expectations from the original PermEn paper)
    --------
    >>> # Our time series...
    >>> x = pd.DataFrame([4, 7, 9, 10, 6, 11, 3], columns=['dtheta'])
    >>> # Our feature extractor
    >>> permen = PermEn()
    >>> print(what2id(permen))
    PermEn(column='dtheta',normalize=False,order=2)
    >>> print(','.join(permen.fnames()))
    PermEn(column='dtheta',normalize=False,order=2)
    >>> # Check expectations
    >>> np.abs(permen.compute(x)[0] - 0.9183) < 1E-4
    True
    >>> # Another configuration
    >>> permen = PermEn(order=2, return_counts=True)
    >>> # Note how return_counts is not part of the id
    >>> # (does not change the result)
    >>> print(what2id(permen))
    PermEn(column='dtheta',normalize=False,order=2)
    >>> # we do get now extra information about the computations
    >>> print('\\n'.join(permen.fnames()))
    PermEn(column='dtheta',normalize=False,order=2)
    PermEn(column='dtheta',info='counts',normalize=False,order=2)
    >>> value, alpha_counts = permen.compute(x)
    >>> # Check expectations
    >>> np.abs(value - 0.9183) < 1E-4
    True
    >>> # How does the alphabet distribution look like?
    >>> for permutation, counts in alpha_counts.items():
    ...     print(PermEn.buffer2permutation(permutation), counts)
    [0 1] 4
    [1 0] 2
    """

    def __init__(self,
                 column='dtheta',
                 order=2,
                 normalize=False,
                 return_counts=False):
        super(PermEn, self).__init__()
        self.column = column
        self.order = order
        self.normalize = normalize
        # return_counts won't change the result
        # we make it not part of the id
        # we could also have done it overriding the what method
        self._return_counts = return_counts

    def _infonames(self):
        return ('counts',) if self._return_counts else ()

    @staticmethod
    def buffer2permutation(data):
        return np.frombuffer(data, dtype=np.int)

    def compute(self, x):

        try:
            # pandas do not play well with flags.writeable
            x = x[self.column].values
        except AttributeError:
            x = x[self.column]

        order = self.order

        # input sanity checks
        if len(x) < self.order:
            raise Exception(
                'Permutation Entropy of vector with length %d '
                'is undefined for embedding dimension %d' % (len(x), order))
        num_permutations = factorial(order)
        if num_permutations > 39916800:
            print('Warning: permEn is O(ordd!). ordd! is %d.'
                  'Expect to wait a long time '
                  '(if we even do not go out-of-memory...)' % num_permutations)

        # vanilla implementation: dictionary counts
        alpha_counts = defaultdict(int)

        # populate counts
        for j in range(len(x) - order + 1):
            # use numpy stride tricks...
            this_permutation = np.argsort(x[j:j + order])
            this_permutation.flags.writeable = False
            alpha_counts[this_permutation.data] += 1

        # convert to frequencies
        alpha_freqs = np.array(alpha_counts.values()) / (len(x) - order + 1)

        # do not allow 0 probs
        SMALL = 1E-6
        alpha_freqs = np.maximum(SMALL, alpha_freqs)

        # permutation entropy
        entropy = -np.sum(alpha_freqs * np.log2(alpha_freqs))

        # make the value to be in [0, 1]
        if self.normalize:
            entropy /= np.log2(num_permutations)

        return (entropy, alpha_counts) if self._return_counts else (entropy,)


# --- Correlations


class LaggedPearson(FeatureExtractor):
    """Computes the Pearson correlation of lagged response over the stimulus.

    Parameters
    ----------
    lag : int, default 0
      The number of observations to lag (can be negative)

    response :

    Examples
    --------
    >>> stimuli = [1, 2, 3]
    >>> reaction = [3, 2, 1]
    >>> df = pd.DataFrame(np.array([reaction, stimuli]).T,
    ...                   columns=['rotation_rate', 'dtheta'])
    >>> lagcorr = LaggedPearson(lag=1)
    >>> corr = lagcorr.compute(df)[0]
    >>> print(corr)
    -1.0
    >>> print(what2id(lagcorr))
    LaggedPearson(lag=1,response='dtheta',stimulus='rotation_rate')
    """

    def __init__(self,
                 stimulus='rotation_rate',
                 response='dtheta',
                 lag=0):
        super(LaggedPearson, self).__init__()
        self.stimulus = stimulus
        self.response = response
        self.lag = lag

    def compute(self, x):
        return x[self.stimulus].shift(self.lag).corr(x[self.response]),
