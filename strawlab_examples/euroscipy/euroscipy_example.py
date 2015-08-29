# coding=utf-8
from __future__ import division, print_function, unicode_literals
from functools import partial
from itertools import chain
import os.path as op

from future.builtins import range
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from jagged.mmap_backend import JaggedByMemMap
from strawlab_examples.euroscipy.features import PermEn, LaggedPearson
from strawlab_examples.minifly import FreeflightHub
from strawlab_examples.euroscipy.misc import (hub2jagged,
                                              download_degradation_dataset,
                                              to_long_form,
                                              compute_cache_features,
                                              pandify, plot_lagged)


# Data and results go here
NEUROPEPTIDE_DEGRADATION_PATH = op.expanduser('~/np-degradation')

# We will work with a smallish dataset:
#   - 51359 trials
#   - 18328689 (18M) observations
#   - 1.7GB (uncompressed, single precision)
# You can download it from:
#   https://zenodo.org/record/29193

# Download released dataset
download_degradation_dataset(dest=NEUROPEPTIDE_DEGRADATION_PATH,
                             force=False)

# Data hub, acess anything in the data
hub = FreeflightHub(path=NEUROPEPTIDE_DEGRADATION_PATH)

# Put the time series in a jagged store
jagged = JaggedByMemMap(path=op.join(NEUROPEPTIDE_DEGRADATION_PATH,
                                     'jagged-mmap'),
                        autoviews=True, contiguity='auto')
hub2jagged(hub, jagged)


# The trials DataFrame
trials_df = hub.trials_df()
# So we can easily combine df queries with series retrieval
trials_df['jagged_index'] = np.arange(len(trials_df))

# Time-series names
columns, human_friendly = hub.series_groups_columns(as_pandas_index=True)

# Fast retrieval of time-series for concrete trials
pandify = partial(pandify, columns=human_friendly)
get_tseries = partial(jagged.get, factory=pandify)

# Find a subset of the time series
trials = trials_df.query('length_s > 2 and exp_group == "DPPIII"').reset_index()
print('There are %d trials' % len(trials))
tseries = list(get_tseries(trials.jagged_index))
# This jagged instance allows "lazy" pandas DataFrames...

# Feature extractors
extractors = [PermEn(order=3), PermEn(order=3, column='speed_xy')]
extractors += [LaggedPearson(lag=lag,
                             stimulus='rotation_rate', response='dtheta')
               for lag in range(-80, 81, 2)]
extractors += [LaggedPearson(lag=lag,
                             stimulus='rotation_rate', response='speed_xy')
               for lag in range(-80, 81, 2)]

# Extract features to a DataFrame
features_pkl = op.join(NEUROPEPTIDE_DEGRADATION_PATH, 'features.pkl')
features_df = compute_cache_features(extractors,
                                     tseries,
                                     features_pkl=features_pkl)

# To long form
melted = to_long_form(trials, features_df,
                      stimulus='rotation_rate',
                      response='dtheta')
print('The tidy dataframe has %d rows and %d columns' % melted.shape)
print('These are the features', melted.fname.unique())
print('These are the lags', melted.lag.unique())

# Draw an interesting plot
sns.set_context('poster')
fig, axes = plt.subplots(nrows=melted.condition.nunique() // 2,
                         ncols=melted.condition.nunique() // 2,
                         sharex=True, sharey=True,
                         figsize=(16, 12))
for ax, (condition, cdf) in zip(chain(*axes),
                                melted.groupby('condition')):
    print(condition)
    plot_lagged(cdf, ax, title='stimulus=%s' % condition)
fig.suptitle('Normalised cross-correlation over different conditions', fontsize=24)
plt.savefig(op.join(NEUROPEPTIDE_DEGRADATION_PATH, 'lagcorr.png'))
print('Plot saved in %s' % op.join(NEUROPEPTIDE_DEGRADATION_PATH, 'lagcorr.png'))


#
# If we had pyopy, hctsa (soon) and matlab...
# from pyopy.hctsa import hctsa
# import numpy as np
# hctsa.prepare()
# extractors = [hctsa.operations.EN_PermEn_4_1, hctsa.bindings.EN_PermEn(m=6)]
# for extractor in extractors:
#     for x in tseries:
#         print(extractor.what().id(), extractor.compute(x))
#
