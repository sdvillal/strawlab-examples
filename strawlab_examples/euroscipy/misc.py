# coding=utf-8
from itertools import chain
import os.path as op

import numpy as np
import pandas as pd
import seaborn as sns

from whatami import id2what, whatid2columns

from strawlab_examples.minifly import split_df


def hub2jagged(hub, jagged, chunk_size=1000, force=False):
    DONE = op.join(jagged.path_or_fail(), 'DONE')
    if op.isfile(DONE) and not force:
        return None
    with jagged:
        indices = []
        for i, tdf in enumerate(split_df(hub.trials_df(),
                                         chunk_size=chunk_size)):
            print('Jaggified chunk %d' % (i + 1))
            for sdf in hub.series_df(tdf).series:
                indices.append(jagged.append(sdf.values))
        with open(DONE, 'w') as writer:
            writer.write('The jagged store is populated')
        return indices


def download_degradation_dataset(dest=op.join(op.expanduser('~'),
                                              'rnai-degradation'),
                                 report_each=1024 * 200,
                                 force=False):
    import humanize
    import requests
    from jagged.misc import ensure_dir

    def download_file(url, dest):
        r = requests.get(url, stream=True)
        total = 0
        countdown = report_each
        with open(dest, 'wb') as writer:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    writer.write(chunk)
                    writer.flush()
                    total += len(chunk)
                    countdown -= 1
                    if countdown == 0:
                        print('\tdownloaded %s' % humanize.naturalsize(total))
                        countdown = report_each
        return dest

    DATASET_URL = 'https://zenodo.org/record/29193'
    FILES = ('README.txt',
             'experiments.csv',
             'genotype2line.csv',
             'trials.csv.gz',
             'series.h5',
             'minifly.py',
             'minifly-out.txt')

    ensure_dir(dest)

    print('Downloading dataset from %s' % DATASET_URL)
    for filename in FILES:
        url = op.join(DATASET_URL, 'files', filename)
        print('Downloading %s' % url)
        dest_file = op.join(dest, filename)
        if op.isfile(dest_file) and not force:
            print('\tAlready downloaded, skipping')
            continue
        if filename == 'series.h5':
            print('(be patient, this is a 1.7GB file)')
        download_file(url, dest_file)
    print('Download successful')


def compute_features(fexes, dfs):
    # N.B. this can be done more efficiently
    # (e.g. avoid last copy / prealloc empty)
    print('Fexing')
    features = [
        np.fromiter(
            chain.from_iterable(fex.compute(df) for fex in fexes),
            dtype=float
        )
        for df in dfs
    ]
    print('FexingEnd')
    columns = list(chain.from_iterable(fex.fnames() for fex in fexes))
    return pd.DataFrame(data=features, columns=columns)


def whatselect(whatids, name=None, kvs=None):
    """
    Selects whatami ids based on several conditions:
      - they should have the same name (if provided)
      - they should have the same key-values (if provided)
    """
    def select(whatid):
        what = id2what(whatid)
        if name is not None and what.name != name:
            return False
        if kvs is not None:
            for k, v in kvs:
                if what[k] != v:
                    return False
        return True

    return [whatid for whatid in whatids if select(whatid)]


def to_long_form(trials_df,
                 features_df,
                 stimulus='rotation_rate', response='speed_xy'):
    """Puts the dataframe in long (aka tidy) form."""
    # We have computed these features
    fnames = list(map(str, features_df.columns))
    # Merge metadata and features
    fdf = pd.concat((trials_df, features_df), axis=1)
    # Each row is a trial
    fdf['unit'] = np.arange(len(fdf))
    # Melt
    melted = pd.melt(fdf,
                     id_vars=['unit', 'exp_group',
                              'condition', 'genotype',
                              'impaired', 'dt'],
                     value_vars=fnames,
                     var_name='fname',
                     value_name='value')
    # Lets just get the lagged correlation we want
    lag_columns = whatselect(fnames,
                             name='LaggedPearson',
                             kvs=[('stimulus', stimulus),
                                  ('response', response)])
    melted = melted[melted['fname'].isin(lag_columns)]
    # Extract the lag to a new column
    melted = whatid2columns(melted, 'fname',
                            columns=['lag'], inplace=False)
    melted['lag'] *= melted['dt']
    # Sort by lag (useful for inspection)
    return melted.sort_values('lag')


def plot_lagged(melted, ax, title):
    sns.set_context('poster')
    sns.lineplot(x='lag', y='value',
                 hue='impaired',
                 data=melted,
                 estimator=np.nanmean,
                 ax=ax,
                 palette={True: 'r', False: 'b'})
    ax.set_title(title)
    ax.set_ylabel('Pearson Correlation')
    ax.set_xlabel('Lag (seconds)')
    ax.set_ylim((-1.05, 1.05))


def compute_cache_features(extractors, tseries, features_pkl=None, force=False):
    if not op.isfile(features_pkl) or force:
        features_df = compute_features(extractors, tseries)
        if features_pkl is not None:
            features_df.to_pickle(features_pkl)
        return features_df
    return pd.read_pickle(features_pkl)


def pandify(array, columns):
    """Make a pandas DataFrames out of an array, hopefully without copying."""
    return pd.DataFrame(array, copy=False, columns=columns)
