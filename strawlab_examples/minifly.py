# coding=utf-8
"""
FreeflightHub provides convenient access to all the released data
(experiments, trials, series) given their path. It heavily relies
in the data analysis library pandas (numpy and h5py must be
installed as well).

Find a simple example at the bottom of this file.
"""
import os.path as op
import pandas as pd
import numpy as np
import h5py


def categorize_df(df, categoricals=None, inplace=False):
    """
    Make some columns of a dataframe categorical.

    Parameters
    ----------
    df : pandas DataFrame

    categoricals : list of strings, default None
      the list of columns to transform; if None, all object columns will be transformed

    inplace : bool, default False
      change the dataframe in place

    :rtype: pandas.DataFrame
    """
    if not inplace:
        df = df.copy()
    if categoricals is None:
        categoricals = [col for col in df.columns if df.dtypes[col] == 'object']
    for categorical in categoricals:
        df[categorical] = df[categorical].astype('category')
    return df


def split_df(df, chunk_size=10000):
    """Generator providing splits of the dataframe df into contiguous
    views of rows of the given chunksize."""
    # numpy.array_split was very slow and returns materialised views.
    chunk_base = 0
    while chunk_base < len(df):
        yield(df.iloc[chunk_base:chunk_base+chunk_size])
        chunk_base += chunk_size


class FreeflightHub(object):
    """One-stop shop for a released free{flight/swim} dataset."""

    def __init__(self, path=None):
        """Creates a new hub, reading data from the provided path.
        If path is None, data is assume to live in the same directory as this script.
        """
        super(FreeflightHub, self).__init__()
        if path is None:
            path = op.dirname(op.abspath(__file__))
        self.path = path
        self._exps = None
        self._g2l = None
        self._trials = None

    def exps_df(self):
        """Returns a pandas dataframe with the experiment metadata.
        :rtype: pandas.DataFrame
        """
        if self._exps is None:
            self._exps = pd.read_csv(op.join(self.path, 'experiments.csv'),
                                     encoding='utf-8',
                                     parse_dates=['start'],
                                     dtype={'tags': str, 'user': str})
        return self._exps

    def genotype2line(self, genotype):
        """Returns the line corresponding to the given genotype string."""
        if self._g2l is None:
            self._g2l = pd.read_csv(op.join(self.path, 'genotype2line.csv'), encoding='utf-8').\
                set_index('genotype')['line'].to_dict()
        return self._g2l.get(genotype, None)

    def trials_df(self,
                  add_relstart=False,
                  add_line=False,
                  index=None, drop_in_index=True,
                  categorize=False,
                  refresh_cache=False):
        """Returns a pandas dataframe with the trials metadata.
        :rtype: pandas.DataFrame

        Parameters
        ----------
        add_relstart : boolean, default False
          If True, a relative start column is added to the dataframe (trial_start - experiment_start)

        add_line : boolena, default False
          If True, add a "line" column with the actual line of the genotype

        index : what to use as index
          If None, a trivial index is used
          If True, a multilevel index is set with the (uuid, oid, startf, exp_group) columns.
            This is the trial unique identifier for the whole rnai dataset
            (we need exp_group since some control experiments are reused in several groups)
          If False, (uuid, oid, startf) is used as index.
          Otherwise index is used verbatim.

        drop_in_index : boolean, default True
          If index is a column specification, whether to drop the selected columns

        categorize : boolean, default False
          If True, string columns in the dataframe are changed to categorical.

        refresh_cache : boolean, default False
          If True, caches for fast loading are recreated.
        """

        trials_pickle = op.join(self.path,
                                'trials(relstart=%r,line=%r,index=%r,drop=%r,cat=%r)' %
                                (add_relstart, add_line, index, drop_in_index, categorize))

        def read_and_cache():
            trials_df = pd.read_csv(op.join(self.path, 'trials.csv.gz'),
                                    encoding='utf-8',
                                    parse_dates=['start', 'start_exp'],
                                    compression='gzip')
            if add_relstart:
                trials_df['rel_start'] = trials_df.start - trials_df.start_exp
            if add_line:
                trials_df['line'] = trials_df.genotype.apply(self.genotype2line)
            if index is not None:
                if index is True:
                    trials_df = trials_df.set_index(['uuid', 'oid', 'startf', 'exp_group'], drop=drop_in_index)
                elif index is False:
                    trials_df = trials_df.set_index(['uuid', 'oid', 'startf'], drop=drop_in_index)
                else:
                    trials_df = trials_df.set_index(index, drop=drop_in_index)
            if categorize:
                trials_df = categorize_df(trials_df)
            self._trials = trials_df
            pd.to_pickle(self._trials, trials_pickle)
        if refresh_cache:
            read_and_cache()
        elif self._trials is None:
            try:
                self._trials = pd.read_pickle(trials_pickle)
            except:
                read_and_cache()
        return self._trials

    def series_group_names(self):
        """Returns the existing time-series group names in the hdf5 file."""
        with h5py.File(op.join(self.path, 'series.h5'), 'r') as h5:
            return sorted(h5.keys())

    def series_groups_columns(self, groups=('basic',), as_pandas_index=False):
        """Returns a two-puple ([column-names], [nicknames]).
        These are the names of the time series in the specified groups.

        Parameters
        ----------
        groups : list of strings, default ('basic',)
          Names of time series groups in the HDF5 file (see "series_group_names")

        as_pandas_index : boolean, default False
          If True, the returned lists are pandas indices instead of lists of strings.
        """
        columns = []
        synonyms = []
        with h5py.File(op.join(self.path, 'series.h5'), 'r') as h5:
            for group in groups:
                group = h5[group]
                columns += list(group.attrs['col_ids'])
                synonyms += list(group.attrs['col_synonyms'])
        if as_pandas_index:
            columns, synonyms = pd.Index(columns), pd.Index(synonyms)
        return columns, synonyms

    def series_df(self,
                  trials_coords,
                  groups=('basic',),
                  lazy=False,
                  as_dataframes=True,
                  to_categories=True):
        """Returns a pandas dataframe with time the time series for the specified trials.
        The pandas dataframe has 4 columns: uuid, oid, startf, series.

        Parameters
        ----------
        trials_coords : pandas dataframe or list providing (uuid, oid, startf) trial coordinates
          Which trials to retrieve.

        groups : list of strings, default ('basic',)
          The name(s) of the time series groups to retrieve.

        lazy : boolean, default False
          If True, and as_dataframes is False, the returned arrays will still be views on the HDF5 file.
          Else, the arrays are fetched into main memory immediatly.

        as_dataframes : boolean, default True
          If True, pandas dataframes are returned in the "series" column; else just numpy arrays.

        to_categories : boolean, default True
          If True, uuid is made categorical in the returned dataframe.
        """
        if isinstance(trials_coords, pd.DataFrame):
            trials_coords = list(zip(trials_coords.uuid, trials_coords.oid, trials_coords.startf))
        trials = []
        columns = None
        # shared columns for all dataframes
        with h5py.File(op.join(self.path, 'series.h5'), 'r') as h5:
            # we can probably optimise the query here
            for uuid, oid, startf in trials_coords:
                arrays = []
                for group in groups:
                    group = h5[group]
                    array = group[uuid]['%d_%d' % (oid, startf)]
                    if not lazy or as_dataframes:
                        array = array[:]
                    arrays.append(array)
                # stack
                array = np.hstack(arrays)
                # to pandas
                if as_dataframes:
                    if columns is None:
                        columns, _ = self.series_groups_columns(groups, as_pandas_index=True)
                    array = pd.DataFrame(data=array, columns=columns, index=None, copy=False)
                trials.append((uuid, oid, startf, array))
        df = pd.DataFrame(data=trials, columns=['uuid', 'oid', 'startf', 'series'])
        if to_categories:
            return categorize_df(df, categoricals=['uuid'])
        return df


if __name__ == '__main__':

    # Put here the path where the data lives
    hub = FreeflightHub(path=None)
    # hub = FreeflightHub(path=op.join(op.expanduser('~'),
    #                                  'data-analysis',
    #                                  'strawlab',
    #                                  'rnai',
    #                                  'release',
    #                                  'rnai-degradation'))

    # Get the "trials" metadata
    trials_df = hub.trials_df(add_relstart=True, add_line=True)

    # Explore the data
    print('There are %d experiments (%d impaired)' % (trials_df.uuid.nunique(),
                                                      trials_df[trials_df.impaired].uuid.nunique()))
    print('There are %d trials (%d impaired)' % (len(trials_df),
                                                 len(trials_df.query('impaired'))))
    print('Experimental groups: %s' % ' '.join(sorted(trials_df.exp_group.unique())))
    print('Genotypes:\n  %s' % '\n  '.join('%s: %s' % (g, hub.genotype2line(g))
                                           for g in sorted(trials_df.genotype.unique())))
    print('Arenas: %s' % ' '.join(sorted(trials_df.arena.unique())))

    # Print length means for different groups
    print(trials_df.groupby(['exp_group', 'impaired']).size())
    print(trials_df.groupby(['exp_group', 'impaired'])['length_s'].mean())

    # Which time series are provided per trial?
    print('Time series groups:')
    for group in hub.series_group_names():
        print('\t%s' % group)
        for sname, snick in zip(*hub.series_groups_columns((group,))):
            print('\t\t%s: %s' % (snick, sname))

    # Print speed means for interesting groups
    # Note this takes a few seconds (series retrieval is not very efficient)
    print('Computing speed means for different groups ')
    print('(this takes around 1 minute in my machine)...')
    for (exp_group, impaired), trials in trials_df.groupby(['exp_group', 'impaired']):
        speed_means = [series['velocity'].mean() for series in hub.series_df(trials).series]
        print('\t%s, impaired=%r, mean(mean(speed_xy)) = %.2f +/- %.2f m/s' % (
            exp_group, impaired,
            float(np.mean(speed_means)), float(np.std(speed_means))))
