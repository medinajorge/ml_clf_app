import os
import pandas as pd
import numpy as np
from numba import njit
import calendar

from phdu import pd_utils

from . import params

def get_leap_year(y):
    """Returns bool array. True if time data belongs to a leap year"""
    year_change = np.argwhere(y[:-1] != y[1:])[:,0]
    is_leap = np.empty((y.size), dtype=bool)
    year_change_edges = np.hstack([0, year_change, y.size])
    for start, end in zip(year_change_edges[:-1], year_change_edges[1:]):
        is_leap[start:end] = calendar.isleap(y[start])
    return is_leap

def rescale_dt(dt, is_leap_year):
    """Rescaling dt has to take into account wether there is a leap year."""
    is_leap = is_leap_year[:dt.size]
    leap_group = [is_leap, ~is_leap]
    end_of_year = [366, 365]
    for leap, end_of_year in zip(leap_group, end_of_year):
        dt[leap & (dt < -end_of_year)] %= (end_of_year + 1)
        dt[leap & (dt < 0)] %= end_of_year
    return dt

def replace_dt_zeros(delta_t, by="mean", threshold=1e-8):
    """
    by: "closest":          When computing velocity, sometimes there are measurements done at the same time => dt = 0 y v=dx/dt leads to errors.
                                                     Replaces zero with idx i by the mean of the closest non-zero dts (mean{dt[j], dt[k]} with dt[j], dt[k] non-zero and j,k closest idxs such that j>i, k<i).
                                 "mean":             Replace zero with the mean dt between measurements in the trajectory.
    """
    zero_dt_bool = delta_t < threshold
    zero_dt = np.argwhere(zero_dt_bool)[:,0]
    num_zeros = zero_dt.size
    default_dt = 0.11 # median dt of the whole dataset

    if num_zeros > 0:
        idxs = set(range(delta_t.size))
        if by == "closest":
            valid_idxs = idxs - set(zero_dt)
            candidates = np.empty((num_zeros))
            start = 0
            end = num_zeros
            if zero_dt[0] == 0:
                candidate_idxs = valid_idxs - set(range(2))
                if len(candidate_idxs) > 0:
                    candidates[0] = delta_t[min(candidate_idxs)]
                else:
                    candidates[0] = default_dt
                start = 1
            if zero_dt[-1] == (delta_t.size - 1) and delta_t.size > 3:
                candidate_idxs = valid_idxs - set(range(num_zeros - 3, num_zeros))
                if len(candidate_idxs) > 0:
                    candidates[-1] = delta_t[max(candidate_idxs)]
                else:
                    candidates[-1] = default_dt
                end -= 1
            for i, idx in enumerate(zero_dt[slice(start, end)], start=start):
                candidates_upper = valid_idxs - set(range(idx))
                candidates_lower = valid_idxs - set(range(idx, num_zeros))

                if len(candidates_upper) > 0 and len(candidates_lower) > 0:
                    closest_upper = min(candidates_upper) if len(candidates_upper) > 0 else max(candidates_lower)
                    closest_lower =  max(candidates_lower) if len(candidates_lower) > 0 else min(candidates_upper)
                    candidates[i] = delta_t[[closest_upper, closest_lower]].mean()
                elif len(candidates_upper) > 0:
                    candidates[i] = delta_t[min(candidates_upper)]
                elif len(candidates_lower) > 0:
                    candidates[i] = delta_t[max(candidates_lower)]
                else:
                    candidates[i] = default_dt

            delta_t[zero_dt] = candidates

        elif by == "mean":
            non_zeros = delta_t.size - num_zeros
            if non_zeros > 0:
                delta_t[zero_dt_bool] = delta_t[~zero_dt_bool].mean()
            else:
                delta_t[zero_dt] = default_dt # IDEA: Declare as NANS in the preprocessing step and try to infer the value using imputation.

    return delta_t

def compute_dt(t, year, replace_zero_by="mean"):
    if t.size < 2:
        return np.array([0])
    else:
        dt = (t[1:] - t[:-1])
        is_leap = get_leap_year(year)
        new_year_expected = np.argwhere(t[1:] < t[:-1])[:,0] # +1 for idx in t array. As it is, for idx in dt array
        new_year = np.argwhere(year[:-1] != year[1:])[:,0]
        new_year_unexpected = set(new_year) - set(new_year_expected)

        dt = rescale_dt(dt, is_leap)
        dt = replace_dt_zeros(dt, by=replace_zero_by)
        for idx in new_year_unexpected:
            dt[idx] += (366 if is_leap[idx-1] else 365)
        return dt

def undersample_trajectories(df, year, dt_threshold=1/24):
    print(f"Undersampling trajectories. Keeping observations separated by at least {dt_threshold} days")

    is_year_series = isinstance(year, pd.Series)
    if not is_year_series:
        year = pd.Series(year, index=df.index)
    T = df.apply(lambda x: x[2])
    T_year = pd_utils.tuple_wise(T.to_frame(), year.to_frame())
    DT = T_year.map(lambda x: compute_dt(*x))

    @njit
    def find_indices(dt, dt_threshold):
        """
        Find the indices of the trajectory that are separated by at least dt_threshold.
        """
        dt_cumsum = np.cumsum(dt)
        idxs = [0] # always include the first index
        last_index = 0

        for i, time in enumerate(dt_cumsum):
            # Check if the time since the last recorded index exceeds dt_threshold
            if time - (dt_cumsum[last_index - 1] if last_index > 0 else 0) >= dt_threshold:
                # Adjust the index to align with the t array
                idx = i + 1
                idxs.append(idx)
                last_index = i + 1

        return np.array(idxs)

    idxs_undersampling = DT[0].apply(find_indices, dt_threshold=dt_threshold)
    df_undersampling = pd_utils.tuple_wise(df.to_frame(), idxs_undersampling.to_frame())[0]
    df_undersampled = df_undersampling.apply(lambda x: x[0][:, x[1]])
    year_undersampling = pd_utils.tuple_wise(year.to_frame(), idxs_undersampling.to_frame())[0]
    year_undersampled = year_undersampling.apply(lambda x: x[0][x[1]])

    if not is_year_series:
        year_undersampled = list(year_undersampled.values)
    return df_undersampled, year_undersampled

def get_maximal_year(t, year, prune_by="time"):
    """
    prune_by:   'data': gets year with max amount of data.
                'time': gets year with max length of time interval.
    """
    #new_year = np.unique(np.hstack([0, 1 + np.argwhere(t[1:] < t[:-1])[:,0], t.size]))
    new_year = np.unique(np.hstack([0, 1 + np.argwhere(year[:-1] != year[1:])[:,0], t.size]))
    if prune_by == "data":
        data_per_year = new_year[1:] - new_year[:-1]
        maximal_data = data_per_year.argmax()
    elif prune_by == "time":
        time_per_year = t[new_year[1:]-1] - t[new_year[:-1]]
        maximal_data = time_per_year.argmax()
    else:
        raise ValueError(f"prune_by {prune_by} not valid. Available: 'data', 'time'")

    year_idxs = slice(new_year[maximal_data], new_year[maximal_data+1])
    return year_idxs

def temporal_extension(X, Year, mode="all", **kwargs):
    """
    mode:   'all':  Time of all the trajectory
            'year': Maximal time extension within one year.
    """
    if mode == "all":
        time_ext = lambda x, year: compute_dt(x[2], year, **kwargs).sum()
    elif mode == "year":
        def time_ext(x, year):
            if x.shape[1] == 1:
                return 0
            else:
                idxs = get_maximal_year(x[2], year, **kwargs)
                t_pruned = x[2, idxs]
                return t_pruned[-1] - t_pruned[0]
    else:
        raise ValueError(f"mode {mode} not valid. Available: 'all', 'year'")
    ndays = np.array([time_ext(x, year) for x, year in zip(X, Year)])
    return ndays

def load_data(path=os.path.join(params.DATA_PATH, 'dataset.csv')):
    df = pd.read_csv(path)
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
    dt = df.DATE_TIME.dt
    df['year'] = df.DATE_TIME.dt.year
    df['day'] = (dt.dayofyear + dt.hour / 24 + dt.minute / (24 * 60) + dt.second / (24 * 60 * 60)) - 1

    # get trajectories and year
    X = df.groupby("ID").apply(lambda x: x[['LATITUDE', 'LONGITUDE', 'day']].values.T)
    year = df.groupby("ID").apply(lambda x: x.year.values)
    year = year.loc[X.index]

    min_dt = 0.25/24
    X, year = undersample_trajectories(X, year, dt_threshold=min_dt)

    # metadata
    min_days = 5
    min_observations = 50
    num_observations = X.apply(lambda x: x.shape[1])
    num_observations.name = 'num_observations'
    ndays = pd.Series(temporal_extension(X, year, mode='all'), index=X.index, name='num_days')
    metadata = pd.concat([num_observations, ndays], axis=1)

    is_hq = (metadata.num_observations >= min_observations) & (metadata.num_days >= min_days)
    metadata.loc[is_hq.values, 'tracking_quality'] = 'high'
    metadata.loc[~is_hq.values, 'tracking_quality'] = 'low'

    N = metadata.shape[0]
    print(f"Loaded {N} trajectories")
    print(f"Low tracking quality trajectories: {(~is_hq).values.sum()}/{N}")
    return X, year, metadata
