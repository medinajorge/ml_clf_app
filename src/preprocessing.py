import os
import pandas as pd
import numpy as np
from numba import njit
import calendar
from copy import deepcopy
from pvlib import solarposition
from typing import Optional, Callable, List
from tkinter import messagebox

from . import params

def _ensure_df(dfs):
    is_series = all(isinstance(df, pd.Series) for df in dfs)
    dfs = [df.to_frame() if isinstance(df, pd.Series) else df for df in dfs]
    return dfs, is_series

def _revert_to_series(out, df_0):
    if out.shape[0] == 1:
        out = out.iloc[:, 0]
        out.name = df_0.iloc[:, 0].name
    else:
        out = out.squeeze()
    return out

def tuple_wise(*dfs, check_index=True, check_columns=True):
    """
    Attributes: Dataframes with same indices and columns. If the input are Series, they are converted to DataFrames.

    Returns dataframe where each element is a tuple containing the elements from other dataframes.
    If the input were Series, the output is a Series.
    """
    dfs, is_series = _ensure_df(dfs)
    df = dfs[0]
    if check_index:
        assert all(df.index.intersection(df2.index).size == df.shape[0] for df2 in dfs[1:]), "Indices do not match. To ignore this, set check_index=False."
    if check_columns:
        assert all(df.columns.intersection(df2.columns).size == df.shape[1] for df2 in dfs[1:]), "Columns do not match. To ignore this, set check_columns=False."
    out = pd.DataFrame(np.rec.fromarrays(tuple(df.values for df in dfs)).tolist(),
                       columns=df.columns,
                       index=df.index)
    if is_series:
        out = _revert_to_series(out, df)
    return out

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
        return np.array([], dtype=np.float64)
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
    print(f"Undersampling trajectories. Keeping observations separated by at least {dt_threshold:.6f} days")

    is_year_series = isinstance(year, pd.Series)
    if not is_year_series:
        year = pd.Series(year, index=df.index)
    T = df.apply(lambda x: x[2])
    T_year = tuple_wise(T.to_frame(), year.to_frame())
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
    df_undersampling = tuple_wise(df.to_frame(), idxs_undersampling.to_frame())[0]
    df_undersampled = df_undersampling.apply(lambda x: x[0][:, x[1]])
    year_undersampling = tuple_wise(year.to_frame(), idxs_undersampling.to_frame())[0]
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

@njit
def vectorized_is_leap(years):
    """
    Create a boolean mask where True indicates leap years
    """
    is_leap_mask = np.logical_or(np.logical_and(years % 4 == 0, years % 100 != 0), years % 400 == 0)
    return is_leap_mask

def _get_hour_angle(time_index, lat, lon):
    """
    Returns the hour angle in radians
    """
    solar_position = solarposition.get_solarposition(time_index, lat, lon, method='nrel_numpy')
    hour_angle = solarposition.hour_angle(time_index, lon, solar_position.equation_of_time.values)
    return hour_angle * np.pi / 180

def hour_angle_from_trajectory(x, year, is_leap=None):
    """
    Returns the hour angle in radians
    """
    if is_leap is None:
        is_leap = vectorized_is_leap(year)
    lat, lon, day = x[:3]
    day %= (365 + is_leap.astype(int))
    T0 = pd.Timestamp(year=year[0], month=1, day=1, tz='UTC')
    time_index = T0 + pd.to_timedelta(day, unit='D')
    return _get_hour_angle(time_index, lat, lon)

@njit
def great_circle_distance(lat1, lon1, lat2, lon2):
    """
    Multiply by radius of Earth to get distance in km.
    Assumes lat1, lon1, lat2, lon2 are all in radians.
    """
    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    d = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return d

@njit
def great_circle_distance_by_time_step(x):
    """
    Great circle distance between consecutive points in a trajectory.
    x is a 2 x n array of latitudes and longitudes in radians.
    """
    n = x.shape[1]
    d = np.empty(n-1)
    for i in range(n-1):
        d[i] = great_circle_distance(x[0, i], x[1, i], x[0, i+1], x[1, i+1])
    return d

def make_periodic(z, year, added_dt=False, to_origin=None, velocity='norm', replace_zero_by="mean", diff=False, diff_idxs=None, add_absolute_z=False, add_hour_angle=True):
    """
    Attributes:

        - z:                     (lat, lon, t) vector with shape (3, length)

        - year:                  array of year values. Used to compute the days per year and dt values.

        - added_dt:              Bool. If true, returns the vector except for the last point, where dt is undefined. Mainly thought for the equal-time case.

        - to_origin:             "time":             Shift initial time to 1 Jan.

        - velocity:              None:               Does not add velocity.
                                 "arch-segment":     Velocities in the SN-WE components.
                                 "x-y-z":            Velocities as the derivatives w.r.t. time of x, y, z.

        - replace_zero_by:       "closest":          When computing velocity, sometimes there are measurements done at the same time => dt = 0 y v=dx/dt leads to errors.
                                                     Replaces zero with idx i by the mean of the closest non-zero dts (mean{dt[j], dt[k]} with dt[j], dt[k] non-zero and j,k closest idxs such that j>i, k<i).
                                 "mean":             Replace zero with the mean dt between measurements in the trajectory.
        - diff:                  Bool. If True, returns the difference in each magnitude (except for the velocity)

        - diff_idxs:             List of indexes of the variables to be differentiated. If None, all variables are differentiated.

    Returns:

        periodic_x := (x, y, z, sin t, cos t, sin h, cos h, {weather vars},  {dt}, {velocity_vars})

    Considerations:
    {x, y, z} = {cos(theta) cos(phi), cos(theta) sin(phi), sin(theta)}
    theta = lat (not the polar angle)
    """
    x = z.copy()
    year_arr = np.array(year)
    is_leap = vectorized_is_leap(year_arr)
    if to_origin in ["time", "all"]:
        x[2] = (x[2] - x[2,0])
        x[2] = rescale_dt(x[2], is_leap)

    t_angle = (2*np.pi) * x[2] # maybe / 366 for leap
    t_angle[is_leap] /= 366
    t_angle[~is_leap] /= 365
    theta = (np.pi/180) * x[0]
    phi = (np.pi/180) * x[1]
    cos_theta = np.cos(theta)
    if add_hour_angle:
        periodic_x = np.empty((7, x.shape[1]))
        hour_angle = hour_angle_from_trajectory(x, year_arr, is_leap=is_leap)
        periodic_x[5] = np.sin(hour_angle)
        periodic_x[6] = np.cos(hour_angle)
    else:
        periodic_x = np.empty((5, x.shape[1]))
    periodic_x[0] = cos_theta * np.cos(phi)
    periodic_x[1] = cos_theta * np.sin(phi)
    periodic_x[2] = np.sin(theta)
    periodic_x[3] = np.sin(t_angle)
    periodic_x[4] = np.cos(t_angle)

    if add_absolute_z:
        periodic_x = np.vstack((periodic_x, np.abs(periodic_x[2])))

    if x.shape[0] > 3: # there are weather variables or/and dt
        periodic_x = np.vstack([periodic_x, x[3:]])

    if velocity is not None:
        delta_t = compute_dt(x[2], year, replace_zero_by=replace_zero_by)

        if velocity == "x-y-z":
            v = (periodic_x[:3, 1:] - periodic_x[:3, :-1]) / delta_t
        elif velocity == "norm":
            v = great_circle_distance_by_time_step((np.pi/180) * x[:2]) / delta_t
            v = v[None]
        else:
            raise ValueError(f"velocity {velocity} not valid. Available: 'x-y-z', 'norm'.")

        if diff_idxs is None:
            periodic_x = np.vstack([np.diff(periodic_x, axis=1) if diff else periodic_x[:, :-1], # velocity undefined for the last point.
                                    v])
        elif len(diff_idxs) > 0:
            no_diff_idxs = np.array([i for i in np.arange(periodic_x.shape[0]) if i not in diff_idxs])
            periodic_x = np.vstack([periodic_x[no_diff_idxs, :-1],
                                    np.diff(periodic_x[diff_idxs], axis=1) if diff else periodic_x[diff_idxs, :-1],
                                    v])
        else:
            periodic_x = np.vstack([periodic_x[:, :-1], v])
    else:
        if added_dt:
            if diff_idxs is None:
                periodic_x = np.diff(periodic_x, axis=1)
            elif len(diff_idxs) > 0:
                no_diff_idxs = np.array([i for i in np.arange(periodic_x.shape[0]) if i not in diff_idxs])
                periodic_x = np.vstack([periodic_x[no_diff_idxs, :-1],
                                        np.diff(periodic_x[diff_idxs], axis=1) if diff else periodic_x[diff_idxs, :-1]
                                       ])
            else:
                periodic_x = periodic_x[:,:-1]
    return periodic_x

def add_bathymetry_data(Z, X=None):
    print("Adding bathymetry data")
    if X is None:
        X = deepcopy(Z)

    bathymetry_data = np.genfromtxt(params.BATHYMETRY_PATH,
                     skip_header=0,
                     skip_footer=0,
                     names=None,
                     delimiter=' ')

    ground = bathymetry_data > 0
    bathymetry_data[ground] = 0

    lon_edges = np.arange(-180, 180.25, 0.25)
    lon_centers = 0.5 * (lon_edges[1:] + lon_edges[:-1])
    lat_edges = np.arange(90, -90.25, -0.25)
    lat_centers = 0.5 * (lat_edges[1:] + lat_edges[:-1])

    @njit
    def find_closest(lat, lon):
        i = np.abs(lat - lat_centers).argmin()
        j = np.abs(lon - lon_centers).argmin()
        return bathymetry_data[i,j]

    Z_new = []
    for z, x in zip(Z, X):
        bathymetry_x = np.array([find_closest(*x[:2, i]) for i in range(x.shape[1])])
        bathymetry_x[(z == 0).all(axis=0)] = 0
        Z_new.append(np.vstack([z, bathymetry_x]))
    return Z_new

def add_time_delta(X, Year):
    print("Computing time delta")
    X = [np.vstack([x, np.hstack([compute_dt(x[2], y), 0])]) for x, y in zip(X, Year)] # added 0 to be able to concatenate. Later will be removed
    return X

def preprocess_periodic_vars(X, Year):
    print("Handling discontinuities:\n\t-(lat, lon) -> (x, y, z)\n\t-day -> (sin, cos)\n\t-hour angle -> (sin, cos)")
    make_periodic_kwargs = dict(added_dt=True, velocity='norm', to_origin=None, replace_zero_by="mean", diff=False, add_absolute_z=False)
    X = [make_periodic(x, year, **make_periodic_kwargs) for (x, year) in zip(X, Year)]
    return X

def transpose_elements(X):
    return [x.T for x in X]

def check_format(df, app_root=None):
    required_cols = {'DATE_TIME', 'ID', 'LATITUDE', 'LONGITUDE'}
    missing = required_cols - set(df.columns)
    if len(missing) > 0:
        if app_root is None:
            raise ValueError(f"Missing columns: {missing}")
        else:
            messagebox.showerror("Format Error", f"Missing columns: {missing}")
            app_root.quit()
    return

def load_data(path=os.path.join(params.DATA_DIR, 'dataset.csv'), app_root=None, status_bar=None):
    df = pd.read_csv(path)
    check_format(df, app_root=app_root)

    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
    dt = df.DATE_TIME.dt
    df['year'] = df.DATE_TIME.dt.year
    df['day'] = (dt.dayofyear + dt.hour / 24 + dt.minute / (24 * 60) + dt.second / (24 * 60 * 60)) - 1

    # get trajectories and year
    X = df.groupby("ID").apply(lambda x: x.sort_values('DATE_TIME')[['LATITUDE', 'LONGITUDE', 'day']].values.T)
    Year = df.groupby("ID").apply(lambda x: x.sort_values('DATE_TIME').year.values)
    Year = Year.loc[X.index]

    min_dt = 0.25/24
    X, Year = undersample_trajectories(X, Year, dt_threshold=min_dt)

    # metadata
    min_days = 5
    min_observations = 50
    num_observations = X.apply(lambda x: x.shape[1])
    num_observations.name = 'num_observations'
    ndays = pd.Series(temporal_extension(X, Year, mode='all'), index=X.index, name='num_days')
    metadata = pd.concat([num_observations, ndays], axis=1)

    is_hq = (metadata.num_observations >= min_observations) & (metadata.num_days >= min_days)
    metadata.loc[is_hq.values, 'tracking_quality'] = 'high'
    metadata.loc[~is_hq.values, 'tracking_quality'] = 'low'

    # Discard trajectories with only 1 observation
    valid = metadata.num_observations > 1
    num_not_valid = (~valid).sum()
    if num_not_valid > 0:
        X = X.loc[valid]
        Year = Year.loc[valid]
        metadata = metadata.loc[valid]
        if status_bar is None:
            print(f"Discarded {num_not_valid} trajectories with only 1 observation.")
        else:
            status_bar.config(text=f"Discarded {num_not_valid} trajectories with only 1 observation.")

    metadata = metadata.reset_index() # index (ID) -> column
    N = metadata.shape[0]
    if status_bar is None:
        print(f"Low tracking quality trajectories: {(~is_hq).values.sum()}/{N}")
    else:
        status_bar.config(text=f"Low tracking quality trajectories (<5 days or <50 observations): {(~is_hq).values.sum()}/{N}")
    return X, Year, metadata

def preprocess(path=os.path.join(params.DATA_DIR, 'dataset.csv'),
               app_root=None,
               status_bar=None,
               progress_callback: Optional[Callable] = None,
               percentages: List = [0, 5, 10, 20],
               ):
    assert len(percentages) == 4, "len(percentages) must be 4"

    if progress_callback is not None:
        progress_callback("Preprocessing CSV file...\nLoading data", percentages[0])
    X, Year, metadata = load_data(path, app_root=app_root, status_bar=status_bar)

    if progress_callback is not None:
        progress_callback("Preprocessing CSV file...\nAdding bathymetry data", percentages[1])
    X = add_bathymetry_data(X)

    if progress_callback is not None:
        progress_callback("Preprocessing CSV file...\nAdding time delta between observations", percentages[2])
    X = add_time_delta(X, Year)

    if progress_callback is not None:
        progress_callback("Preprocessing CSV file...\nHandling discontinuities:\n\t-(lat, lon) -> (x, y, z)\n\t-day -> (sin, cos)\n\t-hour angle -> (sin, cos)",
                          percentages[3])
    X = preprocess_periodic_vars(X, Year)

    X = transpose_elements(X)

    return X, metadata
