# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

import numpy as np
import copy
from scipy.interpolate import interp1d
from pyuvdata import UVData

from . import datacontainer
from .utils import split_pol, get_sun_alt


def solar_flag(flags, times=None, flag_alt=0.0, longitude=21.42830, latitude=-30.72152, inplace=False,
               interp=False, interp_Nsteps=11, verbose=True):
    """
    Apply flags at times when the Sun is above some minimum altitude.

    Parameters
    ----------
    flags : flag ndarray, or DataContainer, or pyuvdata.UVData object

    start_jd : int
        Integer Julian Date to perform calculation for

    times : 1D float ndarray
        If flags is an ndarray or DataContainer, this contains the time bins
        of the data's time axis in Julian Date

    flag_alt : float
        If the Sun is greater than this altitude [degrees], we flag the data.

    longitude : float
        Longitude of observer in degrees East (if flags is a UVData object,
        use its stored longitude instead)

    latitude : float
        Latitude of observer in degrees North (if flags is a UVData object,
        use its stored latitude instead)

    inplace: bool
        If inplace, edit flags instead of returning a new flags object.

    interp : bool
        If True, evaluate Solar altitude with a coarse grid and interpolate at times values.

    interp_Nsteps : int
        Number of steps from times.min() to times.max() to use in get_solar_alt call.
        If the range of times is <= a single day, Nsteps=11 is a good-enough resolution.

    verbose : bool
        if True, print feedback to standard output

    Returns
    -------
    flags : solar-applied flags, same format as input
    """
    # type check
    if isinstance(flags, datacontainer.DataContainer):
        dtype = 'DC'
    elif isinstance(flags, np.ndarray):
        dtype = 'ndarr'
    elif isinstance(flags, UVData):
        if verbose:
            print("Note: using latitude and longitude in given UVData object")
        latitude, longitude, altitude = flags.telescope_location_lat_lon_alt_degrees
        times = np.unique(flags.time_array)
        dtype = 'uvd'
    if dtype in ['ndarr', 'DC']:
        assert times is not None, "if flags is an ndarray or DataContainer, must feed in times"

    # inplace
    if not inplace:
        flags = copy.deepcopy(flags)

    # get solar alts
    if interp:
        # first evaluate coarse grid, then interpolate
        _times = np.linspace(times.min(), times.max(), interp_Nsteps)
        _alts = get_sun_alt(_times, longitude=longitude, latitude=latitude)

        # interpolate _alts
        alts = interp1d(_times, _alts, kind='quadratic')(times)
    else:
        # directly evaluate solar altitude at times
        alts = get_sun_alt(times, longitude=longitude, latitude=latitude)

    # apply flags
    if dtype == 'DC':
        for k in flags.keys():
            flags[k][alts > flag_alt, :] = True
    elif dtype == 'ndarr':
        flags[alts > flag_alt, :] = True
    elif dtype == 'uvd':
        for t, a in zip(times, alts):
            if a > flag_alt:
                flags.flag_array[np.isclose(flags.time_array, t)] = True

    if not inplace:
        return flags


def synthesize_ant_flags(flags, threshold=0.0):
    '''
    Synthesizes flags on visibilities into flags on antennas. For a given antenna and
    a given time and frequency, if the fraction of flagged pixels in all visibilities with that
    antenna exceeds 'threshold', the antenna gain is flagged at that time and frequency. This
    excludes contributions from antennas that are completely flagged, i.e. are dead.

    Arguments:
        flags: DataContainer containing boolean data flag waterfalls
        threshold: float, fraction of flagged pixels across all visibilities (with a common antenna)
            needed to flag that antenna gain at a particular time and frequency.

    Returns:
        ant_flags: dictionary mapping antenna-pol keys like (1,'x') to boolean flag waterfalls
    '''
    # type check
    assert isinstance(flags, datacontainer.DataContainer), "flags must be fed as a datacontainer"
    assert threshold >= 0.0 and threshold <= 1.0, "threshold must be 0.0 <= threshold <= 1.0"
    if np.isclose(threshold, 1.0):
        threshold = threshold - 1e-10

    # get Ntimes and Nfreqs
    Ntimes, Nfreqs = flags[list(flags.keys())[0]].shape

    # get antenna-pol keys
    antpols = set([ap for (i, j, pol) in flags.keys() for ap in [(i, split_pol(pol)[0]),
                                                                 (j, split_pol(pol)[1])]])

    # get dictionary of completely flagged ants to exclude
    is_excluded = {ap: True for ap in antpols}
    for (i, j, pol), flags_here in flags.items():
        if not np.all(flags_here):
            is_excluded[(i, split_pol(pol)[0])] = False
            is_excluded[(j, split_pol(pol)[1])] = False

    # construct dictionary of visibility count (number each antenna touches)
    # and dictionary of number of flagged visibilities each antenna has (excluding dead ants)
    # per time and freq
    ant_Nvis = {ap: 0 for ap in antpols}
    ant_Nflag = {ap: np.zeros((Ntimes, Nfreqs), float) for ap in antpols}
    for (i, j, pol), flags_here in flags.items():
        # get antenna keys
        ap1 = (i, split_pol(pol)[0])
        ap2 = (j, split_pol(pol)[1])
        # only continue if not in is_excluded
        if not is_excluded[ap1] and not is_excluded[ap2]:
            # add to Nvis count
            ant_Nvis[ap1] += 1
            ant_Nvis[ap2] += 1
            # Add to Nflag count
            ant_Nflag[ap1] += flags_here.astype(float)
            ant_Nflag[ap2] += flags_here.astype(float)

    # iterate over antpols and construct antenna gain dictionaries
    ant_flags = {}
    for ap in antpols:
        # create flagged arrays for excluded ants
        if is_excluded[ap]:
            ant_flags[ap] = np.ones((Ntimes, Nfreqs), bool)
        # else create flags based on threshold
        else:
            # handle Nvis = 0 cases
            if ant_Nvis[ap] == 0:
                ant_Nvis[ap] = 1e-10
            # create antenna flags
            ant_flags[ap] = (ant_Nflag[ap] / ant_Nvis[ap]) > threshold

    return ant_flags


def factorize_flags(flags, spw_ranges=None, time_thresh=0.05, inplace=False):
    """
    Factorize flags into two 1D time and frequency masks. This works by
    broadcasting flags across time if the fraction of flagged times exceeds
    time_thresh, otherwise flags are broadcasted across channels in a spw_range.

    Note: although technically allowed, this function may give unexpected results
    if multiple spectral windows in spw_ranges have frequency overlap.

    Note: it is generally not recommended to set time_thresh > 0.5, which
    could lead to substantial amounts of data being flagged.

    Args:
        flags : 2D ndarray or DataContainer
            A 2D boolean ndarray, or a DataContainer containing such ndarrays.
            In the case of an ndarray, it must have shape (Ntimes, Nfreqs).

        spw_ranges : list of tuples
            list of len-2 spectral window tuples, specifying the start (inclusive)
            and stop (exclusive) index of the freq channels for each spw.
            Default is to use the whole band.

        time_thresh : float
            Fractional threshold of flagged pixels across time needed to flag all times
            per freq channel. It is not recommend to set this greater than 0.5.
            Fully flagged integrations do not count towards triggering time_thresh.

        inplace : bool
            If True, edit data in flags directly, otherwise make a deepcopy.

    Returns:
        flags : 2D ndarray or DataContainer
            Input object with flags broadcasted
    """
    # parse datatype
    if isinstance(flags, np.ndarray):

        # spw type check
        Ntimes, Nfreqs = flags.shape
        if spw_ranges is None:
            spw_ranges = [(0, Nfreqs)]
        if not isinstance(spw_ranges, list):
            raise ValueError("spw_ranges must be fed as a list of tuples")

        # inplace
        if not inplace:
            flags = copy.deepcopy(flags)

        # iterate over spws
        for (spw1, spw2) in spw_ranges:

            # identify fully flagged integrations
            fully_flagged_ints = np.isclose(np.sum(flags, axis=1), Nfreqs, atol=1e-6)
            unflagged_Ntimes = np.sum(~fully_flagged_ints, dtype=float)

            # identify channels where flagged fraction exceeds time threshold
            exceeds_thresh = (np.sum(flags[~fully_flagged_ints, spw1:spw2], axis=0, dtype=float)
                              / unflagged_Ntimes > time_thresh)
            exceeds_thresh = np.where(exceeds_thresh)[0]

            # identify integrations with flags that didn't meet broadcasting limit
            flags[:, spw1 + exceeds_thresh] = False
            flag_ints = np.max(flags[:, spw1:spw2], axis=1)

            # flag channels for all times that exceed time_thresh
            flags[:, spw1 + exceeds_thresh] = True

            # flag integrations with flags that didn't meet broadcasting limmits
            flags[flag_ints, :] = True

        return flags

    elif isinstance(flags, datacontainer.DataContainer):
        # iterate over the keys
        if not inplace:
            flags = copy.deepcopy(flags)
        for k in flags.keys():
            factorize_flags(flags[k], spw_ranges=spw_ranges, time_thresh=time_thresh, inplace=True)

        return flags

    else:
        raise ValueError("Didn't recognize data structure of flags")
