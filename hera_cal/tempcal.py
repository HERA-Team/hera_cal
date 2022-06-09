# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

import numpy as np
import copy

from . import utils
from .datacontainer import DataContainer


def gains_from_autos(data, times, flags=None, smooth_frate=1.0, nl=1e-10,
                     Nmirror=0, keys=None, edgeflag=0, xthin=None,
                     freq_avg=True, verbose=False):
    """
    Model temperature fluctuations in auto-correlations
    by dividing auto-correlation by its time-smoothed counterpart.

    This is only valid for timescales significantly shorter than
    the beam crossing timescale. Resultant gains have a "divide" gain_convention.

    Rcommended to factorize the flags beforehand using utils.factorize_flags.

    Args:
        data : 2D ndarray or DataContainer
            Auto-correlation waterfall of shape (Ntimes, Nfreqs) or a
            DataContainer of them, in which case keys are ant-pair-pol tuples.
        times : ndarray
            Holds time array in Julian Date of shape (Ntimes,)
        flags : 2D ndarray of DataContainer
            Data flags or DataContainer of same shape as data.
        smooth_frate : float
            Maximum fringe-rate cutoff [mHz] for "low-pass" smoothing
        nl : float
            Noise level in GP interpolation.
            Recommended to keep this near 1e-10 for non-expert users.
        Nmirror : int
            Number of time bins to mirror about edges before interpolation.
            This can minimimze impact of boundary effects.
        keys : list
            List of ant-pair-pol tuples to operate on if fed DataContainers
        edgeflag : int or len-2 tuple
            Number of channels to flag on low and high end of frequency band.
            Low and high can be specified independently via a len-2 tuple.
        xthin : int
            Thinning factor along time axis of unflagged data.
            Default is no thinning.
        freq_avg : bool
            If True take median of resultant gains across frequency (within edgeflag)
        verbose : bool
            If True, report feedback to stdout.

    Returns:
        ndarray or dictionary
            Gain ndarray (or dict holding ant-pol keys and antenna gain values)
        ndarray or dictionary
            Gain flags (or dict holding ant-pol keys and gain flag values)
        ndarray or DataContainer
            Smoothed data (or DataContainer of such with same shape as input)
    """
    # parse input
    if not isinstance(data, np.ndarray):
        # assume its a DataContainer
        gains, gflags, smooth = {}, {}, DataContainer({})
        if keys is None:
            keys = data.keys()
        # only use auto-ant and auto-pol
        keys = [k for k in keys if (k[0] == k[1]) and (k[2][0] == k[2][1])]
        assert len(keys) > 1, "Can only operate on auto-pol auto-correlations!"
        if flags is None:
            flags = DataContainer(dict([(k, None) for k in keys]))
        for key in keys:
            utils.echo("starting {}".format(key), verbose=verbose)
            gkey = utils.split_bl(key)[0]
            g, gf, s = gains_from_autos(data[key], times, flags=flags[key], smooth_frate=smooth_frate,
                                        nl=nl, Nmirror=Nmirror, edgeflag=edgeflag, freq_avg=freq_avg, verbose=False)
            gains[gkey], gflags[gkey], smooth[key] = g, gf, s
        return gains, gflags, smooth

    # from here onward, assume ndarrays
    # edgeflag
    if edgeflag is not None:
        if flags is None:
            flags = np.zeros_like(data, bool)
        else:
            flags = copy.deepcopy(flags)
        if isinstance(edgeflag, (int, np.integer, float, np.floating)):
            edgeflag = (edgeflag, edgeflag)
        assert len(edgeflag) == 2
        if edgeflag[0] > 0:
            flags[:, :edgeflag[0]] = True
        if edgeflag[1] > 0:
            flags[:, -edgeflag[1]:] = True

    # get length scale in JD
    length_scale = 1.0 / (smooth_frate * 1e-3) / (24.0 * 3600.0)

    # smooth
    data_shape = data.shape
    smooth = utils.gp_interp1d(times, data, flags=flags, length_scale=length_scale, nl=nl,
                               Nmirror=Nmirror, xthin=xthin)

    # take ratio and compute gain term
    gflags = np.isclose(smooth, 0.0)
    gains = np.sqrt(np.true_divide(data, smooth, where=~gflags))

    if freq_avg:
        # use median over mean to help w/ unflagged RFI
        gains[gflags] = np.nan
        gains = np.nanmedian(gains, axis=1, keepdims=True)
        gains[np.isnan(gains)] = 1.0  # catch for fully flagged integrations
        gains = np.repeat(gains, data_shape[1], axis=1)
        gflags = np.repeat(np.all(gflags, axis=1, keepdims=True), data_shape[1], axis=1)

    return gains, gflags, smooth


def gains_from_tempdata(tempdata, times, temp_coeff):
    """
    Convert measurements of temperature data to gains based on temperature
    coefficient and the time chosen initially for absolute calibration.

    Args:
        tempdata : ndarray or DataContainer
            Holds temperature data [Kelvin] for each auto-correlation
            as a function of time.
        times : 1d ndarray
            Times of data in JD
        temp_coeff : float
            Temperature coefficient converting dTemp to dGain
            in units [dGain / dTemp]

    Returns:
        ndarray or dictionary
            Per-antenna gains given temperature data.
    """
    raise NotImplementedError("gains_from_tempdata not yet implemented")


def avg_gain_ants(gains, antkeys, gflags=None, inplace=True):
    """
    Average gain arrays in gains dict according to antkeys
    and re-broadcast to all antkeys.

    Args:
        gains : dict
            Gain dictionary with (ant, pol) keys
        antkeys : list of tuples
            List of (ant, pol) keys in gains to average together.
        gflags : dict
            flag dictionary with same shape as gains
        inplace : bool
            If True edit input dictionaries inplace, otherwise
            return deepcopies

    Returns:
        dict
            averaged gain dictionary if not inplace
        dict
            averaged flag dictionary if not inplace
    """
    # get gain keys
    keys = list(gains.keys())
    assert isinstance(antkeys, list)

    # get gflags
    if gflags is None:
        gflags = dict([(k, np.zeros_like(gains[k], bool)) for k in gains])

    # iterate over antenna lists
    gkeys = [k for k in antkeys if k in gains]
    avg = np.sum([gains[k] * (~gflags[k]).astype(float) for k in gkeys], axis=0) / np.sum([~gflags[k] for k in gkeys], axis=0).clip(1e-10, np.inf)
    avgf = np.any([gflags[k] for k in gkeys], axis=0)

    # update gain dicts
    if inplace:
        _gains, _gflags = gains, gflags
    else:
        _gains, _gflags = copy.deepcopy(gains), copy.deepcopy(gflags)
    for gkey in gkeys:
        _gains[gkey] = avg
        _gflags[gkey] = avgf

    if not inplace:
        return _gains, _gflags


def normalize_tempgains(gains, times, norm_time, inplace=False):
    """
    Normalize temperature-derived gains to unity
    at a chosen time.

    Args:
        gains : ndarray or dict
            Gain array (Ntimes, Nfreqs) or dict of
            such with (ant, pol) keys
        times : ndarray
            Julian Date array of time integrationns
        norm_time : float
            JD time to normalize gains
        inplace : bool
            If True, edit input dicts in place. Else return deepcopies.

    Returns:
        ndarray or dict
            Normalized gain array or dict if not inplace
    """
    if isinstance(gains, dict):
        # operate on dict
        if inplace:
            ngains = gains
        else:
            ngains = copy.deepcopy(gains)
        for k in gains:
            ngains[k] = normalize_tempgains(gains[k], times, norm_time)
        if not inplace:
            return ngains

    # operate on ndarray
    tind = np.argmin(np.abs(times - norm_time))
    if inplace:
        gains /= np.abs(gains[tind])
    else:
        return gains / np.abs(gains[tind])
