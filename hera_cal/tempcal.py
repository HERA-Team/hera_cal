# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

from __future__ import print_function, division, absolute_import

import numpy as np
import os
import copy
from six.moves import map, range
from pyuvdata import UVData, UVCal, utils as uvutils
from functools import wraps

from .vis_clean import VisClean
from . import utils
from .datacontainer import DataContainer
from . import version, io


def gains_from_autos(data, times, flags=None, smooth_frate=1.0, nl=1e-10,
                     Nmirror=0, keys=None, edgeflag=0, verbose=False):
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
                                        nl=nl, Nmirror=Nmirror, edgeflag=edgeflag, verbose=False)
            gains[gkey], gflags[gkey], smooth[key] = g, gf, s
        return gains, gflags, smooth

    # from here onward, assume ndarrays
    # edgeflag
    if edgeflag is not None:
        if flags is None:
            flags = np.zeros_like(data, np.bool)
        else:
            flags = copy.deepcopy(flags)
        if isinstance(edgeflag, (int, np.int, float, np.float)):
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
                               Nmirror=Nmirror)

    # take ratio and compute gain term
    gflags = np.isclose(smooth, 0.0)
    gains = np.sqrt(np.true_divide(data, smooth, where=~gflags))

    # only allow frequency-averaged gains for now via nanmedian
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


class TempCal(VisClean):
    
    def __init__(self, *args, **kwargs):
        """
        Initialize TempCal with vis_clean.VisClean args and kwargs.
        """ 
        super(TempCal, self).__init__(*args, **kwargs)

        # init empty dictionaries
        self.gains = {}
        self.gflags = {}

    def gains_from_autos(self, data, times, flags=None, smooth_frate=1.0, nl=1e-10, Nmirror=0,
                         keys=None, edgeflag=0, verbose=True):
        """
        Model temperature fluctuations in auto-correlations
        by dividing auto-correlation by its time-smoothed counterpart.

        This is only valid for timescales significantly shorter than
        the beam crossing timescale. Resultant gains have a "divide" gain_convention.

        Rcommended to factorize the flags beforehand using utils.factorize_flags.

        Args:
            data : DataContainer
                Auto-correlation waterfalls of shape (Ntimes, Nfreqs) with ant-pair-pol tuple keys
            times : ndarray
                Holds time array in Julian Date of shape (Ntimes,)
            flags : DataContainer
                data flags with matching shape of data
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
            verbose : bool
                If True, report feedback to stdout.

            Result:
                self.gains : dictionary
                    Ant-pol keys and gain ndarray values
                self.gflags : dictionary
                    Ant-pol keys and gain flag ndarray values
                self.smooth : DataContainer
                    Ant-pair-pol keys and smoothed autocorr data values
        """
        g, gf, s = gains_from_autos(data, times, flags=flags, smooth_frate=smooth_frate,
                                    nl=nl, Nmirror=Nmirror, keys=keys, edgeflag=edgeflag, verbose=verbose)
        self.gains.update(g)
        self.gflags.update(gf)
        if not hasattr(self, 'smooth'):
            self.smooth = DataContainer({})
        for k in s:
            self.smooth[k] = s[k]

    def avg_ants(self, avg_ants):
        """
        Average ant-pol keys together in self.gains.

        Args:
            avg_ants : list of tuples
                List of ant-pol tuples to average together.

        Result:
            Averaged gains edited inplace in self.gains and self.gflags.
        """
        # get gain keys
        keys = list(self.gains.keys())
        assert isinstance(avg_ants, list)
        # iterate over antenna lists
        gkeys = [k for k in avg_ants if k in self.gains]
        avg = np.sum([self.gains[k] * ~self.gflags[k] for k in gkeys], axis=0) / np.sum([~self.gflags[k] for k in gkeys], axis=0).clip(1e-10, np.inf)
        avgf = np.any([self.gflags[k] for k in gkeys], axis=0)
        for gkey in gkeys:
            self.gains[gkey] = avg
            self.gflags[gkey] = avgf

    def set_abscal_time(self, times, cal_time):
        """
        Normalize gains to one at time of abscal.

        Args:
            times : ndarray
                Times of data in Julian Date.
            cal_time : float
                Julian Date of when absolute calibration was performed.
                This normalizes the gains at this time to one.

        Result:
            Edits self.gains in place.
        """
        tind = np.argmin(np.abs(times - cal_time))
        for gkey in self.gains:
            self.gains[gkey] /= abs(self.gains[gkey][tind])
