# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

from __future__ import print_function, division, absolute_import

import numpy as np
import os
import copy
from six.moves import map, range
from pyuvdata import UVData, UVCal, utils as uvutils

from .vis_clean import VisClean
from . import utils
from .datacontainer import DataContainer
from . import version, io


class TempCal(VisClean):
    
    def __init__(self, *args, **kwargs):
        """
        Initialize TempCal, see vis_clean.VisClean for kwargs.
        """ 
        super(TempCal, self).__init__(*args, **kwargs)

        # init empty dictionaries
        self.gains = {}
        self.gflags = {}

    def gains_from_autos(self, data, times, flags=None, smooth_frate=1.0, nl=1e-10, Nmirror=0,
                         keys=None, verbose=True):
        """
        Model temperature fluctuations in auto-correlations
        by dividing auto-correlation by its time-smoothed counterpart.

        This is only valid for timescales significantly shorter than
        the beam crossing timescale.
        Rcommended to factorize the flags beforehand using utils.factorize_flags.

        Args:
            data : DataContainer
                Holds auto-correlation waterfalls of shape (Ntimes, Nfreqs)
            times : ndarray
                Holds time array in Julian Date of shape (Ntimes,)
            flags : DataContainer
                Holds data flags of shape (Ntimes, Nflags)
            smooth_frate : float
                Fringe-Rate cutoff for smoothing in mHz
            nl : float
                Noise level in GP interpolation.
            Nmirror : int
                Number of time bins to mirror about edges before interpolation
            keys : list
                List of ant-pair-pol tuples to operate on.
            verbose : bool
                If True, report feedback to stdout.
        """
        # get flags
        if flags is None:
            flags = DataContainer(dict([(k, np.zeros_like(data[k], np.bool)) for k in data]))

        # get length scale in seconds
        length_scale = 1.0 / (smooth_frate * 1e-3) / (24.0 * 3600.0)

        # get keys
        if keys is None:
            keys = list(data.keys())

        # smooth data
        if not hasattr(self, 'smooth'): self.smooth = DataContainer({})
        if not hasattr(self, 'ratio'): self.ratio = DataContainer({})
        for k in keys:
            # smooth data
            utils.echo("working on {}".format(k), verbose=verbose)
            gkey = utils.split_bl(k)[0]
            data_shape = data[k].shape 
            self.smooth[k] = utils.gp_interp1d(times, data[k], flags=flags[k], length_scale=length_scale, nl=nl,
                                               Nmirror=Nmirror)
            self.gflags[gkey] = np.isclose(self.smooth[k], 0.0)
            gwgt = (~self.gflags[gkey]).astype(np.float)

            # take ratio and compute gain term
            self.ratio[k] = np.true_divide(self.data[k], self.smooth[k], where=~self.gflags[gkey])
            self.gains[gkey] = np.sqrt(self.ratio[k])

            # only allow frequency-averaged gains for now!
            self.gains[gkey] = np.sum(self.gains[gkey] * gwgt, axis=1, keepdims=True) / np.sum(gwgt, axis=1, keepdims=True).clip(1e-10, np.inf)
            self.gains[gkey] = np.repeat(self.gains[gkey], data_shape[1], axis=1)
            self.gflags[gkey] = np.repeat(np.min(self.gflags[gkey], axis=1, keepdims=True), data_shape[1], axis=1)

    def avg_ants(self, avg_ants):
        """
        Average ant-pol keys together in self.gains.

        Args:
            avg_ants : list of tuple or list of list of tuples
                List of ant-pol tuples to average together.

        Result:
            Averaged gains edited inplace in self.gains and self.gflags.
        """
        # average antennas
        if avg_ants is not None:
            # get gain keys
            keys = list(self.gains.keys())
            assert isinstance(avg_ants, list)
            if not isinstance(avg_ants[0], list): avg_ants = [avg_ants]
            # iterate over antenna lists
            for antlist in avg_ants:
                gkeys = [k for k in antlist if k in self.gains]
                avg = np.sum([self.gains[k] * ~self.gflags[k] for k in gkeys], axis=0) / np.sum([~self.gflags[k] for k in gkeys], axis=0).clip(1e-10, np.inf)
                avgf = np.min([self.gflags[k] for k in gkeys], axis=0)
                for gkey in gkeys:
                    self.gains[gkey] = avg
                    self.gflags[gkey] = avgf

    def set_abscal_time(self, times, cal_time):
        """
        Normalize gains by Julian Date of abscal.

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

    def gains_from_tempdata(self, tempdata, times, cal_time, temp_coeff):
        """
        Convert measurements of temperature data to gains based on temperature
        coefficient and the time chosen initially for absolute calibration.

        Args:
            tempdata : DataContainer
                Holds temperature data [Kelvin] for each auto-correlation
                as a function of time.
            times : 1d ndarray
                Times of data in JD
            cal_time : float
                Julian date of initial absolute calibration.
            temp_coeff : float
                Temperature coefficient converting dTemp to dGain
                in units [dGain / dTemp] 

        Result:
            self.gains : dictionary
                Holds per-antenna gains given temperature data.
        """
        raise NotImplementedError("gains_from_tempdata not yet implemented")

    def write_gains(self, fname, add_to_history='', overwrite=False, verbose=True, **kwargs):
        """
        Write self.gains to calfits file.

        Args:
            fname : str
                Filepath to output califts file.
            add_to_history : str
                Append to output add_to_historystory.
            overwrite : bool
                If True overwrite output if it exists
            verbose : bool
                Report feedback to stdout
        
        Returns:
            UVCal instance
                Output calibration object
        """
        utils.echo("...writing {}".format(fname), verbose=verbose)
        uvc = io.write_cal(fname, self.gains, self.freqs, self.times, flags=self.gflags,
                           quality=None, total_qual=None, zero_check=False,
                           overwrite=overwrite, history=version.history_string(add_to_history),
                           **kwargs)

