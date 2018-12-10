# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

from __future__ import print_function, division, absolute_import

import numpy as np
from collections import OrderedDict as odict
from copy import deepcopy
import argparse
import datetime
from six.moves import range, zip
from pyuvdata import UVData, UVCal
from uvtools import dspec
import operator

from . import io
from . import apply_cal
from .datacontainer import DataContainer
from .utils import echo


class VisClean(object):
    """
    VisClean object for visibility CLEANing
    and filtering.
    """

    def __init__(self, hd=None, filetype='miriad', **read_kwargs):
        """
        Initialize the object and optionally
        read data if provided.

        Args:
            hd : string, UVData or DataContainer object
                Filepath to a miriad, uvfits or uvh5
                datafile, a UVData object or a DataContainer
                object.
            filetype : str
                If hd is a filepath, this is its filetype.
                See hera_cal.io.HERAData for supported filetypes.
            read_kwargs : dictionary
                Keyword arguments to pass to UVData.read if
                hd is fed as a string.
        """
        if hd is not None:
            self.load_data(hd, filetype=filetype, **read_kwargs)

    def load_data(self, hd, filetype='miriad', inp_cal=None, read_data=True, **read_kwargs):
        """
        Load in visibility data.

        Args:
            hd : string, UVData or HERAData object, or list of such objects
            filetype : str
                IF data is a filepath, this is its filetype.
                See hera_cal.io.HERAData for supported filetypes.
            inp_cal : UVCal, HERACal or string to calfits
                Calibration solutions to apply to the data.
            read_data : bool, if True read the data, otherwise only read metadata.
            read_kwargs : dictionary
                Keyword arguments to pass to UVData.read if data is a string.
        """
        # check for lists etc.
        dtype = None
        if isinstance(hd, (list, tuple, np.ndarray)):
            if isinstance(hd[0], (str, np.str)):
                dtype = 'str'
            elif isinstance(hd[0], UVData):
                assert np.all([_hd.data_array is not None for _hd in hd]), "Cannot feed a list of empty HERAData or UVData objects"
                hd = reduce(operator.add, hd)

        # read HERAData if fed as string
        if isinstance(hd, (str, np.str)) or dtype == 'str':
            self.hd = io.HERAData(hd, filetype=filetype)

        # attach HERAData
        elif isinstance(hd, io.HERAData):
            self.hd = hd

        # promote UVData to HERAData
        elif isinstance(hd, UVData):
            self.hd = hd
            self.hd.__class__ = io.HERAData
            self.hd._determine_blt_slicing()
            self.hd._determine_pol_indexing()

        else:
            raise ValueError("hd must be fed as a HERAData, UVData object, string filepath or list of these")
        self.filetype = filetype

        # read data
        if read_data:
            if self.hd.data_array is None:
                data, flags, nsamples = self.hd.read(read_data=read_data, **read_kwargs)
            else:
                echo("Requested to read_data but hd.data_array already exists.\nUsing existing data_array to build DataContainers")
                data, flags, nsamples = self.hd.build_datacontainers()

            # link to the DataContainers
            self.data = data
            self.flags = flags
            self.nsamples = nsamples

        else:
            self.hd.read(read_data=read_data, **read_kwargs)

        # assign necessary metadata
        mdict = self.hd.get_metadata_dict()
        self.antpos = mdict['antpos']
        self.ants = mdict['ants']
        self.freqs = mdict['freqs']
        self.times = mdict['times']
        self.lsts = mdict['lsts']
        self.pols = mdict['pols']

        self.Nfreqs = len(self.freqs)
        self.Ntimes = len(self.times)
        self.dlst = np.median(np.diff(self.lsts))
        self.dtime = np.median(np.diff(self.times))
        self.dnu = np.median(np.diff(self.freqs))
        self.bls = sorted(set(self.hd.get_antpairs()))
        self.blvecs = odict([(bl, self.antpos[bl[0]] - self.antpos[bl[1]]) for bl in self.bls])
        self.bllens = odict([(bl, np.linalg.norm(self.blvecs[bl]) / 2.99e8) for bl in self.bls])
        self.lat = self.hd.telescope_location_lat_lon_alt[0] * 180 / np.pi
        self.delays = np.fft.fftshift(np.fft.fftfreq(self.Nfreqs, self.dnu)) * 1e9  # ns
        self.frates = np.fft.fftshift(np.fft.fftfreq(self.Ntimes, self.dtime * 24 * 3600)) * 1e3  # mHz

        # load calibration solutions
        if inp_cal is not None:
            self.apply_cal(inp_cal)

    def apply_cal(self, inp_cal):
        """
        Load calibration solutions and apply to the data.

        Args:
            inp_cal : UVCal, HERACal or filepath to calfits file
        """
        assert hasattr(self, 'data'), "Must have loaded data before applying calibration solutions."

        # read HERACal if fed as string
        if isinstance(inp_cal, (str, np.str)):
            self.inp_cal = io.HERACal(inp_cal)
            cal_gains, cal_flags, cal_quals, cal_tquals = self.inp_cal.read()

        # read data if not already done
        elif isinstance(inp_cal, io.HERACal):
            self.inp_cal = inp_cal
            if self.inp_cal.gain_array is None:
                cal_gains, cal_flags, cal_quals, cal_tquals = self.inp_cal.read()
            else:
                cal_gains, cal_flags, cal_quals, cal_tquals = self.inp_cal.build_calcontainers()

        # promote UVCal to HERACal
        elif isinstance(inp_cal, UVCal):
            self.inp_cal = inp_cal
            self.inp_cal.__class__ = io.HERACal
            cal_gains, cal_flags, cal_quals, cal_tquals = self.inp_cal.build_calcontainers()

        else:
            raise ValueError("inp_cal must be fed as a HERACal or UVCal object, or a calfits filepath")

        # apply calibration solutions to data
        apply_cal.calibrate_in_place(self.data, cal_gains, self.flags, cal_flags, gain_convention=self.inp_cal.gain_convention)

    def write_data(self, data, filename, overwrite=False, flags=None, nsamples=None, filetype='miriad',
                   add_to_history=None, verbose=True):
        """
        Write data attached to object to file. If data or flags are fed as DataContainers,
        create a new HERAData with those data and write that to file. Can only write
        data that has associated metadata in the self.hd HERAData object.

        Args:
            data : DataContainer, holding complex visibility data to write to disk.
            filename : string, output filepath
            overwrite : bool, if True, overwrite output file if it exists
            flags : DataContainer, boolean flag arrays to write to disk with data.
            nsamples : DataContainer, float nsample arrays to write to disk with data.
            filetype : string, output filetype. ['miriad', 'uvh5', 'uvfits'] supported.
            add_to_history : string, string to prepend to hd history.
        """
        # get common keys
        keys = [k for k in self.hd.get_antpairpols() if data.has_key(k)]
        if flags is not None:
            keys = [k for k in keys if flags.has_key(k)]
        if nsamples is not None:
            keys = [k for k in keys if nsamples.has_key(k)]

        # select out a copy of hd
        hd = self.hd.select(bls=keys, inplace=False)

        # update HERAData
        hd.update(data=data, flags=flags, nsamples=nsamples)

        # add history
        if add_to_history is not None:
            hd.history = "{} {}".format(add_to_history, hd.history)

        if filetype == 'miriad':
            hd.write_miriad(filename, clobber=overwrite)
        elif filetype == 'uvh5':
            hd.write_uvh5(filename, clobber=overwrite)
        elif filetype == 'uvfits':
            hd.write_uvfits(filename)
        else:
            raise ValueError("filetype {} not recognized".format(filetype))
        echo("...wrote {}".format(filename), verbose=verbose)

    def vis_clean(self, keys=None, data=None, flags=None, wgts=None, ax='freq', horizon=1.0, standoff=0.0,
                  min_dly=0.0, max_frate=None, tol=1e-6, maxiter=100, window='none',
                  gain=1e-1, skip_wgt=0.1, filt2d_mode='rect', alpha=0.5, edgecut_low=0, edgecut_hi=0,
                  overwrite=False, verbose=True):
        """
        Perform a CLEAN deconvolution on data and insert
        results into self.clean_model and self.clean_resid.

        Args:
            keys : list of bl-pol keys in data to CLEAN
            data : DataContainer, data to clean. Default is self.data
            flags : Datacontainer, flags to use. Default is self.flags
            wgts : DataContainer, weights to use. Default is None.
            ax : str, axis to CLEAN, options=['freq', 'time', 'both']
                Where 'freq' and 'time' are 1D CLEANs and 'both' is a 2D CLEAN
            standoff: fixed additional delay beyond the horizon (in ns) to CLEAN [freq cleaning]
            horizon: coefficient to bl_len where 1 is the horizon [freq cleaning]
            min_dly: minimum delay used for freq cleaning [ns]: if bl_len * horizon + standoff < min_dly, use min_dly.
            tol: CLEAN algorithm convergence tolerance (see aipy.deconv.clean)
            gain: The fraction of a residual used in each iteration. If this is too low, clean takes
                unnecessarily long. If it is too high, clean does a poor job of deconvolving.
            window: window function for filtering applied to the filtered axis.
                See uvtools.dspec.gen_window for options.
            alpha : float, if window is Tukey, this is its alpha parameter.
            skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
                Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
                time. Skipped channels are then flagged in self.flags.
                Only works properly when all weights are all between 0 and 1.
            maxiter: Maximum number of iterations for aipy.deconv.clean to converge.
            edgecut_low : int, number of bins to consider zero-padded at low-side of the FFT axis,
                such that the windowing function smoothly approaches zero. If ax is 'both',
                can feed as a tuple specifying for 0th and 1st FFT axis.
            edgecut_hi : int, number of bins to consider zero-padded at high-side of the FFT axis,
                such that the windowing function smoothly approaches zero. If ax is 'both',
                can feed as a tuple specifying for 0th and 1st FFT axis.
            verbose: If True print feedback to stdout

        Notes
        -----
        One can create a "clean_data" DataContainer via
            self.clean_model + self.clean_resid * ~self.flags
        """
        # type checks
        assert ax in ['freq', 'time', 'both'], "ax must be one of ['freq', 'time', 'both']"

        if ax == 'time':
            assert max_frate is not None, "if time cleaning, must feed max_frate parameter"

        # initialize containers
        for dc in ['clean_model', 'clean_resid', 'clean_flags', 'clean_info']:
            if not hasattr(self, dc):
                setattr(self, dc, DataContainer({}))

        # select DataContainers
        if data is None:
            data = self.data
        if flags is None:
            flags = self.flags

        # get keys
        if keys is None:
            keys = data.keys()

        # get weights
        if wgts is None:
            wgts = DataContainer(dict([(k, np.ones_like(flags[k], dtype=np.float)) for k in keys]))

        # parse max_frate if fed
        if max_frate is not None:
            if isinstance(max_frate, (int, np.integer, float, np.float)):
                max_frate = DataContainer(dict([(k, max_frate) for k in data]))
            assert isinstance(max_frate, DataContainer), "If fed, max_frate must be a float, or a DataContainer of floats"

        # iterate over keys
        for k in keys:
            if k in self.clean_model and overwrite is False:
                echo("{} exists in clean_model and overwrite is False, skipping...".format(k), verbose=verbose)
                continue
            echo("Starting CLEAN of {} at {}".format(k, str(datetime.datetime.now())), verbose=verbose)

            # form d and w
            d = data[k]
            f = flags[k]
            w = (~f).astype(np.float) * wgts[k]

            # freq clean
            if ax == 'freq':
                mdl, res, info = dspec.vis_filter(d, w, bl_len=self.bllens[k[:2]], sdf=self.dnu, standoff=standoff, horizon=horizon,
                                                  min_dly=min_dly, tol=tol, maxiter=maxiter, window=window, alpha=alpha,
                                                  gain=gain, skip_wgt=skip_wgt, edgecut_low=edgecut_low,
                                                  edgecut_hi=edgecut_hi)
                flgs = np.zeros_like(mdl, dtype=np.bool)
                for i, _info in enumerate(info):
                    if 'skipped' in _info:
                        flgs[i] = True

            # time clean
            elif ax == 'time':
                # make sure bad channels are zeroed out: this is a common fault case
                # channels are bad (i.e. all zeros) but are not completely flagged
                # and this causes filtering to hang...
                bad_chans = (~np.min(np.isclose(d, 0.0), axis=0, keepdims=True)).astype(np.float)
                w *= bad_chans
                mdl, res, info = dspec.vis_filter(d, w, max_frate=max_frate[k], dt=self.dtime * 24 * 3600, tol=tol, maxiter=maxiter,
                                                  window=window, alpha=alpha, gain=gain, skip_wgt=skip_wgt, edgecut_low=edgecut_low,
                                                  edgecut_hi=edgecut_hi)
                flgs = np.zeros_like(mdl, dtype=np.bool)
                for i, _info in enumerate(info):
                    if 'skipped' in _info:
                        flgs[:, i] = True

            # 2D clean
            elif ax == 'both':
                # check for completely flagged baseline
                if w.max() > 0.0:
                    mdl, res, info = dspec.vis_filter(d, w, bl_len=self.bllens[k[:2]], sdf=self.dnu, max_frate=max_frate[k], dt=self.dtime * 24 * 3600,
                                                      standoff=standoff, horizon=horizon, min_dly=min_dly, tol=tol, maxiter=maxiter, window=window,
                                                      alpha=alpha, gain=gain, edgecut_low=edgecut_low, edgecut_hi=edgecut_hi,
                                                      filt2d_mode=filt2d_mode)
                    flgs = np.zeros_like(mdl, dtype=np.bool)
                else:
                    # flagged baseline
                    mdl = np.zeros_like(d)
                    res = d - mdl
                    flgs = np.ones_like(mdl, dtype=np.bool)
                    info = {'skipped': True}

            # append to new Containers
            self.clean_model[k] = mdl
            self.clean_resid[k] = res
            self.clean_flags[k] = flgs
            self.clean_info[k] = info

    def fft_data(self, data=None, flags=None, keys=None, ax='freq', window='none', alpha=0.1, overwrite=False,
                 edgecut_low=0, edgecut_hi=0, ifft=True, verbose=True):
        """
        Take FFT of data and assign to self.dfft. Note the fourier convention via ifft kwarg.

        Args:
            data : DataContainer
                Object to pull data to FT from. Default is self.data.
            flags : DataContainer
                Object to pull flags in FT from. Default is no flags.
            keys : list of tuples
                List of keys from clean_data to FFT. Default is all keys.
            ax : str, options=['freq', 'time', 'both']
                Axis along with to take FFT.
            window : str
                Windowing function to apply across frequency before FFT. If ax is 'both',
                can feed as a tuple specifying window for 0th and 1st FFT axis.
            alpha : float
                If window is 'tukey' this is its alpha parameter. If ax is 'both',
                can feed as a tuple specifying alpha for 0th and 1st FFT axis.
            edgecut_low : int, number of bins to consider zero-padded at low-side of the FFT axis,
                such that the windowing function smoothly approaches zero. If ax is 'both',
                can feed as a tuple specifying for 0th and 1st FFT axis.
            edgecut_hi : int, number of bins to consider zero-padded at high-side of the FFT axis,
                such that the windowing function smoothly approaches zero. If ax is 'both',
                can feed as a tuple specifying for 0th and 1st FFT axis.
            ifft : bool, if True, use ifft instead of fft
            overwrite : bool
                If self.dfft[key] already exists, overwrite its contents.
        """
        # type checks
        assert ax in ['freq', 'time', 'both'], "ax must be one of ['freq', 'time', 'both']"
        fft2d = ax == 'both'
        if fft2d:
            # 2D fft
            if isinstance(window, (str, np.str)):
                window = (window, window)
            if isinstance(alpha, (int, np.integer, float, np.float)):
                alpha = (alpha, alpha)
            if isinstance(edgecut_low, (int, np.integer)):
                edgecut_low = (edgecut_low, edgecut_low)
            if isinstance(edgecut_hi, (int, np.integer)):
                edgecut_hi = (edgecut_hi, edgecut_hi)

        # generate window
        if ax == 'freq':
            win = dspec.gen_window(window, self.Nfreqs, alpha=alpha, edgecut_low=edgecut_low, edgecut_hi=edgecut_hi)[None, :]
        elif ax == 'time':
            win = dspec.gen_window(window, self.Ntimes, alpha=alpha, edgecut_low=edgecut_low, edgecut_hi=edgecut_hi)[:, None]
        else:
            w1 = dspec.gen_window(window[0], self.Ntimes, alpha=alpha[0], edgecut_low=edgecut_low[0], edgecut_hi=edgecut_hi[0])[:, None]
            w2 = dspec.gen_window(window[1], self.Nfreqs, alpha=alpha[1], edgecut_low=edgecut_low[1], edgecut_hi=edgecut_hi[1])[None, :]
            win = w1 * w2

        # get data
        if data is None:
            data = self.data
        if flags is not None:
            wgts = DataContainer(dict([(k, (~flags[k]).astype(np.float)) for k in flags]))
        else:
            wgts = DataContainer(dict([(k, np.ones_like(data[k], dtype=np.float)) for k in data]))

        # get keys
        if keys is None:
            keys = data.keys()

        if not hasattr(self, 'dfft'):
            self.dfft = DataContainer({})

        # iterate over keys
        for k in keys:
            if k not in data:
                echo("{} not in data, skipping...".format(k), verbose=verbose)
                continue
            if k in self.dfft and not overwrite:
                echo("{} in dfft and overwrite == False, skipping...".format(k), verbose=verbose)
                continue

            # FFT
            if ax == 'time':
                if ifft:
                    dfft = np.fft.fftshift(np.fft.ifft(data[k] * win * wgts[k], axis=0), axes=0)
                else:
                    dfft = np.fft.fftshift(np.fft.fft(data[k] * win * wgts[k], axis=0), axes=0)
            elif ax == 'freq':
                if ifft:
                    dfft = np.fft.fftshift(np.fft.ifft(data[k] * win * wgts[k], axis=1), axes=1)
                else:
                    dfft = np.fft.fftshift(np.fft.fft(data[k] * win * wgts[k], axis=1), axes=1)
            else:
                if ifft:
                    dfft = np.fft.fftshift(np.fft.ifft2(data[k] * win * wgts[k]))
                else:
                    dfft = np.fft.fftshift(np.fft.fft2(data[k] * win * wgts[k]))

            self.dfft[k] = dfft
