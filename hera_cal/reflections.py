# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

"""
Functions for modeling and removing reflections.

Given that a nominal visibility between antennas 1 & 2
is the correlation of two voltage spectra

    V_12 = v_1 v_2^*

a reflection is defined as an additive term in either
v_1 or v_2 that introduces another copy of the voltage as

    v_1 <= v_1 + eps * v_1 = v_1 (1 + eps)

This kind of reflection could be, for example, a cable
reflection or intra-feed reflection.

The reflection itself (eps) can be decomposed into three parameters:
its amplitude (A), delay (tau) and phase offset (phi)

    eps = A * exp(2pi * i * tau * nu + i * phi).

Reflections can be removed from the data within the framework
of standard antenna-based calibration

    V_12^data = V_12^model g_1 g_2^*

The basic idea is that the standard antenna gain, g_1, should be
multiplied by an additional term (1 + eps_1), where eps_1 is the reflection
derived for antenna 1's signal chain, such that an effective gain is formed

    g_1^eff = g_1 (1 + eps)

and the calibration equation becomes

    V_12^data = V_12^model g_1^eff g_2^eff*

Reflections that couple other voltage signals into v_1 cannot
be described in this formalism (for example, over-the-air aka
inter-feed reflections).
"""
from __future__ import print_function, division, absolute_import

import numpy as np
import aipy
import os
import copy
import matplotlib.pyplot as plt
from pyuvdata import UVData, UVCal
import pyuvdata.utils as uvutils

from . import io
from . import abscal_funcs
from . import delay_filter
from .datacontainer import DataContainer


class ReflectionFitter(object):
    def __init__(self, data=None, **load_kwargs):
        """
        Initialize a ReflectionFitter object and optionally
        load / read data if provided.

        Args:
            data : Filepath to a miriad or uvh5 file
                or a pyuvdata.UVData object
            load_kwargs : keyword arguments for self.load_data
        """
        if data is not None:
            self.load_data(data, **load_kwargs)

    def load_data(self, data, filetype='uvh5', **read_kwargs):
        """
        Load data for reflection fitting.

        Args:
            data : Filepath to data file or UVData object
            read_data : If True, read data into Fitter object
            filetype : filetype of data file if filepath provided
            read_kwargs : keyword arguments for HERAData.read
                if filepath provided and read_data is True
        """
        # mount data into hd object
        self.hd = io.to_HERAData(data, filetype=filetype)

        # read data if desired
        if self.hd.data_array is not None:
            self.data, self.flags, _ = self.hd.build_datacontainers()
        else:
            self.data, self.flags, _ = self.hd.read(**read_kwargs)

        # save metadata
        self.antpos, self.ants = self.hd.get_ENU_antpos(center=True, pick_data_ants=True)
        self.pols = np.array([uvutils.polnum2str(p) for p in self.hd.polarization_array])
        self.freqs = np.unique(self.hd.freq_array)
        self.dnu = np.median(np.diff(self.freqs))
        self.Nfreqs = self.hd.Nfreqs
        self.times = np.unique(self.hd.time_array)
        self.Ntimes = self.hd.Ntimes

    def dly_clean_data(self, keys=None, tol=1e-5, maxiter=500, gain=0.1, skip_wgt=0.2,
                       dly_cut=1500, edgecut=0, taper='none', alpha=0.1, timeavg=True,
                       broadcast_flags=False, time_thresh=0.05, overwrite=False, verbose=False):
        """
        Run a Delay Clean on self.data dictionary to derive a model of the
        visibility free of flagged channels. CLEAN data is inserted into
        self.clean_data.

        Args:
            keys : len-3 tuple, keys of self.data to run clean on. Default is all keys.
            tol : float, stopping tolerance for CLEAN. See aipy.deconv.clean
            maxiter : int, maximum number of CLEAN iterations
            gain : float, CLEAN gain
            skip_wgt : float, fraction of flagged channels needed to skip a time integration
            dly_cut : float, maximum delay [nanoseconds] to model FT of CLEAN visibility
            edgecut : int, number of channels to exclude in CLEAN on either side of band.
                Note this is not the same as flagging edge channels. This excludes them
                entirely, such that if a taper is applied, it is not discontinuous at
                the band edges.
            taper : str, Tapering function to apply across freq before FFT
            alpha : float, if taper is Tukey, this its alpha parameter
            timeavg : bool, if True, average data across time weighted by flags
            broadcast_flags : bool, if True, broadcast flags across time using time_thresh
            time_thresh : float, if fraction of flagged times exceeds this ratio,
                flag a channel for all times.
        """
        # initialize containers
        if hasattr(self, "clean_data") and not overwrite:
            raise ValueError("self.clean_data exists and overwrite is False...")
        self.clean_data = DataContainer({})
        self.resid_data = DataContainer({})
        self.clean_info = {}
        self.clean_freqs = self.freqs
        if edgecut > 0:
            self.clean_freqs = self.clean_freqs[edgecut:-edgecut]

        # get keys
        if keys is None:
            keys = self.data.keys()

        # iterate over keys
        for k in keys:
            echo("...Cleaning data key {}".format(k), verbose=verbose)
            (model, flag, residual, dlys,
             info) = reflections_delay_filter(self.data[k].copy(), self.flags[k].copy(), self.dnu, tol=tol,
                                              maxiter=maxiter, gain=gain, skip_wgt=skip_wgt, dly_cut=dly_cut,
                                              edgecut=edgecut, taper=taper, alpha=alpha, timeavg=timeavg,
                                              broadcast_flags=broadcast_flags, time_thresh=time_thresh)
            # add residual back into model
            model += residual * ~flag

            # append to containers
            self.clean_data[k] = model
            self.resid_data[k] = residual
            self.clean_info[k] = info

        self.clean_dlys = dlys
        self.clean_times = self.times
        if timeavg:
            self.clean_times = np.mean(self.clean_times, keepdims=True)

    def model_reflections(self, dly_range, keys=None, data='clean', edgecut=0, taper='none',
                          alpha=0.1, zero_pad=0, overwrite=False, fthin=10, verbose=True):
        """
        Model reflections in (ideally RFI-free) visibility data. To CLEAN data of
        flags see the self.dly_clean_data() function. Recommended to set zero_pad
        to at least as large as Nfreqs.

        Args:
            dly_range : len-2 tuple of delay range [nanoseconds] to search
                within for reflections. Must be either both positive or both negative.
            keys : list of len-3 tuples, keys in data model reflections over. Default
                is to use all auto-correlation keys.
            data : str, data dictionary to find reflections in. Options are ['clean', 'data'].
            edgecut : int, number of channels on band edges to exclude from modeling
            taper : str, tapering function to apply across freq before FFT
            alpha : float, if taper is Tukey, this is its alpha parameter
            zero_pad : int, number of channels to pad band edges with zeros before FFT
            fthin : int, scaling factor to down-select frequency axis when solving for phase
            overwrite : bool, if True, overwrite dictionaries
        """
        # initialize containers
        if hasattr(self, 'epsilon') and not overwrite:
            raise ValueError("reflection dictionaries exist but overwrite is False...")
        self.epsilon = {}
        self.amps = {}
        self.phs = {}
        self.delays = {}
        self.peak_ratio = {}
        self.dfft = DataContainer({})

        # configure data
        if data == 'clean':
            data = self.clean_data
            freqs = self.clean_freqs
            self.reflection_times = self.clean_times
        elif data == 'data':
            # if using unCLEANed data, apply flags now on-the-fly
            data = DataContainer(dict([(k, self.data[k] * ~self.flags[k]) for k in self.data.keys()]))
            freqs = self.freqs
            self.reflection_times = self.times
        else:
            raise ValueError("Didn't recognize data dictionary {}".format(data))

        # get keys: only use auto correlations to model reflections
        if keys is None:
            keys = [k for k in data.keys() if k[0] == k[1]]

        # get window kwargs
        win_kwargs = {}
        if taper == 'tukey':
            win_kwargs['alpha'] = alpha

        # iterate over keys
        for k in keys:
            # get gain key
            if dly_range[0] >= 0:
                rkey = (k[0], uvutils.parse_jpolstr(k[2][0]))
            else:
                rkey = (k[1], uvutils.parse_jpolstr(k[2][1]))

            # find reflection
            echo("...Modeling reflections in {}, assigning to {}".format(k, rkey), verbose=verbose)
            (eps, amp, delays, phs, inds, sig, dfft,
             dly_arr) = fit_reflection(data[k], dly_range, freqs, edgecut=edgecut, taper=taper,
                                       zero_pad=zero_pad, full_freqs=self.freqs, fthin=fthin, **win_kwargs)
            self.epsilon[rkey] = eps
            self.amps[rkey] = amp
            self.phs[rkey] = phs
            self.delays[rkey] = delays
            self.peak_ratio[rkey] = sig
            self.dfft[k] = dfft

    def write_reflections(self, output_calfits, input_calfits=None, overwrite=False):
        """
        Given a filepath to antenna gain calfits file, load the
        calibration, incorporate reflection term from self.epsilon dictionary
        and write to output.

        Args:
            output_calfits : str, filepath to write output calfits file
            input_calfits : str, filepath to input calfits file to multiply in with
                reflection gains.
            overwrite : bool, if True, overwrite output file

        Returns:
            uvc : HERACal object with new gains
        """
        # Create reflection gains
        rgains = _form_gains(self.epsilon)
        flags, quals, tquals = None, None, None

        if input_calfits is not None:
            # Load calfits
            cal = io.HERACal(input_calfits)
            gains, flags, quals, tquals = cal.read()

            # Merge gains
            rgains = abscal_funcs.merge_gains([gains, rgains])

            # resolve possible broadcasting across time and freq
            if cal.Ntimes > self.Ntimes:
                time_array = cal.time_array
            else:
                time_array = self.reflection_times
            if cal.Nfreqs > self.Nfreqs:
                freq_array = cal.freq_array
            else:
                freq_array = self.freqs
            kwargs = dict([(k, getattr(cal, k)) for k in ['gain_convention', 'x_orientation',
                                                          'telescope_name', 'cal_style']])
        else:
            time_array = self.reflection_times
            freq_array = self.freqs
            kwargs = {}

        uvc = io.write_cal(output_calfits, rgains, freq_array, time_array, flags=flags,
                           quality=quals, total_qual=tquals, outdir=os.path.dirname(output_calfits),
                           zero_check=False, overwrite=overwrite, **kwargs)
        return uvc


def _form_gains(epsilon):
    """ Turn epsilon dictionaries into gain dictionaries """
    return dict([(k, 1 + epsilon[k]) for k in epsilon.keys()])


def construct_reflection(freqs, amp, tau, phs, real=False):
    """
    Given a frequency range and reflection parameters,
    construct the complex (or real) valued reflection
    term eps.

    Args:
        freqs : 1-D array of frequencies [Hz]
        amp : N-D array of reflection amplitudes
        tau : N-D array of reflection delays [nanosec]
        phs : N-D array of phase offsets [radians]
        real : bool, if True return as real-valued. Else
            return as complex-valued.

    Returns:
        eps : complex (or real) valued reflection across
            input frequencies.
    """
    # make reflection
    eps = amp * np.exp(2j * np.pi * tau / 1e9 * freqs + 1j * phs)
    if real:
        eps = eps.real

    return eps


def fit_reflection(data, dly_range, freqs, full_freqs=None, edgecut=0,
                   taper='none', zero_pad=0, real=False, fthin=10, **win_kwargs):
    """
    Fourier Transform of RFI-clean visibility data and fit for a reflection
    in a specified region of delay and solve for the reflection coefficients.

    Args:
        data : complex 2D array with shape [Ntimes, Nfreqs]
        dly_range : len-2 tuple specifying range of delays [nanosec]
            to look for reflection within. Must be both positive or both negative.
        freqs : 1D array of frequencies [Hz] matching data.shape[1]
        full_freqs : optional, 1D array of frequencies to evaluate reflection term over.
        edgecut : optional, integer value of edge channels to cut from data initially.
        taper : optional, a taper string to apply to data before FFT (see aipy.dsp.gen_window)
        zero_pad : optional, an integer number of zero-valued bins to add to both edges of data.
        real : optional, if True return reflection eps as real-valued, complex otherwise
        fthin : integer thinning parameter along freq axis when solving for reflection phase
        win_kwargs : extra keyword arguments to feed to aipy.dsp.gen_window

    Returns: (eps, r_amps, r_dlys, r_phs, r_dly_inds, r_significance, dfft, delays)
        eps : N-D array of reflection sinusoids
        r_amps : N-D array holding reflection amplitudes
        r_dlys : N-D array holding reflection delays [nanosec]
        r_phs : N-D array holding reflection phases [radians]
        r_dly_inds : N-D array holding indices of the reflection delays in
            the FT of input data
        r_significance : N-D array holding ref_peak / median(abs_dfft_selection),
            a measure of how significant the reflection is compared to neighboring delays
        dfft : N-D array holding FFT of input data along freq axis
        delays : N-D array holding delay bins of dfft [nanosec]
    """
    # type checks
    assert dly_range[0] * dly_range[1] >= 0, "dly_range must be both positive or both negative"

    # enact edgecut
    if edgecut > 0:
        d = data[:, edgecut:-edgecut].copy()
        freqs = freqs[edgecut:-edgecut]
    else:
        d = data.copy()

    # set a taper
    assert d.ndim == 2, "input data must be 2-dimensional with shape [Ntimes, Nfreqs]"
    Ntimes, Nfreqs = d.shape
    if taper in ['none', 'None', None]:
        t = np.ones(Nfreqs)[None, :]
    else:
        t = aipy.dsp.gen_window(Nfreqs, window=taper, **win_kwargs)[None, :]

    # zero pad
    dnu = np.median(np.diff(freqs))
    if zero_pad > 0:
        z = np.zeros((Ntimes, zero_pad), dtype=d.dtype)
        d = np.concatenate([z, d, z], axis=1)
        f = np.arange(1, zero_pad + 1) * dnu
        freqs = np.concatenate([freqs.min() - f[::-1], freqs, freqs.max() + f])
        t = np.concatenate([z[:1], t, z[:1]], axis=1)

    # get delays
    Ntimes, Nfreqs = d.shape
    assert Nfreqs == len(freqs), "data Nfreqs != len(freqs)"
    dlys = np.fft.fftfreq(Nfreqs, d=dnu) * 1e9

    # fourier transform
    dfft = np.fft.fft(d * t, axis=1)

    # select delay range
    select = np.where((dlys > dly_range[0]) & (dlys < dly_range[1]))[0]
    if len(select) == 0:
        raise ValueError("No delays in specified range {} ns".format(dly_range))

    # locate amplitude peak
    abs_dfft = np.abs(dfft)
    abs_dfft_selection = abs_dfft[:, select]
    ref_peaks = np.max(abs_dfft_selection, axis=1, keepdims=True)
    ref_dly_inds = np.argmin(np.abs(abs_dfft_selection - ref_peaks), axis=1) + select.min()
    ref_dlys = dlys[ref_dly_inds, None]

    # calculate shifted peak for sub-bin resolution
    # https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
    # alpha = a, beta = b, gamma = g
    a = abs_dfft[np.arange(Ntimes), ref_dly_inds - 1, None]
    g = abs_dfft[np.arange(Ntimes), ref_dly_inds + 1, None]
    b = abs_dfft[np.arange(Ntimes), ref_dly_inds, None]
    bin_shifts = 0.5 * (a - g) / (a - 2 * b + g)

    # update delay center and peak value given shifts
    ref_dlys += bin_shifts * np.median(np.diff(dlys))
    ref_peaks = b - 0.25 * (a - g) * bin_shifts

    # get reflection significance, defined as ref_peak / median_abs
    ref_significance = ref_peaks / np.median(abs_dfft_selection, axis=1, keepdims=True)

    # get reflection amplitude
    ref_amps = ref_peaks / np.max(abs_dfft[:, np.abs(dlys) < np.abs(dly_range).min()], axis=1, keepdims=True)

    # get reflection phase by fitting cosines to filtered data thinned along frequency axis
    filt = np.zeros_like(dfft)
    filt[:, select] = dfft[:, select]
    if Nfreqs % 2 == 1:
        filt[:, -select] = dfft[:, -select]
    else:
        filt[:, -select - 1] = dfft[:, -select - 1]
    filt = np.fft.ifft(filt, axis=1)
    if zero_pad > 0:
        filt = filt[:, zero_pad:-zero_pad]
        freqs = freqs[zero_pad:-zero_pad]
        t = t[:, zero_pad:-zero_pad]
    phases = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    cosines = np.array([construct_reflection(freqs[::fthin], ref_amps, ref_dlys, p) for p in phases])
    residuals = np.sum((filt[None, :, ::fthin].real - cosines.real * t[None, :, ::fthin])**2, axis=-1)
    ref_phs = phases[np.argmin(residuals, axis=0)][:, None] % (2 * np.pi)
    dlys = np.fft.fftfreq(len(freqs), d=dnu) * 1e9

    # get reflection phase by interpolation: this didn't work well in practice
    # when the reflection delay wasn't directly centered in a sampled delay bin,
    # but leaving it here for possible future use b/c it should be much faster.
    # x = np.array([-1, 0, 1])
    # a = dfft[np.arange(Ntimes), ref_dly_inds-1]
    # g = dfft[np.arange(Ntimes), ref_dly_inds+1]
    # b = dfft[np.arange(Ntimes), ref_dly_inds]
    # y = np.array([a, b, g]).T
    # real = np.array([interp1d(x, y[i].real, kind='linear')(bin_shifts[i]) for i in range(Ntimes)])
    # imag = np.array([interp1d(x, y[i].imag, kind='linear')(bin_shifts[i]) for i in range(Ntimes)])
    # ref_phs = np.angle(real + 1j*imag)

    # construct reflection
    if full_freqs is None:
        full_freqs = freqs
    eps = construct_reflection(full_freqs, ref_amps, ref_dlys, ref_phs)

    return (eps, ref_amps, ref_dlys, ref_phs, ref_dly_inds, ref_significance,
            np.fft.fftshift(dfft, axes=1), np.fft.fftshift(dlys))


def reflections_delay_filter(data, flags, dnu, dly_cut=200, edgecut=0, taper='none', alpha=0.1, tol=1e-5, maxiter=500,
                             gain=0.1, skip_wgt=0.2, timeavg=False, broadcast_flags=False, time_thresh=0.1):
    """
    Delay (CLEAN) and filter flagged data and return model and residual visibilities.

    Args:
        data : 2D complex visibility data with shape [Ntimes, Nfreqs]
        flags : 2D boolean flags with shape [Ntimes, Nfreqs]
        dnu : channelization width, Hz
        dly_cut : float, maximumm CLEANing delay in nanosec
        edgecut : int, number of channels to exclude on either side of band
        taper : str, tapering function to apply to data across freq before FT. See aipy.dsp.gen_window
        alpha : float, if taper is 'tukey', this is its alpha parameter
        tol : float, tolerance for aipy.deconv CLEAN threshold
        maxiter : int, maximum number of CLEAN iterations
        gain : float, CLEAN gain
        skip_wgt : float, fraction of flagged data across frequency to skip CLEAN of an integration.
        timeavg : bool, average data weighted by flags across time before CLEAN
        broadcast_flags : bool, broadcast flags across time if True, determined by time_thresh
        time_thresh : float, ratio of flagged channels across time to flag a freq channel for all times

    Returns: (mdl, res, dlys, info)
        mdl : 2D array of visibility CLEAN model
        res : 2D array of visibility residual
        dlys : 1D array of delays in ns
        info : dictionary of CLEAN results
    """
    # enact edgecut
    if edgecut > 0:
        d = data[:, edgecut:-edgecut].copy()
        f = flags[:, edgecut:-edgecut].copy()
    else:
        d = data.copy()
        f = flags.copy()

    # average across time
    w = (~f).astype(np.float)
    if timeavg:
        wsum = np.sum(w, axis=0, keepdims=True).clip(1e-10, np.inf)
        d = np.sum(d * w, axis=0, keepdims=True) / wsum
        w = wsum
        f = np.isclose(wsum, 1e-10)

    # broadcast flags
    Ntimes = float(f.shape[0])
    Nfreqs = float(f.shape[1])
    if broadcast_flags:
        # get Ntimes and Nfreqs that aren't completely flagged across opposite axis
        freq_contig_flags = np.sum(f, axis=1) / Nfreqs > 0.99999999
        Ntimes = np.sum(~freq_contig_flags, dtype=np.float)

        # get freq channels where non-contiguous flags exceed threshold
        flag_freq = (np.sum(f[~freq_contig_flags], axis=0, dtype=np.float) / Ntimes) > time_thresh

        # flag integrations holding flags that didn't meet broadcasting limit
        f[:, flag_freq] = False
        f[np.max(f, axis=1)] = True
        f[:, flag_freq] = True

    # delay filter
    kwargs = {}
    if taper == 'tukey':
        kwargs['alpha'] = alpha
    mdl, res, info = delay_filter.delay_filter(d, w, 0., dnu / 1e9, min_dly=dly_cut, skip_wgt=skip_wgt,
                                               window=taper, tol=tol, maxiter=maxiter, gain=gain, **kwargs)
    dlys = np.fft.fftshift(np.fft.fftfreq(d.shape[1], d=dnu)) * 1e9

    return mdl, f, res, dlys, info


def echo(message, verbose=True):
    if verbose:
        print(message)
