# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

"""
Functions for modeling and removing reflections.

Given that a nominal visibility between antennas 1 & 2
is the correlation of two voltage spectra

    V_12 = v_1 v_2^*

an "auto-reflection" is defined as an additive term in either
v_1 or v_2 that introduces another copy of the voltage as

    v_1 <= v_1 + eps * v_1 = v_1 (1 + eps)

This kind of auto-reflection could be, for example, a cable
reflection or intra-feed reflection.

The reflection itself (eps) can be decomposed into three parameters:
its amplitude (A), delay (tau) and phase offset (phi)

    eps = A * exp(2pi * i * tau * nu + i * phi).

Auto-reflections can be removed from the data within the framework
of standard antenna-based calibration

    V_12^data = V_12^model g_1 g_2^*

The basic idea is that the standard antenna gain, g_1, should be
multiplied by an additional term (1 + eps_1), where eps_1 is the reflection
derived for antenna 1's signal chain, such that an effective gain is formed

    g_1^eff = g_1 (1 + eps)

and the calibration equation becomes

    V_12^data = V_12^model g_1^eff g_2^eff*

----------------

Reflections that couple other voltage signals into v_1 cannot
be described in this formalism (for example, over-the-air aka
inter-feed reflections). We refer to these as "cross-reflections"
and their behavior is to introduce a copy of the autocorrelation
into the cross correlation. If an antenna voltage becomes:

    v_1 <= v_1 + eps_1 v_2

then the cross correlation is proportional to

    V_12 = v_1 v_2^* + eps_1 v_2 v_2^* + ...

where we see that the V_22 = v_2 v_2^* autocorrelation is now
present in the V_12 cross correlation. V_22 is a non-fringing
term but time-dependent term, meaning it will introduce an
offset in the visibility at a specific delay that will have
some time-dependent amplitude. This cannot be corrected
via antenna-based calibration. However, we can in principle
subtract this off at the visibility level by taking
some form of a time-average and subtracting off the bias.
Here, we model the time-dependent offset via PCA, project
onto the autocorrelation as a basis function and subtract
off the bias term.
"""
from __future__ import print_function, division, absolute_import

import numpy as np
import aipy
import os
import copy
import matplotlib.pyplot as plt
from pyuvdata import UVData, UVCal
import pyuvdata.utils as uvutils
from scipy.signal import windows
from sklearn import gaussian_process as gp
from uvtools import dspec
import argparse

from . import io
from . import abscal_funcs
from .datacontainer import DataContainer
from .frf import FRFilter
from . import vis_clean
from .utils import echo


class ReflectionFitter(FRFilter):
    """
    A subclass of FRFilter with added reflection modeling capabilities.
    """
    def model_auto_reflections(self, dly_range, data=None, keys=None, Nphs=500, edgecut_low=0, 
                               edgecut_hi=0, window='none', alpha=0.1, zeropad=0,
                               fthin=10, ref_sig_cut=5.0, overwrite=False, verbose=True):
        """
        Model reflections in (ideally RFI-free) autocorrelation data.

        To CLEAN data of flags see the self.vis_clean() function.
        Recommended to set zeropad to at least as large as Nfreqs.

        Args:
            dly_range : len-2 tuple of delay range [nanoseconds] to search
                within for reflections. Must be either both positive or both negative.
            keys : list of len-3 tuples, keys in data model reflections over. Default
                is to use all auto-correlation keys.
            data : str, data dictionary to find reflections in.
                Default is self.clean_model + self.clean_resid * ~self.flags
            edgecut_low : int, number of bins to consider zero-padded at low-side of the FFT axis,
                such that the windowing function smoothly approaches zero. If ax is 'both',
                can feed as a tuple specifying for 0th and 1st FFT axis.
            edgecut_hi : int, number of bins to consider zero-padded at high-side of the FFT axis,
                such that the windowing function smoothly approaches zero. If ax is 'both',
                can feed as a tuple specifying for 0th and 1st FFT axis.
            window : str, tapering function to apply across freq before FFT
            alpha : float, if taper is Tukey, this is its alpha parameter
            zeropad : int, number of channels to pad band edges with zeros before FFT
            fthin : int, scaling factor to down-select frequency axis when solving for phase
            ref_sig_cut : float, if max reflection significance is not above this, do not record solution
            overwrite : bool, if True, overwrite dictionaries
        """
        # initialize containers
        if hasattr(self, 'epsilon') and not overwrite:
            raise ValueError("reflection dictionaries exist but overwrite is False...")
        self.ref_eps = {}
        self.ref_amp = {}
        self.ref_phs = {}
        self.ref_dly = {}
        self.ref_significance = {}

        # get data
        if data is None:
            data = self.data

        # get keys: only use auto correlations to model reflections
        if keys is None:
            keys = [k for k in data.keys() if k[0] == k[1]]

        # Take FFT of data
        self.fft_data(data, keys=keys, assign='dfft', ax='freq', window=window, alpha=alpha,
                      edgecut_low=edgecut_low, edgecut_hi=edgecut_hi, ifft=False, fftshift=True,
                      ifftshift=False, zeropad=zeropad, verbose=verbose, overwrite=overwrite)

        # iterate over keys
        for k in keys:
            # get gain key
            if dly_range[0] >= 0:
                rkey = (k[0], uvutils.parse_jpolstr(k[2][0]))
            else:
                rkey = (k[1], uvutils.parse_jpolstr(k[2][1]))

            # find reflection
            echo("...Modeling reflections in {}, assigning to {}".format(k, rkey), verbose=verbose)

            # fit for reflection delays and amplitude
            amp, dly, inds, sig = fit_reflection_delay(self.dfft[k], dly_range, self.delays, return_peak=False)

            # make significance cut
            if np.max(sig) < ref_sig_cut:
                continue

            # solve for phase (slowest step)
            phs = fit_reflection_phase(self.dfft[k], dly_range, self.delays, amp, dly, fthin=fthin, Nphs=Nphs, ifft=True, ifftshift=True)

            # anchor phase to 0 Hz
            phs = (phs - 2 * np.pi * dly / 1e9 * (self.freqs[0] - zeropad * self.dnu)) % (2 * np.pi)

            # form epsilon term
            eps = construct_reflection(self.freqs, amp, dly / 1e9, phs, real=False)

            self.ref_eps[rkey] = eps
            self.ref_amp[rkey] = amp
            self.ref_phs[rkey] = phs
            self.ref_dly[rkey] = dly
            self.ref_significance[rkey] = sig

        # form gains
        self.ref_gains = form_gains(self.ref_eps)
        for a in self.ants:
            for p in self.pols:
                k = (a, uvutils.parse_jpolstr(p)) 
                if k not in self.ref_gains:
                    self.ref_gains[k] = np.ones((self.Ntimes, self.Nfreqs), dtype=np.complex)

    def write_auto_reflections(self, output_calfits, input_calfits=None, overwrite=False):
        """
        Write auto reflection gain terms from self.ref_gains.

        Given a filepath to antenna gain calfits file, load the
        calibration, incorporate auto-correlation reflection term from the
        self.ref_gains dictionary and write to file.

        Args:
            output_calfits : str, filepath to write output calfits file
            input_calfits : str, filepath to input calfits file to multiply in with
                reflection gains.
            overwrite : bool, if True, overwrite output file

        Returns:
            uvc : UVCal object with new gains
        """
        # Create reflection gains
        rgains = self.ref_gains
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
                time_array = self.times
            if cal.Nfreqs > self.Nfreqs:
                freq_array = cal.freq_array
            else:
                freq_array = self.freqs
            kwargs = dict([(k, getattr(cal, k)) for k in ['gain_convention', 'x_orientation',
                                                          'telescope_name', 'cal_style']])
        else:
            time_array = self.times
            freq_array = self.freqs
            kwargs = {}

        uvc = io.write_cal(output_calfits, rgains, freq_array, time_array, flags=flags,
                           quality=quals, total_qual=tquals, zero_check=False,
                           overwrite=overwrite, **kwargs)
        return uvc

    def pca_decomp(self, dly_range, dfft=None, flags=None, side='both', keys=None, overwrite=False, verbose=True):
        """
        Create a PCA-based model of the FFT data in dfft.

        Args:
            dly_range : len-2 tuple of positive delays in nanosec
            dfft : DataContainer, holding delay-transformed data. Default is self.dfft
            flags : DataContainer, holding dfft flags (e.g. skip_wgts). Default is None.
            side : str, options=['pos', 'neg', 'both']
                Specifies dly_range as positive delays, negative delays or both.
            keys : list of tuples
                List of datacontainer baseline-pol tuples to create model for.
            overwrite : bool
                If dfft exists, overwrite its values.
        """
        # get dfft and flags
        if dfft is None:
            if not hasattr(self, 'dfft'):
                raise ValueError("self.dfft doesn't exist, see self.fft_data and/or self.vis_clean")
            dfft = self.dfft
        if flags is None:
            flags = DataContainer(dict([(k, np.zeros_like(dfft[k], dtype=np.bool)) for k in dfft]))

        # get keys
        if keys is None:
            keys = dfft.keys()

        # setup dictionaries
        if not hasattr(self, 'umodes'):
            self.umodes = DataContainer({})
        if not hasattr(self, 'vmodes'):
            self.vmodes = DataContainer({})
        if not hasattr(self, 'svals'):
            self.svals = DataContainer({})
        if not hasattr(self, 'uflags'):
            self.uflags = DataContainer({})

        # get selection function
        if side == 'pos':
            select = np.where((self.delays >= dly_range[0]) & (self.delays <= dly_range[1]))[0]
        elif side == 'neg':
            select = np.where((self.delays >= -dly_range[1]) & (self.delays <= -dly_range[0]))[0]
        elif side == 'both':
            select = np.where((np.abs(self.delays) > dly_range[0]) & (np.abs(self.delays) < dly_range[1]))[0]

        # iterate over keys
        for k in keys:
            if k in self.svals and not overwrite:
                echo("{} exists in svals and overwrite == False, skipping...".format(k), verbose=verbose)
                continue
            if k not in dfft:
                echo("{} not found in dfft, skipping...".format(k), verbose=verbose)
                continue

            # perform svd to get principal components
            d = np.zeros_like(dfft[k])
            d[:, select] = dfft[k][:, select]
            u, s, v = _svd_waterfall(d)
            Nmodes = len(s)

            # append to containers
            self.umodes[k] = u[:, :Nmodes]
            self.vmodes[k] = v[:Nmodes, :]
            self.svals[k] = s
            self.uflags[k] = np.min(flags[k], axis=1, keepdims=True)

        # get principal components
        self.form_PCs(keys, overwrite=overwrite)

        # append relevant metadata
        if hasattr(dfft, 'times'):
            self.umodes.times = dfft.times
            self.vmodes.times = dfft.times
            self.svals.times = dfft.times
            self.uflags.times = dfft.times

    def form_PCs(self, keys=None, u=None, v=None, overwrite=False, verbose=True):
        """
        Build principal components.

        Take u and v-modes and form outer product to get principal components
        and insert into self.pcomps

        Args:
            keys : list of tuples
                List of baseline-pol DataContainer tuples to operate on.
            u : DataContainer 
                Holds u-modes to use in forming PCs. Default is self.umodes
            v : DataContainer 
                Holds v-modes to use in forming PCs. Default is self.vmodes
            overwrite : bool
                If True, overwrite output data if it exists.
            verbose : bool
                If True, report feedback to stdout.
        """
        if keys is None:
            keys = self.svals.keys()
        if u is None:
            u = self.umodes
        if v is None:
            v = self.vmodes

        if not hasattr(self, 'pcomps'):
            self.pcomps = DataContainer({})

        for k in keys:
            if k in self.pcomps and not overwrite:
                echo("{} exists in pcomps and overwrite == False, skipping...".format(k), verbose=verbose)
                continue

            self.pcomps[k] = _form_PCs(u[k], v[k])
        if hasattr(u, 'times'):
            self.pcomps.times = u.times

    def build_model(self, keys=None, Nkeep=None, overwrite=False, increment=False, verbose=True):
        """
        Sum principal components to get a model.

        Sum principal components dotted with singular values and add to 
        the pcomp_model.

        Args:
            keys : list of tuples
                List of baseline-pol DataContainer tuples to operate on.
            Nkeep : int
                Number of principal components to keep when forming model.
            overwrite : bool
                If True, overwrite output data if it exists.
            increment : bool
                If key already exists in pcomp_model, add the new model
                to it, rather than overwrite it. This supercedes overwrite
                if both are true.
            verbose : bool
                If True, report feedback to stdout.
        """
        # get keys
        if keys is None:
            keys = self.pcomps.keys()

        # setup containers
        if not hasattr(self, "pcomp_model"):
            self.pcomp_model = DataContainer({})

        # iterate over keys
        for k in keys:
            if k not in self.pcomps:
                echo("{} not in pcomps, skipping...".format(k), verbose=verbose)
                continue
            if k in self.pcomp_model and (not overwrite and not increment):
                echo("{} in pcomp_model and overwrite == increment == False, skipping...".format(k), verbose=verbose)
                continue

            # sum PCs
            model_fft = sum_principal_components(self.svals[k], self.pcomps[k], Nkeep=Nkeep)
            if k not in self.pcomp_model:
                self.pcomp_model[k] = model_fft
            elif k in self.pcomp_model and increment:
                self.pcomp_model[k] += model_fft
            else:
                self.pcomp_model[k] = model_fft

    def subtract_model(self, keys=None, data=None, overwrite=False, verbose=True, inplace=False,
                       ifft=True, ifftshift=True):
        """
        FFT pcomp_model and subtract from data.

        Inserts FFT of pcomp_model into self.pcomp_model_fft and 
        residual of data with self.data_pmodel_resid.

        Args:
            keys : list of tuples
                List of baseline-pol DataContainer tuples to operate on.
            data : datacontainer
                Object to pull data from in forming data-model residual.
                Default is self.data.
            overwrite : bool
                If True, overwrite output data if it exists.
            verbose : bool
                If True, report feedback to stdout.
            ifft : bool
                If True, use ifft to go from delay to freq axis
            ifftshift : bool
                If True, ifftshift delay axis before fft.
        """
        # get keys
        if keys is None:
            keys = self.pcomp_model.keys()

        if not hasattr(self, 'pcomp_model_fft'):
            self.pcomp_model_fft = DataContainer({})
        if not hasattr(self, 'data_pmodel_resid'):
            self.data_pmodel_resid = DataContainer({})

        # get data
        if data is None:
            data = self.data

        # inplace
        if not inplace:
            data = copy.deepcopy(data)

        # iterate over keys
        for k in keys:
            if k in self.pcomp_model_fft and not overwrite:
                echo("{} in pcomp_model_fft and overwrite==False, skipping...".format(k), verbose=verbose)
                continue

            # get fft of model
            model = self.pcomp_model[k]

            # ifftshift
            if ifftshift:
                model = np.fft.ifftshift(model, axes=-1)

            # ifft to get to data space
            if ifft:
                model_fft = np.fft.ifft(model, axis=-1)
            else:
                model_fft = np.fft.fft(model, axis=-1)

            # subtract from data
            self.pcomp_model_fft[k] = model_fft
            self.data_pmodel_resid[k] = data[k] - model_fft

    def interp_u(self, u, times, full_times=None, uflags=None, keys=None, overwrite=False, verbose=True,
                 mode='gpr', gp_len=600, gp_nl=0.1, optimizer=None):
        """
        Interpolate u modes along time, inserts into self.umode_interp

        Args:
            u : DataContainer
                u-mode container to interpolate
            times : 1D array
                Holds time_array of input u modes.
            full_times : 1D array
                time_array to interpolate onto. Default is times.
            uflags : DataContainer
                Object to pull target u flags from. Default is None.
            keys : list of tuples
                List of baseline-pol DataContainer tuples to operate on.
            overwrite : bool
                If True, overwrite output data if it exists.
            verbose : bool
                If True, report feedback to stdout.
            mode : str
                Interpolation mode. Options=['gpr']
            gp_len : length-scale of GPR in units of times
            gp_nl : GPR noise-level in units of input u.
        """
        if not hasattr(self, 'umode_interp'):
            self.umode_interp = DataContainer({})

        if uflags is None:
            uflags = self.uflags

        if full_times is None:
            full_times = times

        # get keys
        if keys is None:
            keys = u.keys()

        # setup X predict
        Xmean = np.median(times)
        Xpredict = full_times[:, None] - Xmean

        # iterate over keys
        for k in keys:
            # check overwrite
            if k in self.umode_interp and not overwrite:
                echo("{} in umode_interp and overwrite == False, skipping...".format(k), verbose=verbose)
                continue

            if mode == 'gpr':
                # setup GP kernel
                kernel = 1**2 * gp.kernels.RBF(length_scale=gp_len) + gp.kernels.WhiteKernel(noise_level=gp_nl)
                GP = gp.GaussianProcessRegressor(kernel=kernel, optimizer=optimizer, normalize_y=True)

                # setup regression data: get unflagged data
                select = ~np.max(uflags[k], axis=1)
                X = times[select, None] - Xmean
                y = u[k][select, :]

                # fit gp and predict
                GP.fit(X, y.real)
                ypredict_real = GP.predict(Xpredict)
                GP.fit(X, y.imag)
                ypredict_imag = GP.predict(Xpredict)

                # append
                self.umode_interp[k] = ypredict_real.astype(np.complex) + 1j * ypredict_imag

            else:
                raise ValueError("didn't recognize interp mode {}".format(mode))

    def project_autos_onto_u(self, keys, auto_keys, u=None, flags=None, index=0, auto_delay=0,
                             overwrite=False, verbose=True):
        """
        Project autocorr onto u modes.

        Projects the time dependent dfft of an autocorrelation
        at a specified delay onto the specified u-mode, replacing
        the u-mode with the projected autocorrelation. Inserts
        results into self.umode_interp

        Args:
            keys : list of tuples
                List of baseline-pol DataContainer tuples to operate on.
            auto_keys : list of tuples
                List of autocorr-pol tuples matching input keys in length
                to pull auto-correlation data from in self.dfft.
                Optionally, each auto_key can be itself a list of
                keys that will be used as separate basis functions
                to project onto the u modes. Example: for a given
                cross-corr key, you can provide both auto-corr keys.
            u : DataContainer
                Object to pull u modes from. Default is self.umodes
            flags : DataContainer
                Object to pull data flags from.
            index : int
                Index of the u-mode to project auto-correlation onto.
                All other u-mode indices are copied as-is into umode_interp
                This should almost always be set to zero.
            auto_delay : float
                Delay in nanosec of autocorrelation to project onto the u-mode.
                This should almost always be set to zero.
            overwrite : bool
                If True, overwrite output data if it exists.
            verbose : bool
                If True, report feedback to stdout.
        """
        # type check
        assert len(keys) == len(auto_keys), "len(keys) must equal len(auto_keys)"

        # get u and flags
        if u is None:
            u = self.umodes
        if flags is None:
            flags = DataContainer(dict([(k, np.zeros_like(self.pcomps[k][0], dtype=np.bool)) for k in u]))

        # get dly index
        select = np.argmin(np.abs(self.delays - auto_delay))

        if not hasattr(self, 'umode_interp'):
            self.umode_interp = DataContainer({})
            self.umode_interp.times = u.times
        if not hasattr(self, 'umode_interp_flags'):
            self.umode_interp_flags = DataContainer({})
            self.umode_interp_flags.times = u.times

        # iterate over keys
        for k, ak in zip(keys, auto_keys):
            # check overwrite
            if k in self.umode_interp and not overwrite:
                echo("{} exists in umode_interp and overwrite == False, skipping...".format(k), verbose=verbose)
                continue

            if k not in u:
                echo("{} not in u container, skipping...".format(k, ak), verbose=verbose)
                continue

            # get u-mode
            _u = u[k][:, index]

            # get autocorr
            if isinstance(ak, list):
                if not np.all([_ak in self.dfft for _ak in ak]):
                    echo("{} not in self.dfft, skipping...".format(ak), verbose=verbose)
                    continue
                A = np.array([self.dfft[_ak][:, select] for _ak in ak]).T
                af = np.sum([flags[_ak] for _ak in ak], axis=0)
            else:
                if ak not in self.dfft:
                    echo("{} not in self.dfft, skipping...".format(ak), verbose=verbose)
                    continue
                A = np.array([self.dfft[ak][:, select]]).T
                af = flags[ak]

            # form least squares estimate
            f = np.min(flags[k] + af, axis=1)
            W = np.eye(len(_u)) * (~f).astype(np.float)
            xhat = np.asarray(np.linalg.pinv(A.T.dot(W).dot(A)).dot(A.T.dot(W).dot(_u.real)), dtype=np.complex) \
                + 1j * np.linalg.pinv(A.T.dot(W).dot(A)).dot(A.T.dot(W).dot(_u.imag))
            proj_u = A.dot(xhat)

            self.umode_interp[k] = u[k].copy()
            self.umode_interp[k][:, index] = proj_u
            self.umode_interp_flags[k] = f


def form_gains(epsilon):
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
        tau : N-D array of reflection delays [sec]
        phs : N-D array of phase offsets [radians]
        real : bool, if True return as real-valued. Else
            return as complex-valued.

    Returns:
        eps : complex (or real) valued reflection across
            input frequencies.
    """
    # make reflection
    eps = amp * np.exp(2j * np.pi * tau * freqs + 1j * phs)
    if real:
        eps = eps.real

    return eps


def fit_reflection_delay(dfft, dly_range, dlys, return_peak=False):
    """
    Take FFT'd data and fit for reflection delay and amplitude.

    Args:
        dfft : complex 2D array with shape (Ntimes, Ndlys)
        dly_range : len-2 tuple specifying range of delays [nanosec]
            to look for reflection within.
        dlys : 1D array of data delays [nanosec]
        return_peak : bool, if True, just return peak delay and amplitude

    Returns:
        if return_peak:
            ref_peaks : peak amplitude
            ref_dlys : peak delay [nanosec]
        else:
            ref_amps : reflection amplitudes (Ntimes, 1)
            ref_dlys : reflection delays [nanosec] (Ntimes, 1)
            ref_inds : reflection peak indicies of input dfft array
            ref_sig : reflection significance, SNR of peak relative to neighbors
    """
    # get fft
    assert dfft.ndim == 2, "input dfft must be 2-dimensional with shape [Ntimes, Ndlys]"
    Ntimes, Ndlys = dfft.shape

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

    if return_peak:
        return ref_peaks, ref_dlys

    # get reflection significance, defined as ref_peak / median_abs
    ref_significance = ref_peaks / np.median(abs_dfft_selection, axis=1, keepdims=True)

    # get peak value at tau near zero
    peak, _ = fit_reflection_delay(dfft, (-np.abs(dly_range).min(), np.abs(dly_range).min()),  dlys, return_peak=True)
    peak[np.isclose(peak, 0.0)] = np.inf

    # get reflection amplitude
    ref_amps = ref_peaks / peak

    return ref_amps, ref_dlys, ref_dly_inds, ref_significance


def fit_reflection_phase(dfft, dly_range, dlys, ref_amps, ref_dlys, fthin=1, Nphs=500,
                         ifft=False, ifftshift=True):
    """
    Fit for reflection phases.

    Take FFT'd data and reflection amp and delay and solve for phase.

    Args:
        dfft : complex 2D array with shape (Ntimes, Ndlys)
        dly_range : len-2 tuple specifying range of delays [nanosec]
            to look for reflection within.
        dlys : 1D array of data delays [nanosec]
        ref_amps : 2D array (Ntimes, 1) of reflection amplitudes
        ref_dlys : 2D array (Ntimes, 1) of reflection delays [nanosec]
        fthin : integer thinning parameter along freq axis when solving for reflection phase
        Nphs : int, number of phase bins between 0 and 2pi to use in solving for phase.
        ifft : bool, if True, use ifft to go from delay to freq axis
        ifftshift : bool, if True, ifftshift delay axis before fft to freq in solving for phase.
        return_peak : bool, if True, just return peak delay and amplitude

    Returns:
        ref_phs : reflection phases (Ntimes, 1)
    """
    # get fft
    assert dfft.ndim == 2, "input dfft must be 2-dimensional with shape [Ntimes, Ndlys]"
    Ntimes, Ndlys = dfft.shape

    # get reflection phase by fitting cosines to filtered data thinned along frequency axis
    filt = np.zeros_like(dfft)
    select = np.where((dlys > dly_range[0]) & (dlys < dly_range[1]))[0]
    filt[:, select] = dfft[:, select]
    select = np.where((dlys > -dly_range[1]) & (dlys < -dly_range[0]))[0]
    filt[:, select] = dfft[:, select]
    if ifftshift:
        filt = np.fft.ifftshift(filt, axes=-1)
    if ifft:
        filt = np.fft.ifft(filt, axis=-1)
    else:
        filt = np.fft.fft(filt, axis=-1)
    freqs = np.fft.fftfreq(Ndlys, np.median(np.diff(dlys))/1e9)
    freqs = np.linspace(0, freqs.max() - freqs.min(), Ndlys, endpoint=True)
    phases = np.linspace(0, 2 * np.pi, Nphs, endpoint=False)
    cosines = np.array([construct_reflection(freqs[::fthin], ref_amps, ref_dlys / 1e9, p) for p in phases])
    residuals = np.sum((filt[None, :, ::fthin].real - cosines.real)**2, axis=-1)
    ref_phs = phases[np.argmin(residuals, axis=0)][:, None] % (2 * np.pi)

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

    return ref_phs


def _svd_waterfall(data):
    """
    Take a singular-value decomposition of a 2D
    visibility in either frequency or delay space,
    with shape (Ntimes, Nfreqs) or (Ntimes, Ndelays).

    Parameters
    ----------
    data : 2d complex ndarray

    Returns
    -------
    PCs : 3d ndarray
        Holds principal components of the data
        with shape=(Npcs, Ntimes, Nfreqs)

    svals : 1d ndarray
        Holds the singluar (or eigen) values of the
        principal components.
    """
    # get singular values and eigenbases
    u, svals, v = np.linalg.svd(data)

    return u, svals, v


def _form_PCs(u, v):
    Nu = u.shape[1]
    Nv = v.shape[0]
    Npc = min([Nu, Nv])

    # calculate outer products to get principal components
    PCs = np.einsum("ij,jk->jik", u, v)

    return PCs


def sum_principal_components(svals, PCs, Nkeep=None):
    """
    Dot singular values into principal components,
    keeping a specified number of PCs. svals should
    be rank ordered from highest to lowest.

    Parameters
    ----------
    svals : 1D ndarray
        singular values, with length Nvals

    PCs : 3D ndarray
        principal components with shape=(Nvals, Ntimes, Nfreqs)

    Nkeep : integer
        Number of PCs to keep in summation. Default is all.

    Returns
    -------
    recon : 2D ndarray
        Reconstruction of initial data using
        principal components, shape=(Ntimes, Nfreqs)
    """
    assert len(svals) == len(PCs), "svals must have same len as PCs"

    # assign Nkeep if None
    if Nkeep is None:
        Nkeep = len(svals)

    # get reconstruction
    recon = np.einsum("i,i...->...", svals[:Nkeep], PCs[:Nkeep])

    return recon


def auto_reflection_argparser():
    a = argparse.ArgumentParser(description='Model auto (e.g. cable) reflections')
    a.add_argument("datafile", nargs='*', type=str, help="Data file paths to run auto reflection modeling on")
    a.add_argument("--outfname", type=str, help="Full path to the output .calfits file")
    a.add_argument("--filetype", type=str, default='uvh5', help="Filetype of datafile")
    a.add_argument("--overwrite", default=False, action='store_true', help="Overwrite output file if it already exists")
    a.add_argument("--input_cal", type=str, default=None, help="Path to input .calfits to apply to data before modeling")
    a.add_argument("--dly_range", type=float, nargs='*', help='lower and upper delay [nanosec] within which to search for reflections.')
    a.add_argument("--ants", default=None, type=int, nargs='*', help="List of antenna numbers to operate on.")
    a.add_argument("--pols", default=None, type=str, nargs='*', help="List of polarization strings to operate on.")
    a.add_argument("--zeropad", default=0, type=int, help="Number of channels to zeropad *both* sides of band in auto modeling process.")
    a.add_argument("--window", default='None', type=str, help="FFT window for CLEAN")
    a.add_argument("--alpha", default=0.2, type=float, help="Alpha parameter if window is tukey")
    a.add_argument("--tol", default=1e-6, type=float, help="CLEAN tolerance")
    a.add_argument("--gain", default=1e-1, type=float, help="CLEAN gain")
    a.add_argument("--maxiter", default=100, type=int, help="CLEAN maximum Niter")
    a.add_argument("--skip_wgt", default=0.1, type=float, help="Skip integration if heavily flagged, see hera_cal.delay_filter for details")
    a.add_argument("--edgecut_low", default=0, type=int, help="Number of channels to consider as zeropad on low edge of band")
    a.add_argument("--edgecut_hi", default=0, type=int, help="Number of channels to consider as zeropad on high edge of band")
    a.add_argument("--horizon", default=1.0, type=float, help="Baseline horizon coefficient. See hera_cal.delay_filter for details")
    a.add_argument("--standoff", default=0.0, type=float, help="Baseline horizon standoff. See hera_cal.delay_filter for details")
    a.add_argument("--min_dly", default=0.0, type=float, help="Minimum CLEAN delay horizon. See hera_cal.delay_filter for details")
    a.add_argument("--Nphs", default=500, type=int, help="Number of phase points to evaluate from 0--2pi in solving for phase")
    a.add_argument("--fthin", default=1, type=int, help="Coefficient to thin frequency axis by when solving for phase")
    return a

