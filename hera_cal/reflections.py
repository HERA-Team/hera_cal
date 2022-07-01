# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"""
Functions for modeling and removing reflections.

Given that a nominal visibility between antennas 1 & 2
is the correlation of two voltage spectra

    V_12 = v_1 v_2^*

a "signal chain reflection" is defined as an additive term in either
v_1 or v_2 that introduces another copy of the voltage as

    v_1 <= v_1 + eps * v_1 = v_1 (1 + eps)

This could be, for example, a cable
reflection or dish-to-feed reflection.

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

----------------

Reflections that couple other voltage signals into v_1 cannot
be described in this formalism (for example, over-the-air aka
feed-to-feed reflections). We refer to these as "voltage cross couplings"
and their behavior is to introduce a copy of the autocorrelation
into the cross correlation. If an antenna voltage becomes:

    v_1 <= v_1 + eps_1 v_2

then the cross correlation is proportional to

    V_12 = v_1 v_2^* + eps_1 v_2 v_2^* + ...

where we see that the V_22 = v_2 v_2^* autocorrelation is now
present in the V_12 cross correlation. V_22 is a non-fringing
but still time-dependent term, meaning it will introduce an
offset in the visibility at a specific delay that will have
some time-dependent amplitude. This cannot be corrected
via antenna-based calibration. However, we can in principle
subtract this off at the visibility level by taking
some form of a time-average and subtracting off the bias.
The code here models the time and delay dependent behavior
through a combination of SVD and fringe-rate filtering.
"""
import numpy as np
import os
import copy
from scipy.optimize import minimize
from scipy import sparse
from sklearn import gaussian_process as gp
from hera_filters import dspec

import argparse
import ast
from astropy import constants

from . import io
from . import utils
from .abscal import merge_gains
from .apply_cal import calibrate_in_place
from .datacontainer import DataContainer
from .frf import FRFilter
from . import vis_clean
from .utils import echo, interp_peak, split_pol, split_bl, gp_interp1d, comply_pol


class ReflectionFitter(FRFilter):
    """
    A subclass of FRFilter & VisClean with added reflection
    modeling capabilities. Instantiation inherits from the VisClean class.

    Possible products from this class include
        self.ref_eps : dictionary, see model_auto_reflections()
        self.ref_amp : dictionary, see model_auto_reflections()
        self.ref_dly : dictionary, see model_auto_reflections()
        self.ref_phs : dictionary, see model_auto_reflections()
        self.ref_significance : dictionary, see model_auto_reflections()
        self.ref_gains : dictionary, see model_auto_reflections()
        self.ref_flags : dictionary, see model_auto_reflections()

        self.umodes : DataContainer, see sv_decomp()
        self.vmodes : DataContainer, see sv_decomp()
        self.svals : DataContainer, see sv_decomp()
        self.umode_interp : DataContainer, see interp_u()
        self.pcomp_model : DataContainer, see build_pc_model()
        self.pcomp_model_fft : DataContainer, see subtract_model()
        self.data_pcmodel_resid : DataContainer, see subtract_model()
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize ReflectionFitter, see vis_clean.VisClean for kwargs.
        """
        super(ReflectionFitter, self).__init__(*args, **kwargs)

        # init empty datacontainers
        self.umodes = DataContainer({})
        self.uflags = DataContainer({})
        self.vmodes = DataContainer({})
        self.svals = DataContainer({})
        self.umode_interp = DataContainer({})
        self.pcomp_model = DataContainer({})
        self.pcomp_model_fft = DataContainer({})
        self.data_pcmodel_resid = DataContainer({})

        # init empty dictionaries
        self.ref_eps = {}
        self.ref_amp = {}
        self.ref_phs = {}
        self.ref_dly = {}
        self.ref_flags = {}
        self.ref_significance = {}

    def clear(self, exclude=[]):
        """
        Clear all DataContainers, and
        all reflection dictionaries beginning with ref_*

        Args:
            exclude : list of DataContainer and dictionary names
                attached to self to exclude from purge.
        """
        keys = list(self.__dict__.keys())
        for key in keys:
            if key in exclude:
                continue
            obj = getattr(self, key)
            if isinstance(getattr(self, key), DataContainer):
                setattr(self, key, DataContainer({}))
        self._clear_ref()

    def _clear_ref(self):
        """
        Clear all reflection dictionaries
        """
        keys = list(self.__dict__.keys())
        for key in keys:
            if key[:4] == "ref_":
                setattr(self, key, {})

    def model_auto_reflections(self, clean_resid, dly_range, clean_flags=None, clean_data=None,
                               keys=None, edgecut_low=0, edgecut_hi=0, window='none',
                               alpha=0.1, zeropad=0, fthin=10, Nphs=300, ref_sig_cut=2.0,
                               reject_edges=True, verbose=True):
        """
        Model reflections in (ideally RFI-free) autocorrelation data.

        Recommended to set zeropad to at least as large as Nfreqs/2.
        This function attaches the result to self. See results below.
        To CLEAN data of RFI gaps see the self.vis_clean() function.

        Args:
            clean_resid : DataContainer
                Autocorrelation data (in frequency space) to find reflections in.
                Ideally this is the CLEAN residual, having cleaned out structure
                at delays less than the min(abs(dly_range)). If this is the
                CLEAN residual, you must feed the unfiltered CLEAN data as "clean_data"
                to properly estimate the reflection amplitude.
            dly_range : len-2 tuple
                delay range [nanoseconds] to search within for reflections.
                Must be either both positive or both negative.
            clean_flags : DataContainer
                Flags of the (CLEANed) data. Default is None.
            clean_data : DataContainer
                If clean_resid is fed as the CLEAN residual (and not CLEAN data),
                this should be the unfiltered CLEAN data output.
            keys : list of len-3 tuples
                ant-pair-pols to operate on. Default is to use all auto-corr & auto-pol keys.
            edgecut_low : int
                Nbins to flag and exclude from windowing function on low-side of the band
            edgecut_hi : int
                Nbins to flag and exclude from windowing function on low-side of the band
            window : str
                windwoing function to apply across freq before FFT. See dspec.gen_window for options
            alpha : float
                if taper is Tukey, this is its alpha parameter
            zeropad : int
                number of channels to pad band edges with zeros before FFT
            fthin : int
                scaling factor to down-select frequency axis when solving for phase
            Nphs : int
                Number of samples from 0 - 2pi to estimate reflection phase at.
            ref_sig_cut : float
                if max reflection significance is not above this, do not record solution
            reject_edges : bool
                If True, reject peak solutions at delay edges

        Result:
            self.ref_eps : dict
                reflection epsilon term with (ant, pol) keys and ndarray values
            self.ref_amp : dict
                reflection amplitude with (ant, pol) keys and ndarray values
            self.ref_phs : dict
                reflection phase [radians] with (ant, pol) keys and ndarray values
            self.ref_dly : dict
                reflection delay [nanosec] with (ant, pol) keys and ndarray values
            self.ref_flags : dict
                reflection flags [bool] with (ant, pol) keys and ndarray values
            self.ref_significance : dict
                reflection significance with (ant, pol) keys and ndarray values.
                Significance is max|Vfft| / median|Vfft| near the reflection peak
            self.ref_gains : dict
                reflection gains spanning all keys in clean_resid. keys that
                weren't included in input or didn't have a significant reflection have a gain of 1.0
        """
        # get flags
        if clean_flags is None:
            clean_flags = DataContainer(dict([(k, np.zeros_like(clean_resid[k], dtype=bool)) for k in clean_resid]))

        # get keys: only use auto correlations and auto pols to model reflections
        if keys is None:
            keys = [k for k in clean_resid.keys() if k[0] == k[1] and k[2][0] == k[2][1]]

        # iterate over keys
        for k in keys:
            # get gain key
            if dly_range[0] >= 0:
                rkey = split_bl(k)[0]
            else:
                rkey = split_bl(k)[1]

            # model reflection
            echo("...Modeling reflections in {}, assigning to {}".format(k, rkey), verbose=verbose)
            if clean_data is not None:
                cd = clean_data[k]
            else:
                cd = None
            amp, dly, phs, sig = fit_reflection_params(clean_resid[k], dly_range, self.freqs, clean_flags=clean_flags[k],
                                                       clean_data=cd, window=window, alpha=alpha, edgecut_low=edgecut_low,
                                                       edgecut_hi=edgecut_hi, zeropad=zeropad, ref_sig_cut=ref_sig_cut,
                                                       fthin=fthin, Nphs=Nphs, reject_edges=reject_edges)

            # check for amplitudes greater than 1.0: flag them
            bad_sols = amp > 1.0
            amp[bad_sols] = 0.0

            # form epsilon term
            eps = construct_reflection(self.freqs, amp, dly / 1e9, phs, real=False)

            self.ref_eps[rkey] = eps
            self.ref_amp[rkey] = amp
            self.ref_phs[rkey] = phs
            self.ref_dly[rkey] = dly
            self.ref_significance[rkey] = sig
            self.ref_flags[rkey] = np.all(clean_flags[k], axis=1, keepdims=True) + (sig < ref_sig_cut) + bad_sols

        # form gains
        self.ref_gains = form_gains(self.ref_eps)

        # if ref_gains not empty, fill in missing antenna and polarizations with unity gains
        autopols = [p for p in self.pols if p[0] == p[1]]
        if len(self.ref_gains) > 0:
            antpol = split_bl(keys[0])[0]
            for a in self.data_ants:
                for p in autopols:
                    k = (a, split_pol(p)[0])
                    if k not in self.ref_gains:
                        self.ref_gains[k] = np.ones_like(self.ref_gains[antpol], dtype=complex)
                    if k not in self.ref_flags:
                        self.ref_flags[k] = np.zeros_like(self.ref_flags[antpol], dtype=bool)

    def refine_auto_reflections(self, clean_data, dly_range, ref_amp, ref_dly, ref_phs, ref_flags=None,
                                keys=None, clean_flags=None, clean_model=None, fix_amp=False,
                                fix_dly=False, fix_phs=False,
                                edgecut_low=0, edgecut_hi=0, window='none', alpha=0.1, zeropad=0,
                                skip_frac=0.9, maxiter=50, method='BFGS', tol=1e-2, verbose=True):
        """
        Refine reflection parameters via some minimization technique.

        Iteratively perturbs reflection parameters in ref_* dictionaries,
        and applies to input clean_data until reflection bump amplitude
        inside dly_range is minimimzed to within a tolerance.

        Args:
            clean_data : 2D ndarray of shape (Ntimes, Nfreqs)
                CLEANed auto-correlation visibility data
            dly_range : float or len-2 tuple of floats
                Additive delay [ns] offset (-, +) from ref_dly to create range of
                delays used in objective minimization
            ref_amp : dictionary
                Initial guess for reflection amplitude
            ref_dly : dictionary
                Initial guess for reflection delay [ns]
            ref_phs : dictionary
                Initial guess for reflection phase [radian]
            ref_flags : dictionary
                Flags for reflection fits.
            keys : list
                List of ant-pair-pol tuples in clean_data to iterate over
            clean_model : 2D ndarray of shape (Ntimes, Nfreqs)
                CLEAN model with CLEAN boundary out to at most min(|dly_range|)
                If reflection is well-isolated from foreground power, this is not necessary.
            clean_flags : 2D ndarray boolean of shape (Ntimes, Nfreqs)
                CLEAN flags for FFT
            fix_amp : bool
                If True, fix amplitude solution at ref_amp
            fix_dly : bool
                If True, fix delay solution at ref_dly
            fix_phs : bool
                If True, fix delay solution at ref_phs
            method : str
                Optimization algorithm. See scipy.optimize.minimize for options
            skip_frac : float in range [0, 1]
                fraction of flagged channels (excluding edge flags) above which skip the integration
            tol : float
                Optimization stopping tolerance
            maxiter : int
                Optimization max iterations

        Returns:
            ref_amp : dictionary
                Reflection amplitude
            ref_dly : dictionary
                Reflection delay [ns]
            ref_phs : dictionary
                Reflection phase [radian]
            ref_info : dictionary
                Optimization success [bool]
            ref_eps : dictionary
                Reflection coefficient
            ref_gains : dictionary
                Reflection gains
        """
        # check inputs
        if clean_flags is None:
            clean_flags = DataContainer(dict([(k, np.zeros_like(clean_data[k], dtype=bool)) for k in clean_data]))
        if clean_model is None:
            clean_model = DataContainer(dict([(k, np.zeros_like(clean_data[k])) for k in clean_data]))

        # get keys: only use auto correlations and auto pols to model reflections
        if keys is None:
            keys = [k for k in clean_data.keys() if k[0] == k[1] and k[2][0] == k[2][1]]

        # setup reflection dictionaries
        out_ref_eps = {}
        out_ref_amp = {}
        out_ref_dly = {}
        out_ref_phs = {}
        out_ref_info = {}

        # iterate over keys
        for k in keys:
            # get gain key
            rkey = (k[0], comply_pol(k[2][0]))
            if rkey not in ref_amp:
                rkey = (k[1], comply_pol(k[2][1]))

            # Ensure they exist in reflection dictionaries
            if rkey not in ref_amp or rkey not in ref_dly or rkey not in ref_phs:
                echo("...{} doesn't exist in ref_* dictionaries, skipping".format(rkey), verbose=verbose)
                continue

            # get reflection flags
            if ref_flags is None:
                rflags = False
            else:
                rflags = ref_flags[rkey]

            # run optimization
            echo("...Optimizing reflections in {}, assigning to {}".format(k, rkey), verbose=verbose)
            amp = ref_amp[rkey]
            dly = ref_dly[rkey]
            phs = ref_phs[rkey]
            (opt_amp, opt_dly, opt_phs,
             opt_info) = reflection_param_minimization(clean_data[k], dly_range, self.freqs, amp, dly, phs, fix_amp=fix_amp, fix_dly=fix_dly,
                                                       fix_phs=fix_phs, clean_model=clean_model[k], clean_flags=clean_flags[k] + rflags,
                                                       method=method, tol=tol, maxiter=maxiter, window=window, alpha=alpha,
                                                       edgecut_hi=edgecut_hi, edgecut_low=edgecut_low, zeropad=zeropad)

            amp = np.reshape(opt_amp, amp.shape)
            dly = np.reshape(opt_dly, dly.shape)
            phs = np.reshape(opt_phs, phs.shape)

            # form epsilon term
            eps = construct_reflection(self.freqs, amp, dly / 1e9, phs, real=False)

            out_ref_eps[rkey] = eps
            out_ref_amp[rkey] = amp
            out_ref_phs[rkey] = phs
            out_ref_dly[rkey] = dly
            out_ref_info[rkey] = opt_info

        # form gains
        out_ref_gains = form_gains(out_ref_eps)

        return out_ref_amp, out_ref_dly, out_ref_phs, out_ref_info, out_ref_eps, out_ref_gains

    def write_auto_reflections(self, output_calfits, input_calfits=None, time_array=None,
                               freq_array=None, overwrite=False, write_npz=False,
                               write_calfits=True, add_to_history='', verbose=True):
        """
        Write reflection gain terms from self.ref_gains.

        If input_calfits is provided, merge its gains with the reflection
        gains before writing to disk. Flags, quality and total_quality
        arrays are all empty unless input_calfits is provided, in which
        case its arrays are inherited.

        Optionally, take the values in self.ref_amp, self.ref_phs and self.ref_dly and
        write their values to an NPZ file with the same path name as output_calfits.

        Args:
            output_calfits : str, filepath to write output calfits file to
            input_calfits : str, filepath to input calfits file to multiply in with
                reflection gains.
            time_array : ndarray, Julian Date of times in ref_gains. Default is self.times
            freq_array : ndarray, Frequency array [Hz] of ref_gains. Default is self.freqs
            overwrite : bool, if True, overwrite output file
            write_npz : bool, if True, write an NPZ file holding reflection
                params with the same pathname as output_calfits
            write_calfits : bool, if False, skip writing
            add_to_history: string to add to history of output calfits file
            verbose : bool, report feedback to stdout

        Returns:
            uvc : UVCal object with new gains (or None if write_calfits is False)
        """
        # Create reflection gains
        rgains, rflags = self.ref_gains, self.ref_flags
        quals, tquals = None, None

        # get time and freq array
        if time_array is None:
            time_array = self.times
        Ntimes = len(time_array)
        if freq_array is None:
            freq_array = self.freqs
        Nfreqs = len(freq_array)

        # write npz
        if write_npz:
            output_npz = os.path.splitext(output_calfits)[0] + '.npz'
            if not os.path.exists(output_npz) or overwrite:
                echo("...writing {}".format(output_npz), verbose=verbose)
                np.savez(output_npz, delay=self.ref_dly, phase=self.ref_phs, amp=self.ref_amp,
                         significance=self.ref_significance, times=time_array, freqs=freq_array,
                         lsts=self.lsts, antpos=self.antpos, flags=rflags,
                         history=utils.history_string(add_to_history))

        # return None if we don't want to write a calfits file
        if not write_calfits:
            return None

        if input_calfits is not None:
            # Load calfits
            cal = io.HERACal(input_calfits)
            gains, flags, quals, tquals = cal.read()

            # Merge gains and flags
            rgains = merge_gains([gains, rgains])
            rflags = merge_gains([flags, rflags])

            # resolve possible broadcasting across time and freq
            if cal.Ntimes > Ntimes:
                time_array = cal.time_array
            if cal.Nfreqs > Nfreqs:
                freq_array = cal.freq_array
            kwargs = dict([(k, getattr(cal, k)) for k in ['gain_convention', 'x_orientation',
                                                          'telescope_name', 'cal_style']])
            add_to_history += "\nMerged-in calibration {}".format(input_calfits)
        else:
            kwargs = {'x_orientation': self.hd.x_orientation}

        # write calfits
        antnums2antnames = dict(zip(self.hd.antenna_numbers, self.hd.antenna_names))
        echo("...writing {}".format(output_calfits), verbose=verbose)
        uvc = io.write_cal(output_calfits, rgains, freq_array, time_array, flags=rflags,
                           quality=quals, total_qual=tquals, zero_check=False,
                           overwrite=overwrite, history=utils.history_string(add_to_history),
                           antnums2antnames=antnums2antnames, **kwargs)

        return uvc

    def svd_weights(self, dfft, delays, horizon=1.0, standoff=0.0, min_dly=None, max_dly=None, side='both'):
        """
        Form wgts windowing DataContainer for sv_decomp.

        Args:
            dfft : DataContainer, holding visibilities in time & delay space
            delays : ndarray, 1D array of dfft delays [ns]
            horizon : float, coefficient of baseline geometric horizon
                to set as *lower* delay boundary of window (opposite of CLEANing convention)
            standoff : float, buffer [nanosec] added to baseline horizon
                for lower delay boundary of window
            min_dly : float, minimum |delay| of window
            max_dly : float, maximum |delay| of window
            side : str, options=['pos', 'neg', 'both']
                Specifies window as spanning only positive delays, only negative delays or both.

        Returns:
            wgts : DataContainer, holding sv_decomp weights
        """
        wgts = DataContainer({})
        for k in dfft:
            w = np.ones_like(dfft[k], dtype=float)
            # get horizon
            h = np.linalg.norm(self.antpos[k[1]] - self.antpos[k[0]]) / constants.c.value * 1e9 * horizon + standoff
            if min_dly is not None:
                h = np.max([h, min_dly])
            w[:, np.abs(delays) < h] = 0.0
            if max_dly is not None:
                w[:, np.abs(delays) > max_dly] = 0.0
            if side == 'neg':
                w[:, delays > 0.0] = 0.0
            elif side == 'pos':
                w[:, delays < 0.0] = 0.0
            wgts[k] = w

        return wgts

    def sv_decomp(self, dfft, wgts=None, flags=None, keys=None, Nkeep=None,
                  overwrite=False, sparse_svd=True, verbose=True):
        """
        Create a SVD-based model of the FFT data in dfft.

        This is done via Singular Value Decomposition on the input delay waterfall data
        times the wgts and stores results in self.umodes, self.vmodes, self.svals, and self.uflags.

        Args:
            dfft : DataContainer, holding delay-transformed data.
            wgts : DataContainer, holding weights to multiply with dfft before taking SVD
                See self.svd_weights()
            flags : DataContainer, holding dfft flags (e.g. skip_wgts). Default is None.
            keys : list of tuples
                List of datacontainer baseline-pol tuples to create model for.
            Nkeep : int, number of modes to keep out of total Ntimes number of modes.
                Default is keep all modes.
            overwrite : bool
                If dfft exists, overwrite its values.
            sparse_svd : bool
                If True, use scipy.sparse.linalg.svds, else use scipy.linalg.svd

        Result:
            self.umodes : DataContainer, SVD time-modes, ant-pair-pol keys, 2D ndarray values
            self.vmodes : DataContainer, SVD delay-modes, ant-pair-pol keys, 2D ndarray values
            self.svals : DataContainer, SVD time-modes, ant-pair-pol keys, 1D ndarray values
            self.uflags : DataContainer, flags for umodes, ant-pair-pol keys, 2D ndarray values
        """
        # get flags
        if flags is None:
            flags = DataContainer(dict([(k, np.zeros_like(dfft[k], dtype=bool)) for k in dfft]))

        # get weights
        if wgts is None:
            wgts = DataContainer(dict([(k, np.ones_like(dfft[k], dtype=float)) for k in dfft]))

        # get keys
        if keys is None:
            keys = list(dfft.keys())

        # iterate over keys
        for k in keys:
            if k in self.svals and not overwrite:
                echo("{} exists in svals and overwrite == False, skipping...".format(k), verbose=verbose)
                continue
            if k not in dfft:
                echo("{} not found in dfft, skipping...".format(k), verbose=verbose)
                continue

            # perform svd to get principal components
            # full_matrices = False truncates u or v depending on which has more modes
            if sparse_svd:
                Nk = Nkeep
                if Nk is None:
                    Nk = min(dfft[k].shape) - 2
                u, svals, v = sparse.linalg.svds(dfft[k] * wgts[k], k=Nk, which='LM')
                # some numpy versions flip SV ordering here: make sure its high-to-low
                if svals[-1] > svals[0]:
                    svals = svals[::-1]
                    u = u[:, ::-1]
                    v = v[::-1, :]
            else:
                u, svals, v = np.linalg.svd(dfft[k] * wgts[k], full_matrices=False)

            # append to containers only modes one desires. Default is all modes.
            self.umodes[k] = u[:, :Nkeep]
            self.vmodes[k] = v[:Nkeep, :]
            self.svals[k] = svals[:Nkeep]
            self.uflags[k] = np.min(flags[k], axis=1, keepdims=True)

        # append relevant metadata
        if hasattr(dfft, 'times'):
            self.umodes.times = dfft.times
            self.vmodes.times = dfft.times
            self.svals.times = dfft.times
            self.uflags.times = dfft.times

    def project_svd_modes(self, dfft, umodes=None, svals=None, vmodes=None):
        """
        Given two of the 3 SVD output matrices, project them onto the input dfft data
        to estimate the last remaining SVD matrix. Note U and V are unitary matrices.

        If
            D = U S V
        then
            U = D V_dagger S_inv
            S = U_dagger D V_dagger
            V = S_inv U_dagger D
        where
            S_inv contains the inverse of the S vector along its diagonal.

        Args:
            dfft : DataContainer, holds time-delay waterfall visibilities
            umodes : DataContainer, holds SVD umodes, shape (Ntimes, Nmodes)
            svals : DataContainer, holds SVD singular values, shape (Nmodes,)
            vmodes : DataContainer, holds SVD vmodes, shape (Nmodes, Nfreqs)

        Returns:
            output : DataContainer, estimate of remaining SVD matrix
        """
        output = DataContainer({})
        if umodes is None:
            assert svals is not None and vmodes is not None, "Must feed two of the SVD output matrices"
            # compute umodes
            for k in dfft:
                if k not in svals or k not in vmodes:
                    continue
                output[k] = dfft[k].dot(np.conj(vmodes[k].T).dot(np.eye(len(svals[k])) / svals[k]))

        elif svals is None:
            assert umodes is not None and vmodes is not None, "Must feed two of the SVD output matrices"
            for k in dfft:
                if k not in umodes or k not in vmodes:
                    continue
                output[k] = np.abs(np.conj(umodes[k].T).dot(dfft[k].dot(np.conj(vmodes[k].T))).diagonal())

        elif vmodes is None:
            assert umodes is not None and svals is not None, "Must feed two of the SVD output matrices"
            # compute vmodes
            for k in dfft:
                if k not in svals or k not in umodes:
                    continue
                output[k] = (np.eye(len(svals[k])) / svals[k]).dot(np.conj(umodes[k]).T).dot(dfft[k])

        else:
            raise AssertionError("Must feed two of the SVD output matrices")

        return output

    def build_pc_model(self, umodes, vmodes, svals, keys=None, Nkeep=None, overwrite=False, increment=False, verbose=True):
        """
        Build a data model out of principal components.

        Take outer products of input umodes and vmodes, multiply by singular
        values and sum to form self.pcomp_model.

        Args:
            umodes : DataContainer
                SVD u-modes from self.sv_decomp().
            vmodes : DataContainer
                SVD v-modes from self.sv_decomp().
            svals : DataContainer
                SVD singular values from self.sv_decomp().
            keys : list of tuples
                List of ant-pair-pol tuples to operate on.
            Nkeep : int
                Number of principal components to keep when forming model. Default is all.
            overwrite : bool
                If True, overwrite output data if it exists.
            increment : bool
                If key already exists in pcomp_model, add the new model
                to it, rather than overwrite it. This supercedes overwrite if both are true.
            verbose : bool
                If True, report feedback to stdout.

        Result:
            self.pcomp_model : DataContainer of the PC model, ant-pair-pol key and ndarray value
        """
        # get keys
        if keys is None:
            keys = list(svals.keys())

        # iterate over keys
        for k in keys:
            if k in self.pcomp_model and (not overwrite and not increment):
                echo("{} in pcomp_model and overwrite == increment == False, skipping...".format(k), verbose=verbose)
                continue

            # form principal components
            pcomps = np.einsum("ij,jk->jik", umodes[k], vmodes[k])

            # multiply by svals and sum
            pc_model = np.einsum("i,i...->...", svals[k][:Nkeep], pcomps[:Nkeep])

            # add to pcomp_model
            if k not in self.pcomp_model:
                self.pcomp_model[k] = pc_model
            elif k in self.pcomp_model and increment:
                self.pcomp_model[k] += pc_model
            else:
                self.pcomp_model[k] = pc_model

        if hasattr(umodes, 'times'):
            self.pcomp_model.times = umodes.times

    def subtract_model(self, data, keys=None, overwrite=False, ifft=True, ifftshift=True,
                       window='none', alpha=0.2, edgecut_low=0, edgecut_hi=0, verbose=True):
        """
        Subtract pcomp_model from data.

        FFT pcomp_model to frequency space, divide by window, and subtract from data.
        Inserts FFT of pcomp_model into self.pcomp_model_fft and
        residual of data with self.data_pcmodel_resid.

        Note: The windowing parameters should be the *same* as those that were used
        in constructing the dfft that sv_decomp operated on.

        Args:
            data : DataContainer
                Object to pull data from in forming data-model residual.
            keys : list of tuples
                List of baseline-pol DataContainer tuples to operate on.
            overwrite : bool
                If True, overwrite output data if it exists.
            ifft : bool
                If True, use ifft to go from delay to freq axis
            ifftshift : bool
                If True, ifftshift delay axis before fft.
            window : str
                window function across freq to divide by after FFT.
            alpha : float
                if window is Tukey, this is its alpha parameter.
            edgecut_low : int
                Nbins to flag but not window at low-side of band.
            edgecut_hi : int
                Nbins to flag but not window at high-side of band.
            verbose : bool
                If True, report feedback to stdout.

        Result:
            self.pcomp_model_fft : DataContainer, ant-pair-pol keys and ndarray values
                Holds the FFT of self.pcomp_model, divided by the original windowing
                function applied to the data.
            self.data_pcmodel_resid : DataContainer, ant-pair-pol keys and ndarray values
                Holds the residual between input data and pcomp_model_fft.
        """
        # get keys
        if keys is None:
            keys = list(self.pcomp_model.keys())

        if not hasattr(self, 'pcomp_model_fft'):
            self.pcomp_model_fft = DataContainer({})
        if not hasattr(self, 'data_pcmodel_resid'):
            self.data_pcmodel_resid = DataContainer({})

        # get data
        if data is None:
            data = self.data

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

            # divide by a window: set zeros to inf
            win = dspec.gen_window(window, model_fft.shape[1], alpha=alpha,
                                   edgecut_low=edgecut_low, edgecut_hi=edgecut_hi)
            win[np.isclose(win, 0.0)] = np.inf
            model_fft /= win

            # subtract from data
            self.pcomp_model_fft[k] = model_fft
            self.data_pcmodel_resid[k] = data[k] - model_fft

    def interp_u(self, umodes, times, full_times=None, uflags=None, keys=None, overwrite=False, Ninterp=None,
                 gp_frate=1.0, gp_frate_degrade=0.0, gp_nl=1e-12, kernels=None, optimizer=None, Nmirror=0,
                 xthin=None, verbose=True):
        """
        Interpolate u modes along time with a Gaussian Process.

        Inserts results into self.umode_interp.

        Args:
            umodes : DataContainer
                u-mode container to interpolate, see self.sv_decomp()
            times : 1D array
                Holds time_array of input umodes in Julian Date.
            full_times : 1D array
                Full time_array to interpolate onto in Julian Date. Default is times.
            uflags : DataContainer
                Object to pull target u flags from. Default is None.
            keys : list of tuples
                List of baseline-pol DataContainer tuples to operate on.
            overwrite : bool
                If True, overwrite output data if it exists.
            Ninterp : int
                Number of modes to interpolate. Default (None) is all modes.
            gp_frate : float or DataContainer
                Fringe rate [mHz] associated with GP length scale in time.
                If fed as a DataContainer, must have keys matching umodes.
            gp_frate_degrade : float
                gp_frate * gp_frate_degrade is subtracted from gp_frate before
                being converted to a time length scale to prevent slight
                overfitting of excess frate structure by fall-off of GP covariance
                beyond the set length scale.
            gp_nl : float
                GPR noise-level in units of input umodes.
            kernels : dictionary or sklearn.gaussian_process.kernels.Kernel object
                Dictionary containing sklearn kernels for each key in umodes.
                If kernels is fed, then gp_frate, gp_frate_degrade gp_var and gp_nl are ignored.
            optimizer : str
                GPR optimizer for kernel hyperparameter solution. Default is no regression.
                See sklearn.gaussian_process.GaussianProcessRegressor for details.
            Nmirror : int
                Number of time bins to mirror at ends of input time axis. Default is no mirroring.
            xthin : int
                Factor by which to thin time-axis before GP interpolation. Default is no thinning.
            verbose : bool
                If True, report feedback to stdout.

        Result:
            self.umode_interp : DataContainer, ant-pair-pol keys and ndarray values
                Holds the input umodes Container interpolated onto full_times.
        """
        if not hasattr(self, 'umode_interp'):
            self.umode_interp = DataContainer({})

        if uflags is None:
            uflags = self.uflags

        if full_times is None:
            full_times = times

        # get keys
        if keys is None:
            keys = list(umodes.keys())

        # parse gp_frate
        if isinstance(gp_frate, (int, np.integer, float, np.floating)):
            gp_frate = DataContainer(dict([(k, gp_frate) for k in umodes.keys()]))

        # setup X predict
        Xmean = np.median(times)
        Xtrain = times - Xmean
        Xpredict = full_times - Xmean

        # parse kernels
        if kernels is not None and isinstance(kernels, gp.kernels.Kernel):
            kernels = dict([(k, kernels) for k in keys])

        # iterate over keys
        for k in keys:
            # check overwrite
            if k in self.umode_interp and not overwrite:
                echo("{} in umode_interp and overwrite == False, skipping...".format(k), verbose=verbose)
                continue

            # get kernel
            if kernels is not None:
                kernel = kernels[k]
            else:
                # get length scale in time
                gp_f = np.max([0.0, gp_frate[k] * (1 - gp_frate_degrade)])
                gp_len = 1.0 / (gp_f * 1e-3) / (24.0 * 3600.0)

                # setup GP kernel
                kernel = 1**2 * gp.kernels.RBF(length_scale=gp_len) + gp.kernels.WhiteKernel(noise_level=gp_nl)

            # interpolate
            y = umodes[k][:, :Ninterp]
            yflag = np.repeat(uflags[k], y.shape[1], axis=1)
            self.umode_interp[k] = gp_interp1d(Xtrain, y, x_eval=Xpredict,
                                               flags=yflag, kernel=kernel, Nmirror=Nmirror,
                                               optimizer=optimizer, xthin=xthin)


def form_gains(epsilon):
    """
    Turn reflection coefficients into gains.

    Reflection gains are formed via g = 1 + eps
    where eps is the reflection coefficient
        eps = A exp(2j * pi * tau * freqs + 1j * phs)

    Args:
        epsilon : dictionary, ant-pol keys and ndarray values

    Returns:
        gains : dictionary, ant-pol keys and ndarray values
    """
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
    # reshape if necessary
    if isinstance(amp, np.ndarray):
        if amp.ndim == 1:
            amp = np.reshape(amp, (-1, 1))
    if isinstance(tau, np.ndarray):
        if tau.ndim == 1:
            tau = np.reshape(tau, (-1, 1))
    if isinstance(phs, np.ndarray):
        if phs.ndim == 1:
            phs = np.reshape(phs, (-1, 1))

    # make reflection
    eps = amp * np.exp(2j * np.pi * tau * freqs + 1j * phs)
    if real:
        eps = eps.real

    return eps


def fit_reflection_delay(rfft, dly_range, dlys, dfft=None, return_peak=False, reject_edges=True):
    """
    Take FFT'd data and fit for peak delay and amplitude.

    Args:
        rfft : complex 2D array with shape (Ntimes, Ndlys)
            The FFT of the clean residual, having CLEANed-out
            structure at delays less than dly_range.
            If the reflection is significantly isolated, this works
            fine as the FFT of the clean data.
        dly_range : len-2 tuple
            Specifying range of delays [nanosec] to look for reflection within.
        dlys : 1D array
            data delays [nanosec]
        dfft : complex 2D array with shape (Ntimes, Ndlys)
            The FFT of the unfiltered clean data to use in fitting for
            the main foreground amplitude and delay, if rfft is
            FFT of clean residual.
        return_peak : bool
            If True, just return peak delay and amplitude and rfft
        reject_edges : bool
            If True, reject peak solutions at delay edges

    Returns:
        if return_peak:
            ref_peaks : rfft peak amplitude within dly_range
            ref_dlys : rfft peak delay [nanosec] within dly_range
        else:
            ref_amps : reflection amplitudes (Ntimes, 1)
            ref_dlys : reflection delays [nanosec] (Ntimes, 1)
            ref_inds : reflection peak indicies of input rfft array
            ref_sig : reflection significance, SNR of peak relative to neighbors
    """
    # get fft
    assert rfft.ndim == 2, "input rfft must be 2-dimensional with shape [Ntimes, Ndlys]"
    if dfft is None:
        dfft = rfft
    assert dfft.shape == rfft.shape, "input dfft shape must match rfft shape"
    Ntimes, Ndlys = rfft.shape

    # select delay range
    select = np.where((dlys > dly_range[0]) & (dlys < dly_range[1]))[0]
    if len(select) == 0:
        raise ValueError("No delays in specified range {} ns".format(dly_range))

    # locate peak bin within dly range
    abs_rfft = np.abs(rfft)[:, select]
    ref_dly_inds, bin_shifts, _, ref_peaks = interp_peak(abs_rfft, method='quadratic', reject_edges=reject_edges)
    ref_dly_inds += select.min()
    ref_dlys = dlys[ref_dly_inds, None] + bin_shifts[:, None] * np.median(np.abs(np.diff(dlys)))
    ref_peaks = ref_peaks[:, None]

    if return_peak:
        return ref_peaks, ref_dlys

    # get reflection significance, defined as max|V| / med|V|
    avgmed = np.median(abs_rfft, axis=1, keepdims=True)
    ref_significance = np.true_divide(ref_peaks, avgmed, where=~np.isclose(avgmed, 0.0))

    # get peak value at tau near zero
    fg_peak, fg_dly = fit_reflection_delay(dfft, (-np.abs(dly_range).min(), np.abs(dly_range).min()), dlys, return_peak=True)

    # get reflection amplitude
    ref_amps = np.true_divide(ref_peaks, fg_peak, where=~np.isclose(fg_peak, 0.0))

    # update reflection delay given FG delay
    ref_dlys -= fg_dly

    return ref_amps, ref_dlys, ref_dly_inds, ref_significance


def fit_reflection_phase(dfft, dly_range, dlys, ref_dlys, fthin=1, Nphs=250, ifft=False, ifftshift=True):
    """
    Fit for reflection phases.

    Take FFT'd data and reflection amp and delay and solve for its phase.

    Args:
        dfft : complex 2D array with shape (Ntimes, Ndlys)
        dly_range : len-2 tuple specifying range of delays [nanosec]
            to look for reflection within.
        dlys : 1D array of data delays [nanosec]
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
    filt, freqs = vis_clean.fft_data(filt, np.median(np.diff(dlys)) / 1e9, axis=-1, ifft=ifft, ifftshift=ifftshift, fftshift=False)
    filt /= np.max(np.abs(filt), axis=-1, keepdims=True)
    freqs = np.linspace(0, freqs.max() - freqs.min(), Ndlys, endpoint=True)
    phases = np.linspace(0, 2 * np.pi, Nphs, endpoint=False)
    cosines = np.array([construct_reflection(freqs[::fthin], 1, ref_dlys / 1e9, p) for p in phases])
    residuals = np.sum((filt[None, :, ::fthin].real - cosines.real)**2, axis=-1)

    # quadratic interp of residuals to get reflection phase
    resids = -residuals.T
    inds, bin_shifts, _, _ = interp_peak(resids + np.abs(resids.min()), method='quadratic')
    ref_phs = (phases[inds] + bin_shifts * np.median(np.diff(phases)))[:, None] % (2 * np.pi)

    # fill nans
    ref_phs[np.isnan(ref_phs)] = 0.0

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


def fit_reflection_params(clean_resid, dly_range, freqs, clean_flags=None, clean_data=None, window=None, alpha=0.2,
                          edgecut_low=0, edgecut_hi=0, zeropad=0, ref_sig_cut=2.0, fthin=1, Nphs=250, reject_edges=True):
    """
    Fit for reflection parameters in CLEANed visibility.

    Args:
        clean_resid : 2D ndarray shape (Ntimes, Nfreqs)
            Autocorrelation data (in frequency space) to find reflections in.
            Ideally this is the CLEAN residual, having cleaned out structure
            at delays less than the min(abs(dly_range)). If this is the
            CLEAN residual, you must feed the unfiltered CLEAN data as "clean_data"
            to properly estimate the reflection amplitude.
        dly_range : len-2 tuple
            delay range [nanoseconds] to search within for reflections.
            Must be either both positive or both negative.
        freqs : 1D ndarray
            Frequency array in Hz
        clean_flags : 2D ndarray shape (Ntimes, Nfreqs)
            Flags of the (CLEANed) data. Default is None.
        clean_data : 2D ndarray shape (Ntimes, Nfreqs)
            If clean_resid is fed as the CLEAN residual (and not CLEAN data),
            this should be the unfiltered CLEAN data output.
        window : str
            windwoing function to apply across freq before FFT. See dspec.gen_window for options
        alpha : float
            if taper is Tukey, this is its alpha parameter
        edgecut_low : int
            Nbins to flag and exclude from windowing function on low-side of the band
        edgecut_hi : int
            Nbins to flag and exclude from windowing function on low-side of the band
        zeropad : int
            number of channels to pad band edges with zeros before FFT
        fthin : int
            scaling factor to down-select frequency axis when solving for phase
        Nphs : int
            Number of samples from 0 - 2pi to estimate reflection phase at.
        ref_sig_cut : float
            if max reflection significance is not above this, do not record solution,
            where significance is defined as max(fft) / median(fft).
        reject_edges : bool
            If True, reject peak solutions at delay edges

    Returns:
        amp : reflection amplitude
        dly : reflection delay [nanosec]
        phs : reflection phase [radians]
        sig : reflection significance, defined as max(fft) / median(fft) in dly_range
    """
    # get info
    dnu = np.diff(freqs)[0]

    # get wgts
    if clean_flags is None:
        wgts = np.ones_like(clean_resid, dtype=float)
    else:
        wgts = (~clean_flags).astype(float)

    # fourier transform
    rfft, delays = vis_clean.fft_data(clean_resid, dnu, wgts=wgts, axis=-1, window=window, alpha=alpha, edgecut_low=edgecut_low, edgecut_hi=edgecut_hi, ifft=False, fftshift=True, zeropad=zeropad)
    delays *= 1e9
    if clean_data is None:
        dfft = rfft
    else:
        dfft, _ = vis_clean.fft_data(clean_data, dnu, wgts=wgts, axis=-1, window=window, alpha=alpha, edgecut_low=edgecut_low, edgecut_hi=edgecut_hi, ifft=False, fftshift=True, zeropad=zeropad)
    assert rfft.shape == dfft.shape, "clean_resid and clean_data must have same shape"

    # fit for reflection delays and amplitude
    amp, dly, inds, sig = fit_reflection_delay(rfft, dly_range, delays, dfft=dfft, return_peak=False, reject_edges=reject_edges)

    # make significance cut
    if np.max(sig) < ref_sig_cut:
        # set reflection parameters to zero
        amp[:] = 0.0
        dly[:] = 0.0
        phs = np.zeros_like(amp)

    else:
        # solve for phase (slowest step)
        phs = fit_reflection_phase(rfft, dly_range, delays, dly, fthin=fthin, Nphs=Nphs, ifft=True, ifftshift=True)

        # anchor phase to 0 Hz
        phs = (phs - 2 * np.pi * dly / 1e9 * (freqs[0] - zeropad * dnu)) % (2 * np.pi)

    return amp, dly, phs, sig


def reflection_param_minimization(clean_data, dly_range, freqs, amp0, dly0, phs0,
                                  fix_amp=False, fix_dly=False, fix_phs=False, clean_model=None,
                                  clean_flags=None, method='BFGS', skip_frac=0.9,
                                  tol=1e-4, maxiter=100, **fft_kwargs):
    """
    Perturb reflection parameters to minimize residual in data.

    Args:
        clean_data : 2D ndarray of shape (Ntimes, Nfreqs)
            CLEANed auto-correlation visibility data
        dly_range : float or len-2 tuple of floats
            Additive delay [ns] offset (-, +) from dly0 to create range of
            delays used in objective minimization
        freqs : 1D array of shape (Nfreqs,)
            Frequency array [Hz]
        amp0 : float or ndarray
            Initial guess for reflection amplitude
        dly0 : float or ndarray
            Initial guess for reflection delay [ns]
        phs0 : float or ndarray
            Initial guess for reflection phase [radian]
        fix_amp : bool
            If True, fix amplitude solution at amp0
        fix_dly : bool
            If True, fix delay solution at dly0
        fix_phs : bool
            If True, fix delay solution at phs0
        clean_model : 2D ndarray of shape (Ntimes, Nfreqs)
            CLEAN model with CLEAN boundary out to at most min(|dly0 - dly_range|)
            If reflection is well-isolated from foreground power, this is not necessary.
        clean_flags : 2D ndarray boolean of shape (Ntimes, Nfreqs)
            CLEAN flags for FFT
        method : str
            Optimization algorithm. See scipy.optimize.minimize for options
        skip_frac : float in range [0, 1]
            fraction of flagged channels (excluding edge flags) above which skip the integration
        tol : float
            Optimization stopping tolerance
        maxiter : int
            Optimization max iterations
        fft_kwargs : kwargs to pass to vis_clean.fft_data
            except for axis, wgts, ifft, fftshift

    Returns:
        amp : ndarray
            Reflection amplitude solution
        dly : ndarray
            Reflection delay solution [ns]
        phs : ndarray
            Reflection phase solution [rad]
        info : list
            Optimization success [bool]
    """
    # define removal metric
    def L(x, amp, dly, phs, clean_data, clean_model, clean_wgts, dly_range, freqs, fft_kwargs):
        """
        Metric to minimize is max(dfft) in delay range

        x : ndarray, amp, and/or dly [ns], and/or phs [rad] parameters in this order.
            If any are fed as their own argument below, exclude them from x.
        amp : ndarray, amplitude to hold fixed if not fed in x.
            If fed in x, this should be None
        dly : ndarray, delay [ns] to hold fixed if not fed in x.
            If fed in x, this should be None
        phs : ndarray, phase [rad] to hold fixed if not fed in x.
            If fed in x, this should be None
        clean_data : ndarray, CLEANed visibility data
        clean_model : ndarray, CLEAN visibility model
        clean_wgts : ndarray, FFT weights
        dly_range : len-2 tuple [ns]
        freqs : ndarray, frequency array [Hz]
        fft_kwargs : dictionary, kwargs to pass to fft_data
        """
        # form gains and apply to data
        Nparams = (amp is None) + (dly is None) + (phs is None)
        x = np.reshape(x, (Nparams, -1))
        if amp is None:
            amp = x[0]
            x = x[1:]
        if dly is None:
            dly = x[0]
            x = x[1:]
        if phs is None:
            phs = x[0]

        # return large number if amp is negative
        if amp < 0:
            return 1e10

        # construct reflection and divide gain from data
        eps = construct_reflection(freqs, amp, dly / 1e9, phs, real=False)
        gain = 1 + eps
        cal_data = clean_data / (gain * np.conj(gain))

        # fft to delay space
        dnu = np.diff(freqs)[0]
        dfft, delays = vis_clean.fft_data(cal_data - clean_model, dnu, axis=-1, wgts=clean_wgts, ifft=False, fftshift=True, **fft_kwargs)
        delays *= 1e9

        select = (delays >= dly_range[0]) & (delays <= dly_range[1])
        metric = np.median(np.max(np.abs(dfft[:, select]), axis=1), axis=0)

        return metric

    # input checks
    Ntimes = len(clean_data)
    if clean_model is None:
        clean_model = np.zeros_like(clean_data)
    if clean_flags is None:
        clean_wgts = np.ones_like(clean_data)
    else:
        clean_wgts = (~clean_flags).astype(float)
    edge_flags = np.isclose(np.mean(clean_wgts, axis=0), 0.0)
    edge_flags[np.argmin(edge_flags):-np.argmin(edge_flags[::-1])] = False
    if fix_amp and fix_dly and fix_phs:
        raise ValueError("Can't hold amp, dly and phs fixed")

    # create delay range
    if isinstance(dly_range, (int, np.integer, float, np.floating)):
        dly_range = [dly_range, dly_range]
    dly_range = (np.nanmedian(dly0) - np.abs(dly_range[0]), np.nanmedian(dly0) + np.abs(dly_range[1]))

    # iterate over times
    amp, dly, phs, info = [], [], [], []
    for i in range(Ntimes):
        # skip frac
        if edge_flags.all() or (skip_frac < 1 - np.mean(clean_wgts[i][~edge_flags])):
            amp.append(0)
            dly.append(0)
            phs.append(0)
            info.append(False)
            continue

        # setup starting guess
        x0 = []
        if fix_amp:
            _amp = amp0[i]
        else:
            x0.append(amp0[i])
            _amp = None
        if fix_dly:
            _dly = dly0[i]
        else:
            x0.append(dly0[i])
            _dly = None
        if fix_phs:
            _phs = phs0[i]
        else:
            x0.append(phs0[i])
            _phs = None
        x0 = np.array(x0)

        # optimize
        res = minimize(L, x0, args=(_amp, _dly, _phs, clean_data[i], clean_model[i], clean_wgts[i], dly_range, freqs, fft_kwargs), method=method, tol=tol, options=dict(maxiter=maxiter))

        # collect output
        xf = res.x
        if fix_amp:
            amp.append(amp0[i])
        else:
            amp.append(xf[0])
            xf = xf[1:]
        if fix_dly:
            dly.append(dly0[i])
        else:
            dly.append(xf[0])
            xf = xf[1:]
        if fix_phs:
            phs.append(phs0[i])
        else:
            phs.append(xf[0])
        info.append(res.success)

    return np.array(amp), np.array(dly), np.array(phs), np.array(info)


def auto_reflection_argparser():
    a = argparse.ArgumentParser(description='Model auto (e.g. cable) reflections from auto-correlation visibilities')
    a.add_argument("data", nargs='*', type=str, help="data file paths to CLEAN and run auto reflection modeling on")
    a.add_argument("--output_fname", type=str, help="Full path to the output .calfits file")
    a.add_argument("--dly_ranges", type=str, nargs='*', help='list of 2 comma-delimited delays (ex. 200,300 400,500) specifying delay range to search for reflections [nanosec].')
    a.add_argument("--filetype", type=str, default='uvh5', help="Filetype of datafile")
    a.add_argument("--overwrite", default=False, action='store_true', help="Overwrite output file if it already exists")
    a.add_argument("--write_npz", default=False, action='store_true', help="Write NPZ file with reflection params with same path name as output calfits.")
    a.add_argument("--input_cal", type=str, default=None, help="Path to input .calfits to apply to data before modeling")
    a.add_argument("--antenna_numbers", default=None, type=int, nargs='*', help="List of antenna numbers to operate on. Default is all in data.")
    a.add_argument("--polarizations", default=None, type=str, nargs='*', help="List of polarization strings to operate on.")
    a.add_argument("--window", default='None', type=str, help="FFT window for CLEAN")
    a.add_argument("--alpha", default=0.2, type=float, help="Alpha parameter if window is tukey")
    a.add_argument("--tol", default=1e-6, type=float, help="CLEAN tolerance")
    a.add_argument("--gain", default=1e-1, type=float, help="CLEAN gain")
    a.add_argument("--maxiter", default=100, type=int, help="CLEAN maximum Niter")
    a.add_argument("--skip_wgt", default=0.1, type=float, help="Skip integration if heavily flagged, see hera_cal.delay_filter for details")
    a.add_argument("--edgecut_low", default=0, type=int, help="Number of channels to flag but not window on low edge of band (before zeropadding)")
    a.add_argument("--edgecut_hi", default=0, type=int, help="Number of channels to flag but not window on high edge of band (before zeropadding)")
    a.add_argument("--zeropad", default=0, type=int, help="Number of channels to zeropad *both* sides of band in auto modeling process.")
    a.add_argument("--horizon", default=1.0, type=float, help="Baseline horizon coefficient. See hera_cal.delay_filter for details")
    a.add_argument("--standoff", default=0.0, type=float, help="Baseline horizon standoff. See hera_cal.delay_filter for details")
    a.add_argument("--min_dly", default=0.0, type=float, help="Minimum CLEAN delay horizon. See hera_cal.delay_filter for details")
    a.add_argument("--Nphs", default=500, type=int, help="Number of phase points to evaluate from 0--2pi in solving for phase")
    a.add_argument("--fthin", default=1, type=int, help="Coefficient to thin frequency axis by when solving for phase")
    a.add_argument("--ref_sig_cut", default=2, type=float, help="Reflection minimum 'significance' threshold for fitting.")
    a.add_argument("--add_to_history", default='', type=str, help="String to append to file history")
    a.add_argument("--time_avg", default=False, action='store_true', help='Time average file before reflection fitting.')
    a.add_argument("--compress_tavg_calfits", default=False, action='store_true', help='Save final calfits files with a single integration (at the average JD) '
                   'rather than duplicating identical calibration solutions for every intergration. Ignored if time_avg is False.')
    a.add_argument("--opt_maxiter", default=0, type=int, help="Optimization max Niter. Default is no optimization")
    a.add_argument("--opt_method", default='BFGS', type=str, help="Optimization algorithm. See scipy.optimize.minimize for details")
    a.add_argument("--opt_tol", default=1e-3, type=float, help="Optimization stopping tolerance.")
    a.add_argument("--opt_buffer", default=[25, 25], type=float, nargs='*', help="delay buffer [ns] +/- initial guess for setting range of objective function")
    a.add_argument("--skip_frac", default=0.9, type=float, help="Float in range [0, 1]. Fraction of (non-edge) flagged channels above which integration is skipped in optimization.")
    a.add_argument("--only_write_final_calfits", dest='write_each_calfits', default=True, action='store_false',
                   help="Instead of writing one calfits file for each dly_range, instead only write a single combined calfits file with all reflections multiplied.")
    return a


def auto_reflection_run(data, dly_ranges, output_fname, filetype='uvh5', input_cal=None, time_avg=False, compress_tavg_calfits=False,
                        write_npz=False, antenna_numbers=None, polarizations=None, window='None', alpha=0.2,
                        edgecut_low=0, edgecut_hi=0, zeropad=0, tol=1e-6, gain=1e-1, maxiter=100,
                        skip_wgt=0.2, horizon=1.0, standoff=0.0, min_dly=100.0, Nphs=300, fthin=10,
                        ref_sig_cut=2.0, add_to_history='', skip_frac=0.9, reject_edges=True, opt_maxiter=0,
                        opt_method='BFGS', opt_tol=1e-3, opt_buffer=(25, 25), write_each_calfits=True, overwrite=False):
    """
    Run auto-correlation reflection modeling on files.

    Args:
        data : str or UVData subclass, data to operate on
        dly_ranges : list, len-2 tuples specifying min and max delay range [ns] to fit for reflections
        output_fname : str, full path to output calfits file
        filetype : str, filetype if data is a str, options=['uvh5', 'miriad', 'uvfits']
        input_cal : str or UVCal subclass, calibration to apply to data on-the-fly
        time_avg : bool, if True, time-average the entire input data before reflection modeling.
            This will produce single-integration calfits files.
        compress_tavg_calfits : Save final calfits files with a single integration (at the average JD) rather
            than duplicating identical calibration solutions for every intergration. Ignored if time_avg is False.
        write_npz : bool, if True, write an NPZ with reflection parameters with matching path as output_fname
        antenna_numbers : int list, list of antenna number integers to run on. Default is all in data.
        polarizations : str list, list of polarization strings to run on, default is all
        edgecut_low : int, Nbins to flag but not window at low-side of the FFT axis.
        edgecut_hi : int, Nbins to flag but not window at high-side of the FFT axis.
        window : str, tapering function to apply across freq before FFT
        alpha : float, if taper is Tukey, this is its alpha parameter
        zeropad : int, number of channels to pad band edges with zeros before FFT
        fthin : int, scaling factor to down-select frequency axis when solving for phase
        tol : float, CLEAN tolerance
        gain : float, CLEAN gain
        maxiter : int, CLEAN max Niter
        skip_wgt : float, flagged threshold for skipping integration, see delay_filter.py
        standoff : float ,fixed additional delay beyond the horizon (in nanosec) to CLEAN
        horizon : float, coefficient to baseline horizon where 1 is the horizon
        min_dly : float, upper CLEAN boundary is never below min_dly [ns]
        Nphs : int, Number of points in phase [0=2pi] to evaluate for reflection phase solution
        fthin : int, Thinning number across frequency axis when solving for phase
        ref_sig_cut : float, if max reflection significance is not above this, do not record solution
        add_to_history : str, notes to add to history
        opt_maxiter : int, optimization max iterations. Default is no optimization
        opt_method : str, optimization algorithm. See scipy.optimize.minimize for options
        opt_tol : float, Optimization stopping tolerance
        opt_buffer : float or len-2 tuple, delay buffer [ns] +/- initial guess for setting range of objective function in delay
        skip_frac : float in range [0, 1], fraction of flagged channels (excluding edge flags) above which skip the integration
        reject_edges : bool, If True, reject peak solutions at delay edges
        write_each_calfits : bool, if True, write calfits file for each fit in each dly_ranges, otherwise write one final calfits file
        overwrite : bool, if True, overwrite output files.

    Result:
        A calfits written to output_fname, and if write_npz, an NPZ with the
        same path and filename, except for the .npz suffix.

    Notes:
        The CLEAN min_dly should always be less than the lower boundary of dly_range.
    """
    # dly_ranges type check
    if isinstance(dly_ranges, str):
        dly_ranges = [ast.literal_eval(dly_ranges)]
    if isinstance(dly_ranges, tuple):
        dly_ranges = [dly_ranges]
    if isinstance(dly_ranges, list):
        for i, dlyr in enumerate(dly_ranges):
            if isinstance(dlyr, str):
                dly_ranges[i] = ast.literal_eval(dlyr)

    # initialize reflection fitter
    RF = ReflectionFitter(data, filetype=filetype, input_cal=input_cal)

    # get antennas if possible
    if antenna_numbers is None and hasattr(RF, 'data_ants'):
        bls = [(ant, ant) for ant in RF.data_ants]
    elif antenna_numbers is not None:
        bls = [(ant, ant) for ant in antenna_numbers]
    else:
        bls = None

    # read data
    RF.read(bls=bls, polarizations=polarizations)

    # get all autocorr & autopol keys
    keys = [k for k in RF.data if (k[0] == k[1]) and (k[2][0] == k[2][1])]

    # clean data
    RF.vis_clean(data=RF.data, flags=RF.flags, keys=keys, ax='freq', window=window, alpha=alpha,
                 horizon=horizon, standoff=standoff, min_dly=min_dly, tol=tol, maxiter=maxiter,
                 gain=gain, skip_wgt=skip_wgt, edgecut_low=edgecut_low, edgecut_hi=edgecut_hi)
    data = RF.clean_data
    flags = RF.clean_flags
    model = RF.clean_model
    nsamples = RF.nsamples
    times = RF.times
    lsts = RF.lsts

    # time average file
    if time_avg:
        RF.timeavg_data(data, times, lsts, 1e10, flags=flags, nsamples=nsamples)
        RF.timeavg_data(model, times, lsts, 1e10, flags=flags, nsamples=nsamples, output_prefix='avgm')
        data = RF.avg_data
        flags = RF.avg_flags
        nsamples = RF.avg_nsamples
        model = RF.avgm_data
        if compress_tavg_calfits:
            # Reduce time array to length one, thus making the ouput calfits waterfalls have shape (1, Nfreqs)
            RF.times = np.mean(RF.times, keepdims=True)

    # iterate over dly_ranges
    gains = []
    for i, dly_range in enumerate(dly_ranges):
        _output_fname = list(os.path.splitext(output_fname))
        _output_fname[0] = "{}.ref{}".format(_output_fname[0], i + 1)
        _output_fname = ''.join(_output_fname)
        if i == 0:
            _RF = RF
            cdata = data
            if write_each_calfits:
                _output_fname = output_fname
        else:
            _RF = RF.soft_copy()
            cdata = copy.deepcopy(data)
            calibrate_in_place(cdata, merge_gains(gains, merge_shared=False))

        # model auto reflections in clean data
        _RF.model_auto_reflections(cdata, dly_range, clean_flags=flags, edgecut_low=edgecut_low,
                                   edgecut_hi=edgecut_hi, Nphs=Nphs, window=window, alpha=alpha,
                                   zeropad=zeropad, fthin=fthin, ref_sig_cut=ref_sig_cut, reject_edges=reject_edges)

        # refine reflections
        if opt_maxiter > 0:
            (_RF.ref_amp, _RF.ref_dly, _RF.ref_phs, info, _RF.ref_eps,
             _RF.ref_gains) = RF.refine_auto_reflections(cdata, opt_buffer, _RF.ref_amp, _RF.ref_dly, _RF.ref_phs, ref_flags=_RF.ref_flags,
                                                         keys=keys, window=window, alpha=alpha, edgecut_low=edgecut_low,
                                                         edgecut_hi=edgecut_hi, clean_flags=flags, clean_model=model,
                                                         skip_frac=skip_frac, maxiter=opt_maxiter, method=opt_method, tol=opt_tol)

        # write gains
        RF.ref_gains = _RF.ref_gains
        RF.write_auto_reflections(_output_fname, overwrite=overwrite, add_to_history=add_to_history,
                                  write_npz=write_npz, write_calfits=write_each_calfits)

        # append gains
        gains.append(RF.ref_gains)

    # write out combined gains as a single calfits file
    if not write_each_calfits:
        RF.ref_gains = merge_gains(gains)
        RF.write_auto_reflections(output_fname, overwrite=overwrite, add_to_history=add_to_history,
                                  write_npz=False, write_calfits=True)
