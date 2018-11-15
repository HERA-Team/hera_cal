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

from . import io
from . import abscal_funcs
from . import delay_filter
from .datacontainer import DataContainer
from .frf import FRFilter


class ReflectionFitter(FRFilter):
    """
    A subclass of frf.FRFilter with added
    reflection modeling capabilities.
    """

    def reset(self, force=False):
        """
        Empty all DataContainer attached to class except for
        original data, flags and nsamples. force must be True
        to execute.
        """
        if not force:
            raise ValueError("cannot reset object without force == True")
        else:
            for o in self.__dict__:
                if isinstance(getattr(self, o), DataContainer):
                    if o not in ['data', 'flags', 'nsamples']:
                        setattr(self, o, DataContainer({}))

    def dly_clean_data(self, keys=None, data=None, flags=None, nsamples=None, tol=1e-5, maxiter=500, gain=0.1,
                       skip_wgt=0.2, dly_cut=1500, edgecut=0, taper='none', alpha=0.1, timeavg=False,
                       broadcast_flags=False, time_thresh=0.05, overwrite=False, verbose=True):
        """
        Run a Delay Clean on self.data dictionary to derive a model of the
        visibility free of flagged channels. CLEAN data is inserted into
        self.clean_data.

        Parameters
        ----------
        keys : list of baseline-pol tuples to run clean on. Default is all keys.
        data : DataContainer to pull data from. Default is self.data
        flags : DataContainer to pull flags from. Default is self.flags.
        nsamples : DataContainer to pull nsample from. Default is self.nsamples.
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
        for dc in ['clean_data', 'resid_data', 'clean_flags', 'clean_nsamples']:
            if not hasattr(self, dc):
                setattr(self, dc, DataContainer({}))
        if not hasattr(self, 'clean_info'):
            self.clean_info = {}
        self.clean_freqs = self.freqs
        self.clean_edgecut = edgecut
        if edgecut > 0:
            self.clean_freqs = self.clean_freqs[edgecut:-edgecut]

        # setup data and flags
        if data is None:
            data = self.data
        if flags is None:
            flags = self.flags
        if nsamples is None:
            nsamples = self.nsamples

        # get keys
        if keys is None:
            keys = data.keys()

        # iterate over keys
        for k in keys:
            if k in self.clean_data and not overwrite:
                echo("{} exists in clean_data and overwrite == False, skipping...".format(k), verbose=verbose)
                continue

            echo("...Cleaning data key {}".format(k), verbose=verbose)
            (model, flag, residual, dlys,
             info) = reflections_delay_filter(data[k].copy(), flags[k].copy(), self.dnu, tol=tol,
                                              maxiter=maxiter, gain=gain, skip_wgt=skip_wgt, dly_cut=dly_cut,
                                              edgecut=edgecut, taper=taper, alpha=alpha, timeavg=timeavg,
                                              broadcast_flags=broadcast_flags, time_thresh=time_thresh)
            # add residual back into model
            model += residual * ~flag

            # append to containers
            self.clean_data[k] = model
            self.resid_data[k] = residual
            self.clean_info[k] = info
            self.clean_flags[k] = flag

            # make a band-averaged nsample
            nsamp = nsamples[k]
            if edgecut > 0:
                nsamp = nsamp[:, edgecut:-edgecut]
            self.clean_nsamples[k] = np.ones((len(data.times), 1), dtype=np.float) * np.sum(~flag * nsamp, axis=1, keepdims=True) / np.sum(~flag, axis=1, keepdims=True).clip(1e-10, np.inf)

        self.clean_dlys = dlys
        self.clean_times = data.times
        if timeavg:
            self.clean_times = np.mean(self.clean_times, keepdims=True)

    def fft_data(self, keys=None, taper='none', alpha=0.1, overwrite=False, verbose=True):
        """
        Take FFT of clean data in self.clean_data and assign to self.clean_fft.

        Parameters
        ----------
        keys : list of tuples
            List of keys from clean_data to FFT. Default is all keys.
        taper : str
            Tapering function to apply across frequency before FFT. See aipy.dsp
        alpha : float
            If taper is tukey this is its alpha parameter.
        overwrite : bool
            If self.clean_fft already exists, overwrite its contents.
        """
        # get keys
        if keys is None:
            keys = self.clean_data.keys()

        # assign taper
        if taper in ['none', None, 'boxcar', 'tophat']:
            t = lambda N: windows.boxcar(N)[None, :]
        elif taper == 'blackmanharris':
            t = lambda N: windows.blackmanharris(N)[None, :]
        elif taper == 'tukey':
            t = lambda N: windows.tukey(N, alpha)[None, :]
        elif taper in ['hann', 'hanning']:
            t = lambda N: windows.hann(N)[None, :]
        else:
            raise ValueError("Didn't recognize taper {}".format(taper))

        # iterate over keys
        if not hasattr(self, 'clean_fft'):
            self.clean_fft = DataContainer({})
        self.clean_taper = taper
        for k in keys:
            if k not in self.clean_data:
                echo("{} not in self.clean_data, skipping...".format(k), verbose=verbose)
                continue
            if k in self.clean_fft and not overwrite:
                echo("{} in self.clean_fft and overwrite == False, skipping...".format(k), verbose=verbose)
                continue
            d = self.clean_data[k]
            self.clean_fft[k] = np.fft.fftshift(np.fft.fft(d * t(d.shape[1]), axis=1), axes=1)
   
    def model_auto_reflections(self, dly_range, keys=None, data='clean', edgecut=0, taper='none',
                               alpha=0.1, zero_pad=0, overwrite=False, fthin=10, verbose=True):
        """
        Model reflections in (ideally RFI-free) autocorrelation data. To CLEAN data of
        flags see the self.dly_clean_data() function. Recommended to set zero_pad
        to at least as large as Nfreqs.

        Parameters
        ----------
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

    def write_auto_reflections(self, output_calfits, input_calfits=None, overwrite=False):
        """
        Given a filepath to antenna gain calfits file, load the
        calibration, incorporate auto-correlation reflection term from the
        self.epsilon dictionary and write to file.

        Parameters
        ----------
        output_calfits : str, filepath to write output calfits file
        input_calfits : str, filepath to input calfits file to multiply in with
            reflection gains.
        overwrite : bool, if True, overwrite output file

        Returns
        -------
        uvc : UVCal object with new gains
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

    def pca_decomp(self, dly_range, side='both', keys=None, taper='none', alpha=0.1,
                   overwrite=False, truncate_level=0, verbose=True):
        """
        Create a PCA based model of the FFT of clean data.

        Parameters
        ----------
        dly_range : len-2 tuple of positive delays in nanosec

        side : str, options=['pos', 'neg', 'both']
            Specifies dly_range as positive delays, negative delays or both.

        keys : list of tuples
            List of datacontainer baseline-pol tuples to create model for.

        taper : str
            Tapering function to apply during FFT. See self.fft_data

        alpha : float
            If taper is tukey, this is its alpha parameter.

        overwrite : bool
            If clean_fft exists, overwrite its values.

        truncate_level : float
            Variance ratio relative to zeroth eigenmode to truncate
            eigenmodes.
        """
        # get keys
        if keys is None:
            keys = self.clean_data.keys()

        # form fft of clean data
        self.fft_data(keys=keys, taper=taper, alpha=alpha, overwrite=overwrite)

        # setup dictionaries
        if not hasattr(self, 'clean_u'):
            self.clean_u = DataContainer({})
        if not hasattr(self, 'clean_v'):
            self.clean_v = DataContainer({})
        if not hasattr(self, 'clean_svals'):
            self.clean_svals = DataContainer({})
        if not hasattr(self, 'clean_u_flags'):
            self.clean_u_flags = DataContainer({})

        # get selection function
        if side == 'pos':
            select = np.where((self.clean_dlys >= dly_range[0]) & (self.clean_dlys <= dly_range[1]))[0]
        elif side == 'neg':
            select = np.where((self.clean_dlys >= -dly_range[1]) & (self.clean_dlys <= -dly_range[0]))[0]
        elif side == 'both':
            select = np.where((np.abs(self.clean_dlys) > dly_range[0]) & (np.abs(self.clean_dlys) < dly_range[1]))[0]

        # iterate over keys
        for k in keys:
            if k in self.clean_svals and not overwrite:
                echo("{} exists in clean_svals and overwrite == False, skipping...".format(k), verbose=verbose)
                continue

            # perform svd to get principal components
            d = np.zeros_like(self.clean_fft[k])
            d[:, select] = self.clean_fft[k][:, select]
            u, svals, v = _svd_waterfall(d)

            # append to containers
            keep = np.where(svals / svals[0] > truncate_level)[0]
            self.clean_u[k] = u[:, keep]
            self.clean_v[k] = v[keep, :]
            self.clean_svals[k] = svals[keep]
            self.clean_u_flags[k] = np.min(self.clean_flags[k], axis=1)

        # get principal components
        self.form_PCs(keys, overwrite=overwrite)

        # append relevant metadata
        self.clean_u.times = self.clean_times
        self.clean_v.times = self.clean_times
        self.clean_svals.times = self.clean_times
        self.clean_u_flags.times = self.clean_times

    def form_PCs(self, keys=None, u=None, v=None, overwrite=False, verbose=True):
        """
        Take u and v-modes and form outer product to get principal components.

        Parameters
        ----------
        keys : list of tuples
            List of baseline-pol DataContainer tuples to operate on.

        u : DataContainer 
            Holds u-modes to use in forming PCs. Default is self.clean_u

        v : DataContainer 
            Holds v-modes to use in forming PCs. Default is self.clean_v

        overwrite : bool
            If True, overwrite output data if it exists.

        verbose : bool
            If True, report feedback to stdout.
        """
        if keys is None:
            keys = self.clean_svals.keys()
        if u is None:
            u = self.clean_u
        if v is None:
            v = self.clean_v

        if not hasattr(self, 'clean_PCs'):
            self.clean_PCs = DataContainer({})

        for k in keys:
            if k in self.clean_PCs and not overwrite:
                echo("{} exists in clean_PCs and overwrite == False, skipping...".format(k), verbose=verbose)
                continue

            self.clean_PCs[k] = _form_PCs(u[k], v[k])

    def build_model(self, keys=None, Nkeep=None, overwrite=False, increment=False, verbose=True):
        """
        Sum principal components dotted with singular values and add to 
        the clean_pc_model.

        Parameters
        ----------
        keys : list of tuples
            List of baseline-pol DataContainer tuples to operate on.

        Nkeep : int
            Number of principal components to keep when forming model.

        overwrite : bool
            If True, overwrite output data if it exists.

        increment : bool
            If key already exists in clean_pc_model, add the new model
            to it, rather than overwrite it. This supercedes overwrite
            if both are true.

        verbose : bool
            If True, report feedback to stdout.
        """
        if not hasattr(self, "clean_PCs"):
            raise ValueError("self.clean_PCs must exist. See self.pcs_decomp()")

        # get keys
        if keys is None:
            keys = self.clean_PCs.keys()

        # form PCs
        self.form_PCs(keys=keys, overwrite=overwrite)

        # setup containers
        if not hasattr(self, "clean_pc_model"):
            self.clean_pc_model = DataContainer({})
        if not hasattr(self, "data_pc_model"):
            self.data_pc_model = DataContainer({})
        if not hasattr(self, "data_pc_sub"):
            self.data_pc_sub = DataContainer({})

        # iterate over keys
        for k in keys:
            if k not in self.clean_PCs:
                echo("{} not in clean_PCs, skipping...".format(k), verbose=verbose)
                continue
            if k in self.clean_pc_model and (not overwrite or increment):
                echo("{} in clean_pc_model and overwrite == increment == False, skipping...".format(k), verbose=verbose)
                continue

            # sum PCs
            model_fft = sum_principal_components(self.clean_svals[k], self.clean_PCs[k], Nkeep=Nkeep)
            if k not in self.clean_pc_model:
                self.clean_pc_model[k] = model_fft
            elif k in self.clean_pc_model and increment:
                self.clean_pc_model[k] += model_fft
            else:
                self.clean_pc_model[k] = model_fft

    def subtract_model(self, keys=None, data=None, overwrite=False, verbose=True):
        """
        iFFT clean_pc_model (with necessary zero-padding) and then
        subtract from original self.data container.

        Parameters
        ----------
        keys : list of tuples
            List of baseline-pol DataContainer tuples to operate on.

        data : datacontainer
            Object to pull data from in forming data-model residual.
            Default is self.data.

        overwrite : bool
            If True, overwrite output data if it exists.

        verbose : bool
            If True, report feedback to stdout.
        """
        # get keys
        if keys is None:
            keys = self.clean_pc_model.keys()

        if not hasattr(self, 'data_pc_model'):
            self.data_pc_model = DataContainer({})
        if not hasattr(self, 'data_pc_sub'):
            self.data_pc_sub = DataContainer({})

        # get data
        if data is None:
            data = self.data

        # iterate over keys
        for k in keys:
            if k in self.data_pc_model and not overwrite:
                echo("{} in data_pc_model and overwrite==False, skipping...".format(k), verbose=verbose)
                continue

            # get fft of model
            model_fft = self.clean_pc_model[k]

            # zero-pad if necessary
            if self.clean_edgecut > 0:
                z = np.zeros((model_fft.shape[0], self.clean_edgecut), dtype=model_fft.dtype)
                model_fft = np.concatenate([z, model_fft, z], axis=1)

            # ifft to get to data space
            model = np.fft.fftshift(np.fft.ifft(model_fft, axis=1), axes=1)

            # subtract from data
            self.data_pc_model[k] = model
            self.data_pc_sub[k] = data[k] - model

    def interp_u(self, u=None, uflags=None, keys=None, overwrite=False, verbose=True,
                 mode='gpr', gp_len=600, gp_nl=0.1, optimizer=None):
        """
        Interpolate u modes along time. This can cover
        flagged gaps, can interpolate a time-averaged u-mode
        onto the original full time resolution, and can also
        act as a smoothing process for noisey u-modes. Pulls
        from self.clean_u and inserts into self.clean_u_interp

        Parameters
        ----------
        u : DataContainer
            Object to pull target u from. Default is self.clean_u

        uflags : DataContainer
            Object to pull target u flags from. Default is self.clean_u_flags

        keys : list of tuples
            List of baseline-pol DataContainer tuples to operate on.

        overwrite : bool
            If True, overwrite output data if it exists.

        verbose : bool
            If True, report feedback to stdout.

        mode : str
            Interpolation mode. Options=['gpr']

        """
        if not hasattr(self, 'clean_u_interp'):
            self.clean_u_interp = DataContainer({})

        if u is None:
            u = self.clean_u
        if uflags is None:
            uflags = self.clean_u_flags
        times = np.asarray(u.times)

        # get keys
        if keys is None:
            keys = u.keys()

        # setup X predict
        Xpredict = self.times[:, None] * 24 * 3600
        Xmean = np.median(Xpredict)
        Xpredict -= Xmean

        # iterate over keys
        for k in keys:
            # check overwrite
            if k in self.clean_u_interp and overwrite == False:
                echo("{} in clean_u_interp and overwrite == False, skipping...".format(k), verbose=verbose)
                continue

            if mode == 'gpr':
                # setup GP kernel
                kernel = 1**2 * gp.kernels.RBF(length_scale=gp_len) + gp.kernels.WhiteKernel(noise_level=gp_nl)
                GP = gp.GaussianProcessRegressor(kernel=kernel, optimizer=optimizer, normalize_y=True)

                # setup regression data: get unflagged data
                X = times[~uflags[k], None] * 3600 * 24 - Xmean
                y = u[k][~uflags[k], :]

                # fit gp
                GP.fit(X, y)

                # predict across all times
                ypredict = GP.predict(Xpredict)

                # append
                self.clean_u_interp[k] = ypredict

            else:
                raise ValueError("didn't recognize interp mode {}".format(mode))

    def project_autos_onto_u(self, keys, auto_keys, u=None, umode=0, auto_delay=0,
                             overwrite=False, verbose=True):
        """
        Project the time dependent clean_fft of an autocorrelation
        at a specified delay onto the specified u-mode, replacing
        the u-mode with the projected autocorrelation. Inserts
        results into self.clean_u_interp

        Parameters
        ----------
        keys : list of tuples
            List of baseline-pol DataContainer tuples to operate on.

        auto_keys : list of tuples
            List of autocorr-pol tuples matching input keys in length
            to pull auto-correlation data from in self.clean_fft.
            Optionally, each auto_key can be itself a list of
            keys that will be used as separate basis functions
            to project onto the u modes. Example: for a given
            cross-corr key, you can provide both auto-corr keys.

        u : DataContainer
            Object to pull u modes from. Default is self.clean_u.

        umode : int
            Index of the u-mode to project auto-correlation onto.
            All other u-mode indices are copied as-is into clean_u_interp.
            This should almost always be set to zero.

        auto_dly : float
            Delay in nanosec of autocorrelation to project onto the u-mode.
            This should almost always be set to zero.

        overwrite : bool
            If True, overwrite output data if it exists.

        verbose : bool
            If True, report feedback to stdout.
        """
        # type check
        assert len(keys) == len(auto_keys), "len(keys) must equal len(auto_keys)"

        # get u
        if u is None:
            u = self.clean_u

        # get dly index
        select = np.argmin(np.abs(self.clean_dlys - auto_dly))

        if not hasattr(self, 'clean_u_interp'):
            self.clean_u_interp = DataContainer({})
            self.clean_u_interp.times = u.times
        if not hasattr(self, 'clean_u_interp_flags'):
            self.clean_u_interp_flags = DataContainer({})
            self.clean_u_interp_flags.times = u.times

        # iterate over keys
        for k, ak in zip(keys, auto_keys):
            # check overwrite
            if k in self.clean_u_interp and not overwrite:
                echo("{} exists in clean_u_interp and overwrite == False, skipping...".format(k), verbose=verbose)
                continue

            if k not in u or ak not in self.clean_fft:
                echo("{} not in u container or {} not in self.clean_fft, skipping...".format(k, ak), verbose=verbose)
                continnue

            # get u-mode
            _u = u[k][:, umode]

            # get autocorr
            if isinstance(ak, list):
                A = np.array([self.clean_fft[_ak][:, select] for _ak in ak]).T
            else:
                A = np.array([self.clean_fft[ak][:, select]]).T

            # form least squares estimate
            f = np.min(RF.clean_flags[k] + RF.clean_flags[ak], axis=1)
            W = np.eye(len(_u)) * (~f).astype(np.float)
            xhat = np.asarray(np.linalg.pinv(A.T.dot(W).dot(A)).dot(A.T.dot(W).dot(_u.real)), dtype=np.complex) \
                    + 1j * np.linalg.pinv(A.T.dot(W).dot(A)).dot(A.T.dot(W).dot(_u.imag))
            proj_u = A.dot(xhat)

            self.clean_u_interp[k] = proj_u
            self.clean_u_interp_flags[k] = f


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

    # average across time
    w = (~f).astype(np.float)
    if timeavg:
        wsum = np.sum(w, axis=0, keepdims=True).clip(1e-10, np.inf)
        d = np.sum(d * w, axis=0, keepdims=True) / wsum
        w = wsum
        f = np.isclose(wsum, 1e-10)

    # delay filter
    kwargs = {}
    if taper == 'tukey':
        kwargs['alpha'] = alpha
    mdl, res, info = delay_filter.delay_filter(d, w, 0., dnu / 1e9, min_dly=dly_cut, skip_wgt=skip_wgt,
                                               window=taper, tol=tol, maxiter=maxiter, gain=gain, **kwargs)
    dlys = np.fft.fftshift(np.fft.fftfreq(d.shape[1], d=dnu)) * 1e9

    return mdl, f, res, dlys, info


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


def echo(message, verbose=True):
    if verbose:
        print(message)
