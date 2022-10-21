# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

import numpy as np
import scipy
from copy import deepcopy
import warnings
import argparse
import pyuvdata
from collections.abc import Iterable
import hera_filters

try:
    import aipy
    AIPY = True
except ImportError:
    AIPY = False

from . import io
from . import utils
from . import flag_utils
from .vis_clean import truncate_flagged_edges, restore_flagged_edges
from .noise import interleaved_noise_variance_estimate


def single_iterative_fft_dly(gains, wgts, freqs, conv_crit=1e-5, maxiter=100):
    '''Iteratively find a single best-fit delay for a given gain waterfall for all times.

    Arguments:
        gains: ndarray of shape=(Ntimes,Nfreqs) of complex calibration solutions
        wgts: ndarray of shape=(Ntimes,Nfreqs) of real linear multiplicative weights
            For the purposes of this function, wghts <= 0 are considered flags.
        freqs: ndarray of frequency channels in Hz
        conv_crit: convergence criterionf or relative change in the rephasor
        maxiter: maximum number of

    Returns:
        tau: float, single best fit delay in s
    '''
    if np.sum(wgts) == 0:  # if all flagged, return 0 delay
        return 0.0

    df = np.median(np.diff(freqs))
    gains = deepcopy(gains)
    gains[wgts <= 0] = np.nan
    avg_gains = np.nanmean(gains, axis=0, keepdims=True)
    unflagged_channels = np.nonzero(np.isfinite(avg_gains[0]))
    unflagged_range = slice(np.min(unflagged_channels), np.max(unflagged_channels) + 1)
    avg_gains[~np.isfinite(avg_gains)] = 0

    taus = []
    for i in range(maxiter):
        tau, _ = utils.fft_dly(avg_gains[:, unflagged_range], df)
        taus.append(tau)

        rephasor = np.exp(-2.0j * np.pi * tau[0][0] * freqs)
        avg_gains *= rephasor
        if np.mean(np.abs(rephasor - 1.0)) < conv_crit:
            break

    return np.sum(taus)


def dpss_filters(freqs, times, freq_scale=10, time_scale=1800, eigenval_cutoff=1e-9):
    """Generate a set of 2D discrete prolate spheroidal sequence (DPSS) filters
    to filter calibration solutions along the time and frequency axes simultaneously.

    Arguments:
        freqs: ndarray of frequency channels in Hz
        times: ndarray of shape=(Ntimes) of Julian dates as floats in units of days
        freq_scale: frequency scale in MHz to use for the low-pass filter. freq_scale^-1 corresponds
            to the half-width (i.e. the width of the positive part) of the region in fourier
            space, symmetric about 0, that is retained after filtering.
            Note that freq_scale is in MHz while freqs is in Hz.
        time_scale: time scale in seconds. Defined analogously to freq_scale.
            Note that time_scale is in seconds, times is in days.
        eigenval_cutoff: sinc_matrix eigenvalue cutoff to use for included dpss modes.
            Only used when the filtering method is 'DPSS'

    Returns:
        time_filters: DPSS filtering vectors along the time axis, ndarray of size (Ntimes, N_time_vectors)
        freq_filters: DPSS filtering vectors along the frequency axis, (Nfreqs, N_freq_vectors)
    """
    delay_scale = (freq_scale * 1e6) ** -1  # Puts it in seconds
    fringe_scale = (time_scale) ** -1  # fringe scale in Hz
    time_in_seconds = (times - times.min()) * 60 * 60 * 24  # time array in seconds

    dpss_windows = []

    for x0, fw in zip([time_in_seconds, freqs], [fringe_scale, delay_scale]):
        nf = x0.shape[0]
        df = np.abs(x0[1] - x0[0])
        W = nf * fw * df

        # Estimate the number of eigenvalues > eigenval_cutoff - Slepian 1978 + Karnik 2020
        Nw = 2 * W + 2 / np.pi ** 2 * np.log(4 * nf) * np.log(4 / (eigenval_cutoff * (1 - eigenval_cutoff)))

        # Generate DPSS vectors and keep vectors with eigenvalues greater than eigenval_cutoff
        windows, eigvals = hera_filters.dspec.windows.dpss(nf, W, int(min(Nw, nf)), return_ratios=True)
        windows = windows[eigvals > eigenval_cutoff].T
        dpss_windows.append(windows)

    return dpss_windows[0], dpss_windows[1]


def solve_2D_DPSS(gains, weights, time_filters, freq_filters, XTXinv=None):
    """
    Filters gain solutions by solving the weighted linear least squares problem
    for a design matrix that can be factored by a kronecker product in the following way

         y = X b = (A outer B) b

    where A and B are DPSS vector matrices for the time and frequency axes. More memory
    and computationally efficient than computing the design matrix from the time and frequency
    filters for most input matrix sizes.

    Arguments:
        gains: ndarray of shape=(Ntimes,Nfreqs) of complex calibration solutions to filter
        weights: ndarray of shape (Ntimes, Nfreqs) of calibration flags
        time_filters: DPSS filtering vectors along the time axis, ndarray of size (Ntimes, N_time_vectors)
            obtained from hera_cal.smooth_cal.dpss_filters
        freq_filters: DPSS filtering vectors along the frequency axis, (Nfreqs, N_freq_vectors)
            obtained from hera_cal.smooth_cal.dpss_filters
        XTXinv: Matrix of (X^T W X)^{-1} for the input DPSS filters and weights. Useful for filtering
            many gain grids with similar flagging patterns. np.ndarray of shape (N_time_vectors * N_freq_vectors,
            N_time_vectors * N_freq_vectors)

    Returns:
        filtered: filtered gain grids from least squares fit, np.ndarray with the
            same shape as the input gain grid
        info: dictionary containing fit components and XTXinv
    """
    # If (X^T W X)^-1 is not precalculated, calculate it
    if XTXinv is None:
        # Make sure data types between weights and filters are compatible for einsum optimization
        weights = weights.astype(time_filters.dtype)

        # Use einsum to calculate (X^T W X) in a memory efficient way
        XTX = np.einsum(
            "ij,kl,jl,jm,ln->ikmn", np.transpose(time_filters), np.transpose(freq_filters),
            weights, time_filters, freq_filters, optimize=True
        )
        ncomps = time_filters.shape[1] * freq_filters.shape[1]
        XTX = np.reshape(XTX, (ncomps, ncomps))
        XTXinv = np.linalg.pinv(XTX)

    # Calculate X^T W y using the property (A \otimes B) vec(y) = (A Y B)
    XTWy = np.ravel(np.transpose(time_filters) @ (gains * weights) @ freq_filters)

    # Calculate beta
    beta = np.reshape(XTXinv @ XTWy, (time_filters.shape[1], freq_filters.shape[1]))

    # Produce an estimate of the filtered gains
    filtered = time_filters @ beta @ np.transpose(freq_filters)

    # Dictionary for storing fitting information
    info = {"beta": beta, "XTXinv": XTXinv}

    return filtered, info


def filter_1d(gains, wgts, xvals, filter_scale=None, skip_wgt=0.1, ax='freq',
              mode='clean', tol=1e-6, skip_flagged_edges=False, cache=None, **filter_kwargs):
    '''Frequency-filter calibration solutions on a given scale in MHz using hera_filters.dspec.high_pass_fourier_filter.
    Before filtering, removes a single average delay, then puts it back in after filtering.

    Arguments:
        gains: ndarray of shape=(Ntimes,Nfreqs) of complex calibration solutions to filter
        wgts: ndarray of shape=(Ntimes,Nfreqs) of real linear multiplicative weights
        freqs: ndarray of frequency channels in Hz
        filter_scale: frequency scale in MHz to use for the low-pass filter. filter_scale^-1 corresponds
            to the half-width (i.e. the width of the positive part) of the region in fourier
            space, symmetric about 0, that is filtered out.
        skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
            filtered is left unchanged and info is {'skipped': True} for that time.
            Only works properly when all weights are all between 0 and 1.
        ax: str either 'freq' or 'time' specifying which axis to filter.
        mode: deconvolution method to use. See hera_filters.dspec.fourier_filter for full list of supported modes.
              examples include 'dpss_leastsq', 'clean'.
        skip_flagged_edges: bool, optional
            if True, truncate flagged edges before filtering.
        filter_kwargs : any keyword arguments for the filtering mode being used.
        See vis_clean.fourier_filter or hera_filters.dspec.fourier_filter for a full description.
    Returns:
        filtered: filtered gains, ndarray of shape=(Ntimes,Nfreqs)
        info: info object from hera_filters.dspec.high_pass_fourier_filter
    '''
    if filter_scale is None:
        if ax == 'time':
            filter_scale = 1800.
        else:
            filter_scale = 10.
    if ax == 'freq':
        filter_size = (filter_scale * 1e6) ** -1  # Puts it in MHz
        dly = single_iterative_fft_dly(gains, wgts, xvals)  # dly in s
    else:
        dly = 0.
        filter_size = (filter_scale) ** -1  # units of inverse days
    if mode == 'DPSS' or mode == 'dpss_leastsq':
        filter_kwargs['suppression_factors'] = [tol]
        mode = 'dpss_leastsq'
        # Check to see if the grid size changes after removing the flagged edges
        if skip_flagged_edges:
            xin, din, win, edges, chunks = truncate_flagged_edges(gains, wgts, xvals, ax=ax)
        else:
            xin = xvals
            win = wgts
            din = gains
    else:
        xin = xvals
        win = wgts
        din = gains
        filter_kwargs['tol'] = tol
    if ax == 'freq':
        rephasor = np.exp(-2.0j * np.pi * dly * xin)
    else:
        rephasor = 1.
    din = din * rephasor
    fdim = {'freq': 1, 'time': 0}[ax]
    filtered, res, info = hera_filters.dspec.fourier_filter(x=xin, data=din, wgts=win, mode=mode, filter_centers=[0.],
                                                            skip_wgt=skip_wgt, filter_half_widths=[filter_size],
                                                            filter_dims=fdim, cache=cache,
                                                            **filter_kwargs)
    # put back in unfilted values if skip_wgt is triggered
    filtered /= rephasor
    if skip_flagged_edges:
        filtered = restore_flagged_edges(filtered, chunks, edges, ax=ax)

    if ax == 'freq':
        for i in info['status']['axis_1']:
            if info['status']['axis_1'][i] == 'skipped':
                filtered[i, :] = gains[i, :]
    else:
        for i in info['status']['axis_0']:
            if info['status']['axis_0'][i] == 'skipped':
                filtered[:, i] = gains[:, i]

    return filtered, info


def time_kernel(nInt, tInt, filter_scale=1800.0):
    '''Build time averaging gaussian kernel.

    Arguments:
        nInt: number of integrations to be filtered
        tInt: length of integrations (seconds)
        filter_scale: float in seconds of FWHM of Gaussian smoothing kernel in time

    Returns:
        kernel: numpy array of length 2 * nInt + 1
    '''
    kernel_times = np.append(-np.arange(0, nInt * tInt + tInt / 2, tInt)[-1:0:-1], np.arange(0, nInt * tInt + tInt / 2, tInt))
    filter_std = filter_scale / (2 * (2 * np.log(2))**.5)
    kernel = np.exp(-kernel_times**2 / 2 / (filter_std)**2)
    return kernel / np.sum(kernel)


def time_filter(gains, wgts, times, filter_scale=1800.0, nMirrors=0):
    '''Time-filter calibration solutions with a rolling Gaussian-weighted average. Allows
    the mirroring of gains and wgts and appending the mirrored gains and wgts to both ends,
    ensuring temporal smoothness of the rolling average.

    Arguments:
        gains: ndarray of shape=(Ntimes,Nfreqs) of complex calibration solutions to filter
        wgts: ndarray of shape=(Ntimes,Nfreqs) of real linear multiplicative weights
        times: ndarray of shape=(Ntimes) of Julian dates as floats in units of days
        filter_scale: float in seconds of FWHM of Gaussian smoothing kernel in time
        nMirrors: Number of times to reflect gains and wgts (each one increases nTimes by 3)

    Returns:
        conv_gains: gains conolved with a Gaussian kernel in time
    '''
    padded_gains, padded_wgts = deepcopy(gains), deepcopy(wgts)
    nBefore = 0
    for n in range(nMirrors):
        nBefore += (padded_gains[1:, :]).shape[0]
        padded_gains = np.vstack((np.flipud(padded_gains[1:, :]), gains, np.flipud(padded_gains[:-1, :])))
        padded_wgts = np.vstack((np.flipud(padded_wgts[1:, :]), wgts, np.flipud(padded_wgts[:-1, :])))

    nInt, nFreq = padded_gains.shape
    conv_gains = padded_gains * padded_wgts
    conv_weights = padded_wgts
    kernel = time_kernel(nInt, np.median(np.diff(times)) * 24 * 60 * 60, filter_scale=filter_scale)
    for i in range(nFreq):
        conv_gains[:, i] = scipy.signal.convolve(conv_gains[:, i], kernel, mode='same')
        conv_weights[:, i] = scipy.signal.convolve(conv_weights[:, i], kernel, mode='same')
    conv_gains /= conv_weights
    conv_gains[np.logical_not(np.isfinite(conv_gains))] = 0
    return conv_gains[nBefore: nBefore + len(times), :]


def time_freq_2D_filter(gains, wgts, freqs, times, freq_scale=10.0, time_scale=1800.0,
                        tol=1e-09, filter_mode='rect', maxiter=100, window='tukey', method='CLEAN',
                        dpss_vectors=None, XTXinv=None, eigenval_cutoff=1e-9, skip_flagged_edges=True,
                        **win_kwargs):
    '''Filter calibration solutions in both time and frequency simultaneously. First rephases to remove
    a time-average delay from the gains, then performs the low-pass 2D filter in time and frequency,
    then puts back in the delay rephasor. Uses aipy.deconv.clean to account for weights/flags.

    Arguments:
        gains: ndarray of shape=(Ntimes,Nfreqs) of complex calibration solutions to filter
        wgts: ndarray of shape=(Ntimes,Nfreqs) of real linear multiplicative weights
        freqs: ndarray of frequency channels in Hz
        times: ndarray of shape=(Ntimes) of Julian dates as floats in units of days
        freq_scale: frequency scale in MHz to use for the low-pass filter. freq_scale^-1 corresponds
            to the half-width (i.e. the width of the positive part) of the region in fourier
            space, symmetric about 0, that is retained after filtering.
            Note that freq_scale is in MHz while freqs is in Hz.
        time_scale: time scale in seconds. Defined analogously to freq_scale.
            Note that time_scale is in seconds, times is in days.
        tol: CLEAN algorithm convergence tolerance (see aipy.deconv.clean)
        filter_mode: either 'rect' or 'plus':
            'rect': perform 2D low-pass filter, keeping modes in a small rectangle around delay = 0
                    and fringe rate = 0
            'plus': produce a separable calibration solution by only keeping modes with 0 delay,
                    0 fringe rate, or both. Only used when method is 'CLEAN'
        window: window function for filtering applied to the filtered axis. Only used with when method used is 'CLEAN'.
            See aipy.dsp.gen_window for options.
        maxiter: Maximum number of iterations for aipy.deconv.clean to converge.
        method: Algorithm used to smooth calibration solutions. Either 'CLEAN' or 'DPSS':
            'CLEAN': uses the CLEAN algorithm to smooth calibration solutions
            'DPSS': uses discrete prolate spheroidal sequences (DPSS) to filter calibration solutions
        dpss_vectors: Tuple of 2 1D DPSS filters, one for the time axis and one for the frequency axis
            that form the least squares design matrix, X, when the outer product of the two is taken.
            If dpss_vectors is not provided, one will be calculated using smooth_cal.dpss_filters and the
            time and frequency scale. Only used when the method is 'DPSS'
        XTXinv: Matrix operation of (X^T W X)^{-1} where X are the set of 2D DPSS vectors and W is the
            wgts grid. Useful when filtering many gain grids where wgts grid is the same between two or more
            gain solutions. Only used when the filtering method is 'DPSS'
        eigenval_cutoff: sinc_matrix eigenvalue cutoffs to use for included dpss modes.
            Only used when the filtering method is 'DPSS'
        skip_flagged_edges : bool, optional
            if True, do not filter over flagged edge times/freqs (filter over sub-region). Only used when method used is 'DPSS'.
            Default is True
        win_kwargs : any keyword arguments for the window function selection in aipy.dsp.gen_window.
            Currently, the only window that takes a kwarg is the tukey window with a alpha=0.5 default.

    Returns:
        filtered: filtered gains, ndarray of shape=(Ntimes,Nfreqs)
        info: dictionary of metadata from aipy.deconv.clean or DPSS, depending on the filter method
            chosen.
    '''
    df = np.median(np.diff(freqs))
    dt = np.median(np.diff(times)) * 24.0 * 3600.0  # in seconds
    delays = np.fft.fftfreq(freqs.size, df)
    fringes = np.fft.fftfreq(times.size, dt)
    delay_scale = (freq_scale * 1e6)**-1  # Puts it in seconds
    fringe_scale = (time_scale)**-1  # in Hz

    # Build rephasor to take out average delay
    dly = single_iterative_fft_dly(gains, wgts, freqs)  # dly in seconds
    rephasor = np.exp(-2.0j * np.pi * dly * freqs)

    if method == 'DPSS' or method == 'dpss_leastsq':
        info = {}
        if skip_flagged_edges:
            xout, gout, wout, edges, chunks = truncate_flagged_edges(gains * rephasor, wgts, (times, freqs), ax='both')

        else:
            gout = deepcopy(gains * rephasor)
            wout = deepcopy(wgts)
            xout = (times, freqs)

        if filter_mode == 'rect':
            # Generate filters if not provided
            if dpss_vectors is None:
                dpss_vectors = dpss_filters(
                    freqs=xout[1], times=xout[0], freq_scale=freq_scale, time_scale=time_scale,
                    eigenval_cutoff=eigenval_cutoff
                )

            time_filters, freq_filters = dpss_vectors

            # Filter gain solutions
            filtered, fit_info = solve_2D_DPSS(
                gains=gout, weights=wout, time_filters=time_filters, freq_filters=freq_filters, XTXinv=XTXinv
            )

            if skip_flagged_edges:
                ((tstart, tend),), ((fstart, fend),) = edges
                # Create a mask to fill-in flagged region
                mask = np.ones(gains.shape, dtype=bool)
                mask[tstart:gains.shape[0] - tend, fstart:gains.shape[1] - fend] = False
                # Restore flagged region with zeros and fill-in with original data
                filtered = restore_flagged_edges(filtered, chunks, edges, ax='both')
                filtered[mask] = gains[mask]

            # Store design matrices and XTXinv for computational speed-up
            info['XTXinv'] = fit_info['XTXinv']
            info['dpss_vectors'] = dpss_vectors

        elif filter_mode == 'plus':
            raise NotImplementedError("filter_mode 'plus' only implemented for CLEAN")

        else:
            raise ValueError("DPSS mode {} not recognized. Must be 'rect' or 'plus'.".format(filter_mode))

    elif method == 'CLEAN':
        assert AIPY, "You need aipy to use this function"
        # Build fourier space image and kernel for deconvolution
        window = aipy.dsp.gen_window(len(freqs), window=window, **win_kwargs)
        image = np.fft.ifft2(gains * rephasor * wgts * window)
        kernel = np.fft.ifft2(wgts * window)

        # set up "area", the set of Fourier modes that are allowed to be non-zero in the CLEAN
        if filter_mode == 'rect':
            area = np.outer(np.where(np.abs(fringes) < fringe_scale, 1, 0),
                            np.where(np.abs(delays) < delay_scale, 1, 0))
        elif filter_mode == 'plus':
            area = np.zeros(image.shape, dtype=int)
            area[0] = np.where(np.abs(delays) < delay_scale, 1, 0)
            area[:, 0] = np.where(np.abs(fringes) < fringe_scale, 1, 0)
        else:
            raise ValueError("CLEAN mode {} not recognized. Must be 'rect' or 'plus'.".format(filter_mode))

        # perform deconvolution
        CLEAN, info = aipy.deconv.clean(image, kernel, tol=tol, area=area, stop_if_div=False, maxiter=maxiter)
        filtered = np.fft.fft2(CLEAN + info['res'] * area)
        del info['res']  # this matches the convention in hera_filters.dspec.high_pass_fourier_filter

    else:
        raise ValueError("Filter method {} not recognized. Must be 'CLEAN' or 'DPSS'".format(method))

    return filtered / rephasor, info


def flag_threshold_and_broadcast(flags, freq_threshold=0.35, time_threshold=0.5, ant_threshold=0.5):
    '''Thesholds and broadcast flags along three axes. First, it loops through frequency and time thresholding
    until no new flags are found. Then it does antenna thresholding.

    Arguments:
        flags: dictionary mapping antenna keys (e.g. (0, 'Jxx')) to Ntimes x Nfreq flag waterfalls.
            Modified in place.
        freq_threshold: float. Finds the times that flagged for all antennas at a single channel but not flagged
            for all channels. If the ratio of of such times for a given channel compared to all times that are not
            completely flagged is greater than freq_threshold, flag the entire channel for all antennas.
            Setting this to 1.0 means no additional flagging.
        time_threshold: float. Finds the channels that flagged for all antennas at a single time but not flagged
            for all times. If the ratio of of such channels for a given time compared to all channels that are not
            completely flagged is greater than time_threshold, flag the entire time for all antennas.
            Setting this to 1.0 means no additional flagging.
        ant_threshold: float. If, after time and freq thesholding and broadcasting, an antenna is left unflagged
            for a number of visibilities less than ant_threshold times the maximum among all antennas, flag that
            antenna for all times and channels. Setting this to 1.0 means no additional flagging.
    '''
    # figure out which frequencies and times are flagged for all antennas
    all_flagged = np.mean(1.0 - np.array(list(flags.values())), axis=0) == 0
    sum_flagged = 0

    # apply thresholding for times and frequencies and keep looping until no new flags are produced
    while np.sum(all_flagged) > sum_flagged:
        sum_flagged = np.sum(all_flagged)
        # consider times that are not flagged for all chans, then flag chans that are flagged too often
        flagged_chans = np.mean(all_flagged, axis=0) == 1
        all_flagged[np.mean(all_flagged[:, ~flagged_chans], axis=1) > freq_threshold, :] = True
        # consider chans that are not flagged for all times, then flag times that are flagged too often
        flagged_times = np.mean(all_flagged, axis=1) == 1
        all_flagged[:, np.mean(all_flagged[~flagged_times, :], axis=0) > time_threshold] = True

    # apply flags to all antennas
    for ant in flags.keys():
        flags[ant] |= all_flagged

    # flag antennas with fewer unflagged visibilities than the maximum times 1 - ant_threshold
    most_unflagged = np.max([np.sum(~f) for f in flags.values()])
    for ant in flags.keys():
        if np.sum(~flags[ant]) < (1.0 - ant_threshold) * most_unflagged:
            flags[ant] |= np.ones_like(flags[ant])  # flag the whole antenna


def pick_reference_antenna(gains, flags, freqs, per_pol=True):
    '''Pick a refrence antenna that has the minimum number of per-antenna flags and produces
    the least noisy phases on other antennas when used as a reference antenna.

    Arguments:
        gains: Dictionary mapping antenna keys to gain waterfalls.
        flags: dictionary mapping antenna keys to flag waterfall where True means flagged
        freqs: ndarray of frequency channels in Hz

    Returns:
        refant: key(s) of the antenna(s) with the minimum number of flags and the least noisy phases
            if per_pol: dictionary mapping gain polarizations string to ant-pol tuples
            else: ant-pol tuple e.g. (0, 'Jxx')
    '''
    # compute delay for all gains to flatten them as well as possible. Average over times.
    rephasors = {}
    for ant in gains.keys():
        wgts = np.array(~(flags[ant]), dtype=float)
        dly = single_iterative_fft_dly(gains[ant], wgts, freqs)
        rephasors[ant] = np.exp(-2.0j * np.pi * dly * freqs)

    def narrow_refant_candidates(candidates):
        '''Helper function for comparing refant candidates to another another looking for the one with the
        least noisy phases in other antennas when its the reference antenna (after taking out delays)'''
        median_angle_noise_as_refant = {}
        for ref in candidates:
            refant_rephasor = np.abs(gains[ref] * rephasors[ref]) / (gains[ref] * rephasors[ref])
            angle_noise = [interleaved_noise_variance_estimate(np.angle(gains[ant] * rephasors[ant] * refant_rephasor),
                           kernel=[[-.5, 1, -.5]])[~(flags[ant] | flags[ref])] for ant in candidates]
            median_angle_noise_as_refant[ref] = np.median(angle_noise)
        return sorted(median_angle_noise_as_refant, key=median_angle_noise_as_refant.__getitem__)[0]

    # loop over pols (if per_pol)
    refant = {}
    pols = set([ant[1] for ant in gains])
    for pol in (pols if per_pol else [pols]):
        # pick antennas with the mininum number of flags
        flags_per_ant = {ant: np.sum(f) for ant, f in flags.items() if ant[1] in pol}
        refant_candidates = sorted([ant for ant, nflags in flags_per_ant.items()
                                    if nflags == np.min(list(flags_per_ant.values()))])
        while len(refant_candidates) > 1:  # loop over groups of 3 (the smallest, non-trivial size)
            # compare phase noise imparted by reference antenna candidates on two other reference antenna candidates
            refant_candidates = [narrow_refant_candidates(candidates) for candidates in [refant_candidates[i:i + 3]
                                 for i in range(0, len(refant_candidates), 3)]]
        if not per_pol:
            return refant_candidates[0]
        else:
            refant[pol] = refant_candidates[0]

    return refant


def rephase_to_refant(gains, refant, flags=None, propagate_refant_flags=False):
    '''Rephase all gains in place so that the phase of the reference antenna is 0 for all times and frequencies.

    Arguments:
        gains: Dictionary mapping antenna keys to gain waterfalls. Modified in place.
        refant: Antenna key of antenna to make the reference antenna (or dictionary mapping pols to keys)
        flags: Optional dictionary mapping antenna keys to flag waterfall.
            Used only to verify that all gains are flagged where the refant is flagged.
        propagate_refant_flags: If True and flags is not None, update flags so that all antennas are flagged
            at the specific frequencies and times that the reference antenna is also flagged. If False and
            there exists times and frequencies where the reference antenna is flagged but another antenna
            is not flagged, a ValueError will be raised.
    '''
    for pol, ref in (refant.items() if not isinstance(refant, tuple) else [(None, refant)]):
        refant_phasor = gains[ref] / np.abs(gains[ref])
        for ant in gains.keys():
            if ((pol is None) or (ant[1] == pol)):
                if flags is not None:
                    if propagate_refant_flags:
                        flags[ant][flags[ref]] = True
                    elif np.any(flags[ref][np.logical_not(flags[ant])]):
                        raise ValueError('The chosen reference antenna', refant, 'is flagged in at least one place where antenna',
                                         ant, 'is not, so automatic reference antenna selection has failed.')
                gains[ant] = gains[ant] / refant_phasor


def build_time_blacklist(time_grid, time_blacklists=[], lst_blacklists=[], lat_lon_alt_degrees=None, telescope_name='HERA'):
    '''Converts pairs of bounds on blacklisted times/LSTs into a boolean array of blacklisted times.

    Arguments:
        time_grid: numpy array of times in Julian Day
        time_blacklists: list of pairs of times in Julian Day bounding (inclusively) regions in time that are to be marked
            as True in the time_blacklist_array
        lst_blacklists:  list of pairs of LSTs in hours bounding (inclusively) regions of LST that are to be marked as True
            in the time_blacklist_array. Regions crossing the 24h brach cut, e.g. [(23, 1)], are allowed.
        lat_lon_alt_degrees: length 3 array of telescope location in degreees and altitude in meters. Only used to convert
            times to LSTs if lst_blacklists is not empty
        telescope_name: string name of telescope. Only used if lst_blacklists is not empty and lat_lon_alt_degrees
            is not None. Currently, only "HERA" will work since it's position is hard-coded in this module.

    Returns:
        time_blacklist_array: boolean array with the same shape as time_grid in which blacklisted integrations are True'''

    time_blacklist_array = np.zeros(len(time_grid), dtype=bool)

    # Calculate blacklisted times
    if len(time_blacklists) > 0:
        for bounds in time_blacklists:
            assert isinstance(bounds, Iterable) and len(bounds) == 2, 'time_blacklists must be list of pairs of bounds'
            assert bounds[0] <= bounds[1], 'time_blacklist bounds must be in chronological order'
            time_blacklist_array[(time_grid >= bounds[0]) & (time_grid <= bounds[1])] = True

    # Calculate blacklisted LSTs
    if len(lst_blacklists) > 0:
        # If lat_lon_alt is not specified, try to infer it from the telescope name, which calfits files generally carry around
        if lat_lon_alt_degrees is None:
            if telescope_name.upper() == 'HERA':
                lat_lon_alt_degrees = np.array(pyuvdata.utils.LatLonAlt_from_XYZ(utils.HERA_TELESCOPE_LOCATION))
                lat_lon_alt_degrees *= [180 / np.pi, 180 / np.pi, 1]
            else:
                raise NotImplementedError(f'No known position for telescope {telescope_name}. lat_lon_alt_degrees must be specified.')

        # calculate LST grid in hours from time grid and lat_lon_alt
        lst_grid = pyuvdata.utils.get_lst_for_time(time_grid, *lat_lon_alt_degrees) * 12 / np.pi

        # add blacklisted times from lst_blacklists
        for bounds in lst_blacklists:
            assert isinstance(bounds, Iterable) and len(bounds) == 2, 'lst_blacklists must be list of pairs of bounds'
            if bounds[0] < bounds[1]:
                time_blacklist_array[(lst_grid >= bounds[0]) & (lst_grid <= bounds[1])] = True
            else:  # the bounds span the 24 hours --> 0 hours branch cut
                time_blacklist_array[(lst_grid >= bounds[0]) | (lst_grid <= bounds[1])] = True

    return time_blacklist_array


def build_freq_blacklist(freqs, freq_blacklists=[], chan_blacklists=[]):
    '''Converts pairs of bounds on blacklisted frequencies/channels into a boolean array of blacklisted freqs.

    Arguments:
        freqs: numpy array of frequencies
        freq_blacklists: list of pairs of frequencies in the same units bounding (inclusively) the spectral regions that
            are to be marked as True in the freq_blacklist_array
        chan_blacklists: list of pairs of channel numbers bounding (inclusively) spectral regions that are to be marked
            as True in the freq_blacklist_array.

    Returns:
        freq_blacklist_array: boolean array with the same shape as freqs with blacklisted frequencies set to True'''

    freq_blacklist_array = np.zeros(len(freqs), dtype=bool)

    # Calculate blacklisted frequencies
    if len(freq_blacklists) > 0:
        for bounds in freq_blacklists:
            assert isinstance(bounds, Iterable) and len(bounds) == 2, 'freq_blacklists must be list of pairs of bounds'
            assert bounds[0] <= bounds[1], 'freq_blacklists bounds must be in ascending order'
            freq_blacklist_array[(freqs >= bounds[0]) & (freqs <= bounds[1])] = True

    # Calculate blacklisted channels
    if len(chan_blacklists) > 0:
        for bounds in chan_blacklists:
            assert isinstance(bounds, Iterable) and len(bounds) == 2, 'chan_blacklists must be list of pairs of bounds'
            assert bounds[0] <= bounds[1], 'chan_blacklists bounds must be in ascending order'
            freq_blacklist_array[(np.arange(len(freqs)) >= bounds[0]) & (np.arange(len(freqs)) <= bounds[1])] = True

    return freq_blacklist_array


def _build_wgts_grid(flag_grid, time_blacklist=None, freq_blacklist=None, blacklist_wgt=0.0):
    '''Builds a wgts_grid float array (Ntimes, Nfreqs) with 0s flagged or blacklisted data and 1s otherwise.'''
    wgts_grid = np.ones_like(flag_grid, dtype=float)
    if time_blacklist is not None:
        wgts_grid[time_blacklist, :] = blacklist_wgt
    if freq_blacklist is not None:
        wgts_grid[:, freq_blacklist] = blacklist_wgt
    wgts_grid[flag_grid] = 0.0
    return wgts_grid


class CalibrationSmoother():

    def __init__(self, calfits_list, flag_file_list=[], flag_filetype='h5', antflag_thresh=0.0, load_cspa=False, load_chisq=False,
                 time_blacklists=[], lst_blacklists=[], lat_lon_alt_degrees=None, freq_blacklists=[], chan_blacklists=[],
                 blacklist_wgt=0.0, pick_refant=False, freq_threshold=1.0, time_threshold=1.0, ant_threshold=1.0, verbose=False):
        '''Class for smoothing calibration solutions in time and frequency for a whole day. Initialized with a list of
        calfits files and, optionally, a corresponding list of flag files, which must match the calfits files
        one-to-one in time. This function sets up a time grid that spans the whole day with dt = integration time.
        Gains and flags are assigned to the nearest gridpoint using np.searchsorted. It is assumed that:
        1) All calfits and flag files have the same frequencies
        2) The flag times and calfits time map one-to-one to the same set of integrations
        Also contains functionality to broadcasting flags beyond certain thresholds and for automatically picking
        a reference antenna for the whole day.

        Arguments:
            calfits_list: list of string paths to calfits files containing calibration solutions and flags
            flag_file_list: list of string paths to files containing flags as a function of baseline, times
                and frequency. Must have all baselines for all times. Flags on baselines are broadcast to both
                antennas involved, unless either antenna is completely flagged for all times and frequencies.
            flag_filetype: filetype of flag_file_list to pass into io.load_flags. Either 'h5' (default) or legacy 'npz'.
            antflag_thresh: float, fraction of flagged pixels across all visibilities (with a common antenna)
                needed to flag that antenna gain at a particular time and frequency. antflag_thresh=0.0 is
                aggressive flag broadcasting, antflag_thresh=1.0 is conservative flag_broadcasting.
                Only used for converting per-baseline flags to per-antenna flags if flag_file_list is specified.
            load_cspa: if True, also save chisq_per_ant into self.cspa, analogously to self.gain_grids. Does not
                currently affect the smoothing functions, but is useful for interactive work.
            load_chisq: if True, also save chisq into self.chisq, analogously to self.gain_grids except that the
                keys are jpols, e.g. 'Jee' and not antennas. Does not currently affect the smoothing functions.
            time_blacklists: list of pairs of times in Julian Day bounding (inclusively) regions in time that are
                to receive 0 weight during smoothing, forcing the smoother to interpolate/extrapolate.
                N.B. Blacklisted times are not necessarily flagged.
            lst_blacklists:  list of pairs of LSTs in hours bounding (inclusively) regions of LST that are
                to receive 0 weight (by default) during smoothing, forcing the smoother to interpolate/extrapolate.
                Regions crossing the 24h brach cut are acceptable (e.g. [(23, 1)] blacklists two total hours).
                N.B. Blacklisted LSTS are not necessarily flagged.
            lat_lon_alt_degrees: length 3 list or array of latitude (deg), longitude (deg), and altitude (m) of
                the array. Only used to convert LSTs to JD times. If the telescope_name in the calfits file is 'HERA',
                this is not required.
            freq_blacklists: list of pairs of frequencies in Hz hours bounding (inclusively) spectral regions
                that are to receive 0 weight during smoothing, forcing the smoother to interpolate/extrapolate.
                N.B. Blacklisted frequencies are not necessarily flagged.
            chan_blacklists: list of pairs of channel numbers bounding (inclusively) spectral regions
                that are to receive 0 weight during smoothing, forcing the smoother to interpolate/extrapolate.
                N.B. Blacklisted channels are not necessarily flagged.
            blacklist_wgt: float weight to give to blacklisted times or frequencies. 0.0 weight will create
                problems at edge times and frequencies when using DPSS filtering.
            pick_refant: if True, automatically picks one reference anteanna per polarization. The refants chosen have the
                fewest total flags and causes the least noisy phases on other antennas when made the phase reference.
            freq_threshold: float. Finds the times that flagged for all antennas at a single channel but not flagged
                for all channels. If the ratio of of such times for a given channel compared to all times that are not
                completely flagged is greater than freq_threshold, flag the entire channel for all antennas.
                Default1.0 means no additional flagging.
            time_threshold: float. Finds the channels that flagged for all antennas at a single time but not flagged
                for all times. If the ratio of of such channels for a given time compared to all channels that are not
                completely flagged is greater than time_threshold, flag the entire time for all antennas.
                Default 1.0 means no additional flagging.
            ant_threshold: float. If, after time and freq thesholding and broadcasting, an antenna is left unflagged
                for a number of visibilities less than ant_threshold times the maximum among all antennas, flag that
                antenna for all times and channels. Default 1.0 means no additional flagging.
            verbose: print status updates
        '''
        self.verbose = verbose

        # load calibration files---gains, flags, freqs, times, and if desired, cspa and chisq
        utils.echo('Now loading calibration files...', verbose=self.verbose)
        self.cals = calfits_list
        gains, cal_flags, chisq, cspa, self.cal_freqs, self.cal_times = {}, {}, {}, {}, {}, {}
        for cal in self.cals:
            hc = io.HERACal(cal)
            gains[cal], cal_flags[cal], quals, total_qual = hc.read()
            if load_cspa:
                cspa[cal] = quals
            if load_chisq:
                chisq[cal] = total_qual
            self.cal_freqs[cal], self.cal_times[cal] = hc.freqs, hc.times

        # load flag files
        self.flag_files = flag_file_list
        if len(self.flag_files) > 0:
            utils.echo('Now loading external flag files...', verbose=self.verbose)
            self.ext_flags, self.flag_freqs, self.flag_times = {}, {}, {}
            for ff in self.flag_files:
                flags, meta = io.load_flags(ff, filetype=flag_filetype, return_meta=True)
                self.ext_flags[ff] = flag_utils.synthesize_ant_flags(flags, threshold=antflag_thresh)
                self.flag_freqs[ff] = meta['freqs']
                self.flag_times[ff] = meta['times']

        # set up time grid (note that it is offset by .5 dt so that times always map to the same
        # index, even if they are slightly different between the flag_files and the calfits files)
        utils.echo('Now setting up gain and flag grids...', verbose=self.verbose)
        all_file_times = sorted(np.array([t for times in self.cal_times.values() for t in times]))
        self.dt = np.median(np.diff(all_file_times))
        self.time_grid = np.arange(all_file_times[0] + self.dt / 2.0, all_file_times[-1] + self.dt, self.dt)
        self.time_indices = {cal: np.searchsorted(self.time_grid, times) for cal, times in self.cal_times.items()}
        if len(self.flag_files) > 0:
            self.flag_time_indices = {ff: np.searchsorted(self.time_grid, times) for ff, times in self.flag_times.items()}

        # build empty multi-file grids for each antenna's gains and flags (and optionally for cspa)
        self.freqs = self.cal_freqs[self.cals[0]]
        self.ants = sorted(list(set([k for gain in gains.values() for k in gain.keys()])))
        self.gain_grids = {ant: np.ones((len(self.time_grid), len(self.freqs)), dtype=complex) for ant in self.ants}
        self.flag_grids = {ant: np.ones((len(self.time_grid), len(self.freqs)), dtype=bool) for ant in self.ants}
        if load_cspa:
            self.cspa_grids = {ant: np.ones((len(self.time_grid), len(self.freqs)), dtype=float) for ant in self.ants}
        # Now fill those grid
        for ant in self.ants:
            for cal in self.cals:
                if ant in gains[cal]:
                    self.gain_grids[ant][self.time_indices[cal], :] = gains[cal][ant]
                    self.flag_grids[ant][self.time_indices[cal], :] = cal_flags[cal][ant]
                    if load_cspa:
                        self.cspa_grids[ant][self.time_indices[cal], :] = cspa[cal][ant]
            if len(self.flag_files) > 0:
                for ff in self.flag_files:
                    if ant in self.ext_flags[ff]:
                        self.flag_grids[ant][self.flag_time_indices[ff], :] += self.ext_flags[ff][ant]

        # Now build grid and fill it for chisq_grid, if desired
        if load_chisq:
            jpols = set([ant[1] for ant in self.ants])
            self.chisq_grids = {jpol: np.ones((len(self.time_grid), len(self.freqs)), dtype=float) for jpol in jpols}
            for jpol in jpols:
                for cal in self.cals:
                    self.chisq_grids[jpol][self.time_indices[cal], :] = chisq[cal][jpol]

        # perform data quality checks and flag thresholding
        self.check_consistency()
        flag_threshold_and_broadcast(self.flag_grids, freq_threshold=freq_threshold,
                                     time_threshold=time_threshold, ant_threshold=time_threshold)

        # build blacklists
        self.blacklist_wgt = blacklist_wgt
        self.time_blacklist = build_time_blacklist(self.time_grid, time_blacklists=time_blacklists, lst_blacklists=lst_blacklists,
                                                   lat_lon_alt_degrees=lat_lon_alt_degrees, telescope_name=hc.telescope_name)
        self.freq_blacklist = build_freq_blacklist(self.freqs, freq_blacklists=freq_blacklists, chan_blacklists=chan_blacklists)

        # pick a reference antenna that has the minimum number of flags (tie goes to lower antenna number) and then rephase
        if pick_refant:
            utils.echo('Now picking reference antenna(s)...', verbose=self.verbose)
            self.refant = pick_reference_antenna(self.gain_grids, self.flag_grids, self.freqs, per_pol=True)
            utils.echo('\n'.join(['Reference Antenna ' + str(self.refant[pol][0]) + ' selected for ' + pol + '.'
                                  for pol in sorted(list(self.refant.keys()))]), verbose=self.verbose)
            self.rephase_to_refant()

    def check_consistency(self):
        '''Checks the consistency of the input calibration files (and, if loaded, flag files).
        Ensures that all files have the same frequencies, that they are time-ordered, that
        times are internally contiguous in a file and that calibration and flagging times match.
        '''
        all_time_indices = np.array([i for indices in self.time_indices.values() for i in indices])
        assert len(all_time_indices) == len(np.unique(all_time_indices)), \
            'Multiple calibration integrations map to the same time index.'
        for cal in self.cals:
            assert np.all(np.abs(self.cal_freqs[cal] - self.freqs) < 1e-4), \
                '{} and {} have different frequencies.'.format(cal, self.cals[0])
        if len(self.flag_files) > 0:
            all_flag_time_indices = np.array([i for indices in self.flag_time_indices.values() for i in indices])
            assert len(all_flag_time_indices) == len(np.unique(all_flag_time_indices)), \
                'Multiple flag file integrations map to the same time index.'
            assert np.all(np.unique(all_flag_time_indices) == np.unique(all_time_indices)), \
                'The number of unique indices for the flag files does not match the calibration files.'
            for ff in self.flag_files:
                assert np.all(np.abs(self.flag_freqs[ff] - self.freqs) < 1e-4), \
                    '{} and {} have different frequencies.'.format(ff, self.cals[0])

    def rephase_to_refant(self, warn=True):
        '''If the CalibrationSmoother object has a refant attribute, this function rephases the
        filtered gains to it.'''
        if hasattr(self, 'refant'):
            rephase_to_refant(self.gain_grids, self.refant, flags=self.flag_grids)
        elif warn:
            warnings.warn('No rephasing done because self.refant has not been set.', RuntimeWarning)

    def time_filter(self, filter_scale=1800.0, mirror_kernel_min_sigmas=5):
        '''Time-filter calibration solutions with a rolling Gaussian-weighted average. Allows
        the mirroring of gains and flags and appending the mirrored gains and wgts to both ends,
        ensuring temporal smoothness of the rolling average.

        Arguments:
            filter_scale: float in seconds of FWHM of Gaussian smoothing kernel in time
            mirror_kernel_min_sigmas: Number of stdev into the Gaussian kernel one must go before edge
                effects can be ignored.
        '''
        # Make sure that the gain_grid will be sufficiently padded on each side to avoid edge effects
        needed_buffer = filter_scale / (2 * (2 * np.log(2))**.5) * mirror_kernel_min_sigmas
        duration = self.dt * len(self.time_grid) * 24 * 60 * 60
        nMirrors = 0
        while (nMirrors * duration < needed_buffer):
            nMirrors += 1

        # Now loop through and apply running Gaussian averages
        for ant, gain_grid in self.gain_grids.items():
            utils.echo('    Now smoothing antenna' + str(ant[0]) + ' ' + str(ant[1]) + ' in time...', verbose=self.verbose)
            if not np.all(self.flag_grids[ant]):
                wgts_grid = _build_wgts_grid(self.flag_grids[ant], self.time_blacklist, self.freq_blacklist, self.blacklist_wgt)
                self.gain_grids[ant] = time_filter(gain_grid, wgts_grid, self.time_grid,
                                                   filter_scale=filter_scale, nMirrors=nMirrors)

    def filter_1d(self, filter_scale=None, tol=1e-09, skip_wgt=0.1, mode='clean', ax='freq', **filter_kwargs):
        '''Frequency-filter stored calibration solutions on a given scale in MHz.

        Arguments:
            filter_scale: scale to use for the low-pass filter. filter_scale^-1 corresponds
                to the half-width (i.e. the width of the positive part) of the region in fourier
                space, symmetric about 0, that is filtered out.
                if ax='freq' then units are MHz
                if ax='time' then units are days.
            tol: CLEAN algorithm convergence tolerance (see aipy.deconv.clean)
            window: window function for filtering applied to the filtered axis. Default tukey has alpha=0.5.
                See aipy.dsp.gen_window for options.
            skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
                gains are left unchanged and self.wgts and self.flag_grids are set to 0 and True,
                respectively. Only works properly when all weights are all between 0 and 1.
            mode: str, optional
                specify whether to use 'clean' or 'dpss_leastsq' filtering of gains.
            ax: str, optional
                specify axis to perform 1d smoothing over. Options are 'time' and 'freq'. default is 'freq'.
            filter_kwargs : any keyword arguments for hera_filters.dspec.fourier_filter.
        '''
        # Loop over all antennas and perform a low-pass delay filter on gains
        cache = {}
        for ant, gain_grid in self.gain_grids.items():
            utils.echo('    Now filtering antenna' + str(ant[0]) + ' ' + str(ant[1]) + f' in {ax}...', verbose=self.verbose)
            wgts_grid = _build_wgts_grid(self.flag_grids[ant], self.time_blacklist, self.freq_blacklist)
            if ax == 'freq':
                xaxis = self.freqs
            else:
                xaxis = self.time_grid
            self.gain_grids[ant], info = filter_1d(gain_grid, wgts_grid, xaxis, ax=ax,
                                                   filter_scale=filter_scale, tol=tol, mode=mode,
                                                   skip_wgt=skip_wgt, cache=cache, **filter_kwargs)
            # flag all channels for any time that triggers the skip_wgt
            if ax == 'freq':
                for i in info['status']['axis_1']:
                    if info['status']['axis_1'][i] == 'skipped':
                        self.flag_grids[ant][i, :] = np.ones_like(self.flag_grids[ant][i, :])
            else:
                for i in info['status']['axis_0']:
                    if info['status']['axis_0'][i] == 'skipped':
                        self.flag_grids[ant][:, i] = np.ones_like(self.flag_grids[ant][:, i])
        self.rephase_to_refant(warn=False)

    def time_freq_2D_filter(self, freq_scale=10.0, time_scale=1800.0, tol=1e-09, filter_mode='rect',
                            window='tukey', maxiter=100, method="CLEAN", eigenval_cutoff=1e-9, skip_flagged_edges=True,
                            **win_kwargs):
        '''2D time and frequency filter stored calibration solutions on a given scale in seconds and MHz respectively.

        Arguments:
            freq_scale: frequency scale in MHz to use for the low-pass filter. freq_scale^-1 corresponds
                to the half-width (i.e. the width of the positive part) of the region in fourier
                space, symmetric about 0, that is filtered out.
            time_scale: time scale in seconds. Defined analogously to freq_scale.
            tol: CLEAN algorithm convergence tolerance (see aipy.deconv.clean)
            filter_mode: either 'rect' or 'plus':
                'rect': perform 2D low-pass filter, keeping modes in a small rectangle around delay = 0
                        and fringe rate = 0
                'plus': produce a separable calibration solution by only keeping modes with 0 delay,
                        0 fringe rate, or both
            window: window function for filtering applied to the frequency axis. Only used when the method is 'CLEAN'
                See aipy.dsp.gen_window for options.
            maxiter: Maximum number of iterations for aipy.deconv.clean to converge. Only used when the method is 'CLEAN'
            eigenval_cutoff: sinc_matrix eigenvalue cutoffs to use for included dpss modes.
                Only used when the filtering method is 'DPSS'
            method: Algorithm used to smooth calibration solutions. Either 'CLEAN' or 'DPSS':
                'CLEAN': uses the CLEAN algorithm to
                'DPSS': uses discrete prolate spheroidal sequences to filter calibration solutions
            skip_flagged_edges : if True, do not filter over flagged edge times (filter over sub-region)
                Default is True, only used when method='DPSS'
            win_kwargs : any keyword arguments for the window function selection in aipy.dsp.gen_window.
                Currently, the only window that takes a kwarg is the tukey window with a alpha=0.5 default
        '''
        # Keep track of the size of the weights grid after removing flagged edges
        if skip_flagged_edges:
            edges_old = None

        # Create default variables to be passed in to time_freq_2D_filter
        info = {'dpss_vectors': None, 'XTXinv': None}
        dpss_vectors = None
        XTXinv = None
        wgts_old = np.zeros(1)

        # loop over all antennas that are not completely flagged and filter
        for ant, gain_grid in self.gain_grids.items():
            if not np.all(self.flag_grids[ant]):
                utils.echo('    Now filtering antenna ' + str(ant[0]) + ' ' + str(ant[1]) + ' in time and frequency...', verbose=self.verbose)
                wgts_grid = _build_wgts_grid(self.flag_grids[ant], self.time_blacklist, self.freq_blacklist, self.blacklist_wgt)

                # If the weights grid is the same as the previous, the solution matrix can be reused to speed up computation
                if method == 'DPSS' or method == 'dpss_leastsq':
                    XTXinv = info['XTXinv'] if np.allclose(wgts_grid, wgts_old) else None
                    wgts_old = np.copy(wgts_grid)

                    # Check to see if the grid size changes after removing the flagged edges
                    if skip_flagged_edges:
                        xout, _, _, edges, chunks = truncate_flagged_edges(gain_grid, wgts_grid, (self.time_grid, self.freqs), ax='both')
                        # If the grid size changes, recompute filters for new grid size
                        dpss_vectors = info['dpss_vectors'] if edges == edges_old else None
                        edges_old = deepcopy(edges)
                filtered, info = time_freq_2D_filter(gain_grid, wgts_grid, self.freqs, self.time_grid, freq_scale=freq_scale,
                                                     time_scale=time_scale, tol=tol, filter_mode=filter_mode, maxiter=maxiter,
                                                     window=window, dpss_vectors=dpss_vectors, method=method, XTXinv=XTXinv,
                                                     skip_flagged_edges=skip_flagged_edges, **win_kwargs)

                self.gain_grids[ant] = filtered

        self.rephase_to_refant(warn=False)

    def write_smoothed_cal(self, output_replace=('.flagged_abs.', '.smooth_abs.'), add_to_history='', clobber=False, **kwargs):
        '''Writes time and/or frequency smoothed calibration solutions to calfits, updating input calibration.
        Also compares the input and output calibration and saves that result in the quals/total_quals fields.

        Arguments:
            output_replace: tuple of input calfile substrings: ("to_replace", "to_replace_with")
            add_to_history: appends a string to the history of the output file (in addition to the )
            clobber: if True, overwrites existing file at outfilename
            kwargs: dictionary mapping updated attributes to their new values.
                See pyuvdata.UVCal documentation for more info.
        '''
        utils.echo('Now writing results to disk...', verbose=self.verbose)
        for cal in self.cals:
            hc = io.HERACal(cal)
            gains, flags, _, _ = hc.read()
            if hasattr(self, 'refant'):
                rephase_to_refant(gains, self.refant)
            out_gains = {ant: self.gain_grids[ant][self.time_indices[cal], :] for ant in self.ants}
            out_flags = {ant: self.flag_grids[ant][self.time_indices[cal], :] for ant in self.ants}
            rel_diff, avg_rel_diff = utils.gain_relative_difference(gains, out_gains, out_flags)
            hc.update(gains=out_gains, flags=out_flags, quals=rel_diff, total_qual=avg_rel_diff)
            hc.history += utils.history_string(add_to_history)
            for attribute, value in kwargs.items():
                hc.__setattr__(attribute, value)
            hc.check()
            outfilename = cal.replace(output_replace[0], output_replace[1])
            hc.write_calfits(outfilename, clobber=clobber)


def _pair(dash_sep_arg_pair):
    '''Helper function for argparser to turn dash-separted numbers into tuples of floats.'''
    return tuple([float(arg) for arg in dash_sep_arg_pair.split('-', maxsplit=1)])


def smooth_cal_argparser():
    '''Arg parser for commandline operation of 2D calibration smoothing.'''
    a = argparse.ArgumentParser(description="Smooth calibration solutions in time and frequency using the hera_cal.smooth_cal module.")
    a.add_argument("calfits_list", type=str, nargs='+', help="list of paths to chronologically sortable calfits files (usually a full day)")
    a.add_argument("--infile_replace", type=str, default='.flagged_abs.', help="substring of files in calfits_list to replace for output files")
    a.add_argument("--outfile_replace", type=str, default='.smooth_abs.', help="replacement substring for output files")
    a.add_argument("--clobber", default=False, action="store_true", help='overwrites existing file at cal_outfile (default False)')
    a.add_argument("--pick_refant", default=False, action="store_true", help='Automatically picks one reference anteanna per polarization. \
                   The refants chosen have the fewest total flags and causes the least noisy phases on other antennas when made the phase reference.')
    a.add_argument("--run_if_first", default=None, type=str, help='only run smooth_cal if the first item in the sorted calfits_list\
                   matches run_if_first (default None means always run)')
    a.add_argument("--verbose", default=False, action="store_true", help="Print status updates while smoothing.")

    # Options relating to optional external flags and flag thresholding and broadcasting
    flg_opts = a.add_argument_group(title='Flagging options.')
    flg_opts.add_argument("--flag_file_list", type=str, nargs='+', default=[], help="optional list of paths to chronologically\
                                  sortable flag files (usually a full day)")
    flg_opts.add_argument("--flag_filetype", type=str, default='h5', help="filetype of flag_file_list (either 'h5' or legacy 'npz'")
    flg_opts.add_argument("--antflag_thresh", default=0.0, type=float, help="fraction of flagged pixels across all visibilities (with a common antenna) \
                                  needed to flag that antenna gain at a particular time and frequency. 0.0 is aggressive flag broadcasting, while 1.0 is \
                                  conservative flag broadcasting.")
    flg_opts.add_argument("--freq_threshold", default=0.35, type=float, help="Finds the times that flagged for all antennas at a single channel but not \
                          flagged for all channels. If the ratio of of such times for a given channel compared to all times that are not completely flagged \
                          is greater than freq_threshold, flag the entire channel for all antennas. 1.0 means no additional flagging (default 0.35).")
    flg_opts.add_argument("--time_threshold", default=0.5, type=float, help="Finds the channels that flagged for all antennas at a single time but not \
                          flagged for all times. If the ratio of of such channels for a given time compared to all channels that are not completely flagged \
                          is greater than time_threshold, flag the entire time for all antennas. 1.0 means no additional flagging (default 0.5).")
    flg_opts.add_argument("--ant_threshold", default=0.5, type=float, help="If, after time and freq thesholding and broadcasting, an antenna is left \
                          unflagged for a number of visibilities less than ant_threshold times the maximum among all antennas, flag that antenna for all \
                          times and channels. 1.0 means no additional flagging (default 0.5).")

    # Options relating to blacklisting time or frequency ranges
    bkl_opts = a.add_argument_group(title="Blacklisting options used for assigning 0 weight (by default) to times/frequencies so that smooth_cal\n"
                                    "interpolates/extrapoaltes over them (though they aren't necessarily flagged).")
    bkl_opts.add_argument("--time_blacklists", type=_pair, default=[], nargs='+', help="space-separated list of dash-separted pairs of times in Julian Day \
                          bounding (inclusively) blacklisted times, e.g. '2458098.1-2458098.4'.")
    bkl_opts.add_argument("--lst_blacklists", type=_pair, default=[], nargs='+', help="space-separated list of dash-separted pairs of LSTs in hours \
                          bounding (inclusively) blacklisted LSTs, e.g. '3-4 10-12 23-.5'")
    bkl_opts.add_argument("--chan_blacklists", type=_pair, default=[], nargs='+', help="space-separated list of dash-separted pairs of channel numbers \
                          bounding (inclusively) blacklisted spectral ranges, e.g. '0-256 800-900'")
    bkl_opts.add_argument("--freq_blacklists", type=_pair, default=[], nargs='+', help="space-separated list of dash-separted pairs of frequencies in Hz \
                          bounding (inclusively) blacklisted spectral ranges, e.g. '88e6-110e6 136e6-138e6'")
    bkl_opts.add_argument("--blacklist_wgt", type=float, default=0.0, help="Relative weight to assign to blacklisted times/freqs compared to 1.0. Default 0.0 \
                          means no weight. Note that 0.0 will create problems for DPSS at edge times and frequencies.")

    # Options relating to performing the filter in time and frequency
    flt_opts = a.add_argument_group(title='Filtering options.')
    flt_opts.add_argument("--freq_scale", type=float, default=10.0, help="frequency scale in MHz for the low-pass filter\
                          (default 10.0 MHz, i.e. a 100 ns delay filter)")
    flt_opts.add_argument("--time_scale", type=float, default=1800.0, help="time scale in seconds, defined analogously to freq_scale (default 1800 s).")
    flt_opts.add_argument("--tol", type=float, default=1e-9, help='CLEAN algorithm convergence tolerance (default 1e-9)')
    flt_opts.add_argument("--filter_mode", type=str, default="rect", help='Mode for CLEAN algorithm that defines the shape of the area that can have\
                          non-zero CLEAN components. Default "rect". "plus" creates calibration solutions that are separable in time and frequency.')
    flt_opts.add_argument("--window", type=str, default="tukey", help='window function for frequency filtering (default "tukey",\
                          see aipy.dsp.gen_window for options')
    flt_opts.add_argument("--maxiter", type=int, default=100, help='maximum iterations for aipy.deconv.clean to converge (default 100)')
    flt_opts.add_argument("--alpha", type=float, default=.3, help='alpha parameter to use for Tukey window (ignored if window is not Tukey)')
    flt_opts.add_argument("--method", type=str, default='CLEAN', help='Algorithm used to smooth calibration solutions. Default is "CLEAN". "DPSS" uses \
                          discrete prolate spheroidal sequences to smooth calibration solutions.')
    flt_opts.add_argument("--eigenval_cutoff", type=str, default=1e-9, help="sinc_matrix eigenvalue cutoff to use for included dpss modes. \
                          Only used when the filtering method is 'DPSS'")
    flt_opts.add_argument("--dont_skip_flagged_edges", action="store_true", default=False, help="if True, use DPSS over integrations with flagged edge channels.\
                          Only used when method used is 'DPSS'")
    flt_opts.add_argument("--axis", default="both", type=str, help="smooth either in 'freq', or 'both' (time and freq) axes.")
    flt_opts.add_argument("--skip_wgt", default=0.1, type=float, help="skip if this fraction is flagged.")
    args = a.parse_args()
    return args
