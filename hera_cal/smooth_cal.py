# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

from __future__ import absolute_import, division, print_function

import numpy as np
import scipy
import aipy
from collections import OrderedDict as odict
from copy import deepcopy
from six.moves import range
import warnings
import argparse
import uvtools

from . import io
from . import version
from . import utils
from . import flag_utils
from .noise import interleaved_noise_variance_estimate


def freq_filter(gains, wgts, freqs, filter_scale=10.0, tol=1e-09, window='tukey', skip_wgt=0.1,
                maxiter=100, **win_kwargs):
    '''Frequency-filter calibration solutions on a given scale in MHz using uvtools.dspec.high_pass_fourier_filter.
    Before filtering, removes a per-integration delay using abscal.fft_dly, then puts it back in after filtering.

    Arguments:
        gains: ndarray of shape=(Ntimes,Nfreqs) of complex calibration solutions to filter
        wgts: ndarray of shape=(Ntimes,Nfreqs) of real linear multiplicative weights
        freqs: ndarray of frequency channels in Hz
        filter_scale: frequency scale in MHz to use for the low-pass filter. filter_scale^-1 corresponds
            to the half-width (i.e. the width of the positive part) of the region in fourier
            space, symmetric about 0, that is filtered out.
        tol: CLEAN algorithm convergence tolerance (see aipy.deconv.clean)
        window: window function for filtering applied to the frequency axis.
            See aipy.dsp.gen_window for options.
        skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
            filtered is left unchanged and info is {'skipped': True} for that time.
            Only works properly when all weights are all between 0 and 1.
        maxiter: Maximum number of iterations for aipy.deconv.clean to converge.
        win_kwargs : any keyword arguments for the window function selection in aipy.dsp.gen_window.
            Currently, the only window that takes a kwarg is the tukey window with a alpha=0.5 default.

    Returns:
        filtered: filtered gains, ndarray of shape=(Ntimes,Nfreqs)
        info: info object from uvtools.dspec.high_pass_fourier_filter
    '''
    sdf = np.median(np.diff(freqs)) / 1e9  # in GHz
    filter_size = (filter_scale / 1e3)**-1  # Puts it in ns
    (dlys, phi) = utils.fft_dly(gains, sdf * 1e9, wgts, medfilt=False)  # delays are in seconds
    rephasor = np.exp(-2.0j * np.pi * np.outer(dlys, freqs))
    filtered, res, info = uvtools.dspec.high_pass_fourier_filter(gains * rephasor, wgts, filter_size, sdf, tol=tol, window=window,
                                                                 skip_wgt=skip_wgt, maxiter=maxiter, **win_kwargs)
    filtered /= rephasor
    # put back in unfilted values if skip_wgt is triggered
    for i, info_dict in enumerate(info):
        if info_dict.get('skipped', False):
            filtered[i, :] = gains[i, :]
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
                        tol=1e-09, filter_mode='rect', maxiter=100, window='tukey', **win_kwargs):
    '''Filter calibration solutions in both time and frequency simultaneously. First rephases to remove
    a time-smoothed delay from the gains, then performs the low-pass 2D filter in time and frequency,
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
                    0 fringe rate, or both
        window: window function for filtering applied to the filtered axis.
            See aipy.dsp.gen_window for options.
        maxiter: Maximum number of iterations for aipy.deconv.clean to converge.
        win_kwargs : any keyword arguments for the window function selection in aipy.dsp.gen_window.
            Currently, the only window that takes a kwarg is the tukey window with a alpha=0.5 default.

    Returns:
        filtered: filtered gains, ndarray of shape=(Ntimes,Nfreqs)
        info: dictionary of metadata from aipy.deconv.clean
    '''
    df = np.median(np.diff(freqs))
    dt = np.median(np.diff(times)) * 24.0 * 3600.0  # in seconds
    delays = np.fft.fftfreq(freqs.size, df)
    fringes = np.fft.fftfreq(times.size, dt)
    delay_scale = (freq_scale * 1e6)**-1  # Puts it in seconds
    fringe_scale = (time_scale)**-1  # in Hz

    # find per-integration delays, smooth on the time_scale of gain smoothing, and rephase
    taus = utils.fft_dly(gains, df, wgts, medfilt=False)[0].astype(np.complex)  # delays are in seconds
    if not np.all(taus == 0):  # this breaks CLEAN, but it means we don't need smoothing anyway
        taus = uvtools.dspec.high_pass_fourier_filter(taus.T, np.sum(wgts, axis=1, keepdims=True).T,
                                                      fringe_scale, dt, tol=tol, maxiter=maxiter)[0].T  # 0th index is the CLEAN components
    rephasor = np.exp(-2.0j * np.pi * np.outer(np.abs(taus), freqs))

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
    del info['res']  # this matches the convention in uvtools.dspec.high_pass_fourier_filter
    return filtered / rephasor, info


def pick_reference_antenna(gains, flags, freqs):
    '''Pick a refrence antenna that has the minimum number of per-antenna flags and produces
    the least noisy phases on other antennas when used as a reference antenna.

    Arguments:
        gains: Dictionary mapping antenna keys to gain waterfalls.
        flags: dictionary mapping antenna keys to flag waterfall where True means flagged
        freqs: ndarray of frequency channels in Hz

    Returns:
        refant: key of the antenna with the minimum number of flags and the least noisy phases
    '''
    # pick antennas with the mininum number of flags
    flags_per_ant = {ant: np.sum(f) for ant, f in flags.items()}
    refant_candidates = [ant for ant, nflags in flags_per_ant.items() if nflags == np.min(list(flags_per_ant.values()))]
    
    # compute delay and phase for all gains to flatten them as well as possible. Average  over times.
    df = np.median(np.diff(freqs))
    rephasors = {}
    for ant in gains.keys():
        wgts = np.array(~(flags[ant]), dtype=float)
        (dlys, phis) = utils.fft_dly(gains[ant], df, wgts, medfilt=False)
        rephasors[ant] = np.exp(-2.0j * np.pi * np.mean(dlys), freqs - 1.0j * np.mean(phis))

    # assess each candidate reference antenna: estimate the median noise of the angle after rephasing to a given refant
    # (after taking out delays from both to minimize phase wraps) and return least noisy refant
    median_angle_noise_as_refeant = {}
    for refant in refant_candidates:
        refant_rephasor = np.abs(gains[refant] * rephasors[refant]) / (gains[refant] * rephasors[refant])
        median_phase_noise = [np.median(interleaved_noise_variance_estimate(np.angle(gains[ant] * rephasors[ant] * refant_rephasor)))
                              for ant in gains.keys() if not np.all(flags[ant])]
        median_angle_noise_as_refeant[refant] = np.median(median_phase_noise)
    return sorted(median_angle_noise_as_refeant, key=median_angle_noise_as_refeant.__getitem__)[0]


def rephase_to_refant(gains, refant, flags=None):
    '''Rephase all gains in place so that the phase of the reference antenna is 0 for all times and frequencies.

    Arguments:
        gains: Dictionary mapping antenna keys to gain waterfalls. Modified in place.
        refant: Antenna key of antenna to make the reference antenna
        flags: Optional dictionary mapping antenna keys to flag waterfall.
            Used only to verify that all gains are flagged where the refant is flagged.
    '''
    refant_phasor = gains[refant] / np.abs(gains[refant])
    for ant in gains.keys():
        if flags is not None and np.any(flags[refant][np.logical_not(flags[ant])]):
            raise ValueError('The chosen reference antenna', refant, 'is flagged in at least one place where antenna',
                             ant, 'is not, so automatic reference antenna selection has failed.')
        else:
            gains[ant] = gains[ant] / refant_phasor


class CalibrationSmoother():

    def __init__(self, calfits_list, flag_file_list=[], flag_filetype='h5', pick_refant=False, antflag_thresh=0.0):
        '''Class for smoothing calibration solutions in time and frequency for a whole day. Initialized with a list of
        calfits files and, optionally, a corresponding list of flag files, which must match the calfits files
        one-to-one in time. This function sets up a time grid that spans the whole day with dt = integration time.
        Gains and flags are assigned to the nearest gridpoint using np.searchsorted. It is assumed that:
        1) All calfits and flag files have the same frequencies
        2) The flag times and calfits time map one-to-one to the same set of integrations

        Arguments:
            calfits_list: list of string paths to calfits files containing calibration solutions and flags
            flag_file_list: list of string paths to files containing flags as a function of baseline, times
                and frequency. Must have all baselines for all times. Flags on baselines are broadcast to both
                antennas involved, unless either antenna is completely flagged for all times and frequencies.
            flag_filetype: filetype of flag_file_list to pass into io.load_flags. Either 'h5' (default) or legacy 'npz'.
            pick_refant: if True, automatically pick as the reference anteanna the antenna with the fewest total
                flags and then rephase all gains so that that reference antenna has purely real gains.
            antflag_thresh: float, fraction of flagged pixels across all visibilities (with a common antenna)
                needed to flag that antenna gain at a particular time and frequency. antflag_thresh=0.0 is
                aggressive flag broadcasting, antflag_thresh=1.0 is conservative flag_broadcasting.
        '''
        # load calibration files
        self.cals = calfits_list
        gains, cal_flags, self.cal_freqs, self.cal_times = odict(), odict(), odict(), odict()
        for cal in self.cals:
            hc = io.HERACal(cal)
            gains[cal], cal_flags[cal], _, _ = hc.read()
            self.cal_freqs[cal], self.cal_times[cal] = hc.freqs, hc.times

        # load flag files
        self.flag_files = flag_file_list
        if len(self.flag_files) > 0:
            self.ext_flags, self.flag_freqs, self.flag_times = odict(), odict(), odict()
            for ff in self.flag_files:
                flags, meta = io.load_flags(ff, filetype=flag_filetype, return_meta=True)
                self.ext_flags[ff] = flag_utils.synthesize_ant_flags(flags, threshold=antflag_thresh)
                self.flag_freqs[ff] = meta['freqs']
                self.flag_times[ff] = meta['times']

        # set up time grid (note that it is offset by .5 dt so that times always map to the same
        # index, even if they are slightly different between the flag_files and the calfits files)
        all_file_times = sorted(np.array([t for times in self.cal_times.values() for t in times]))
        self.dt = np.median(np.diff(all_file_times))
        self.time_grid = np.arange(all_file_times[0] + self.dt / 2.0, all_file_times[-1] + self.dt, self.dt)
        self.time_indices = {cal: np.searchsorted(self.time_grid, times) for cal, times in self.cal_times.items()}
        if len(self.flag_files) > 0:
            self.flag_time_indices = {ff: np.searchsorted(self.time_grid, times) for ff, times in self.flag_times.items()}

        # build multi-file grids for each antenna's gains and flags
        self.freqs = self.cal_freqs[self.cals[0]]
        self.ants = sorted(list(set([k for gain in gains.values() for k in gain.keys()])))
        self.gain_grids = {ant: np.ones((len(self.time_grid), len(self.freqs)), dtype=np.complex) for ant in self.ants}
        self.flag_grids = {ant: np.ones((len(self.time_grid), len(self.freqs)), dtype=bool) for ant in self.ants}
        for ant in self.ants:
            for cal in self.cals:
                if ant in gains[cal]:
                    self.gain_grids[ant][self.time_indices[cal], :] = gains[cal][ant]
                    self.flag_grids[ant][self.time_indices[cal], :] = cal_flags[cal][ant]
            if len(self.flag_files) > 0:
                for ff in self.flag_files:
                    if ant in self.ext_flags[ff]:
                        self.flag_grids[ant][self.flag_time_indices[ff], :] += self.ext_flags[ff][ant]

        # perform data quality checks
        self.check_consistency()
        self.reset_filtering()

        # pick a reference antenna that has the minimum number of flags (tie goes to lower antenna number) and then rephase
        if pick_refant:
            self.refant = pick_reference_antenna(self.gain_grids, self.flag_grids, self.freqs)
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

    def reset_filtering(self):
        '''Reset gain smoothing to the original input gains.'''
        self.filtered_gain_grids = deepcopy(self.gain_grids)
        self.filtered_flag_grids = deepcopy(self.flag_grids)

    def rephase_to_refant(self):
        '''If the CalibrationSmoother object has a refant attribute, this function rephases to it.'''
        if hasattr(self, 'refant'):
            rephase_to_refant(self.filtered_gain_grids, self.refant, flags=self.flag_grids)

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
        for ant, gain_grid in self.filtered_gain_grids.items():
            if not np.all(self.filtered_flag_grids[ant]):
                wgts_grid = np.logical_not(self.filtered_flag_grids[ant]).astype(float)
                self.filtered_gain_grids[ant] = time_filter(gain_grid, wgts_grid, self.time_grid,
                                                            filter_scale=filter_scale, nMirrors=nMirrors)

    def freq_filter(self, filter_scale=10.0, tol=1e-09, window='tukey', skip_wgt=0.1, maxiter=100, **win_kwargs):
        '''Frequency-filter stored calibration solutions on a given scale in MHz.

        Arguments:
            filter_scale: frequency scale in MHz to use for the low-pass filter. filter_scale^-1 corresponds
                to the half-width (i.e. the width of the positive part) of the region in fourier
                space, symmetric about 0, that is filtered out.
            tol: CLEAN algorithm convergence tolerance (see aipy.deconv.clean)
            window: window function for filtering applied to the filtered axis. Default tukey has alpha=0.5.
                See aipy.dsp.gen_window for options.
            skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
                filtered_gains are left unchanged and self.wgts and self.filtered_flag_grids are set to 0 and True,
                respectively. Only works properly when all weights are all between 0 and 1.
            maxiter: Maximum number of iterations for aipy.deconv.clean to converge.
            win_kwargs : any keyword arguments for the window function selection in aipy.dsp.gen_window.
                    Currently, the only window that takes a kwarg is the tukey window with a alpha=0.5 default.
        '''

        # Loop over all antennas and perform a low-pass delay filter on gains
        for ant, gain_grid in self.filtered_gain_grids.items():
            wgts_grid = np.logical_not(self.filtered_flag_grids[ant]).astype(float)
            self.filtered_gain_grids[ant], info = freq_filter(gain_grid, wgts_grid, self.freqs,
                                                              filter_scale=filter_scale, tol=tol, window=window,
                                                              skip_wgt=skip_wgt, maxiter=maxiter, **win_kwargs)
            # flag all channels for any time that triggers the skip_wgt
            for i, info_dict in enumerate(info):
                if info_dict.get('skipped', False):
                    self.filtered_flag_grids[ant][i, :] = np.ones_like(self.filtered_flag_grids[ant][i, :])
        self.rephase_to_refant()

    def time_freq_2D_filter(self, freq_scale=10.0, time_scale=1800.0, tol=1e-09,
                            filter_mode='rect', window='tukey', maxiter=100, **win_kwargs):
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
            window: window function for filtering applied to the frequency axis.
                See aipy.dsp.gen_window for options.
            maxiter: Maximum number of iterations for aipy.deconv.clean to converge.
            win_kwargs : any keyword arguments for the window function selection in aipy.dsp.gen_window.
                Currently, the only window that takes a kwarg is the tukey window with a alpha=0.5 default
        '''
        # loop over all antennas that are not completely flagged and filter
        for ant, gain_grid in self.filtered_gain_grids.items():
            if not np.all(self.filtered_flag_grids[ant]):
                wgts_grid = np.logical_not(self.filtered_flag_grids[ant]).astype(float)
                filtered, info = time_freq_2D_filter(gain_grid, wgts_grid, self.freqs, self.time_grid, freq_scale=freq_scale,
                                                     time_scale=time_scale, tol=tol, filter_mode=filter_mode, maxiter=maxiter,
                                                     window=window, **win_kwargs)
                self.filtered_gain_grids[ant] = filtered
        self.rephase_to_refant()

    def write_smoothed_cal(self, output_replace=('.abs.', '.smooth_abs.'), add_to_history='', clobber=False, **kwargs):
        '''Writes time and/or frequency smoothed calibration solutions to calfits, updating input calibration.

        Arguments:
            output_replace: tuple of input calfile substrings: ("to_replace", "to_replace_with")
            add_to_history: appends a string to the history of the output file (in addition to the )
            clobber: if True, overwrites existing file at outfilename
            kwargs: dictionary mapping updated attributes to their new values.
                See pyuvdata.UVCal documentation for more info.
        '''
        for cal in self.cals:
            outfilename = cal.replace(output_replace[0], output_replace[1])
            out_gains = {ant: self.filtered_gain_grids[ant][self.time_indices[cal], :] for ant in self.ants}
            out_flags = {ant: self.filtered_flag_grids[ant][self.time_indices[cal], :] for ant in self.ants}
            io.update_cal(cal, outfilename, gains=out_gains, flags=out_flags,
                          add_to_history=version.history_string(add_to_history), clobber=clobber, **kwargs)


def smooth_cal_argparser():
    '''Arg parser for commandline operation of 2D calibration smoothing.'''
    a = argparse.ArgumentParser(description="Smooth calibration solutions in time and frequency using the hera_cal.smooth_cal module.")
    a.add_argument("calfits_list", type=str, nargs='+', help="list of paths to chronologically sortable calfits files (usually a full day)")
    a.add_argument("--flag_file_list", type=str, nargs='+', default=[], help="optional list of paths to chronologically\
                   sortable flag files (usually a full day)")
    a.add_argument("--flag_filetype", type=str, default='h5', help="filetype of flag_file_list (either 'h5' or legacy 'npz'")
    a.add_argument("--infile_replace", type=str, default='.abs.', help="substring of files in calfits_list to replace for output files")
    a.add_argument("--outfile_replace", type=str, default='.smooth_abs.', help="replacement substring for output files")
    a.add_argument("--clobber", default=False, action="store_true", help='overwrites existing file at cal_outfile (default False)')
    a.add_argument("--antflag_thresh", default=0.0, type=float, help="fraction of flagged pixels across all visibilities (with a common antenna) \
                   needed to flag that antenna gain at a particular time and frequency. 0.0 is aggressive flag broadcasting, while 1.0 is \
                   conservative flag broadcasting.")
    a.add_argument("--pick_refant", default=False, action="store_true", help='automatically pick as the reference anteanna the antenna with the \
                  fewest total flags and then rephase all gains so that that reference antenna has purely real gains.')
    a.add_argument("--run_if_first", default=None, type=str, help='only run smooth_cal if the first item in the sorted calfits_list\
                   matches run_if_first (default None means always run)')

    # Options relating to performing the filter in time and frequency
    filter_options = a.add_argument_group(title='Filtering options.')
    filter_options.add_argument("--freq_scale", type=float, default=10.0, help="frequency scale in MHz for the low-pass filter\
                              (default 10.0 MHz, i.e. a 100 ns delay filter)")
    filter_options.add_argument("--time_scale", type=float, default=1800.0, help="time scale in seconds, defined analogously to freq_scale (default 1800 s).")
    filter_options.add_argument("--tol", type=float, default=1e-9, help='CLEAN algorithm convergence tolerance (default 1e-9)')
    filter_options.add_argument("--filter_mode", type=str, default="rect", help='Mode for CLEAN algorithm that defines the shape of the area that can have\
                                non-zero CLEAN components. Default "rect". "plus" creates calibration solutions that are separable in time and frequency.')
    filter_options.add_argument("--window", type=str, default="tukey", help='window function for frequency filtering (default "tukey",\
                                see aipy.dsp.gen_window for options')
    filter_options.add_argument("--maxiter", type=int, default=100, help='maximum iterations for aipy.deconv.clean to converge (default 100)')
    filter_options.add_argument("--alpha", type=float, default=.3, help='alpha parameter to use for Tukey window (ignored if window is not Tukey)')

    args = a.parse_args()
    return args
