import numpy as np
import scipy
from hera_cal import io, utils
from collections import OrderedDict as odict
from copy import deepcopy
import warnings
import uvtools
import argparse
from hera_cal.abscal import fft_dly


def freq_filter(gains, wgts, freqs, filter_scale=10.0, tol=1e-09, window='tukey', skip_wgt=0.1,
                maxiter=100, **win_kwargs):
    '''Frequency-filter calibration solutions on a given scale in MHz using uvtools.dspec.high_pass_fourier_filter.
    Befor filtering, removes a per-integration delay using abscal.fft_dly, then puts it back in after filtering.

    Arguments:
        gains: ndarray of shape=(Ntimes,Nfreqs) of complex calibration solutions to filter
        wgts: ndarray of shape=(Ntimes,Nfreqs) of real linear multiplicative weights
        freqs: ndarray of frequency channels in Hz
        filter_scale: frequency scale in MHz to use for the low-pass filter. filter_scale^-1 corresponds
            to the half-width (i.e. the width of the positive part) of the region in fourier
            space, symmetric about 0, that is filtered out.
        tol: CLEAN algorithm convergence tolerance (see aipy.deconv.clean)
        window: window function for filtering applied to the filtered axis.
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
    (dlys, phi) = fft_dly(gains, wgts, df=sdf * 1e9, medfilt=False)  # delays are in seconds
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


class CalibrationSmoother():

    def __init__(self, calfits_list, flags_npz_list=[], antflag_thresh=0.0):
        '''Class for smoothing calibration solutions in time and frequency for a whole day. Initialized with a list of
        calfits files and, optionally, a corresponding list of flag npz files, which must match the calfits files
        one-to-one in time. This function sets up a time grid that spans the whole day with dt = integration time.
        Gains and flags are assigned to the nearest gridpoint using np.searchsorted. It is assumed that:
        1) All calfits and npzs have the same frequencies
        2) The npz times and calfits time map one-to-one to the same set of integrations

        Arguments:
            calfits_list: list of string paths to calfits files containing calibration solutions and flags
            flags_npz_list: list of string paths to npz files containing flags as a function of baseline, times
                and frequency. Must have all baselines for all times. Flags on baselines are broadcast to both
                antennas involved, unless either antenna is completely flagged for all times and frequencies.
            antflag_thresh: float, fraction of flagged pixels across all visibilities (with a common antenna)
                needed to flag that antenna gain at a particular time and frequency. antflag_thresh=0.0 is
                aggressive flag broadcasting, antflag_thresh=1.0 is conservative flag_broadcasting.
        '''
        # load calibration files
        self.cals = calfits_list
        self.gains, self.cal_flags, self.cal_freqs, self.cal_times = odict(), odict(), odict(), odict()
        for cal in self.cals:
            (self.gains[cal], self.cal_flags[cal], _, _, _, self.cal_freqs[cal],
             self.cal_times[cal], _) = io.load_cal(cal, return_meta=True)

        # load flags files
        self.npzs = flags_npz_list
        if len(self.npzs) > 0:
            self.npz_flags, self.npz_freqs, self.npz_times = odict(), odict(), odict()
            for npz in self.npzs:
                self.npz_flags[npz] = utils.synthesize_ant_flags(io.load_npz_flags(npz), threshold=antflag_thresh)
                npz_dict = np.load(npz)
                self.npz_freqs[npz] = npz_dict['freq_array']
                self.npz_times[npz] = np.unique(npz_dict['time_array'])

        # set up time grid (note that it is offset by .5 dt so that times always map to the same
        # index, even if they are slightly different between the npzs and the calfits files)
        all_file_times = sorted(np.array([t for times in self.cal_times.values() for t in times]))
        self.dt = np.median(np.diff(all_file_times))
        self.time_grid = np.arange(all_file_times[0] + self.dt / 2.0, all_file_times[-1] + self.dt, self.dt)
        self.time_indices = {cal: np.searchsorted(self.time_grid, times) for cal, times in self.cal_times.items()}
        if len(self.npzs) > 0:
            self.npz_time_indices = {npz: np.searchsorted(self.time_grid, times) for npz, times in self.npz_times.items()}

        # build multi-file grids for each antenna's gains and flags
        self.freqs = self.cal_freqs[self.cals[0]]
        self.ants = sorted(list(set([k for gain in self.gains.values() for k in gain.keys()])))
        self.gain_grids = {ant: np.ones((len(self.time_grid), len(self.freqs)),
                                        dtype=np.complex) for ant in self.ants}
        self.flag_grids = {ant: np.ones((len(self.time_grid), len(self.freqs)),
                                        dtype=bool) for ant in self.ants}
        for ant in self.ants:
            for cal in self.cals:
                if ant in self.gains[cal]:
                    self.gain_grids[ant][self.time_indices[cal], :] = self.gains[cal][ant]
                    self.flag_grids[ant][self.time_indices[cal], :] = self.cal_flags[cal][ant]
            if len(self.npzs) > 0:
                for npz in self.npzs:
                    if ant in self.npz_flags[npz]:
                        self.flag_grids[ant][self.npz_time_indices[npz], :] += self.npz_flags[npz][ant]

        # perform data quality checks
        self.check_consistency()
        self.reset_filtering()

    def check_consistency(self):
        '''Checks the consistency of the input calibration files (and, if loaded, flag npzs).
        Ensures that all files have the same frequencies, that they are time-ordered, that
        times are internally contiguous in a file and that calibration and flagging times match.
        '''
        all_time_indices = np.array([i for indices in self.time_indices.values() for i in indices])
        assert len(all_time_indices) == len(np.unique(all_time_indices)), \
            'Multiple calibration integrations map to the same time index.'
        for n, cal in enumerate(self.cals):
            assert np.all(np.abs(self.cal_freqs[cal] - self.freqs) < 1e-4), \
                '{} and {} have different frequencies.'.format(cal, self.cals[0])
        if len(self.npzs) > 0:
            all_npz_time_indices = np.array([i for indices in self.npz_time_indices.values() for i in indices])
            assert len(all_npz_time_indices) == len(np.unique(all_npz_time_indices)), \
                'Multiple flagging npz integrations map to the same time index.'
            assert np.all(np.unique(all_npz_time_indices) == np.unique(all_time_indices)), \
                'The number of unique indices for the flag npzs does not match the calibration files.'
            for n, npz in enumerate(self.npzs):
                assert np.all(np.abs(self.npz_freqs[npz] - self.freqs) < 1e-4), \
                    '{} and {} have different frequencies.'.format(npz, self.cals[0])

    def reset_filtering(self):
        '''Reset gain smoothing to the original input gains.'''
        self.filtered_gain_grids = deepcopy(self.gain_grids)
        self.filtered_flag_grids = deepcopy(self.flag_grids)
        self.freq_filtered, self.time_filtered = False, False

    def time_filter(self, filter_scale=1800.0, mirror_kernel_min_sigmas=5):
        '''Time-filter calibration solutions with a rolling Gaussian-weighted average. Allows
        the mirroring of gains and flags and appending the mirrored gains and wgts to both ends,
        ensuring temporal smoothness of the rolling average.

        Arguments:
            filter_scale: float in seconds of FWHM of Gaussian smoothing kernel in time
            mirror_kernel_min_sigmas: Number of stdev into the Gaussian kernel one must go before edge
                effects can be ignored.
        '''
        print 'Now performing time filtering...'
        if self.freq_filtered:
            warnings.warn('It is usually better to time-filter first, then frequency-filter.')
        self.time_filtered = True

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
                filtered_gains are left unchanged and self.wgts and self.cal_flags are set to 0 and True,
                respectively. Only works properly when all weights are all between 0 and 1.
            maxiter: Maximum number of iterations for aipy.deconv.clean to converge.
            win_kwargs : any keyword arguments for the window function selection in aipy.dsp.gen_window.
                    Currently, the only window that takes a kwarg is the tukey window with a alpha=0.5 default.
        '''
        print 'Now performing frequency filtering...'
        if not self.time_filtered:
            warnings.warn('It is usually better to time-filter first, then frequency-filter.')
        self.freq_filtered = True

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

    def write_smoothed_cal(self, output_replace=('.abs.', '.smooth_abs.'), add_to_history='', clobber=False, **kwargs):
        '''Writes time and/or frequency smoothed calibration solutions to calfits, updating input calibration.

        Arguments:
            output_replace: tuple of input calfile substrings: ("to_replace", "to_replace_with")
            add_to_history: appends a string to the history of the output file (in addition to the )
            clobber: if True, overwrites existing file at outfilename
            kwargs: dictionary mapping updated attributes to their new values.
                See pyuvdata.UVCal documentation for more info.
        '''
        print 'Now writing smoothed calibration solutions...'
        for cal in self.cals:
            outfilename = cal.replace(output_replace[0], output_replace[1])
            out_gains = {ant: self.filtered_gain_grids[ant][self.time_indices[cal], :] for ant in self.ants}
            out_flags = {ant: self.filtered_flag_grids[ant][self.time_indices[cal], :] for ant in self.ants}
            io.update_cal(cal, outfilename, gains=out_gains, flags=out_flags,
                          add_to_history=add_to_history, clobber=clobber, **kwargs)


def smooth_cal_argparser():
    '''Arg parser for commandline operation of calibration smoothing.'''
    a = argparse.ArgumentParser(description="Smooth calibration solutions in time and frequency using the hera_cal.smooth_cal module.")
    a.add_argument("calfits_list", type=str, nargs='+', help="list of paths to chronologically sortable calfits files (usually a full day)")
    a.add_argument("--flags_npz_list", type=str, nargs='+', default=[], help="optional list of paths to chronologically\
                   sortable flag npz files (usually a full day)")
    a.add_argument("--infile_replace", type=str, default='.abs.', help="substring of files in calfits_list to replace for output files")
    a.add_argument("--outfile_replace", type=str, default='.smooth_abs.', help="replacement substring for output files")
    a.add_argument("--clobber", default=False, action="store_true", help='overwrites existing file at cal_outfile (default False)')
    a.add_argument("--antflag_thresh", default=0.0, type=float, help="fraction of flagged pixels across all visibilities (with a common antenna) \
                   needed to flag that antenna gain at a particular time and frequency. 0.0 is aggressive flag broadcasting, while 1.0 is \
                   conservative flag broadcasting.")
    a.add_argument("--run_if_first", default=None, type=str, help='only run smooth_cal if the first item in the sorted calfits_list\
                   matches run_if_first (default None means always run)')
    # Options relating to smoothing in time
    time_options = a.add_argument_group(title='Time smoothing options')
    time_options.add_argument("--disable_time", default=False, action="store_true", help="turn off time smoothing")
    time_options.add_argument("--time_scale", type=float, default=1800.0, help="FWHM in seconds of time smoothing Gaussian kernel (default 1800 s)")
    time_options.add_argument("--mirror_sigmas", type=float, default=5.0, help="number of stdev into the Gaussian kernel\
                              one must go before edge effects can be ignored (default 5)")

    # Options relating to smoothing in frequency
    freq_options = a.add_argument_group(title='Frequency smoothing options')
    freq_options.add_argument("--disable_freq", default=False, action="store_true", help="turn off frequency smoothing")
    freq_options.add_argument("--freq_scale", type=float, default=10.0, help="frequency scale in MHz for the low-pass filter\
                              (default 10.0 MHz, i.e. a 100 ns delay filter)")
    freq_options.add_argument("--tol", type=float, default=1e-9, help='CLEAN algorithm convergence tolerance (default 1e-9)')
    freq_options.add_argument("--window", type=str, default="tukey", help='window function for frequency filtering (default "tukey",\
                              see aipy.dsp.gen_window for options')
    freq_options.add_argument("--skip_wgt", type=float, default=0.1, help='skips filtering and flags times with unflagged fraction ~< skip_wgt (default 0.1)')
    freq_options.add_argument("--maxiter", type=int, default=100, help='maximum iterations for aipy.deconv.clean to converge (default 100)')
    freq_options.add_argument("--alpha", type=float, default=.3, help='alpha parameter to use for Tukey window (ignored if window is not Tukey)')
    args = a.parse_args()
    return args
