import numpy as np
import scipy
from hera_cal import io
from pyuvdata import UVData, UVCal
from hera_cal.datacontainer import DataContainer
from collections import OrderedDict as odict
from copy import deepcopy
import warnings
import uvtools
import argparse

def drop_cross_vis(data):
    '''Delete all entries from a DataContiner that not autocorrelations in order to save memory.'''
    for (i,j,pol) in data.keys():
        if i != j or pol[0] != pol[1]:
            del data[(i,j,pol)]
    return data


def synthesize_ant_flags(flags):
    '''Conservatively synthesizes flags on visibilities into flags on antennas. Any flag at any time or
    frequency for visibility Vij is propagated to both antenna i and antenna j, unless that flag is
    due to either antenna i or antenna j being flagged for all times and all frequencies for all 
    visibilities that its involved in, in which case its flags are just applied to that antenna.

    Arguments:
        flags: DataContainer containing boolean data flag waterfalls

    Returns:
        ant_flags: dictionary mapping antenna-pol keys like (1,'x') to boolean flag waterfalls
    '''
    antpols = set([ap for (i,j,pol) in flags.keys() for ap in [(i, pol[0]), (j, pol[1])]])
    is_excluded = {ap: True for ap in antpols}
    for (i,j,pol), flags_here in flags.items():
        if not np.all(flags_here): 
            is_excluded[(i,pol[0])] = False
            is_excluded[(j,pol[1])] = False
    ant_flags = {}
    for (antpol) in antpols:
        if is_excluded[antpol]:
            ant_flags[antpol] = flags[(antpol[0], antpol[0], antpol[1]+antpol[1])]
    for (i,j,pol), flags_here in flags.items():
        if not is_excluded[(i,pol[0])] and not is_excluded[(j,pol[1])]:
            for antpol in [(i,pol[0]), (j,pol[1])]:
                if ant_flags.has_key(antpol):
                    ant_flags[antpol] = np.logical_or(ant_flags[antpol], flags_here)
                else:
                    ant_flags[antpol] = flags_here
    return ant_flags


def build_weights(unnorm_chisq_per_ant, autocorr, flags, binary_wgts = False):
    '''Builds waterfall of linear multiplicative weights to use in smoothing. 
    Our model treats flagged visibilities get 0 weight. The idea is that (chi^2)**-1 is a reasonable
    proxy for an inverse variance weight, but our omnical chisq are unnormalized by the visibility noise
    variance. To get around this, we use autocorreations as a noise proxy, which gets us weights that
    are proportional to (chi^2)**-1.

    Arguments:
        unnorm_chisq_per_ant: numpy array of Omnical's chi^2 per antenna. Units of visibility^2.
        autocorr: numpy array of autocorrelations, taken as a noise level as a function of time and freq
        flags: numpy array of booleans. True means flagged and thus 0 weight.
        binary_wgts: if True, set all weights that are not zero to 1.
    
    Returns:
        wgts: numpy array weights normalized so that the non-zero entries average to 1.
    '''
    # weights ~ (unnorm_chi^2 / sigma^2)**-1
    wgts = np.abs(autocorr)**2 / unnorm_chisq_per_ant
    # Anywhere with chisq == 0 or autocorr == 0 is also treated as flagged
    wgts[np.logical_not(np.isfinite(wgts))] = 0
    wgts[flags] = 0.0
    # Renormalize weights to make skip_wgt work properly
    wgts[wgts > 0] /= np.mean(wgts[wgts > 0])
    if binary_wgts:
        wgts[wgts > 0] = 1.0
    return wgts


def freq_filter(gains, wgts, freqs, filter_scale = 10.0, tol=1e-09, window='none', skip_wgt=0.1, maxiter=100):
    '''Frequency-filter calibration solutions on a given scale in MHz using uvtools.dspec.high_pass_fourier_filter.
    
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
        skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt) 
            Only works properly when all weights are all between 0 and 1. 
        maxiter: Maximum number of iterations for aipy.deconv.clean to converge.

    Returns:
        filtered: filtered gains, ndarray of shape=(Ntimes,Nfreqs) 
    '''
    sdf = np.median(np.diff(freqs)) / 1e9 #in GHz
    filter_size = (filter_scale / 1e3)**-1 #Puts it in ns
    filtered, res, info = uvtools.dspec.high_pass_fourier_filter(gains, wgts, filter_size, sdf, tol=tol, window=window, 
                                                                 skip_wgt=skip_wgt, maxiter=maxiter)
    return filtered


def time_kernel(nInt, tInt, filter_scale = 120.0):
    '''Build time averaging gaussian kernel.
    
    Arguments:
        nInt: number of integrations to be filtered
        tInt: length of integrations (seconds)
        filter_scale: float in seconds of FWHM of Gaussian smoothing kernel in time

    Returns:
        kernel: numpy array of length 2 * nInt + 1
    '''
    kernel_times = np.append(-np.arange(0,nInt*tInt+tInt/2,tInt)[-1:0:-1], np.arange(0,nInt*tInt+tInt/2,tInt))
    filter_std = filter_scale / (2*(2*np.log(2))**.5) # now in seconds
    kernel = np.exp(-kernel_times**2/2/(filter_std)**2)
    return kernel / np.sum(kernel)



def time_filter(gains, wgts, times, filter_scale = 120.0, nMirrors = 0):
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
        nBefore += (padded_gains[1:,:]).shape[0]
        padded_gains = np.vstack((np.flipud(padded_gains[1:,:]),gains,np.flipud(padded_gains[:-1,:])))
        padded_wgts = np.vstack((np.flipud(padded_wgts[1:,:]),gains,np.flipud(padded_wgts[:-1,:])))

    nInt, nFreq = padded_gains.shape
    conv_gains = padded_gains * padded_wgts
    conv_weights = padded_wgts
    kernel = time_kernel(nInt, np.median(np.diff(times))*24*60*60, filter_scale=filter_scale)
    for i in range(nFreq):
        conv_gains[:,i] = scipy.signal.convolve(conv_gains[:,i], kernel, mode='same')
        conv_weights[:,i] = scipy.signal.convolve(conv_weights[:,i], kernel, mode='same')
    conv_gains /= conv_weights
    conv_gains[np.logical_not(np.isfinite(conv_gains))] = 0
    return conv_gains[nBefore: nBefore+len(times), :]



class Calibration_Smoother():

    def __init__(self, binary_wgts = False):
        '''Class for smoothing calibration solutions in time and frequency. Contains functions for 
        loading calfits files and assocaited data (for flags and autocorrelations), performing the
        smoothing, and then writing the results to disk.

        Arguments:
            binary_wgts: if True, set all weights that are not zero to 1. Otherwise, use renormalized
                omnical chi^2 per antenna as a proxy for variance and perform inverse variance weighting.
        '''
        self.has_cal, self.has_prev_cal, self.has_next_cal = False, False, False
        self.has_data, self.has_prev_data, self.has_next_data = False, False, False
        self.binary_wgts = binary_wgts


    def reset_filtering(self):
        '''Reset gain smoothing to the original input gains.'''
        self.filtered_gains = deepcopy(self.gains)
        self.freq_filtered, self.time_filtered = False, False


    def check_consistency(self):
        '''After loading both the data and the calibration solutions, this function 
        makes sure everything (times, freqs, ants, etc.) lines up.'''
        if not (self.has_cal and self.has_data):
            raise AttributeError('Data consistency cannot be checked unless both load_cal() and load_data() have been run.')

        assert(len(self.times) == len(self.data_times))
        #assert(self.times == self.data_times) #TODO: revisit this
        assert(np.all(self.freqs == self.data_freqs))
        for (ant,pol) in self.gains.keys():
            assert((ant, ant, pol+pol) in self.data) #assert data has autocorrelations
        
        if self.has_prev_cal or self.has_prev_data:
            assert(self.has_prev_cal and self.has_prev_data) #should have both or it doesn't make sense
            assert(np.all(self.prev_freqs == self.freqs))
            assert(np.all(self.prev_data_freqs == self.freqs))
            assert(len(self.prev_times) == len(self.prev_data_times))
            #test time contiguity
            assert(np.abs(np.median(np.diff(self.times)) - self.times[0] + self.prev_times[-1]) < 1e-6)
            assert(np.abs(np.median(np.diff(self.times)) - np.median(np.diff(self.prev_times))) < 1e-6)
            #assert(np.all(self.prev_times == self.prev_data_times)) #TODO: revisit this
            for (ant,pol) in self.gains.keys():
                assert((ant,pol) in self.prev_gains.keys()) #assert prev_gains has all the same keys
                assert((ant, ant, pol+pol) in self.prev_data) #assert data has autocorrelations
        
        if self.has_next_cal or self.has_next_data:
            assert(self.has_next_cal and self.has_next_data) #should have both or it doesn't make sense
            assert(np.all(self.next_freqs == self.freqs))
            assert(np.all(self.next_data_freqs == self.freqs))
            assert(len(self.next_times) == len(self.next_data_times))
            #test time contiguity
            assert(np.abs(np.median(np.diff(self.times)) + self.times[-1] - self.next_times[0]) < 1e-6)
            assert(np.abs(np.median(np.diff(self.times)) - np.median(np.diff(self.next_times))) < 1e-6)
            #assert(np.all(self.next_times == self.next_data_times)) #TODO: revisit this
            for (ant,pol) in self.gains.keys():
                assert((ant,pol) in self.next_gains.keys()) #assert next_gains has all the same keys
                assert((ant, ant, pol+pol) in self.next_data) #assert data has autocorrelations


    def build_weights(self, binary_wgts = False):
        '''Builds weights and stores then internally. Runs automatically after loading data and cals.

        Arguments:
            binary_wgts: if True, set all weights that are not zero to 1.
        '''
        self.wgts, self.prev_wgts, self.next_wgts = {}, {}, {}
        for antpol in self.gains.keys():
            auto_key = (antpol[0], antpol[0], antpol[1]+antpol[1])
            self.wgts[antpol] = build_weights(self.quals[antpol], self.data[auto_key], np.logical_or(self.flags[antpol], 
                                              self.data_ant_flags[antpol]), binary_wgts= self.binary_wgts)
            if self.has_prev_cal:
                self.prev_wgts[antpol] = build_weights(self.prev_quals[antpol], self.prev_data[auto_key], 
                                                       np.logical_or(self.prev_flags[antpol], self.prev_data_ant_flags[antpol]), 
                                                       binary_wgts= self.binary_wgts)
            if self.has_next_cal:
                self.next_wgts[antpol] = build_weights(self.next_quals[antpol], self.next_data[auto_key], 
                                                       np.logical_or(self.next_flags[antpol], self.next_data_ant_flags[antpol]), 
                                                       binary_wgts= self.binary_wgts)
        self.cal_flags = {antpol: wgts == 0.0 for antpol, wgts in self.wgts.items()}


    def load_cal(self, cal, prev_cal=None, next_cal=None):
        '''Loads in calibration solutions for smoothing. Optionally, load in previous and
        subsequent calibration files to help avoid edge effects in time smoothing. 
    
        Arguments:
            cal: UVCal object or path to calfits file that is to be smoothed
            prev_cal: UVCal object, path to calfits file, or a list of either in chronological order,
                that comes before the calibration to be smoothed.
            next_cal: UVCal object, path to calfits file, or a list of either in chronological order,
                that comes after the calibration to be smoothed.
        '''
        assert(isinstance(cal, (str, UVCal)))
        self.input_cal = cal
        self.gains, self.flags, self.quals, _, _, self.freqs, self.times, self.pols, hist =  io.load_cal(cal, return_meta=True)
        self.nFreq = len(self.freqs)
        self.nInt =len(self.times)
        self.tInt = np.median(np.diff(self.times))*24.0*60.0*60.0
        self.has_cal = True
        
        if prev_cal is not None:
            assert(isinstance(prev_cal, (str, UVCal, tuple, list, np.ndarray)))
            self.prev_gains, self.prev_flags, self.prev_quals, _, _, self.prev_freqs, self.prev_times, self.prev_pols, prev_hist = io.load_cal(prev_cal, return_meta=True)
            self.has_prev_cal = True

        if next_cal is not None:
            assert(isinstance(next_cal, (str, UVCal, tuple, list, np.ndarray)))
            self.next_gains, self.next_flags, self.next_quals, _, _, self.next_freqs, self.next_times, self.next_pols, next_hist = io.load_cal(next_cal, return_meta=True)
            self.has_next_cal = True

        if self.has_data:
            self.check_consistency()
            self.build_weights(binary_wgts = self.binary_wgts)
        self.reset_filtering()


    def load_data(self, data, prev_data=None, next_data=None, filetype='miriad'):
        '''Loads in data associated with the calibration to be smoothed. Used only to produce
        weights from flags and autocorrelations. If previous and subsequent calibration files
        are also to be loaded, must load associated data files as well.

        Arguments:
            data: UVData object or path to data file corresponding to the calibration to be smoothed
            prev_cal: UVData object or path to data file, or a list of either in chronological order,
                that correspond to the prev_cal file(s) loaded.
            next_cal: UVData object, path to data file, or a list of either in chronological order,
                that correspond to the next_cal file(s) loaded.
            filetype: file format of data. Default 'miriad.' Ignored if input_data is UVData object(s).
        '''
        assert(isinstance(data, (str, UVData)))
        self.data_filetype = filetype
        self.data, data_flags, _, _, self.data_freqs, self.data_times, _, self.data_pols = io.load_vis(data, return_meta=True, filetype=filetype)
        #TODO: speed this up by only loading the autocorrelations and the flags
        self.data = drop_cross_vis(self.data)
        self.data_ant_flags = synthesize_ant_flags(data_flags)
        self.has_data = True

        if prev_data is not None:
            assert(isinstance(prev_data, (str, UVData, tuple, list, np.ndarray)))
            self.prev_data, prev_data_flags, _, _, self.prev_data_freqs, self.prev_data_times, _, _ = io.load_vis(prev_data, return_meta=True, filetype=filetype)
            self.prev_data = drop_cross_vis(self.prev_data)
            self.prev_data_ant_flags = synthesize_ant_flags(prev_data_flags)
            del prev_data_flags
            self.has_prev_data = True

        if next_data is not None:
            assert(isinstance(next_data, (str, UVData, tuple, list, np.ndarray)))
            self.next_data, next_data_flags, _, _, self.next_data_freqs, self.next_data_times, _, _ = io.load_vis(next_data, return_meta=True, filetype=filetype)
            self.next_data = drop_cross_vis(self.next_data)
            self.next_data_ant_flags = synthesize_ant_flags(next_data_flags)
            del next_data_flags
            self.has_next_data = True

        if self.has_cal:
            self.check_consistency()
            self.build_weights()


    def time_filter(self, filter_scale = 120.0, mirror_kernel_min_sigmas = 5):
        '''Time-filter calibration solutions with a rolling Gaussian-weighted average. Uses both 
        prev_cal and next_cal calibration solutions to help avoid edge effects. Also allows
        the mirroring of gains and wgts and appending the mirrored gains and wgts to both ends, 
        ensuring temporal smoothness of the rolling average.
    
        Arguments:
            filter_scale: float in seconds of FWHM of Gaussian smoothing kernel in time
            mirror_kernel_min_sigmas: Number of stdev into the Gaussian kernel one must go before edge
                effects can be ignored. If after adding prev_gains and next_gains on to the end, we still
                have edge effects, then the calibration solutions are iteratively mirrored in time.
        '''
        if self.freq_filtered:
            warnings.warn('It is usually better to time-filter first, then frequency-filter.')

        # Handle times and mirroring
        start_time_index = 0
        times = deepcopy(self.times)
        tInt = np.median(np.diff(times))*24*60*60
        duration = tInt * len(times)
        if self.has_prev_cal:
            start_time_index += len(self.prev_times)
            times = np.append(self.prev_times, times)
            prev_duration = np.median(self.prev_times)*24*60*60 * (len(self.prev_times) - 1) # -1 is for mirroring
        else:
            prev_duration = 0
        if self.has_next_cal:
            times = np.append(times, self.next_times)
            next_duration = np.median(self.next_times)*24*60*60 * (len(self.next_times) - 1) # -1 is for mirroring
        else:
            next_duration = 0
        total_duration = duration + prev_duration + next_duration
        # This is how much duration we need before the main gain and after to ensure no edge effects
        needed_buffer = filter_scale / (2*(2*np.log(2))**.5) * mirror_kernel_min_sigmas 
        # Make sure that the gain array will be sufficiently padded on each side 
        nMirrors = 0
        while (next_duration + nMirrors*total_duration < needed_buffer) and (prev_duration + nMirrors*total_duration < needed_buffer):
            nMirrors += 1

        # Now loop through and apply running Gaussian averages
        for antpol, gains in self.filtered_gains.items():
            if not np.all(self.cal_flags[antpol]):
                g = deepcopy(gains)
                w = deepcopy(self.wgts[antpol])
                if self.has_prev_cal:
                    g = np.vstack((self.prev_gains[antpol], g))
                    w = np.vstack((self.prev_wgts[antpol], w))
                if self.has_next_cal:
                    g = np.vstack((g, self.next_gains[antpol]))
                    w = np.vstack((w, self.next_wgts[antpol]))
                assert(g.shape[0] > 1) #time filtering doesn't make sense if nInt < 2
                time_filtered = time_filter(g, w, times, filter_scale = filter_scale, nMirrors = nMirrors)
                # keep only the part corresponding to the gain of interest
                self.filtered_gains[antpol] = time_filtered[start_time_index:start_time_index + len(self.times), :]

        self.time_filtered = True


    def freq_filter(self, filter_scale = 10.0, tol=1e-09, window='none', skip_wgt=0.1, maxiter=100):
        '''Frequency-filter stored calibration solutions on a given scale in MHz.
    
        Arguments:
            filter_scale: frequency scale in MHz to use for the low-pass filter. filter_scale^-1 corresponds 
                to the half-width (i.e. the width of the positive part) of the region in fourier 
                space, symmetric about 0, that is filtered out. 
            tol: CLEAN algorithm convergence tolerance (see aipy.deconv.clean)
            window: window function for filtering applied to the filtered axis. 
                See aipy.dsp.gen_window for options.        
            skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt) 
                Only works properly when weights are normalized to be 1 on average.
            maxiter: Maximum number of iterations for aipy.deconv.clean to converge.
        '''
        if not self.time_filtered:
            warnings.warn('It is usually better to time-filter first, then frequency-filter.')

        for antpol, gains in self.filtered_gains.items():
            if not np.all(self.cal_flags[antpol]):
                w = self.wgts[antpol]
                self.filtered_gains[antpol] = freq_filter(gains, w, self.freqs, filter_scale=filter_scale,
                                                          tol=tol, window=window, skip_wgt=skip_wgt, maxiter=maxiter)
        self.freq_filtered = True


    def write_smoothed_cal(self, outfilename, add_to_history = '', clobber = False, **kwargs):
        '''Writes time and/or frequency smoothed calibration solutions to calfits, updating input calibration.
        
        Arguments:
            outfilename: filename of the filtered calibration file to be written to disk
            add_to_history: appends a string to the history of the output file
            clobber: if True, overwrites existing file at outfilename
            kwargs: dictionary mapping updated attributes to their new values.
                See pyuvdata.UVCal documentation for more info.
        '''
        io.update_cal(self.input_cal, outfilename, gains = self.filtered_gains, flags = self.cal_flags, 
                      add_to_history = add_to_history, clobber = clobber, **kwargs)



def smooth_cal_argparser():
    '''Arg parser for commandline operation of calibration smoothing.'''
    a = argparse.ArgumentParser(description="Smooth calibration solutions in time and frequency using the hera_cal.smooth_cal module.")
    a.add_argument("cal_infile", type=str, help="path to calfits file to smooth")
    a.add_argument("data", type=str, help="path to associated visibility data file (used only for flags and autocorrelations)")
    a.add_argument("cal_outfile", type=str, help="path to smoothed calibration calfits file")
    a.add_argument("--filetype", type=str, default='miriad', help='filetype of input data files (default "miriad")')
    a.add_argument("--clobber", default=False, action="store_true", help='overwrites existing file at cal_outfile (default False)')
    a.add_argument("--binary_wgts", default=False, action="store_true", help='give all non-flagged times and frequencies equal weights (default False)')

    # Optional neightboring data and calibration solutions
    neighbors = a.add_argument_group(title='Optional neighboring data and calibrations', description='Additional calibration and data files used\
                                     to ensure temporal smoothness of the smoothed solution, but unaffected by this script. Must be\
                                     contiguous with cal_infile and data and in chronological order.')
    neighbors.add_argument("--prev_cal", default=None, nargs='+', help='path to previous calibration file (or files)')
    neighbors.add_argument("--prev_data", default=None, nargs='+', help='path to previous data file (or files)')
    neighbors.add_argument("--next_cal", default=None, nargs='+', help='path to subsequent calibration file (or files)')
    neighbors.add_argument("--next_data", default=None, nargs='+', help='path to subsequent data file (or files)')

    # Options relating to smoothing in time
    time_options = a.add_argument_group(title='Time smoothing options')
    time_options.add_argument("--disable_time", default=False, action="store_true", help="turn off time smoothing")
    time_options.add_argument("--time_scale", type=float, default=120.0, help="FWHM in seconds of time smoothing Gaussian kernel (default 120 sec)")
    time_options.add_argument("--mirror_sigmas", type=float, default=5.0, help="number of stdev into the Gaussian kernel\
                              one must go before edge effects can be ignored (default 5)")

    # Options relating to smoothing in frequency
    freq_options = a.add_argument_group(title='Frequency smoothing options')
    freq_options.add_argument("--disable_freq", default=False, action="store_true", help="turn off frequency smoothing")
    freq_options.add_argument("--freq_scale", type=float, default=10.0, help="frequency scale in MHz for the low-pass filter\
                              (default 10.0 MHz, i.e. a 100 ns delay filter)")
    freq_options.add_argument("--tol", type=float, default=1e-9, help='CLEAN algorithm convergence tolerance (default 1e-9)')
    freq_options.add_argument("--window", type=str, default="none", help='window function for frequency filtering (default "none",\
                              see aipy.dsp.gen_window for options')
    freq_options.add_argument("--skip_wgt", type=float, default=0.1, help='skips filtering rows with unflagged fraction ~< skip_wgt (default 0.1)')
    freq_options.add_argument("--maxiter", type=int, default=100, help='maximum iterations for aipy.deconv.clean to converge (default 100)')
    args = a.parse_args()
    return args
