# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

"""Module for predicting noise on visibilties using autocorrelations and for producing noise standard deviation data files."""

import numpy as np
import argparse
import scipy
from astropy import units

from . import io
from . import utils
from .utils import split_pol, join_pol
from .apply_cal import calibrate_in_place
from .datacontainer import DataContainer


def interleaved_noise_variance_estimate(vis, kernel=[[1, -2, 1], [-2, 4, -2], [1, -2, 1]]):
    '''Estimate the noise on a visibility per frequency and time using weighted differecing of
    neighboring frequency and time samples.

    Arguments:
        vis: complex visibility waterfall, usually a numpy array of size (Ntimes, Nfreqs)
        kernel: differencing kernel for how to weight each visibility relative to its neighbors
            in time and frequency. Must sum to zero and must be 2D (either dimension could be length 1)

    Returns:
        variance: estimate of the noise variance on the input visibility with the same shape
    '''
    assert (np.sum(kernel) == 0), 'The kernal must sum to zero for difference-based noise estimation.'
    assert (np.array(kernel).ndim == 2), 'The kernel must be 2D.'
    variance = np.abs(scipy.signal.convolve2d(vis, kernel, mode='same', boundary='wrap'))**2
    variance /= np.sum(np.array(kernel)**2)
    return variance


def infer_dt(times_by_bl, bl, default_dt=None):
    '''Attemps to infer the length of each integration (i.e. Delta t) for a baseline from times by bl.
    If len(times_by_bl[bl]) is 1, this is inferred from Delta t of another baseline where len(times_by_bl[bl]) > 1.

    Arguments:
        times_by_bl: dictionary mapping antenna pair tuples to 1D time arrays (usually in JD)
        bl: antpair tuple e.g. (0, 1) or basebline tuple e.g. (0, 1, 'ee'). Polarization ignored.
        default_dt: default value to return if times cannot be infered. Default None raises ValueError.
            Units should match those in times_by_bl (typically in JD).

    Returns:
        dt: float. Delta time in units of times_by_bl
    '''
    # normal operation
    if len(times_by_bl[bl[:2]]) > 1:
        return np.median(np.ediff1d(times_by_bl[bl[0:2]]))

    # try to infer dt from other baselines
    elif len(times_by_bl[bl[:2]]) == 1:
        for bl2 in times_by_bl:
            if len(times_by_bl[bl2]) > 1:
                # this assumes that all baselines have the same total time
                return np.median(np.ediff1d(times_by_bl[bl2])) * len(times_by_bl[bl2])

    # if dt cannot be inferred
    if default_dt is not None:
        return default_dt
    if len(times_by_bl[bl[:2]]) == 0:
        raise ValueError('Cannot infer dt when len(times_by_bl[bl[:2]]) == 0.')
    else:
        raise ValueError('Cannot infer dt when all len(times_by_bl) == 1 or fewer.')


def predict_noise_variance_from_autos(bl, data, dt=None, df=None, nsamples=None,
                                      auto_ant=None, noise_excess=1.0):
    '''Predict the noise variance on a baseline using autocorrelation data
    using the formula sigma^2 = Vii * Vjj / Delta t / Delta nu * noise_excess.

    Arguments:
        bl: baseline tuple of the form (0, 1, 'nn')
        data: DataContainer containing autocorrelation data of the two antennas in bl
        dt: integration time in seconds. If None, will try infer this
            from the times stored in the DataContainer.
        df: channel width in Hz. If None, will try to infer this from
            from the frequencies stored in the DataContainer
        nsamples: DataContainer mapping bl tuples to numpy arrays of the number
            integrations for that given baseline. Must include nsamples[bl].
        auto_ant: int
            For some cases, like redundant averaging, the auto corresponding
            to the individual antennas is not available. In this case, the
            user should use this keyword to specify a single auto antenna from
            which to derive this statistic.
        noise_excess: factor by which the noise variance exceeds the radiometer
            equation, Vii * Vjj / Delta t / Delta nu. The default, 1.0, is the
            radiometer equation alone. Typically this is the first value returned
            by correlator_noise_covariance(), which accounts for how the correlator
            forms and accumulates spectra, e.g.
            noise_excess=correlator_noise_covariance(kind='sum')[0] for HERA's sums.

    Returns:
        Noise variance predicted on baseline bl in units of data squared.
    '''
    if dt is None:
        dt = infer_dt(data.times_by_bl, bl) * units.si.day.to(units.si.s)
    if df is None:
        assert (len(data.freqs) > 1)  # cannot infer channel width if only one channel is present
        df = np.median(np.ediff1d(data.freqs))

    ap1, ap2 = split_pol(bl[2])
    if auto_ant is None:
        auto_bl1, auto_bl2 = (bl[0], bl[0], join_pol(ap1, ap1)), (bl[1], bl[1], join_pol(ap2, ap2))
    else:
        auto_bl1, auto_bl2 = (auto_ant, auto_ant, join_pol(ap1, ap1)), (auto_ant, auto_ant, join_pol(ap2, ap2))

    var = noise_excess * np.abs(data[auto_bl1] * data[auto_bl2] / dt / df)
    if nsamples is not None:
        return var / nsamples[bl]
    return var


def correlator_noise_covariance(kind='sum', n_taps=4, window='hamming', fwidth=1.0, n_fft=16384,
                                spectra_per_interleave=2, fine_channels_summed=4, max_lag=1):
    '''Predict how a polyphase filterbank (PFB) correlator makes visibility noise depart from the radiometer equation.

    A PFB forms each spectrum from n_taps * n_fft input samples, so consecutive spectra share samples, and each channel's
    response overlaps its neighbors'. For Gaussian noise that is independent between the two antennas, the noise on the
    per-spectrum visibility X_k Y_k^* is then correlated between nearby spectra and channels by |rho|^2, where rho is the
    PFB's own correlation of white noise. Accumulating spectra (all of them for sums, or alternating blocks of them for
    even and odd) and summing fine channels into output channels turns that into a variance excess over the radiometer
    equation and a covariance between nearby output channels. Separate integrations are treated as independent: each
    holds many spectra, and only the few at its edges overlap the next one's. Quantization is not modeled.

    The defaults describe HERA's SNAP-based correlator:
        * hera_corr_f's pfb_fir_8192ch_core is a CASPER pfb_fir_real with TotalTaps = 4, WindowType = hamming, fwidth = 1,
          and PFBSize = 14 (a 16384-point FFT of real samples, so 8192 channels). The coefficients built into it match
          the prototype filter used here to within rounding.
        * paper_gpu's X-engines alternate between even and odd every 2 spectra (N_TIME_PER_PACKET = TIME_DEMUX = 2).
        * The catcher sums 4 fine channels into each output channel (CATCHER_CHAN_SUM_BDA = 4).
    With these, sums have 1.0926 times the radiometer equation's variance, with a correlation of 0.0063 between adjacent
    channels. Diffs have 1.0190 and 0.0035, and even or odd alone 1.0558 and 0.0049. Channels two or more apart are
    uncorrelated to better than 1e-7.

    Arguments:
        kind: which accumulation of spectra the visibilities are: 'sum', 'diff', 'even', or 'odd'
        n_taps: number of PFB taps (CASPER's TotalTaps)
        window: window applied to the PFB's sinc prototype filter. Any name scipy.signal.get_window accepts.
        fwidth: scales the prototype's sinc, whose zeros are n_fft / fwidth samples apart (CASPER's fwidth)
        n_fft: length of the PFB's FFT in input samples, i.e. 2**PFBSize in CASPER. For real input, this is twice the
            number of PFB channels.
        spectra_per_interleave: number of consecutive spectra that go into even before switching to odd. This only
            matters for diffs, even, and odd.
        fine_channels_summed: number of adjacent PFB channels summed into each output channel
        max_lag: largest separation between output channels, in channels, for which to compute the covariance

    Returns:
        covariance: numpy array of length max_lag + 1 in units of the radiometer equation's variance, Vii * Vjj / dt / df.
            covariance[0] is the variance excess, typically passed to predict_noise_variance_from_autos() as noise_excess.
            covariance[l] is the covariance with the output channel l channels away, so covariance[l] / covariance[0] is
            the correlation. In delay, the noise power is proportional to
            covariance[0] + 2 * sum_l covariance[l] * cos(2 pi tau l df).
    '''
    # weights of one period of the even/odd interleave, standing in for an integration of many spectra
    position = np.arange(2 * spectra_per_interleave)
    even = (position // spectra_per_interleave) % 2 == 0
    weights = {'sum': np.ones(len(position)), 'diff': np.where(even, 1.0, -1.0), 'even': 1.0 * even, 'odd': 1.0 * ~even}
    if kind not in weights:
        raise ValueError(f"kind must be 'sum', 'diff', 'even', or 'odd', not {kind!r}.")
    a = weights[kind]

    # CASPER's pfb_fir prototype filter: the window times a sinc with zeros n_fft / fwidth samples apart
    n = np.arange(n_taps * n_fft)
    h = scipy.signal.get_window(window, n_taps * n_fft, fftbins=False) * np.sinc(fwidth * (n / n_fft - n_taps / 2))

    # Covariance of the per-spectrum visibility noise between fine channels dk apart and spectra dm apart, |rho(dk, dm)|^2.
    # The sum over samples that gives rho only depends on the sample index modulo n_fft, so it folds into one FFT per dm.
    dms = np.arange(1 - n_taps, n_taps)
    dks = np.arange(1 - fine_channels_summed, fine_channels_summed * (max_lag + 1))
    per_spectrum_cov = np.zeros((len(dks), len(dms)))
    for j, dm in enumerate(dms):
        lo, hi = max(0, -dm * n_fft), len(h) - max(0, dm * n_fft)
        overlap = np.zeros(len(h))
        overlap[lo:hi] = h[lo:hi] * h[lo + dm * n_fft:hi + dm * n_fft]
        rho = np.fft.fft(overlap.reshape(n_taps, n_fft).sum(axis=0))[dks % n_fft] / np.sum(h**2)
        per_spectrum_cov[:, j] = np.abs(rho)**2

    # accumulate over spectra (the mean of a_m a_{m + dm}), then over the pairs of fine channels in two output channels
    accumulated = per_spectrum_cov @ np.array([np.mean(a * np.roll(a, -dm)) for dm in dms])
    offsets = np.arange(1 - fine_channels_summed, fine_channels_summed)  # fine-channel offsets, each made by this many pairs:
    pairs = fine_channels_summed - np.abs(offsets)
    covariance = [np.sum(pairs * accumulated[fine_channels_summed * lag + offsets - dks[0]]) for lag in range(max_lag + 1)]
    return np.array(covariance) / (fine_channels_summed * np.mean(a**2))


def per_antenna_noise_std(autos, dt=None, df=None):
    '''Predicts per-antenna noise using autocorrelation data. The result is a noise standard deviation. To predict
    the noise variance on e.g. (1, 2, 'ne'), one would use noise[1, 1, 'nn'] * noise[2, 2, 'ee'].

    Arguments:
        autos: DataContainer containing autocorrelation data
        dt: integration time in seconds. If None, will try infer this from the times stored in the DataContainer
        df: channel width in Hz. If None, will try to infer this from the frequencies stored in the DataContainer

    Returns:
        noise: DataContainer mapping autocorrelation baselines like (1, 1, 'nn') to predictions for the standard
            deviation of noise for that antenna (in the same units as the autocorrelations themselves).
    '''
    noise = {}
    for bl in autos:
        if (bl[0] == bl[1]) and (split_pol(bl[2])[0] == split_pol(bl[2])[1]):
            noise[bl] = np.sqrt(predict_noise_variance_from_autos(bl, autos, dt=dt, df=df))
    return DataContainer(noise)


def write_per_antenna_noise_std_from_autos(infile, outfile, calfile=None, gain_convention='divide', add_to_history='', clobber=False):
    '''Loads autocorrelations and uses them to predict the per-antenna noise standard deviation on visibilities (e.g. to predict
    the noise variance on e.g. (1, 2, 'ne'), one would use noise[1, 1, 'nn'] * noise[2, 2, 'ee']). Optionally applies calibration
    solutions with associated flags. Only reads and writes to .uhv5 files.

    Arguments:
        infile: string path to .uvh5 visibility data file from which to extract autocorrelations
        outfile: string path to .uvh5 output data file of noise standard deviations (saved as if they were autocorrelations).
        calfile: optional string path to .calfits calibration file to apply to autocorrelations before computing noise.
            Can also take a list of paths if the .calfits files can be combined into a single UVCal object.
        gain_convention: str, either 'divide' or 'multiply'. 'divide' means V_obs = gi gj* V_true,
            'multiply' means V_true = gi gj* V_obs.
        add_to_history: appends a string to the history of the infile when writing the outfile
        clobber: if True, overwrites existing file at outfile
    '''
    hd = io.HERAData(infile)
    auto_bls = [bl for bl in hd.bls if (bl[0] == bl[1] and split_pol(bl[2])[0] == split_pol(bl[2])[1])]
    autos, auto_flags, _ = hd.read(bls=auto_bls)
    if calfile is not None:
        hc = io.HERACal(calfile)
        gains, cal_flags, _, _ = hc.read()
        calibrate_in_place(autos, gains, data_flags=auto_flags, cal_flags=cal_flags, gain_convention=gain_convention)
    noise = per_antenna_noise_std(autos)
    hd.update(data=noise, flags=auto_flags)
    hd.history += utils.history_string(add_to_history)
    hd.write_uvh5(outfile, clobber=clobber)


def noise_std_argparser():
    '''Arg parser for commandline operation of noise_from_autos.py'''
    a = argparse.ArgumentParser(description="Read autocorrelations from a .uvh5 file and write predictions for noise standard \
                                             deviations to disk as a .uvh5 file, optionally calibrating")
    a.add_argument("infile", type=str, help="path to .uvh5 visibility data file from which to extract autocorrelations")
    a.add_argument("outfile", type=str, help="path to .uvh5 output data file of noise standard deviations")
    a.add_argument("--calfile", type=str, default=None, nargs="+", help="optional path to new calibration calfits file (or files) to apply")
    a.add_argument("--gain_convention", type=str, default='divide',
                   help="'divide' means V_obs = gi gj* V_true, 'multiply' means V_true = gi gj* V_obs.")
    a.add_argument("--clobber", default=False, action="store_true", help='overwrites existing file at outfile')
    return a
