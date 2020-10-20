# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

import numpy as np
try:
    from uvtools import dspec
    HAVE_UVTOOLS = True
except ImportError:
    HAVE_UVTOOLS = False

from . import utils

from .datacontainer import DataContainer
from .vis_clean import VisClean
import argparse
import copy
from pyuvdata import UVFlag

def timeavg_waterfall(data, Navg, flags=None, nsamples=None, wgt_by_nsample=True, rephase=False,
                      lsts=None, freqs=None, bl_vec=None, lat=-30.72152, extra_arrays={}, verbose=True):
    """
    Calculate the time average of a visibility waterfall. The average is optionally
    weighted by a boolean flag array (flags) and also optionally by an Nsample array (nsample),
    such that, for a single frequency channel, the time average is constructed as

    avg_data = sum( data * flag_wgt * nsample ) / sum( flag_wgt * nsample )

    where flag_wgt is constructed as (~flags).astype(np.float).

    Additionally, one can rephase each integration in the averaging window to the LST of the
    window-center before taking their average. This assumes the
    input data are drift-scan phased. See hera_cal.utils.lst_rephase
    for details on the rephasing algorithm. By feeding an nsample array,
    the averaged nsample for each averaging window is computed and returned.

    Parameters
    ----------
    data : ndarray
        2D complex ndarray of complex visibility with shape=(Ntimes, Nfreqs)
        The rows of data are assumed to be ordered chronologically.

    Navg : int
        Number of time samples to average together, with the condition
        that 0 < Navg <= Ntimes. Navg = 1 is no averaging. Navg = Ntimes
        is complete averaging.

    flags : ndarray
        2D boolean ndarray containing data flags with matching shape of data.
        Flagged pixels are True, otherwise False.

    nsamples : ndarray, optional
        2D float ndarray containing the number of pre-averages behind each pixel
        in data. Default is to assume unity for all pixels.

    wgt_by_nsample : bool, optional
        If True, perform time average weighted by nsample, otherwise perform uniform
        average. Default is True.

    rephase : boolean, optional
        If True, phase each integration to the LST of the averaging window-center
        before averaging. Need to feed lsts, freqs and bl_vec if True.

    lsts : ndarray, optional
        1D float array holding the LST [radians] of each time integration in
        data. Shape=(Ntimes,)

    freqs : ndarray, optional
        1D float array holding the starting frequency of each frequency bin [Hz]
        in data. Shape=(Nfreqs,)

    bl_vec : ndarray, optional
        3D float ndarray containing the visibility baseline vector in meters
        in the ENU (TOPO) frame.

    lat : float, optional
        Latitude of observer in degrees North. Default is HERA coordinates.

    extra_arrays : dict, optional
        Dictionary of extra 1D arrays with shape=(Ntimes,) to push through
        averaging windows. For example, a time_array, or
        anything that has length Ntimes.

    verbose : bool, optional
        If True, report feedback to standard output.

    Returns
    -------
    avg_data : ndarray
        2D complex array with time-average spectrum, shape=(Navg_times, Nfreqs)

    win_flags : ndarray
        2D boolean array with OR of flags in averaging window, shape=(Navg_times, Nfreqs)

    avg_nsamples : ndarray
        2D array containing the sum of nsamples of each averaging window, weighted
        by the input flags, if fed. Shape=(Navg_times, Nfreqs)

    avg_lsts : ndarray
        1D float array holding the center LST of each averaging window, if
        lsts was fed. Shape=(Navg_times,).

    avg_extra_arrays : dict
        Dictionary of 1D arrays holding average of input extra_arrays for
        each averaging window, shape=(Navg_times,).
    """
    # type check
    assert isinstance(data, np.ndarray), "data must be fed as an ndarray"
    if rephase:
        assert lsts is not None and freqs is not None and bl_vec is not None, "" \
            "If rephase is True, must feed lsts, freqs and bl_vec."

    # unwrap lsts if fed
    if lsts is not None:
        lsts = np.unwrap(lsts)

    # form flags if None
    if flags is None:
        flags = np.zeros_like(data, dtype=np.bool)
    assert isinstance(flags, np.ndarray), "flags must be fed as an ndarray"

    # turn flags into weights
    flagw = (~flags).astype(np.float)

    # form nsamples if None
    if nsamples is None:
        nsamples = np.ones_like(data, dtype=np.float)
    assert isinstance(nsamples, np.ndarray), "nsamples must be fed as an ndarray"

    # assert Navg makes sense
    Ntimes = data.shape[0]
    assert Navg <= Ntimes and Navg > 0, "Navg must satisfy 0 < Navg <= Ntimes"

    # calculate Navg_times, the number of remaining time samples after averaging
    Navg_times = float(Ntimes) / Navg
    if Navg_times % 1 > 1e-10:
        if verbose:
            print("Warning: Ntimes is not evenly divisible by Navg, "
                  "meaning the last output time sample will be noisier "
                  "than the others.")
    Navg_times = int(np.ceil(Navg_times))

    # form output avg list
    avg_data = []
    win_flags = []
    avg_lsts = []
    avg_nsamples = []
    avg_extra_arrays = dict([('avg_{}'.format(a), []) for a in extra_arrays])

    # iterate through Navg_times
    for i in range(Navg_times):
        # get starting and stopping indices
        start = i * Navg
        end = (i + 1) * Navg
        d = data[start:end, :]
        f = flags[start:end, :]
        fw = flagw[start:end, :]
        n = nsamples[start:end, :]

        # calculate mean_l and l, if lsts was fed
        if lsts is not None:
            lst = lsts[start:end]
            mean_l = np.mean(lst)
            avg_lsts.append(mean_l)

        # rephase data if desired
        if rephase:
            # get dlst and rephase
            dlst = mean_l - lst
            d = utils.lst_rephase(d, bl_vec, freqs, dlst, lat=lat, inplace=False, array=True)

        # form data weights
        if wgt_by_nsample:
            w = fw * n
        else:
            w = fw
        w_sum = np.sum(w, axis=0, keepdims=False).clip(1e-10, np.inf)

        # perfom weighted average of data along time
        ad = np.sum(d * w, keepdims=False, axis=0) / w_sum
        an = np.sum(n * fw, keepdims=False, axis=0)

        # append to data lists
        avg_data.append(ad)
        win_flags.append(np.min(f, axis=0, keepdims=False))
        avg_nsamples.append(an)

        # average arrays in extra_arrays
        for a in extra_arrays:
            avg_extra_arrays['avg_{}'.format(a)].append(np.mean(extra_arrays[a][start:end]))

    avg_data = np.asarray(avg_data, np.complex)
    win_flags = np.asarray(win_flags, np.bool)
    avg_nsamples = np.asarray(avg_nsamples, np.float)
    avg_lsts = np.asarray(avg_lsts, np.float)

    # wrap lsts
    avg_lsts = avg_lsts % (2 * np.pi)

    return avg_data, win_flags, avg_nsamples, avg_lsts, avg_extra_arrays


def apply_fir(data, fir, wgts=None, axis=0):
    """
    Convolves an FIR filter with visibility data.

    Args:
        data : complex ndarray of shape (Ntimes, Nfreqs)
        fir : complex 2d array of shape (Ntimes, Nfreqs)
            holding FIR filter to convolve against data
        wgts : float ndarray of shape (Ntimes, Nfreqs)
            Default is all ones.
        axis : int
            data axis along which to apply FIR

    Returns:
        new_data : complex ndarray of shape (Ntimes, Nfreqs)
            Contains data convolved with fir across
            time for each frequency channel independently.
    """
    # shape checks
    shape = list(data.shape)
    Ntimes, Nfreqs = shape
    assert isinstance(fir, np.ndarray), "fir must be an ndarray"
    if fir.ndim == 1:
        # try to broadcast given axis
        if axis == 0:
            fir = np.repeat(fir[:, None], Nfreqs, axis=1)
        elif axis == 1:
            fir = np.repeat(fir[None, :], Ntimes, axis=0)

    assert (Ntimes, Nfreqs) == fir.shape, "fir shape must match input data along time and frequency"

    # get weights
    if wgts is None:
        wgts = np.ones_like(data, dtype=np.float)

    new_data = np.empty_like(data, dtype=np.complex)

    shape.pop(axis)
    for i in range(shape[0]):
        slices = [i, i]
        slices[axis] = slice(None)
        slices = tuple(slices)
        new_data[slices] = np.convolve(data[slices] * wgts[slices], fir[slices], mode='same')

    return new_data


def frp_to_fir(frp, delta_bin=None, axis=0, undo=False):
    '''
    Transform a fourier profile to an FIR, or vice versa.

    This function assumes the convention of fft for real->fourier space and ifft
    for fourier->real space. The input fourier profile must have monotonically increasing fourier bins.

    Args:
        frp : 1D or 2D ndarray of the fourier profile.
        delta_bin : frp bin width along axis of fourier transform.
        axis : int, axis of frp along which to take fourier transform
        undo : bool, if True converts an fir to frp, else converts frp to fir

    Returns:
        fir : ndarray of the FIR filter, else undo == True then ndarray of frp
        frbins : 1D ndarray of fourier bins [1/delta_bin] if delta_bin is provided, else is None.
    '''
    # generate fir
    frp = np.fft.ifftshift(frp, axes=axis)
    if undo:
        fir = np.fft.fft(frp, axis=axis)
    else:
        fir = np.fft.ifft(frp, axis=axis)
    fir = np.fft.fftshift(fir, axes=axis)

    # generate frbins
    if delta_bin is None:
        frbins = None
    else:
        frbins = np.fft.fftshift(np.fft.fftfreq(len(frp), delta_bin), axes=axis)

    return fir, frbins


def fr_tavg(frp, noise_amp=None, axis=0):
    """
    Calculate the attenuation induced by fourier filtering a noise signal.

    See Ali et al. 2015 Eqn (9)

    Args:
        frp : A 1D or 2D fourier profile
        noise_amp : The noise amplitude (stand dev. not variance) in frate space
            with shape matching input frp
        axis : int, axis of frp along which filtering is done

    Returns:
        t_ratio : ndarray, effective integration ratio t_after / t_before
    """
    if noise_amp is None:
        noise_amp = np.ones_like(frp, dtype=np.float)

    t_ratio = np.sum(np.abs(noise_amp)**2, axis=axis, keepdims=True) / np.sum(np.abs(frp)**2 * np.abs(noise_amp)**2, axis=axis, keepdims=True).clip(1e-10, np.inf)

    return t_ratio


class FRFilter(VisClean):
    """
    FRFilter object. See hera_cal.vis_clean.VisClean.__init__ for instantiation options.
    """
    def timeavg_data(self, data, times, lsts, t_avg, flags=None, nsamples=None, wgt_by_nsample=True,
                     rephase=False, verbose=True, output_prefix='avg', keys=None, overwrite=False):
        """
        Time average data attached to object given a averaging time-scale t_avg [seconds].
        The resultant averaged data, flags, time arrays, etc. are attached to self
        with the name "{}_data".format(output_prefix), etc

        Note: The t_avg provided will be rounded to the nearest time that makes Navg
            an integer, and is stored as self.t_avg and self.Navg.

        Note: Time-averaging data with differing time-dependent flags per freq channel
            can create artificial spectral structure in the averaged data products.
            One can mitigate this by factorizing the flags into time-freq separable masks,
            see self.factorize_flags.

        Args :
            data : DataContainer
                data to time average, must be consistent with self.lsts and self.freqs
            times : 1D array
                Holds Julian Date time array for input data
            lsts : 1D array
                Holds LST time array for input data
            t_avg : float
                Width of time-averaging window in seconds.
            flags : DataContainer
                flags to use in averaging. Default is None.
                Must be consistent with self.lsts, self.freqs, etc.
            nsamples : DataContainer
                nsamples to use in averaging. Default is None.
                Must be consistent with self.lsts, self.freqs, etc.
            wgt_by_nsample : bool
                If True, perform time average weighted by nsample, otherwise perform
                uniform average. Default is True.
            rephase : bool
                If True, rephase data in averaging window to the window-center.
            keys : list of len-3 antpair-pol tuples
                List of data keys to operate on.
            overwrite : bool
                If True, overwrite existing keys in output DataContainers.
        """
        # turn t_avg into Navg
        Ntimes = len(times)
        dtime = np.median(np.abs(np.diff(times))) * 24 * 3600
        Navg = int(np.round((t_avg / dtime)))
        assert Navg > 0, "A t_avg of {:0.5f} makes Navg=0, which is too small.".format(t_avg)
        if Navg > Ntimes:
            Navg = Ntimes
        old_t_avg = t_avg
        t_avg = Navg * dtime

        if verbose:
            print("The t_avg provided of {:.3f} has been shifted to {:.3f} to make Navg = {:d}".format(
                old_t_avg, t_avg, Navg))

        # setup containers
        for n in ['data', 'flags', 'nsamples']:
            name = "{}_{}".format(output_prefix, n)
            if not hasattr(self, name):
                setattr(self, name, DataContainer({}))
            if n == 'data':
                avg_data = getattr(self, name)
            elif n == 'flags':
                avg_flags = getattr(self, name)
            elif n == 'nsamples':
                avg_nsamples = getattr(self, name)

        # setup averaging quantities
        if flags is None:
            flags = DataContainer(dict([(k, np.zeros_like(data[k], np.bool)) for k in data]))
        if nsamples is None:
            nsamples = DataContainer(dict([(k, np.ones_like(data[k], np.float)) for k in data]))

        if keys is None:
            keys = data.keys()

        # iterate over keys
        al = None
        at = None
        for i, k in enumerate(keys):
            if k in avg_data and not overwrite:
                utils.echo("{} exists in output DataContainer and overwrite == False, skipping...".format(k), verbose=verbose)
                continue
            (ad, af, an, al,
             ea) = timeavg_waterfall(data[k], Navg, flags=flags[k], nsamples=nsamples[k],
                                     rephase=rephase, lsts=lsts, freqs=self.freqs, bl_vec=self.blvecs[k[:2]],
                                     lat=self.lat, extra_arrays=dict(times=times), wgt_by_nsample=wgt_by_nsample,
                                     verbose=verbose)
            avg_data[k] = ad
            avg_flags[k] = af
            avg_nsamples[k] = an
            at = ea['avg_times']

        setattr(self, "{}_times".format(output_prefix), np.asarray(at))
        setattr(self, "{}_lsts".format(output_prefix), np.asarray(al))
        self.t_avg = t_avg
        self.Navg = Navg

    def filter_data(self, data, frps, flags=None, nsamples=None,
                    output_prefix='filt', keys=None, overwrite=False,
                    edgecut_low=0, edgecut_hi=0, axis=0, verbose=True):
        """
        Apply an FIR filter to data.

        Args :
            data : DataContainer
                data to time average, must be consistent with self.lsts and self.freqs
            frps : DataContainer
                DataContainer holding 2D fringe-rate profiles for each key in data,
                with values the same shape as data.
            flags : DataContainer
                flags to use in averaging. Default is None.
                Must be consistent with self.lsts, self.freqs, etc.
            nsamples : DataContainer
                nsamples to use in averaging. Default is None.
                Must be consistent with self.lsts, self.freqs, etc.
            keys : list of len-3 antpair-pol tuples
                List of data keys to operate on.
            overwrite : bool
                If True, overwrite existing keys in output DataContainers.
            edgecut_low : int, number of bins to flag on low side of axis
            edgecut_hi : int, number of bins to flag on high side of axis
        """
        if not HAVE_UVTOOLS:
            raise ImportError("FRFilter.filter_data requires uvtools to be installed. Install hera_cal[all]")
        # setup containers
        for n in ['data', 'flags', 'nsamples']:
            name = "{}_{}".format(output_prefix, n)
            if not hasattr(self, name):
                setattr(self, name, DataContainer({}))
            if n == 'data':
                filt_data = getattr(self, name)
            elif n == 'flags':
                filt_flags = getattr(self, name)
            elif n == 'nsamples':
                filt_nsamples = getattr(self, name)

        # setup averaging quantities
        if flags is None:
            flags = DataContainer(dict([(k, np.zeros_like(data[k], np.bool)) for k in data]))
        if nsamples is None:
            nsamples = DataContainer(dict([(k, np.ones_like(data[k], np.float)) for k in data]))

        if keys is None:
            keys = data.keys()

        # iterate over keys
        for i, k in enumerate(keys):
            if k in filt_data and not overwrite:
                utils.echo("{} exists in ouput DataContainer and overwrite == False, skipping...".format(k), verbose=verbose)
                continue

            # get wgts
            w = (~flags[k]).astype(np.float)
            shape = [1, 1]
            shape[axis] = -1
            w *= dspec.gen_window('none', w.shape[axis], edgecut_low=edgecut_low, edgecut_hi=edgecut_hi).reshape(tuple(shape))
            f = np.isclose(w, 0.0)

            # calculate effective nsamples
            eff_nsamples = np.zeros_like(nsamples[k])
            eff_nsamples += np.sum(nsamples[k] * w, axis=axis, keepdims=True) / np.sum(w, axis=axis, keepdims=True).clip(1e-10, np.inf)
            eff_nsamples *= fr_tavg(frps[k], axis=axis) * np.sum(w, axis=axis, keepdims=True).clip(1e-10, np.inf) / w.shape[axis]

            # setup FIR
            fir, _ = frp_to_fir(frps[k], axis=axis, undo=False)

            # apply fir
            dfilt = apply_fir(data[k], fir, wgts=w, axis=axis)

            # append
            filt_data[k] = dfilt
            filt_flags[k] = f
            filt_nsamples[k] = eff_nsamples

    def _write_time_averaged_data(self, output_file, output_prefix='avg', clobber=False):
        """Internal method for writing time-averaged data to pyuvdata objects.

        Parameters
        ----------
        output_file: str, optional
            Name of outputfile to write. uvh5 only :)
        output_prefix: str, optional
            Output prefix for time-averaged arrays to write out.
        clobber: bool, optional
            overwrite output if it exists already. default is False.
        Returns
        -------
            Nothing!
        """
        avg_data = getattr(self, f"{output_prefix}_data")
        avg_flags = getattr(self, f"{output_prefix}_flags")
        avg_nsamples = getattr(self, f"{output_prefix}_nsamples")
        avg_times = getattr(self, f"{output_prefix}_times")
        avg_lsts = getattr(self, f"{output_prefix}_lsts")
        # copy the attached data
        hd_output = copy.deepcopy(self.hd)
        nt_avg = len(avg_times)
        # select arbitrary times to line up with average number of times
        hd_output.select(times=self.times[:nt_avg])
        hd_output.times = avg_times
        hd_output.lsts = avg_lsts
        # set data to averaged data
        hd_output.update(data=avg_data, flags=avg_flags, nsamples=avg_nsamples)
        # change values of time and lst arrays.
        nbls = hd_output.Nbls
        for tind in range(nt_avg):
            hd_output.time_array[nbls * tind: nbls * (tind + 1)] = avg_times[tind]
            hd_output.lst_array[nbls * tind: nbls * (tind + 1)] = avg_lsts[tind]
        hd_output.write_uvh5(output_file, clobber=clobber)
        return hd_output


def time_avg_data_and_write(input_data, output_data, t_avg, wgt_by_nsample=True, rephase=False,
                            verbose=False, clobber=False, flag_output=None):
    """Driver method for averaging and writing out data.

    Parameters
    ----------
    intput_data: str
        name of input data file.
    output_data: str
        name of output data file.
    t_avg: float
        width of time-averaging interval in seconds.
    wgt_by_nsample: bool, optional
        weight by nsamples in time average
        default is True
    verbose: bool, optional
        if true, more outputs.
        default is False
    clobber: bool, optional
        if true, overwrite output ata if it already exists.
        default is False
    flag_output: str, optional
        optional string path to output UVFlag storing time-averaged flags
        separately.

    Returns
    -------
    frf object with time averaged data attached.
    """
    fr = FRFilter(input_data)
    fr.read()
    fr.timeavg_data(fr.data, fr.times, fr.lsts, t_avg,
                    wgt_by_nsample=wgt_by_nsample, rephase=rephase)
    hdo = fr._write_time_averaged_data(output_data, clobber=clobber)
    if flag_output is not None:
        uvf=UVFlag(hdo, mode='flag', copy_flags=True)
        uvf.to_waterfall(keep_pol=False, method='and')
        uvf.write(flag_output)

def time_avg_data_and_write_baseline_list(input_data_list, baseline_list, output_data, t_avg, wgt_by_nsample=True, rephase=False,
                                          verbose=False, clobber=False, flag_output=None):
    """Time-averaging with a baseline cornerturn


    Parameters
    ----------
    intput_data_list: list of strings.
        list of names of input data file to read baselines across.
    baseline_list: list
        list of antpolpairs or antpairs

    output_data: str
        name of output data file.
    t_avg: float
        width of time-averaging interval in seconds.
    wgt_by_nsample: bool, optional
        weight by nsamples in time average
        default is True
    verbose: bool, optional
        if true, more outputs.
        default is False
    clobber: bool, optional
        if true, overwrite output ata if it already exists.
        default is False
    flag_output: str, optional
        string to write flag output. Optional.
    Returns
    -------
    frf object with time averaged data attached.
    """
    fr = FRFilter(input_data_list)
    fr.read(bls=baseline_list, axis='blt')
    fr.timeavg_data(fr.data, fr.times, fr.lsts, t_avg,
                    wgt_by_nsample=wgt_by_nsample, rephase=rephase)
    hdo = fr._write_time_averaged_data(output_data, clobber=clobber)
    if flag_output is not None:
        uvf=UVFlag(hdo, mode='flag', copy_flags=True)
        uvf.to_waterfall(keep_pol=False, method='and')
        uvf.write(flag_output)

def time_average_argparser(multifile=False):
    """argparser for time averaging data.
    """
    a = argparse.ArgumentParser(description="Time-average data.")
    a.add_argument("input_data", type=str, help="name of data file to determine baselines based on posotion in input_data_list.")
    a.add_argument("output_data", type=str, help="name of data file to write out time-average.")
    if multifile:
        a.add_argument("input_data_list", type=str, nargs="+", help="list of data files to read baselines across.")
    a.add_argument("t_avg", type=float, help="number of seconds to average over.")
    a.add_argument("--rephase", default=False, action="store_true", help="rephase to averaging window center.")
    a.add_argument("--dont_wgt_by_nsample", default=False, action="store_true", help="don't weight averages by nsample.")
    a.add_argument("--clobber", default=False, action="store_true", help="Overwrite output files.")
    a.add_argument("--verbose", default=False, action="store_true", help="verbose output.")
    a.add_argument("--flag_output", default=None, type=str, help="optional filename to save a separate copy of the time-averaged flags as a uvflag object.")
    return a
