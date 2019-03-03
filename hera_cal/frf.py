# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

from __future__ import print_function, division, absolute_import

import numpy as np
from collections import OrderedDict as odict
import copy
import os
from six.moves import range
from pyuvdata import UVData
import pyuvdata.utils as uvutils
from uvtools import dspec

from . import io
from . import version
from . import utils

from .datacontainer import DataContainer
from .vis_clean import VisClean


def timeavg_waterfall(data, Navg, flags=None, nsamples=None, rephase=False, lsts=None,
                      freqs=None, bl_vec=None, lat=-30.72152, extra_arrays={}, verbose=True):
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

        # form data weights : flag weights * nsample
        w = fw * n
        w_sum = np.sum(w, axis=0, keepdims=False).clip(1e-10, np.inf)

        # perfom weighted average of data along time
        ad = np.sum(d * w, keepdims=False, axis=0) / w_sum
        an = np.sum(w, keepdims=False, axis=0)

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


def apply_fir(data, fir, wgts=None):
    """
    Applies an FIR filter to data across zeroth axis.

    Convolve data against an FIR filter across time (zeroth) axis.
    If fir is 1D, will repeat along frequency to match shape of data.

    Args:
        data : complex ndarray of shape (Ntimes, Nfreqs)
        fir : complex 1d or 2d array of shape (Ntimes,) or (Ntimes, Nfreqs)
            Holds FIR filter to convolve against data
        wgts : float ndarray of shape (Ntimes, Nfreqs)
            Default is all ones.

    Returns:
        new_data : complex ndarray of shape (Ntimes, Nfreqs)
            Contains data convolved with fir across
            time for each frequency channel independently.
    """
    # shape checks
    Ntimes, Nfreqs = data.shape
    assert isinstance(fir, np.ndarray), "fir must be an ndarray"
    if fir.ndim == 1:
        fir = np.repeat(fir[:, None], Nfreqs, axis=1)
    assert (Ntimes, Nfreqs) == fir.shape, "fir shape must match input data along time and frequency"

    # get weights
    if wgts is None:
        wgts = np.ones_like(data, dtype=np.float)

    new_data = np.empty_like(data)

    for i in range(Nfreqs):
        new_data[:, i] = np.convolve(data[:, i] * wgts[:, i], fir[:, i], mode='same')

    return new_data


def fir_to_frp(fir, dt=None):
    '''
    Transform a FIR filter (time domain fr filter) to a fringe rate profile.

    This function assumes the convention of fft for time->fringe-rate and
    ifft for fringe-rate->time. The input FIR must have a monotonically increasing time axis.

    Args:
        fir : 1D or 2D ndarray of the FIR filter with shape (Ntimes,) or (Ntimes, Nfreqs)
        dt : Time spacing [sec] of the time axis.

    Returns:
        frp : 1D or 2D ndarray of the fringe-rate profile.
        frbins : 1D ndarray of fringe-rate bins [Hz] if dt is provided, else is None.
    '''
    # generate frp
    fir = np.fft.ifftshift(fir, axes=0)
    frp = np.fft.fft(fir, axis=0)
    frp = np.fft.fftshift(frp, axes=0)

    # generate frbins
    if dt is None:
        frbins = None
    else:
        frbins = np.fft.fftshift(np.fft.fftfreq(len(fir), dt), axes=0)

    return frp, frbins


def frp_to_fir(frp, dfr=None):
    '''
    Transform a fringe-rate profile into an FIR filter.

    This function assumes the convention of fft for time->fringe-rate and
    ifft for fringe-rate->time. The input FR profile must have a monotonically increasing FR axis.

    Args:
        frp : 1D or 2D ndarray of the FR profile with shape (Nfrates,) or (Nfrates, Nfreqs)
        dfr : Fringe-rate spacing [Hz] of the fringe-rate axis.

    Returns:
        fir : 1D or 2D ndarray of the FIR filter.
        tbins : 1D ndarray of time bins [sec] if dfr is provided, else is None.
    '''
    # generate fir
    frp = np.fft.ifftshift(frp, axes=0)
    fir = np.fft.ifft(frp, axis=0)
    fir = np.fft.fftshift(fir, axes=0)

    # generate frbins
    if dfr is None:
        tbins = None
    else:
        tbins = np.fft.fftshift(np.fft.fftfreq(len(frp), dfr), axes=0)

    return fir, tbins


def fr_tavg(frp, noise_amp=None):
    """
    Calculate the attenuation induced by an FR filter on a noise signal.

    See Ali et al. 2015 Eqn (9)

    Args:
        frp : A 2D fringe-rate profile of shape (Nfrates, Nfreqs)
        noise_amp : The noise amplitude (stand dev. not variance) in frate space
            with shape matching input frp

    Returns:
        t_ratio : ndarray, effective integration t_after / t_before with shape (Nfrates, Nfreqs)
    """
    if frp.ndim == 1:
        frp = np.reshape(frp, (-1, 1))
    if noise_amp is None:
        noise_amp = np.ones_like(frp, dtype=np.float)

    t_ratio = np.sum(np.abs(noise_amp)**2, axis=0, keepdims=True) / np.sum(np.abs(frp)**2 * np.abs(noise_amp)**2, axis=0, keepdims=True).clip(1e-10, np.inf)
    t_ratio = np.repeat(t_ratio, len(frp), axis=0)

    return t_ratio


class FRFilter(VisClean):
    """
    Fringe Rate Filter object. See hera_cal.vis_clean.VisClean.__init__ for instantiation options.
    """

    def timeavg_data(self, data, times, lsts, t_avg, flags=None, nsamples=None, rephase=False,
                     verbose=True, output_prefix='avg', keys=None, overwrite=False):
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
                utils.echo("{} exists in ouput DataContainer and overwrite == False, skipping...".format(k), verbose=verbose)
                continue
            (ad, af, an, al,
             ea) = timeavg_waterfall(data[k], Navg, flags=flags[k], nsamples=nsamples[k],
                                     rephase=rephase, lsts=lsts, freqs=self.freqs, bl_vec=self.blvecs[k[:2]],
                                     lat=self.lat, extra_arrays=dict(times=times), verbose=verbose)
            avg_data[k] = ad
            avg_flags[k] = af
            avg_nsamples[k] = an
            at = ea['avg_times']

        setattr(self, "{}_times".format(output_prefix), at)
        setattr(self, "{}_lsts".format(output_prefix), al)
        self.t_avg = t_avg
        self.Navg = Navg

    def frfilter_data(self, data, times, lsts, frps, flags=None, nsamples=None,
                      output_prefix='frf', keys=None, overwrite=False,
                      edgecut_low=0, edgecut_hi=0, verbose=True):
        """
        Fringe-rate filter (i.e. apply an FIR filter) to data.

        Args : 
            data : DataContainer
                data to time average, must be consistent with self.lsts and self.freqs
            times : 1D array
                Holds Julian Date time array for input data
            lsts : 1D array
                Holds LST time array for input data
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
            edgecut_low : int, number of bins to flag on low side of time axis
            edgecut_hi : int, number of bins to flag on high side of time axis
        """
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
        for i, k in enumerate(keys):
            if k in avg_data and not overwrite:
                utils.echo("{} exists in ouput DataContainer and overwrite == False, skipping...".format(k), verbose=verbose)
                continue

            # setup FIR
            fir, _ = frp_to_fir(frps[k])

            # get wgts
            w = (~flags[k]).astype(np.float)
            w *= dspec.gen_window('none', len(w), edgecut_low=edgecut_low, edgecut_hi=edgecut_hi)[:, None]
            f = np.isclose(w, 0.0)

            # calculate effective nsamples
            eff_nsamples = np.zeros_like(nsamples[k])
            eff_nsamples += np.sum(nsamples[k] * w, axis=0, keepdims=True) / np.sum(w, axis=0, keepdims=True).clip(1e-10, np.inf)
            eff_nsamples *= fr_tavg(frps[k]) * np.sum(w, axis=0, keepdims=True).clip(1e-10, np.inf) / len(w)

            # apply fir
            avg_data[k] = apply_fir(data[k], fir, wgts=w)
            avg_flags[k] = f
            avg_nsamples[k] = eff_nsamples

        setattr(self, "{}_times".format(output_prefix), times)
        setattr(self, "{}_lsts".format(output_prefix), lsts)

    def write_data(self, outfilename, write_avg=True, filetype='uvh5', add_to_history='', overwrite=False,
                   run_check=True):
        """
        Write data in FRFringe object.

        If write_avg == True, write the self.avg_data dictionary,
        else write the self.data dictionary.

        Parameters
        ----------
        outfilename : str
            Path to output visibility data.

        write_avg : bool
            If True, write the avg_data dictionary, else write the data dictionary.

        filetype : str
            Output file format. Currently only miriad is supported.

        add_to_history : str
            History string to add to the HERAData object before writing to disk.

        overwrite : bool
            If True, overwrite output if it exists.

        run_check : bool
            If True, run UVData check before write.

        Returns
        -------
        new_hd : HERAData object
            A copy of the hd object, but with updated data
            and relevant metadata.
        """
        # check output
        if os.path.exists(outfilename) and not overwrite:
            print("{} already exists, not overwriting...".format(outfilename))
            return

        # create new HERAData object
        new_hd = copy.deepcopy(self.hd)
        new_hd.history += version.history_string(add_to_history)

        # set write data references
        if write_avg:
            data = self.avg_data
            flags = self.avg_flags
            nsamples = self.avg_nsamples
            lsts = self.avg_lsts
            times = self.avg_times
        else:
            data = self.data
            flags = self.flags
            nsamples = self.nsamples
            lsts = self.lsts
            times = self.times

        # strip down to appropriate Ntimes
        Ntimes = len(times)
        new_hd.select(times=self.times[:Ntimes], inplace=True)

        # get telescope coords
        lat, lon, alt = new_hd.telescope_location_lat_lon_alt
        lat = lat * 180 / np.pi
        lon = lon * 180 / np.pi

        # Overwrite data
        for k in data.keys():
            blts_inds = new_hd.antpair2ind(*k[:2])
            p = uvutils.polstr2num(k[2])
            pol_ind = np.argmax(p in new_hd.polarization_array)
            new_hd.data_array[blts_inds, 0, :, pol_ind] = data[k]
            new_hd.flag_array[blts_inds, 0, :, pol_ind] = flags[k]
            new_hd.nsample_array[blts_inds, 0, :, pol_ind] = nsamples[k]
            new_hd.time_array[blts_inds] = times
            new_hd.lst_array[blts_inds] = lsts

        if run_check:
            new_hd.check()

        # write data
        if filetype == 'miriad':
            new_hd.write_miriad(outfilename, clobber=True)
        elif filetype == 'uvh5':
            new_hd.write_uvh5(outfilename, clobber=True)
        else:
            raise NotImplementedError("filetype {} not recognized".format(filetype))

        return new_hd
