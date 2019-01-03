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
    one can also construct the averaged nsample for each averaging window.

    Parameters
    ----------
    data : ndarray
        2D complex ndarray of complex visibility with shape=(Ntimes, Nfreqs)
        The rows of data are assumed to be ordered chronologically, in either
        asending or descending order.

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
        if True, shift the phase center of each integration to the
        LST of the averaging window-center before averaging. Need to
        feed lsts, freqs and bl_vec if True.

    lsts : ndarray, optional
        1D float array holding the LST [radians] of each time integration in
        data. Shape=(Ntimes,)

    freqs : ndarray, optional
        1D float array holding the starting frequency of each frequency bin [Hz]
        in data. Shape=(Nfreqs,)

    bl_vec : ndarray, optional
        3D float ndarray containing baseline vector of visibility in meters
        in the ENU (TOPO) frame.

    lat : float, optional
        Latitude of observer in degrees North. Default is HERA coordinates.

    extra_arrays : dict, optional
        Dictionary of extra 1D arrays with shape=(Ntimes,) to push through
        averaging windows. For example, a time_array, or
        anything that has length Ntimes.

    verbose : bool, optional
        if True, report feedback to standard output.

    Returns (output_dictionary)
    -------
    output_dictionary : dictionary
        A dictionary containing the following variables

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

    avg_data = np.array(avg_data, np.complex)
    win_flags = np.array(win_flags, np.bool)
    avg_nsamples = np.array(avg_nsamples, np.float)
    avg_lsts = np.array(avg_lsts, np.float)

    # wrap lsts
    avg_lsts = avg_lsts % (2 * np.pi)

    return dict(avg_data=avg_data, win_flags=win_flags, avg_nsamples=avg_nsamples,
                avg_lsts=avg_lsts, avg_extra_arrays=avg_extra_arrays)


class FRFilter(VisClean):
    """
    Fringe Rate Filter object.
    """

    def timeavg_data(self, t_avg, rephase=False, verbose=True):
        """
        Time average data attached to object given a averaging time-scale t_avg [seconds].
        The time-averaged data, flags, time arrays, etc. are stored in avg_* attributes.
        Note that although denoted avg_flags for consistency, this array stores the AND
        of flags in each averaging window.

        The t_avg provided will be rounded to the nearest time that makes Navg
        an integer, and is stored as self.t_avg.

        Parameters
        ----------
        t_avg : float
            Width of time-averaging window in seconds.
        """
        # turn t_avg into Navg given dtime
        Navg = int(np.round((t_avg / self.dtime)))
        assert Navg > 0, "A t_avg of {:0.5f} makes Navg=0, which is too small.".format(t_avg)
        if Navg > self.Ntimes:
            Navg = self.Ntimes
        old_t_avg = t_avg
        t_avg = Navg * self.dtime

        if verbose:
            print("The t_avg provided of {:.1f} has been shifted to {:.1f} to make Navg = {:d}".format(
                old_t_avg, t_avg, Navg))

        # setup lists
        avg_data = odict()
        avg_flags = odict()
        avg_nsamples = odict()

        # iterate over keys
        for i, k in enumerate(self.data.keys()):
            output = timeavg_waterfall(self.data[k], Navg, flags=self.flags[k], nsamples=self.nsamples[k],
                                       rephase=rephase, lsts=self.lsts, freqs=self.freqs, bl_vec=self.blvecs[k[:2]],
                                       lat=self.lat, extra_arrays=dict(times=self.times), verbose=verbose)
            ad, af, an, al, ea = (output['avg_data'], output['win_flags'], output['avg_nsamples'],
                                  output['avg_lsts'], output['avg_extra_arrays'])
            avg_data[k] = ad
            avg_flags[k] = af
            avg_nsamples[k] = an

        self.avg_data = DataContainer(avg_data)
        self.avg_flags = DataContainer(avg_flags)
        self.avg_nsamples = DataContainer(avg_nsamples)
        self.avg_lsts = al
        self.avg_times = ea['avg_times']
        self.t_avg = t_avg
        self.Navg = Navg

    def write_data(self, outfilename, write_avg=True, filetype='miriad', add_to_history='', overwrite=False,
                   run_check=True):
        """
        Write data in FRFringe object. If write_avg == True, write the self.avg_data dictionary,
        else write the self.data dictionary.

        Parameters
        ----------
        outfilename : str
            Path to output visibility data.

        write_avg : bool
            If True, write the avg_data dictionary, else write the data dictionary.

        filetype : str
            Output file format. Currently only miriad is supported.

        add_to_history = str
            History string to add to the HERAData object before writing to disk.

        overwrite: bool
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
        new_hd.history += add_to_history + version.history_string()

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
        else:
            raise NotImplementedError("filetype {} not recognized".format(filetype))

        return new_hd
