import numpy as np
from hera_cal import io, utils, datacontainer
from pyuvdata import UVData
import pyuvdata.utils as uvutils
from collections import OrderedDict as odict
import copy
import os


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
            print "Warning: Ntimes is not evenly divisible by Navg, " \
                "meaning the last output time sample will be noisier " \
                "than the others."
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


class FRFilter(object):
    """
    Fringe Rate Filter
    """

    def __init__(self):
        """
        Fringe Rate Filter
        """
        pass

    def load_data(self, input_data, filetype='miriad'):
        """
        Load in visibility data for filtering

        Parameters
        ----------
        input_data : HERAData object or str
            HERAData object or string filepath to visibility data

        filetype : str
            File format of the data. Only miriad is currently supported.
        """
        assert isinstance(input_data, (UVData, str, np.str, io.HERAData)), "input_data must be fed as a HERAData, UVData object or a string filepath"

        # load HERAData if fed as string
        if isinstance(input_data, (str, np.str)):
            # TODO: Need HERAData to take UVData objects
            self.input_data = io.HERAData(input_data, filetype=filetype)
            self.input_data.read()
        elif isinstance(input_data, UVData):
            # promote UVData to HERAData
            self.input_data = input_data
            self.input_data.__class__ = io.HERAData
            self.input_data._determine_blt_slicing()
            self.input_data._determine_pol_indexing()
        else:
            self.input_data = input_data

        self.filetype = filetype

        # read all the data into datacontainers
        self.data, self.flags, self.nsamples = self.input_data.build_datacontainers()

        # read the metadata: assign individually to guard against code
        # changes within hera_cal.io implicitly changing variable names
        mdict = self.input_data.get_metadata_dict()
        self.antpos = mdict['antpos']
        self.ants = mdict['ants']
        self.freqs = mdict['freqs']
        self.times = mdict['times']
        self.lsts = mdict['lsts']
        self.pols = mdict['pols']

        self.Nfreqs = len(self.freqs)
        self.Ntimes = len(self.times)
        self.dlst = np.median(np.diff(self.lsts))
        self.dtime = np.median(np.diff(self.times))
        self.bls = sorted(set([k[:2] for k in self.data.keys()]))
        self.blvecs = odict([(bl, self.antpos[bl[0]] - self.antpos[bl[1]]) for bl in self.bls])
        self.lat = self.input_data.telescope_location_lat_lon_alt[0] * 180 / np.pi

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
        Navg = int(np.round((t_avg / (3600.0 * 24) / self.dtime)))
        assert Navg > 0, "A t_avg of {:0.5f} makes Navg=0, which is too small.".format(t_avg)
        if Navg > self.Ntimes:
            Navg = self.Ntimes
        old_t_avg = t_avg
        t_avg = Navg * self.dtime * 3600.0 * 24

        if verbose:
            print "The t_avg provided of {:.1f} has been shifted to {:.1f} to make Navg = {:d}".format(old_t_avg, t_avg, Navg)

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

        self.avg_data = datacontainer.DataContainer(avg_data)
        self.avg_flags = datacontainer.DataContainer(avg_flags)
        self.avg_nsamples = datacontainer.DataContainer(avg_nsamples)
        self.avg_lsts = al
        self.avg_times = ea['avg_times']
        self.t_avg = t_avg
        self.Navg = Navg

    def write_data(self, outfilename, write_avg=True, filetype='miriad', add_to_history='', overwrite=False):
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

        Returns
        -------
        new_hd : HERAData object
            A copy of the input_data object, but with updated data
            and relevant metadata.
        """
        # check output
        if os.path.exists(outfilename) and not overwrite:
            print "{} already exists, not overwriting...".format(outfilename)
            return

        # create new HERAData object
        new_hd = copy.deepcopy(self.input_data)

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
            pol_ind = np.argmax(p in new_uvd.polarization_array)
            new_uvd.data_array[blts_inds, 0, :, pol_ind] = data[k]
            new_uvd.flag_array[blts_inds, 0, :, pol_ind] = flags[k]
            new_uvd.nsample_array[blts_inds, 0, :, pol_ind] = nsamples[k]
            new_uvd.time_array[blts_inds] = times
            new_uvd.lst_array[blts_inds] = lsts

        # write data
        if filetype == 'miriad':
            new_hd.write_miriad(outfilename, clobber=True)
        else:
            raise NotImplementedError("filetype {} not recognized".format(filetype))

        return new_hd
