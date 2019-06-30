# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

from __future__ import print_function, division, absolute_import

import numpy as np
from collections import OrderedDict as odict
import datetime
from six.moves import range, zip
from uvtools import dspec
from astropy import constants
import copy
import fnmatch
from scipy import signal

from . import io
from . import apply_cal
from . import version
from .datacontainer import DataContainer
from .utils import echo
from .flag_utils import factorize_flags


class VisClean(object):
    """
    VisClean object for visibility CLEANing and filtering.
    """

    def __init__(self, input_data, filetype='uvh5', input_cal=None, link_data=True):
        """
        Initialize the object.

        Args:
            input_data : string, UVData or HERAData object
                Filepath to a miriad, uvfits or uvh5
                datafile, or a UVData or HERAData object.
            filetype : string, options=['miriad', 'uvh5', 'uvfits']
            input_cal : string, UVCal or HERACal object holding
                gain solutions to apply to DataContainers
                as they are built.
            link_data : bool, if True, attempt to link DataContainers
                from HERAData object, otherwise only link metadata if possible.
        """
        # attach HERAData
        self.clear_containers()
        self.hd = io.to_HERAData(input_data, filetype=filetype)

        # attach calibration
        if input_cal is not None:
            self.attach_calibration(input_cal)

        # attach data and/or metadata to object if exists
        self.attach_data(link_data=link_data)

    def soft_copy(self, references=[]):
        """
        Make and return a new object with references (not copies)
        to the data objects in self.

        By default, self.hd, self.data, self.flags and self.nsamples
        are referenced into the new object. Additional attributes
        can be specified by references.

        Args:
            references : list of string
                List of extra attributes to copy references from self to output.
                Accepts wildcard * and ? values.

        Returns:
            VisClean object : A VisClean object with references
                to self.hd, and all attributes specified in references.
        """
        # make a new object w/ only copies of metadata
        newobj = self.__class__(self.hd, link_data=False)
        newobj.hd = self.hd
        newobj.data = self.data
        newobj.flags = self.flags
        newobj.nsamples = self.nsamples

        # iterate through extra attributes
        refs = list(self.__dict__.keys())
        for ref in references:
            atrs = fnmatch.filter(refs, ref)
            for atr in atrs:
                setattr(newobj, atr, getattr(self, atr))

        return newobj

    def attach_data(self, link_data=True):
        """
        Attach DataContainers to self.

        If they exist, attach metadata and/or data from self.hd
        and apply calibration solutions from self.hc if it exists.

        Args:
            link_data : bool, if True, attempt to link DataContainers
                from HERAData object, otherwise only link metadata if possible.
        """
        # link the metadata if they exist
        if self.hd.antenna_numbers is not None:
            mdict = self.hd.get_metadata_dict()
            self.antpos = mdict['antpos']
            self.ants = mdict['ants']
            self.freqs = mdict['freqs']
            self.times = mdict['times']
            self.lsts = mdict['lsts']
            self.pols = mdict['pols']

            self.Nfreqs = len(self.freqs)
            self.Ntimes = len(self.times)  # Does not support BDA for now
            self.dlst = np.median(np.diff(self.lsts))
            self.dtime = np.median(np.diff(self.times)) * 24 * 3600
            self.dnu = np.median(np.diff(self.freqs))
            self.bls = sorted(set(self.hd.get_antpairs()))
            self.blvecs = odict([(bl, self.antpos[bl[0]] - self.antpos[bl[1]]) for bl in self.bls])
            self.bllens = odict([(bl, np.linalg.norm(self.blvecs[bl]) / constants.c.value) for bl in self.bls])
            self.lat = self.hd.telescope_location_lat_lon_alt[0] * 180 / np.pi  # degrees
            self.lon = self.hd.telescope_location_lat_lon_alt[1] * 180 / np.pi  # degrees

        # link the data if they exist
        if self.hd.data_array is not None and link_data:
            data, flags, nsamples = self.hd.build_datacontainers()
            self.data = data
            self.flags = flags
            self.nsamples = nsamples

            # apply calibration solutions if they exist
            if hasattr(self, 'hc'):
                self.apply_calibration(self.hc)

    def clear_containers(self, exclude=[]):
        """
        Clear all DataContainers attached to self.

        Args:
            exclude : list of DataContainer names attached
                to self to exclude from purge.
        """
        keys = list(self.__dict__.keys())
        for key in keys:
            if key in exclude:
                continue
            if isinstance(getattr(self, key), DataContainer):
                setattr(self, key, DataContainer({}))

    def attach_calibration(self, input_cal):
        """
        Attach input_cal to self.

        Attach calibration so-as to apply or unapply
        to visibility data on-the-fly as it
        is piped into DataContainers upon read-in.
        """
        # attach HERACal
        self.hc = io.to_HERACal(input_cal)

    def clear_calibration(self):
        """
        Remove calibration object self.hc to clear memory
        """
        if hasattr(self, 'hc'):
            delattr(self, 'hc')

    def apply_calibration(self, input_cal, unapply=False):
        """
        Apply input_cal self.data.

        Args:
            input_cal : UVCal, HERACal or filepath to calfits file
            unapply : bool, if True, reverse gain convention to
                unapply the gains from the data.
        """
        # ensure its a HERACal
        hc = io.to_HERACal(input_cal)

        # load gains
        cal_gains, cal_flags, cal_quals, cal_tquals = hc.read()

        # apply calibration solutions to data and flags
        gain_convention = hc.gain_convention
        if unapply:
            if gain_convention == 'multiply':
                gain_convention = 'divide'
            elif gain_convention == 'divide':
                gain_convention = 'multiply'
        apply_cal.calibrate_in_place(self.data, cal_gains, self.flags, cal_flags, gain_convention=gain_convention)

    def read(self, **read_kwargs):
        """
        Read from self.hd and attach data and/or metadata to self.

        Args:
            read_kwargs : dictionary
                Keyword arguments to pass to HERAData.read().
        """
        # read data
        self.hd.read(return_data=False, **read_kwargs)

        # attach data
        self.attach_data()

    def write_data(self, data, filename, overwrite=False, flags=None, nsamples=None,
                   times=None, lsts=None, filetype='uvh5', partial_write=False,
                   add_to_history='', verbose=True, **kwargs):
        """
        Write data to file.

        Create a new HERAData and update it with data and write to file. Can only write
        data that has associated metadata in the self.hd HERAData object.

        Args:
            data : DataContainer, holding complex visibility data to write to disk.
            filename : string, output filepath
            overwrite : bool, if True, overwrite output file if it exists
            flags : DataContainer, boolean flag arrays to write to disk with data.
            nsamples : DataContainer, float nsample arrays to write to disk with data.
            times : ndarray, list of Julian Date times to replace in HD
            lsts : ndarray, list of LST times [radian] to replace in HD
            filetype : string, output filetype. ['miriad', 'uvh5', 'uvfits'] supported.
            partial_write : bool, if True, begin (or continue) a partial write to
            the output filename and store file descriptor in self.hd._writers.
            add_to_history : string, string to append to hd history.
            kwargs : additional attributes to update before write to disk.
        """
        # get common keys
        keys = [k for k in self.hd.get_antpairpols() if data.has_key(k)]
        if flags is not None:
            keys = [k for k in keys if flags.has_key(k)]
        if nsamples is not None:
            keys = [k for k in keys if nsamples.has_key(k)]

        # if time_array is fed, select out appropriate times
        if times is not None:
            assert lsts is not None, "Both times and lsts must be fed"
            _times = np.unique(self.hd.time_array)[:len(times)]
        else:
            _times = None

        # select out a copy of hd
        hd = self.hd.select(bls=keys, inplace=False, times=_times)
        hd._determine_blt_slicing()
        hd._determine_pol_indexing()

        # update HERAData data arrays
        hd.update(data=data, flags=flags, nsamples=nsamples)

        # update extra blt arrays
        for ap in hd.get_antpairs():
            s = hd._blt_slices[ap]
            if times is not None:
                hd.time_array[s] = times
            if lsts is not None:
                hd.lst_array[s] = lsts

        # add history
        hd.history += version.history_string(add_to_history)

        # update other kwargs
        for attribute, value in kwargs.items():
            hd.__setattr__(attribute, value)

        # write to disk
        if filetype == 'miriad':
            hd.write_miriad(filename, clobber=overwrite)
        elif filetype == 'uvh5':
            if partial_write:
                hd.partial_write(filename, clobber=overwrite, inplace=True)
                self.hd._writers.update(hd._writers)
            else:
                hd.write_uvh5(filename, clobber=overwrite)
        elif filetype == 'uvfits':
            hd.write_uvfits(filename)
        else:
            raise ValueError("filetype {} not recognized".format(filetype))
        echo("...writing to {}".format(filename), verbose=verbose)

    def vis_clean(self, keys=None, data=None, flags=None, wgts=None, ax='freq', horizon=1.0, standoff=0.0,
                  min_dly=0.0, max_frate=None, tol=1e-6, maxiter=100, window='none', zeropad=0,
                  gain=1e-1, skip_wgt=0.1, filt2d_mode='rect', alpha=0.5, edgecut_low=0, edgecut_hi=0,
                  overwrite=False, output_prefix='clean', add_clean_residual=False, dtime=None, dnu=None,
                  verbose=True):
        """
        Perform a CLEAN deconvolution.

        Run a CLEAN on data and insert the CLEAN components
        into self.clean_model, the CLEAN residual into self.clean_resid,
        the CLEAN flags into self.clean_flags and other relevant info
        into self.clean_info. CLEAN flags are by definition all False
        unless a skip_wgt is triggered, in which case all pixels
        along the CLEAN axis are set to True.

        Args:
            keys : list of bl-pol keys in data to CLEAN
            data : DataContainer, data to clean. Default is self.data
            flags : Datacontainer, flags to use. Default is self.flags
            wgts : DataContainer, weights to use. Default is None.
            ax : str, axis to CLEAN, options=['freq', 'time', 'both']
                Where 'freq' and 'time' are 1D CLEANs and 'both' is a 2D CLEAN
            standoff: fixed additional delay beyond the horizon (in nanosec) to CLEAN [freq cleaning]
            horizon: coefficient to bl_len where 1 is the horizon [freq cleaning]
            min_dly: max delay (in nanosec) used for freq CLEAN is never below this.
            max_frate : max fringe rate (in milli-Hz) used for time CLEANing. See uvtools.dspec.vis_filter for options.
            tol: CLEAN algorithm convergence tolerance (see aipy.deconv.clean)
            gain: The fraction of a residual used in each iteration. If this is too low, clean takes
                unnecessarily long. If it is too high, clean does a poor job of deconvolving.
            window: window function for filtering applied to the filtered axis.
                See uvtools.dspec.gen_window for options.
            alpha : float, if window is Tukey, this is its alpha parameter.
            skip_wgt: skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
                Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
                time. Skipped channels are then flagged in self.flags.
                Only works properly when all weights are all between 0 and 1.
            maxiter: Maximum number of iterations for aipy.deconv.clean to converge.
            edgecut_low : int, number of bins to consider zero-padded at low-side of the FFT axis,
                such that the windowing function smoothly approaches zero. If ax is 'both',
                can feed as a tuple specifying for 0th and 1st FFT axis.
            edgecut_hi : int, number of bins to consider zero-padded at high-side of the FFT axis,
                such that the windowing function smoothly approaches zero. If ax is 'both',
                can feed as a tuple specifying for 0th and 1st FFT axis.
            zeropad : int, number of bins to zeropad on both sides of FFT axis.
            output_prefix : str, attach output model, resid, etc, to self as output_prefix + '_model' etc.
            add_clean_residual : bool, if True, adds the CLEAN residual within the CLEAN bounds
                in fourier space to the CLEAN model. Note that the residual actually returned is
                not the CLEAN residual, but the residual of data - model in real (data) space.
            dtime : float, time spacing of input data [sec], not necessarily integration time!
                Default is self.dtime.
            dnu : float, frequency spacing of input data [Hz]. Default is self.dnu.
            verbose: If True print feedback to stdout
        """
        # type checks
        if ax not in ['freq', 'time', 'both']:
            raise ValueError("ax must be one of ['freq', 'time', 'both']")

        if ax == 'time':
            if max_frate is None:
                raise ValueError("if time cleaning, must feed max_frate parameter")

        # initialize containers
        containers = ["{}_{}".format(output_prefix, dc) for dc in ['model', 'resid', 'flags', 'data']]
        for i, dc in enumerate(containers):
            if not hasattr(self, dc):
                setattr(self, dc, DataContainer({}))
            containers[i] = getattr(self, dc)
        clean_model, clean_resid, clean_flags, clean_data = containers
        clean_info = "{}_{}".format(output_prefix, 'info')
        if not hasattr(self, clean_info):
            setattr(self, clean_info, {})
        clean_info = getattr(self, clean_info)

        # select DataContainers
        if data is None:
            data = self.data
        if flags is None:
            flags = self.flags

        # get keys
        if keys is None:
            keys = data.keys()

        # get weights
        if wgts is None:
            wgts = DataContainer(dict([(k, np.ones_like(flags[k], dtype=np.float)) for k in keys]))

        # get delta bin
        dtime, dnu = self._get_delta_bin(dtime=dtime, dnu=dnu)

        # parse max_frate if fed
        if max_frate is not None:
            if isinstance(max_frate, (int, np.integer, float, np.float)):
                max_frate = DataContainer(dict([(k, max_frate) for k in data]))
            if not isinstance(max_frate, DataContainer):
                raise ValueError("If fed, max_frate must be a float, or a DataContainer of floats")
            # convert kwargs to proper units
            max_frate = DataContainer(dict([(k, np.asarray(max_frate[k])) for k in max_frate]))

        if max_frate is not None:
            max_frate = max_frate * 1e-3
        min_dly /= 1e9
        standoff /= 1e9

        # iterate over keys
        for k in keys:
            if k in clean_model and overwrite is False:
                echo("{} exists in clean_model and overwrite is False, skipping...".format(k), verbose=verbose)
                continue
            echo("Starting CLEAN of {} at {}".format(k, str(datetime.datetime.now())), verbose=verbose)

            # form d and w
            d = data[k]
            f = flags[k]
            fw = (~f).astype(np.float)
            w = fw * wgts[k]

            # freq clean
            if ax == 'freq':
                # zeropad the data
                if zeropad > 0:
                    d, _ = zeropad_array(d, zeropad=zeropad, axis=1)
                    w, _ = zeropad_array(w, zeropad=zeropad, axis=1)

                mdl, res, info = dspec.vis_filter(d, w, bl_len=self.bllens[k[:2]], sdf=dnu, standoff=standoff, horizon=horizon,
                                                  min_dly=min_dly, tol=tol, maxiter=maxiter, window=window, alpha=alpha,
                                                  gain=gain, skip_wgt=skip_wgt, edgecut_low=edgecut_low,
                                                  edgecut_hi=edgecut_hi, add_clean_residual=add_clean_residual)

                # un-zeropad the data
                if zeropad > 0:
                    mdl, _ = zeropad_array(mdl, zeropad=zeropad, axis=1, undo=True)
                    res, _ = zeropad_array(res, zeropad=zeropad, axis=1, undo=True)

                flgs = np.zeros_like(mdl, dtype=np.bool)
                for i, _info in enumerate(info):
                    if 'skipped' in _info:
                        flgs[i] = True

            # time clean
            elif ax == 'time':
                # make sure bad channels are flagged: this is a common failure mode where
                # channels are bad (i.e. data is identically zero) but are not flagged
                # and this causes filtering to hang. Particularly band edges...
                bad_chans = (~np.min(np.isclose(d, 0.0), axis=0, keepdims=True)).astype(np.float)
                w = w * bad_chans  # not inplace for broadcasting

                # zeropad the data
                if zeropad > 0:
                    d, _ = zeropad_array(d, zeropad=zeropad, axis=0)
                    w, _ = zeropad_array(w, zeropad=zeropad, axis=0)

                mdl, res, info = dspec.vis_filter(d, w, max_frate=max_frate[k], dt=dtime, tol=tol, maxiter=maxiter,
                                                  window=window, alpha=alpha, gain=gain, skip_wgt=skip_wgt, edgecut_low=edgecut_low,
                                                  edgecut_hi=edgecut_hi)

                # un-zeropad the data
                if zeropad > 0:
                    mdl, _ = zeropad_array(mdl, zeropad=zeropad, axis=0, undo=True)
                    res, _ = zeropad_array(res, zeropad=zeropad, axis=0, undo=True)

                flgs = np.zeros_like(mdl, dtype=np.bool)
                for i, _info in enumerate(info):
                    if 'skipped' in _info:
                        flgs[:, i] = True

            # 2D clean
            elif ax == 'both':
                # check for completely flagged baseline
                if w.max() > 0.0:
                    # zeropad the data
                    if zeropad > 0:
                        d, _ = zeropad_array(d, zeropad=zeropad, axis=(0, 1))
                        w, _ = zeropad_array(w, zeropad=zeropad, axis=(0, 1))

                    mdl, res, info = dspec.vis_filter(d, w, bl_len=self.bllens[k[:2]], sdf=dnu, max_frate=max_frate[k], dt=dtime,
                                                      standoff=standoff, horizon=horizon, min_dly=min_dly, tol=tol, maxiter=maxiter, window=window,
                                                      alpha=alpha, gain=gain, edgecut_low=edgecut_low, edgecut_hi=edgecut_hi,
                                                      filt2d_mode=filt2d_mode)

                    # un-zeropad the data
                    if zeropad > 0:
                        mdl, _ = zeropad_array(mdl, zeropad=zeropad, axis=(0, 1), undo=True)
                        res, _ = zeropad_array(res, zeropad=zeropad, axis=(0, 1), undo=True)

                    flgs = np.zeros_like(mdl, dtype=np.bool)
                else:
                    # flagged baseline
                    mdl = np.zeros_like(d)
                    res = d - mdl
                    flgs = np.ones_like(mdl, dtype=np.bool)
                    info = {'skipped': True}

            # append to new Containers
            clean_model[k] = mdl
            clean_resid[k] = res
            clean_data[k] = mdl + res * fw
            clean_flags[k] = flgs
            clean_info[k] = info

        # add metadata
        if hasattr(data, 'times'):
            clean_data.times = data.times
            clean_model.times = data.times
            clean_resid.times = data.times
            clean_flags.times = data.times

    def fft_data(self, data=None, flags=None, keys=None, assign='dfft', ax='freq', window='none', alpha=0.1,
                 overwrite=False, edgecut_low=0, edgecut_hi=0, ifft=False, ifftshift=False, fftshift=True,
                 zeropad=0, dtime=None, dnu=None, verbose=True):
        """
        Take FFT of data and attach to self.

        Results are stored as self.assign. Default is self.dfft.
        Take note of the adopted fourier convention via ifft and fftshift kwargs.

        Args:
            data : DataContainer
                Object to pull data to FT from. Default is self.data.
            flags : DataContainer
                Object to pull flags in FT from. Default is no flags.
            keys : list of tuples
                List of keys from clean_data to FFT. Default is all keys.
            assign : str
                Name of DataContainer to attach to self. Default is self.dfft
            ax : str, options=['freq', 'time', 'both']
                Axis along with to take FFT.
            window : str
                Windowing function to apply across frequency before FFT. If ax is 'both',
                can feed as a tuple specifying window for 0th and 1st FFT axis.
            alpha : float
                If window is 'tukey' this is its alpha parameter. If ax is 'both',
                can feed as a tuple specifying alpha for 0th and 1st FFT axis.
            edgecut_low : int, number of bins to consider zero-padded at low-side of the FFT axis,
                such that the windowing function smoothly approaches zero. If ax is 'both',
                can feed as a tuple specifying for 0th and 1st FFT axis.
            edgecut_hi : int, number of bins to consider zero-padded at high-side of the FFT axis,
                such that the windowing function smoothly approaches zero. If ax is 'both',
                can feed as a tuple specifying for 0th and 1st FFT axis.
            ifft : bool, if True, use ifft instead of fft
            ifftshift : bool, if True, ifftshift data along FT axis before FFT.
            fftshift : bool, if True, fftshift along FFT axes.
            zeropad : int, number of zero-valued channels to append to each side of FFT axis.
            dtime : float, time spacing of input data [sec], not necessarily integration time!
                Default is self.dtime.
            dnu : float, frequency spacing of input data [Hz]. Default is self.dnu.
            overwrite : bool
                If dfft[key] already exists, overwrite its contents.
        """
        # type checks
        if ax not in ['freq', 'time', 'both']:
            raise ValueError("ax must be one of ['freq', 'time', 'both']")

        # generate home
        if not hasattr(self, assign):
            setattr(self, assign, DataContainer({}))

        # get home
        dfft = getattr(self, assign)

        # get data
        if data is None:
            data = self.data
        if flags is not None:
            wgts = DataContainer(dict([(k, (~flags[k]).astype(np.float)) for k in flags]))
        else:
            wgts = DataContainer(dict([(k, np.ones_like(data[k], dtype=np.float)) for k in data]))

        # get keys
        if keys is None:
            keys = data.keys()
        if len(keys) == 0:
            raise ValueError("No keys found")

        # get delta bin
        if ax == 'freq':
            _, delta_bin = self._get_delta_bin(dtime=dtime, dnu=dnu)
            axis = 1
        elif ax == 'time':
            delta_bin, _ = self._get_delta_bin(dtime=dtime, dnu=dnu)
            axis = 0
        else:
            delta_bin = self._get_delta_bin(dtime=dtime, dnu=dnu)
            axis = (0, 1)

        # iterate over keys
        j = 0
        for k in keys:
            if k not in data:
                echo("{} not in data, skipping...".format(k), verbose=verbose)
                continue
            if k in dfft and not overwrite:
                echo("{} in self.{} and overwrite == False, skipping...".format(k, assign), verbose=verbose)
                continue

            # FFT
            dfft[k], fourier_axes = fft_data(data[k], delta_bin, wgts=wgts[k], axis=axis, window=window,
                                             alpha=alpha, edgecut_low=edgecut_low, edgecut_hi=edgecut_hi,
                                             ifft=ifft, ifftshift=ifftshift, fftshift=fftshift, zeropad=zeropad)
            j += 1

        if j == 0:
            raise ValueError("No FFT run with keys {}".format(keys))

        if hasattr(data, 'times'):
            dfft.times = data.times
        if ax == 'freq':
            self.delays = fourier_axes
            self.delays *= 1e9
        elif ax == 'time':
            self.frates = fourier_axes
            self.frates *= 1e3
        else:
            self.frates, self.delays = fourier_axes
            self.delays *= 1e9
            self.frates *= 1e3

    def factorize_flags(self, keys=None, spw_ranges=None, time_thresh=0.05, inplace=False):
        """
        Factorize self.flags into two 1D time and frequency masks.

        This works by broadcasting flags across time if the fraction of
        flagged times exceeds time_thresh, otherwise flags are broadcasted
        across channels in a spw_range.

        Note: although technically allowed, this function may give unexpected
        results if multiple spectral windows in spw_ranges have overlap.

        Note: it is generally not recommended to set time_thresh > 0.5, which
        could lead to substantial amounts of data being flagged.

        Args:
            keys : list of antpairpol tuples to operate on
            spw_ranges : list of tuples
                list of len-2 spectral window tuples, specifying the start (inclusive)
                and stop (exclusive) index of the freq channels for each spw.
                Default is to use the whole band.

            time_thresh : float
                Fractional threshold of flagged pixels across time needed to flag all times
                per freq channel. It is not recommend to set this greater than 0.5.
                Fully flagged integrations do not count towards triggering time_thresh.

            inplace : bool, if True, edit self.flags in place, otherwise return a copy
        """
        # get flags
        flags = self.flags
        if not inplace:
            flags = copy.deepcopy(flags)

        # get keys
        if keys is None:
            keys = flags.keys()

        # iterate over keys
        for k in keys:
            factorize_flags(flags[k], spw_ranges=spw_ranges, time_thresh=time_thresh, inplace=True)

        if not inplace:
            return flags

    def zeropad_data(self, data, binvals=None, zeropad=0, axis=-1, undo=False):
        """
        Iterate through DataContainer "data" and zeropad it inplace.

        Args:
            data : DataContainer to zero-pad (or un-pad)
            binvals : bin for data (e.g. times or freqs) to also pad out
                by relevant amount. If axis is an iterable, binvals must also be
            zeropad : int, number of bins on each axis to pad
                If axis is an iterable, zeropad must be also be
            axis : int, axis to zeropad. Can be a tuple
                to zeropad mutliple axes.
            undo : If True, remove zero-padded edges along axis.
        """
        # iterate over data
        for k in data:
            data[k], bvals = zeropad_array(data[k], binvals=binvals, zeropad=zeropad, axis=axis, undo=undo)

        data.binvals = bvals

    def _get_delta_bin(self, dtime=None, dnu=None):
        """
        Get visibility time & frequency spacing.

        Defaults are self.dtime and self.dnu

        Args:
            dtime : float, time spacing [sec]
            dnu : float, frequency spacing [Hz]

        Returns:
            (dtime, dnu)
        """
        if dtime is None:
            dtime = self.dtime

        if dnu is None:
            dnu = self.dnu

        return dtime, dnu


def fft_data(data, delta_bin, wgts=None, axis=-1, window='none', alpha=0.2, edgecut_low=0,
             edgecut_hi=0, ifft=False, ifftshift=False, fftshift=True, zeropad=0):
    """
    FFT data along specified axis.

    Note the fourier convention of ifft and fftshift.

    Args:
        data : complex ndarray
        delta_bin : bin size (seconds or Hz). If axis is a tuple can feed
            as tuple with bin size for time and freq axis respectively.
        wgts : float ndarray of shape (Ntimes, Nfreqs)
        axis : int, FFT axis. Can feed as tuple for 2D fft.
        window : str
            Windowing function to apply across frequency before FFT. If axis is tuple,
            can feed as a tuple specifying window for each FFT axis.
        alpha : float
            If window is 'tukey' this is its alpha parameter. If axis is tuple,
            can feed as a tuple specifying alpha for each FFT axis.
        edgecut_low : int, number of bins to consider zero-padded at low-side of the FFT axis,
            such that the windowing function smoothly approaches zero. If axis is tuple,
            can feed as a tuple specifying for each FFT axis.
        edgecut_hi : int, number of bins to consider zero-padded at high-side of the FFT axis,
            such that the windowing function smoothly approaches zero. If axis is tuple,
            can feed as a tuple specifying for each FFT axis.
        ifft : bool, if True, use ifft instead of fft
        ifftshift : bool, if True, ifftshift data along FT axis before FFT.
        fftshift : bool, if True, fftshift along FT axes after FFT.
        zeropad : int, number of zero-valued channels to append to each side of FFT axis.
            If axis is tuple, can feed as a tuple specifying for each FFT axis.
    Returns:
        dfft : complex ndarray FFT of data
        fourier_axes : fourier axes, if axis is ndimensional, so is this.
    """
    # type checks
    if not isinstance(axis, (tuple, list)):
        axis = [axis]
    if not isinstance(window, (tuple, list)):
        window = [window for i in range(len(axis))]
    if not isinstance(alpha, (tuple, list)):
        alpha = [alpha for i in range(len(axis))]
    if not isinstance(edgecut_low, (tuple, list)):
        edgecut_low = [edgecut_low for i in range(len(axis))]
    if not isinstance(edgecut_hi, (tuple, list)):
        edgecut_hi = [edgecut_hi for i in range(len(axis))]
    if not isinstance(zeropad, (tuple, list)):
        zeropad = [zeropad for i in range(len(axis))]
    if not isinstance(delta_bin, (tuple, list)):
        if len(axis) > 1:
            raise ValueError("delta_bin must have same len as axis")
        delta_bin = [delta_bin]
    else:
        if len(delta_bin) != len(axis):
            raise ValueError("delta_bin must have same len as axis")
    Nax = len(axis)

    # get a copy
    data = data.copy()

    # set fft convention
    fourier_axes = []
    if ifft:
        fft = np.fft.ifft
    else:
        fft = np.fft.fft

    # get wgts
    if wgts is None:
        wgts = np.ones_like(data, dtype=np.float)
    data *= wgts

    # iterate over axis
    for i, ax in enumerate(axis):
        Nbins = data.shape[ax]

        # generate and apply window
        win = dspec.gen_window(window[i], Nbins, alpha=alpha[i], edgecut_low=edgecut_low[i], edgecut_hi=edgecut_hi[i])
        wshape = np.ones(data.ndim, dtype=np.int)
        wshape[ax] = Nbins
        win.shape = tuple(wshape)
        data *= win

        # zeropad
        data, _ = zeropad_array(data, zeropad=zeropad[i], axis=ax)

        # ifftshift
        if ifftshift:
            data = np.fft.ifftshift(data, axes=ax)

        # FFT
        data = fft(data, axis=ax)

        # get fourier axis
        fax = np.fft.fftfreq(data.shape[ax], delta_bin[i])

        # fftshift
        if fftshift:
            data = np.fft.fftshift(data, axes=ax)
            fax = np.fft.fftshift(fax)

        fourier_axes.append(fax)

    if len(axis) == 1:
        fourier_axes = fourier_axes[0]

    return data, fourier_axes


def trim_model(clean_model, clean_resid, dnu, keys=None, noise_thresh=5.0, delay_cut=3000,
               flags=None, kernel_size=None, edgecut_low=0, edgecut_hi=0,):
    """
    Truncate CLEAN model components below some amplitude threshold.

    Estimate the noise in Fourier space and truncate CLEAN model
    components below a specified value times the noise.

    Args:
        clean_model : DataContainer
            Holds clean_model output of self.vis_clean
        clean_resid : DataContainer
            Holds clean_resid output of self.vis_clean
        dnu : float
            Frequency channel width [Hz]
        keys : list of antpairpol tuples
            List of keys to operate on
        noise_thresh : float
            Coefficient times noise to truncate model components below
        delay_cut : float
            Minimum |delay| [ns] above which to use in estimating noise
        kernel_size : int
            Time median filter kernel_size. None is no median filter.
        edgecut_low : int
            Edgecut bins to apply to low edge of frequency band
        edgecut_hi : int
            Edgecut bins to apply to high edge of frequency band

    Returns:
        model : DataContainer
            Truncated clean_model
        noise : DataContainer
            Per integration noise estimate from clean_resid
    """
    # get keys
    if keys is None:
        keys = [k for k in sorted(set(clean_model.keys() + clean_resid.keys())) if k in clean_model and k in clean_resid]

    # estimate noise in Fourier space
    model = DataContainer({})
    noise = DataContainer({})    
    for k in keys:
        # get rfft
        rfft, delays = fft_data(clean_resid[k], dnu, axis=1, window='none', edgecut_low=edgecut_low, edgecut_hi=edgecut_hi, ifft=False, ifftshift=False, fftshift=False)
        delays *= 1e9

        # get clean_resid flags and its NEB
        neb = noise_eq_bandwidth(~np.isclose(clean_resid[k], 0.0))[:, None]

        # get noise estimate
        noise[k] = np.median(np.abs((rfft * neb)[:, np.abs(delays) > delay_cut]), axis=1)

        # median filter it
        if kernel_size is not None:
            n = noise[k]
            nlen = len(n)
            n = np.pad(n, nlen, 'reflect', reflect_type='odd')
            noise[k] = signal.medfilt(n, kernel_size=kernel_size)[nlen:-nlen]

        # get mfft
        mfft, _ = fft_data(clean_model[k], dnu, axis=1, window='none', edgecut_low=edgecut_low, edgecut_hi=edgecut_hi, ifft=False, ifftshift=False, fftshift=False)
        mfft[np.abs(mfft) < (noise[k][:, None] * noise_thresh)] = 0.0

        # re-fft
        mdl, _ = fft_data(mfft, dnu, axis=1, window='none', edgecut_low=0, edgecut_hi=0, ifft=True, ifftshift=False, fftshift=False)
        model[k] = mdl

    return model, noise


def zeropad_array(data, binvals=None, zeropad=0, axis=-1, undo=False):
    """
    Zeropad data ndarray along axis.

    If data is float, int or complex, zeropads with zero.
    If data is boolean, zeropads with True.

    Args:
        data : ndarray to zero-pad (or un-pad)
        binvals : bin values for data (e.g. times or freqs) to also pad out
            by relevant amount. If axis is an iterable, binvals must also be
        zeropad : int, number of bins on each axis to pad
            If axis is an iterable, zeropad must be also be
        axis : int, axis to zeropad. Can be a tuple
            to zeropad mutliple axes.
        undo : If True, remove zero-padded edges along axis.

    Returns:
        padded_data : zero-padded (or un-padded) data
        padded_bvals : bin array(s) padded (or un-padded) if binvals is fed, otherwise None
    """
    # get data type
    bool_dtype = np.issubdtype(data.dtype, np.bool_)

    # type checks
    if not isinstance(axis, (list, tuple, np.ndarray)):
        axis = [axis]
    binvals = copy.deepcopy(binvals)
    if not isinstance(binvals, (list, tuple)):
        binvals = [binvals]
    if not isinstance(zeropad, (list, tuple, np.ndarray)):
        zeropad = [zeropad]
    if isinstance(axis, (list, tuple, np.ndarray)) and not isinstance(zeropad, (list, tuple, np.ndarray)):
        raise ValueError("If axis is an iterable, so must be zeropad.")
    if len(axis) != len(zeropad):
        raise ValueError("len(axis) must equal len(zeropad)")

    for i, ax in enumerate(axis):
        if zeropad[i] > 0:
            if undo:
                s = [slice(None) for j in range(data.ndim)]
                s[ax] = slice(zeropad[i], -zeropad[i])
                s = tuple(s)
                data = data[s]
                if binvals[i] is not None:
                    binvals[i] = binvals[i][s[i]]

            else:
                zshape = list(data.shape)
                zshape[ax] = zeropad[i]
                if bool_dtype:
                    z = np.ones(zshape, np.bool)
                else:
                    z = np.zeros(zshape, data.dtype)
                data = np.concatenate([z, data, z], axis=ax)
                if binvals[i] is not None:
                    dx = np.median(np.diff(binvals[i]))
                    Nbin = binvals[i].size
                    z = np.arange(1, zeropad[i] + 1)
                    binvals[i] = np.concatenate([binvals[i][0] - z[::-1] * dx, binvals[i], binvals[i][-1] + z * dx])

    if len(binvals) == 1:
        binvals = binvals[0]

    return data, binvals


def noise_eq_bandwidth(window, axis=-1):
    """
    Calculate the noise equivalent bandwidth (NEB) of a windowing function
    as
        sqrt(window.size * window.max ** 2 / sum(window ** 2))

    Args:
        window : float ndarray
        axis : int, axis along which to calculate NEB

    Returns
        neb : float or ndarray
            Noise equivalent bandwidth of the window
    """
    return np.sqrt(window.shape[axis] * np.max(window, axis=axis)**2 / np.sum(window**2, dtype=np.float, axis=axis))
