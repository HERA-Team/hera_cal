# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

import numpy as np
from collections import OrderedDict as odict
import datetime
from uvtools import dspec
import argparse


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
import warnings
from pyuvdata import UVFlag


class VisClean(object):
    """
    VisClean object for visibility CLEANing and filtering.
    """

    def __init__(self, input_data, filetype='uvh5', input_cal=None, link_data=True,
                 round_up_bllens=False,
                 **read_kwargs):
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
            round_up_bllens : bool, optional
                If True, round up baseline lengths to nearest meter for delay-filtering.
                this allows linear filters for baselines with slightly different lengths
                to be hashed to the same matrix. Saves lots of time by only computing
                one unique filtering matrix per flagging pattern and baseline length group.
            read_kwargs : kwargs to pass to UVData.read (e.g. run_check, check_extra and
                run_check_acceptability). Only used for uvh5 filetype
        """
        # attach HERAData
        self.clear_containers()
        self.hd = io.to_HERAData(input_data, filetype=filetype, **read_kwargs)
        # attach calibration
        if input_cal is not None:
            self.attach_calibration(input_cal)

        # attach data and/or metadata to object if exists
        self.attach_data(link_data=link_data)
        self.round_up_bllens = round_up_bllens

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
            self.Nfreqs = len(self.freqs)
        # link the data if they exist
        if self.hd.data_array is not None and link_data:
            self.hd.select(frequencies=self.freqs)
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
        # get overlapping frequency bins
        cal_freqs_in_data = []
        for f in self.freqs:
            match = np.isclose(hc.freqs, f, rtol=1e-10)
            if True in match:
                cal_freqs_in_data.append(np.argmax(match))
        # assert all frequencies in data are found in uvcal
        assert len(cal_freqs_in_data) == len(self.freqs), "Not all freqs in uvd are in uvc"

        for ant in cal_gains:
            cal_gains[ant] = cal_gains[ant][:, cal_freqs_in_data]
            cal_flags[ant] = cal_flags[ant][:, cal_freqs_in_data]
            cal_quals[ant] = cal_quals[ant][:, cal_freqs_in_data]
        if cal_tquals is not None:
            for pol in cal_tquals:
                cal_tquals[pol] = cal_tquals[pol][:, cal_freqs_in_data]

        # apply calibration solutions to data and flags
        gain_convention = hc.gain_convention
        if unapply:
            if gain_convention == 'multiply':
                gain_convention = 'divide'
            elif gain_convention == 'divide':
                gain_convention = 'multiply'
        apply_cal.calibrate_in_place(self.data, cal_gains, self.flags, cal_flags,
                                     gain_convention=gain_convention)

    def apply_flags(self, external_flags=None, overwrite_data_flags=False,
                    flag_zero_times=True, a_priori_flag_yaml=None):
        """
        apply external flags.
        Parameters
        ----------
        external_flags: str, optional.
            Str or list of strings pointing to flag files to apply.

        overwrite_data_flags: bool, optional
            If true, overwrite all data flags for bls that are not entirely flagge.d

        flag_zero_times: bool, optional
            if true, don't overwrite flags where the entire time is flagged.

        a_priori_flag_yaml: str, optional
            path to a yaml file containing manual flags.
        """
        if external_flags is not None:
            external_flags = UVFlag(external_flags)
            # select frequencies and times that match data.
            flag_times = np.unique(external_flags.time_array)
            flag_freqs = np.unique(external_flags.freq_array)
            times_overlapping = []
            freqs_overlapping = []
            for t in flag_times:
                if np.any(np.isclose(self.times, t)):
                    times_overlapping.append(t)
            for f in flag_freqs:
                if np.any(np.isclose(self.freqs, f)):
                    freqs_overlapping.append(f)
            # select frequencies and times that overlap with data.
            external_flags.select(frequencies=freqs_overlapping, times=times_overlapping)
        from hera_qm.xrfi import flag_apply
        # set all flags to False on waterfalls that are not fully flagged
        # if overwrite_data_flags is True.
        if overwrite_data_flags:
            for bl in self.flags:
                if not np.all(self.flags[bl]):
                    if not flag_zero_times:
                        self.flags[bl][:] = False
                    else:
                        self.flags[bl][~np.all(self.flags[bl], axis=1), :] = False
            self.hd.update(flags=self.flags)
        # explicitly keep_existing since we already reset flags.
        if external_flags is not None:
            flag_apply(external_flags, self.hd, force_pol=True, keep_existing=True)
        # apply apriori flag yaml too.
        if a_priori_flag_yaml is not None:
            import hera_qm.utils as qm_utils
            self.hd = qm_utils.apply_yaml_flags(self.hd, a_priori_flag_yaml)

        _, self.flags, _ = self.hd.build_datacontainers()


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
                   add_to_history='', verbose=True, extra_attrs={}, **kwargs):
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
            extra_attrs : additional UVData/HERAData attributes to update before writing
            kwargs : extra kwargs to pass to UVData.write_*() call
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
        hd = self.hd.select(bls=keys, inplace=False, times=_times, frequencies=self.freqs)
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

        # update other extra attrs
        for attribute, value in extra_attrs.items():
            hd.__setattr__(attribute, value)

        # write to disk
        if filetype == 'miriad':
            hd.write_miriad(filename, clobber=overwrite, **kwargs)
        elif filetype == 'uvh5':
            if partial_write:
                hd.partial_write(filename, clobber=overwrite, inplace=True, **kwargs)
                self.hd._writers.update(hd._writers)
            else:
                hd.write_uvh5(filename, clobber=overwrite, **kwargs)
        elif filetype == 'uvfits':
            hd.write_uvfits(filename, **kwargs)
        else:
            raise ValueError("filetype {} not recognized".format(filetype))
        echo("...writing to {}".format(filename), verbose=verbose)

    def vis_clean(self, keys=None, x=None, data=None, flags=None, wgts=None,
                  ax='freq', horizon=1.0, standoff=0.0, cache=None, mode='clean',
                  min_dly=10.0, max_frate=None, output_prefix='clean',
                  skip_wgt=0.1, verbose=False, tol=1e-9,
                  overwrite=False,
                  skip_flagged_edge_freqs=False,
                  skip_flagged_edge_times=False,
                   **filter_kwargs):
        """
        Filter the data

        Parameters
        -----------
        keys : list of bl-pol keys in data to filter
        x : array-like, x-values of axes to be filtered. Numpy array if 1d filter.
            2-list/tuple of numpy arrays if 2d filter.
        data : DataContainer, data to clean. Default is self.data
        flags : Datacontainer, flags to use. Default is self.flags
        wgts : DataContainer, weights to use. Default is None.
        ax: str, axis to filter, options=['freq', 'time', 'both']
            Where 'freq' and 'time' are 1d filters and 'both' is a 2d filter.
        horizon: coefficient to bl_len where 1 is the horizon [freq filtering]
        standoff: fixed additional delay beyond the horizon (in nanosec) to filter [freq filtering]
        cache: dictionary containing pre-computed filter products.
        mode: string specifying filtering mode. See fourier_filter or uvtools.dspec.fourier_filter for supported modes.
        min_dly: max delay (in nanosec) used for freq filter is never below this.
        max_frate : max fringe rate (in milli-Hz) used for time filtering. See uvtools.dspec.fourier_filter for options.
        output_prefix : str, attach output model, resid, etc, to self as output_prefix + '_model' etc.
        cache: dict, optional
            dictionary for caching fitting matrices.
        skip_wgt : skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
            Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
            time. Skipped channels are then flagged in self.flags.
            Only works properly when all weights are all between 0 and 1.
        verbose : Lots of outputs
        overwrite : bool, if True, overwrite output modules with the same name
                    if they already exist.
        skip_flagged_edge_freqs : bool, optional
            if true, do not filter over flagged edge frequencies (filter over sub-region)
            defualt is False
        skip_flagged_edge_times : bool, optional
            if true, do not filter over flagged edge times (filter over sub-region)
            defualt is False
        tol : float, optional. To what level are foregrounds subtracted.
        filter_kwargs : optional dictionary, see fourier_filter **filter_kwargs.
                        Do not pass suppression_factors (non-clean)!
                        instead, use tol to set suppression levels in linear filtering.
        """
        if cache is None and not mode == 'clean':
            cache = {}
        if data is None:
            data = self.data
        if flags is None:
            flags = self.flags
        if keys is None:
            keys = data.keys()
        if wgts is None:
            wgts = DataContainer(dict([(k, np.ones_like(flags[k], dtype=np.float)) for k in keys]))
        # make sure flagged channels have zero weight, regardless of what user supplied.
        wgts = DataContainer(dict([(k, (~flags[k]).astype(float) * wgts[k]) for k in keys]))
        suppression_factors = [tol]
        if max_frate is not None:
            if isinstance(max_frate, (int, np.integer, float, np.float)):
                max_frate = DataContainer(dict([(k, max_frate) for k in data]))
            if not isinstance(max_frate, DataContainer):
                raise ValueError("If fed, max_frate must be a float, or a DataContainer of floats")
            # convert kwargs to proper units
            max_frate = DataContainer(dict([(k, np.asarray(max_frate[k])) for k in max_frate]))

        for k in keys:
            if ax == 'freq' or ax == 'both':
                filter_centers_freq = [0.]
                if self.round_up_bllens:
                    bl_dly = np.ceil(self.bllens[k[:2]] * constants.c.value) / constants.c.value * horizon + standoff / 1e9
                else:
                    bl_dly = self.bllens[k[:2]] * horizon + standoff / 1e9
                filter_half_widths_freq = [np.max([bl_dly, min_dly / 1e9])]
            if ax == 'time' or ax == 'both':
                filter_centers_time = [0.]
                if max_frate is not None:
                    max_fr = max_frate[k] * 1e-3
                    filter_centers_time = [0.]
                    filter_half_widths_time = [max_fr]
                else:
                    raise ValueError("Must provide a maximum ringe-rate (or max frate dict) for time filtering.")
            if ax == 'both':
                filter_centers = [filter_centers_time, filter_centers_freq]
                filter_half_widths = [filter_half_widths_time, filter_half_widths_freq]
                filter_centers = [filter_centers_time, filter_centers_freq]
                if not mode == 'clean':
                    suppression_factors = [[tol], [tol]]
            else:
                if not mode == 'clean':
                    suppression_factors = [tol]
                if ax == 'freq':
                    filter_centers = filter_centers_freq
                    filter_half_widths = filter_half_widths_freq
                elif ax == 'time':
                    filter_centers = filter_centers_time
                    filter_half_widths = filter_half_widths_time
            if mode != 'clean':
                self.fourier_filter(keys=[k], filter_centers=filter_centers, filter_half_widths=filter_half_widths,
                                    mode=mode, suppression_factors=suppression_factors,
                                    x=x, data=data, flags=flags, wgts=wgts, output_prefix=output_prefix,
                                    ax=ax, cache=cache, skip_wgt=skip_wgt, verbose=verbose, overwrite=overwrite,
                                    skip_flagged_edge_freqs=skip_flagged_edge_freqs,
                                    skip_flagged_edge_times=skip_flagged_edge_times, **filter_kwargs)
            else:
                self.fourier_filter(keys=[k], filter_centers=filter_centers, filter_half_widths=filter_half_widths,
                                    mode=mode, tol=tol, x=x, data=data, flags=flags, wgts=wgts, output_prefix=output_prefix,
                                    ax=ax, skip_wgt=skip_wgt, verbose=verbose, overwrite=overwrite,
                                    skip_flagged_edge_freqs=skip_flagged_edge_freqs,
                                    skip_flagged_edge_times=skip_flagged_edge_times,
                                    **filter_kwargs)

    def fourier_filter(self, filter_centers, filter_half_widths, mode,
                       x=None, keys=None, data=None, flags=None, wgts=None,
                       output_prefix='clean', zeropad=None, cache=None,
                       ax='freq', skip_wgt=0.1, verbose=False, overwrite=False,
                       skip_flagged_edge_freqs=False, skip_flagged_edge_times=False,
                       flag_filled=False,
                       **filter_kwargs):
        """
        Generalized fourier filtering of attached data.
        It can filter 1d or 2d data with x-axis(es) x and wgts in fourier domain
        rectangular windows centered at filter_centers or filter_half_widths
        perform filtering along any of 2 dimensions in 2d or 1d!
        the 'dft' and 'dayenu' modes support irregularly sampled data.

        Parameters
        -----------
        filter_centers: array-like
            if not 2dfilter: 1d np.ndarray or list or tuple of floats
            specifying centers of rectangular fourier regions to filter.
            If 2dfilter: should be a 2-list or 2-tuple. Each element
            should be a list or tuple or np.ndarray of floats that include
            centers of rectangular regions to filter.
        filter_half_widths: array-like
            if not 2dfilter: 1d np.ndarray or list of tuples of floats
            specifying the half-widths of rectangular fourier regions to filter.
            if 2dfilter: should be a 2-list or 2-tuple. Each element should
            be a list or tuple or np.ndarray of floats that include centers
            of rectangular bins.
        mode: string
            specify filtering mode. Currently supported are
            'clean', iterative clean
            'dpss_lsq', dpss fitting using scipy.optimize.lsq_linear
            'dft_lsq', dft fitting using scipy.optimize.lsq_linear
            'dpss_matrix', dpss fitting using direct lin-lsq matrix
                           computation. Slower then lsq but provides linear
                           operator that can be used to propagate
                           statistics and the matrix is cached so
                           on average, can be faster for data with
                           many similar flagging patterns.
            'dft_matrix', dft fitting using direct lin-lsq matrix
                          computation. Slower then lsq but provides
                          linear operator that can be used to propagate
                          statistics and the matrix is cached so
                          on average, can be faster for data with
                          many similar flagging patterns.
                          !!!WARNING: In my experience,
                          'dft_matrix' option is numerical unstable.!!!
                          'dpss_matrix' works much better.
            'dayenu', apply dayenu filter to data. Does not
                     deconvolve subtracted foregrounds.
            'dayenu_dft_leastsq', apply dayenu filter to data
                     and deconvolve subtracted foregrounds using
                    'dft_leastsq' method (see above).
            'dayenu_dpss_leastsq', apply dayenu filter to data
                     and deconvolve subtracted foregrounds using
                     'dpss_leastsq' method (see above)
            'dayenu_dft_matrix', apply dayenu filter to data
                     and deconvolve subtracted foregrounds using
                    'dft_matrix' mode (see above).
                    !!!WARNING: dft_matrix mode is often numerically
                    unstable. I don't recommend it!
            'dayenu_dpss_matrix', apply dayenu filter to data
                     and deconvolve subtracted foregrounds using
                     'dpss_matrix' method (see above)
            'dayenu_clean', apply dayenu filter to data. Deconvolve
                     subtracted foregrounds with 'clean'.
        x : array-like, optional, numpy ndarray
        keys : list, optional, list of tuple ant-pol pair keys of visibilities to filter.
        data : DataContainer, data to clean. Default is self.data
        flags : Datacontainer, flags to use. Default is self.flags
        wgts : DataContainer, weights to use. Default is None.
        output_prefix : string, prefix for attached filter data containers.
        zeropad : int, number of bins to zeropad on both sides of FFT axis. Provide 2-tuple if axis='both'
        ax : string, optional, string specifying axis to filter.
            Where 'freq' and 'time' are 1d filters and 'both' is a 2d filter.
        skip_wgt : skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
            Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
            time. Skipped channels are then flagged in self.flags.
            Only works properly when all weights are all between 0 and 1.
        max_contiguous_edge_flags : int, optional
            if the number of contiguous samples at the edge is greater then this
            at either side, skip.
        verbose : Lots of outputs.
        overwrite : bool, if True, overwrite output modules with the same name
                    if they already exist.
        skip_flagged_edge_freqs : bool, optional
            if true, do not filter over flagged edge frequencies (filter over sub-region)
            defualt is False
        skip_flagged_edge_times : bool, optional
            if true, do not filter over flagged edge times (filter over sub-region)
            defualt is False
        flag_filled : bool, optional
            if true, set filter flags equal to the original flags (do not unflag interpolated channels)
            This is useful for cross-talk filtering where the cross-talk modes will not completely interpolate
            over the channel gaps since there are substantial contributions to the foreground power from fringe-rates
            that are not being modeled as cross-talk. In this case, we may want a file with the modelled cross-talk included but
            not used to in-paint flagged integrations.
        filter_kwargs: dict. NOTE: Unlike the dspec.fourier_filter function, cache is not passed in filter_kwargs.
            dictionary with options for fitting techniques.
            if filter2d is true, this should be a 2-tuple or 2-list
            of dictionaries. The dictionary for each dimension must
            specify the following for each fitting method.
                * 'dft':
                    'fundamental_period': float or 2-tuple
                        the fundamental_period of dft modes to fit. The number of
                        modes fit within each window in 'filter_half_widths' will
                        equal fw / fundamental_period where fw is the filter_half_width of the window.
                        if filter2d, must provide a 2-tuple with fundamental_period
                        of each dimension.
                * 'dayenu':
                    No parameters necessary if you are only doing 'dayenu'.
                    For 'dayenu_dpss', 'dayenu_dft', 'dayenu_clean' see below
                    and use the appropriate fitting options for each method.
                    suppression_factors: array-like
                        if not 2dfilter: 1d np.ndarray or list of tuples of floats
                        specifying the fractional residuals of model to leave in the data.
                        For example, 1e-6 means that the filter will leave in 1e-6 of data fitted
                        by the model.
                        if 2dfilter: should be a 2-list or 2-tuple. Each element should
                        be a list or tuple or np.ndarray of floats that include centers
                        of rectangular bins.
                    ax: str, axis to filter, options=['freq', 'time', 'both']
                * 'dpss':
                    'eigenval_cutoff': array-like
                        list of sinc_matrix eigenvalue cutoffs to use for included dpss modes.
                    'nterms': array-like
                        list of integers specifying the order of the dpss sequence to use in each
                        filter window.
                    'edge_supression': array-like
                        specifies the degree of supression that must occur to tones at the filter edges
                        to calculate the number of DPSS terms to fit in each sub-window.
                    'avg_suppression': list of floats, optional
                        specifies the average degree of suppression of tones inside of the filter edges
                        to calculate the number of DPSS terms. Similar to edge_supression but instead checks
                        the suppression of a since vector with equal contributions from all tones inside of the
                        filter width instead of a single tone.
                    suppression_factors: array-like
                        if not 2dfilter: 1d np.ndarray or list of tuples of floats
                        specifying the fractional residuals of model to leave in the data.
                        For example, 1e-6 means that the filter will leave in 1e-6 of data fitted
                        by the model.
                        if 2dfilter: should be a 2-list or 2-tuple. Each element should
                        be a list or tuple or np.ndarray of floats that include centers
                        of rectangular bins.
                    ax: str, axis to filter, options=['freq', 'time', 'both']
                *'clean':
                     'tol': float,
                        clean tolerance. 1e-9 is standard.
                     'maxiter' : int
                        maximum number of clean iterations. 100 is standard.
                     'pad': int or array-like
                        if filt2d is false, just an integer specifing the number of channels
                        to pad for CLEAN (sets Fourier interpolation resolution).
                        if filt2d is true, specify 2-tuple in both dimensions.
                     'filt2d_mode' : string
                        if 'rect', clean withing a rectangular region of Fourier space given
                        by the intersection of each set of windows.
                        if 'plus' only clean the plus-shaped shape along
                        zero-delay and fringe rate.
                    'edgecut_low' : int, number of bins to consider zero-padded at low-side of the FFT axis,
                        such that the windowing function smoothly approaches zero. For 2D cleaning, can
                        be fed as a tuple specifying edgecut_low for first and second FFT axis.
                    'edgecut_hi' : int, number of bins to consider zero-padded at high-side of the FFT axis,
                        such that the windowing function smoothly approaches zero. For 2D cleaning, can
                        be fed as a tuple specifying edgecut_hi for first and second FFT axis.
                    'add_clean_residual' : bool, if True, adds the CLEAN residual within the CLEAN bounds
                        in fourier space to the CLEAN model. Note that the residual actually returned is
                        not the CLEAN residual, but the residual in input data space.
                    'taper' : window function for filtering applied to the filtered axis.
                        See dspec.gen_window for options. If clean2D, can be fed as a list
                        specifying the window for each axis in data.
                    'skip_wgt' : skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
                        Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
                        time. Only works properly when all weights are all between 0 and 1.
                    'gain': The fraction of a residual used in each iteration. If this is too low, clean takes
                        unnecessarily long. If it is too high, clean does a poor job of deconvolving.
                    'alpha': float, if window is 'tukey', this is its alpha parameter.
        x : array-like, x-values of axes to be filtered. Numpy array if 1d filter.
            2-list/tuple of numpy arrays if 2d filter.
     """
        # type checks

        if ax == 'both':
            if zeropad is None:
                zeropad = [0, 0]
            filterdim = [1, 0]
            filter2d = True
            if x is None:
                x = [(self.times - np.mean(self.times)) * 3600. * 24., self.freqs]
        elif ax == 'time':
            filterdim = 0
            filter2d = False
            if x is None:
                x = (self.times - np.mean(self.times)) * 3600. * 24.
            if zeropad is None:
                zeropad = 0
        elif ax == 'freq':
            filterdim = 1
            filter2d = False
            if zeropad is None:
                zeropad = 0
            if x is None:
                x = self.freqs
        else:
            raise ValueError("ax must be one of ['freq', 'time', 'both']")

        # initialize containers
        containers = ["{}_{}".format(output_prefix, dc) for dc in ['model', 'resid', 'flags', 'data', 'resid_flags']]
        for i, dc in enumerate(containers):
            if not hasattr(self, dc):
                setattr(self, dc, DataContainer({}))
            containers[i] = getattr(self, dc)
        filtered_model, filtered_resid, filtered_flags, filtered_data, resid_flags = containers
        filtered_info = "{}_{}".format(output_prefix, 'info')
        if not hasattr(self, filtered_info):
            setattr(self, filtered_info, {})
        filtered_info = getattr(self, filtered_info)

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
            wgts = DataContainer(dict([(k, (~flags[k]).astype(float)) for k in keys]))
        else:
            # make sure flagged channels have zero weight, regardless of what user supplied.
            wgts = DataContainer(dict([(k, (~flags[k]).astype(float) * wgts[k]) for k in keys]))
        if mode != 'clean':
            if cache is None:
                cache = {}
            filter_kwargs['cache'] = cache
        # iterate over keys
        for k in keys:
            if k in filtered_model and overwrite is False:
                echo("{} exists in clean_model and overwrite is False, skipping...".format(k), verbose=verbose)
                continue
            echo("Starting fourier filter of {} at {}".format(k, str(datetime.datetime.now())), verbose=verbose)
            d = data[k]
            f = flags[k]
            fw = (~f).astype(np.float)
            w = fw * wgts[k]
            # avoid modifying x in-place with zero-padding.
            xp = copy.deepcopy(x)
            if ax == 'freq':
                # zeropad the data
                if zeropad > 0:
                    d, _ = zeropad_array(d, zeropad=zeropad, axis=1)
                    w, _ = zeropad_array(w, zeropad=zeropad, axis=1)
                    xp = np.hstack([x.min() - (1 + np.arange(zeropad)[::-1]) * np.mean(np.diff(x)), x,
                                    x.max() + (1 + np.arange(zeropad)) * np.mean(np.diff(x))])
            elif ax == 'time':
                # zeropad the data
                if zeropad > 0:
                    d, _ = zeropad_array(d, zeropad=zeropad, axis=0)
                    w, _ = zeropad_array(w, zeropad=zeropad, axis=0)
                    xp = np.hstack([x.min() - (1 + np.arange(zeropad)[::-1]) * np.mean(np.diff(x)), x,
                                   x.max() + (1 + np.arange(zeropad)) * np.mean(np.diff(x))])
            elif ax == 'both':
                if not isinstance(zeropad, (list, tuple)) or not len(zeropad) == 2:
                    raise ValueError("zeropad must be a 2-tuple or 2-list of integers")
                if not (isinstance(zeropad[0], (int, np.int)) and isinstance(zeropad[0], (int, np.int))):
                    raise ValueError("zeropad values must all be integers. You provided %s" % (zeropad))
                for m in range(2):
                    if zeropad[m] > 0:
                        d, _ = zeropad_array(d, zeropad=zeropad[m], axis=m)
                        w, _ = zeropad_array(w, zeropad=zeropad[m], axis=m)
                        xp[m] = np.hstack([x[m].min() - (np.arange(zeropad[m])[::-1] + 1) * np.mean(np.diff(x[m])),
                                           x[m], x[m].max() + (1 + np.arange(zeropad[m])) * np.mean(np.diff(x[m]))])
            mdl, res = np.zeros_like(d), np.zeros_like(d)
            if skip_flagged_edge_freqs:
                unflagged_chans = np.where(~np.all(np.isclose(w, 0.0), axis=0))[0]
                if len(unflagged_chans) > 0:
                    ind_left = np.min(unflagged_chans)
                    ind_right = np.max(unflagged_chans) + 1
                else:
                    ind_left = 0
                    ind_right = d.shape[1]
            else:
                ind_left = 0
                ind_right = d.shape[1]
            if skip_flagged_edge_times:
                unflagged_times = np.where(~np.all(np.isclose(w, 0.0), axis=1))[0]
                if len(unflagged_times) > 0:
                    ind_lower = np.min(unflagged_times)
                    ind_upper = np.max(unflagged_times) + 1
                else:
                    ind_lower = 0
                    ind_upper = d.shape[0]
            else:
                ind_lower = 0
                ind_upper = d.shape[0]
            din = d[ind_lower: ind_upper][:, ind_left: ind_right]
            win = w[ind_lower: ind_upper][:, ind_left: ind_right]
            if ax == 'both':
                xp[0] = xp[0][ind_lower: ind_upper]
                xp[1] = xp[1][ind_left: ind_right]
            elif ax == 'time':
                xp = xp[ind_lower: ind_upper]
            elif ax == 'freq':
                xp = xp[ind_left: ind_right]

            mdl[ind_lower: ind_upper][:, ind_left: ind_right], res[ind_lower: ind_upper][:, ind_left: ind_right], info \
            = dspec.fourier_filter(x=xp, data=din, wgts=win, filter_centers=filter_centers,
                                   filter_half_widths=filter_half_widths,
                                   mode=mode, filter_dims=filterdim, skip_wgt=skip_wgt,
                                   **filter_kwargs)

            # unzeropad array and put in skip flags.
            if ax == 'freq':
                if zeropad > 0:
                    mdl, _ = zeropad_array(mdl, zeropad=zeropad, axis=1, undo=True)
                    res, _ = zeropad_array(res, zeropad=zeropad, axis=1, undo=True)
            elif ax == 'time':
                if zeropad > 0:
                    mdl, _ = zeropad_array(mdl, zeropad=zeropad, axis=0, undo=True)
                    res, _ = zeropad_array(res, zeropad=zeropad, axis=0, undo=True)
            elif ax == 'both':
                for i in range(2):
                    if zeropad[i] > 0:
                        mdl, _ = zeropad_array(mdl, zeropad=zeropad[i], axis=i, undo=True)
                        res, _ = zeropad_array(res, zeropad=zeropad[i], axis=i, undo=True)
                    _trim_status(info, i, zeropad[i - 1])

            skipped = np.zeros_like(mdl, dtype=np.bool)
            for dim in range(2):
                if len(info['status']['axis_%d' % dim]) > 0:
                    for i in range(len(info['status']['axis_%d' % dim])):
                        if info['status']['axis_%d' % dim][i] == 'skipped':
                            if dim == 0:
                                skipped[:, i] = True
                            elif dim == 1:
                                skipped[i] = True
            # also flag skipped edge channels and integrations.
            skipped[:, :ind_left] = True
            skipped[:, ind_right:] = True
            skipped[:ind_lower, :] = True
            skipped[ind_upper:, :] = True
            filtered_model[k] = mdl
            filtered_model[k][skipped] = 0.
            filtered_resid[k] = res * fw
            filtered_resid[k][skipped] = 0.
            filtered_data[k] = filtered_model[k] + filtered_resid[k]
            if not flag_filled:
                filtered_flags[k] = skipped
            else:
                filtered_flags[k] = copy.deepcopy(flags[k]) | skipped
            filtered_info[k] = info
            resid_flags[k] = copy.deepcopy(flags[k]) | skipped

        if hasattr(data, 'times'):
            filtered_data.times = data.times
            filtered_model.times = data.times
            filtered_resid.times = data.times
            filtered_flags.times = data.times

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

    def trim_edges(self, ax='freq'):
        """Trim edge times and frequencies that are fully flagged. Always in place.

        Function to remove edge times and frequencies from data that are completely flagged.
        such flagged edges and times can cause problems for linear filtering methods.
        since a set number of times and frequencies are assumed in vis_clean objects, this only
        works for datasets where the flags are identical for all baselines.

        This function clears all datacontainers that are not data, flags, and nsamples.

        """
        # first check that all flags are the same or completely flagged.
        ntimes_before_trim = self.Ntimes
        nfreqs_before_trim = self.Nfreqs
        template = None
        trimmed = False
        for k in self.flags:
            if not np.all(self.flags[k]):
                if template is None:
                    template = self.flags[k]
                else:
                    if not np.all(template == self.flags[k]):
                        raise ValueError("Flag Trimming only supported when flagging for all baselines is identical!")

        for k in self.flags:
            if not np.all(self.flags[k]):
                unflagged_chans = np.where(~np.all(self.flags[k], axis=0))[0]
                unflagged_times = np.where(~np.all(self.flags[k], axis=1))[0]
                ind_left = np.min(unflagged_chans)
                ind_right = np.max(unflagged_chans) + 1
                ind_lower = np.min(unflagged_times)
                ind_upper = np.max(unflagged_times) + 1
                # if we are only trimming freq axis, restore ind_upper/lower
                if ax.lower() == 'freq':
                    ind_upper = self.Ntimes
                    ind_lower = 0
                # if we are only trimming time axis, restore ind_left/right
                elif ax.lower() == 'time':
                    ind_left = 0
                    ind_right = self.Nfreqs
                elif ax.lower() != 'both':
                    raise ValueError("Invalid ax=%s provided! Must be either ['freq', 'time', 'both']"%ax)
                # back up trimmed versions of
                bls = list(self.data.keys())
                # flags, data, and nsamples
                data_bk = DataContainer({k: self.data[k][ind_lower: ind_upper][:, ind_left: ind_right] for k in self.data})
                flags_bk = DataContainer({k: self.flags[k][ind_lower: ind_upper][:, ind_left: ind_right] for k in self.flags})
                nsamples_bk = DataContainer({k: self.nsamples[k][ind_lower: ind_upper][:, ind_left: ind_right] for k in self.nsamples})
                # clear datacontainers
                self.clear_containers()
                # reread data over trimmed frequencies and times
                dt = np.mean(np.diff(self.times))
                self.read(time_range=[self.times[ind_lower]-dt/10, self.times[ind_upper-1]+dt/10],
                          freq_chans=np.arange(ind_left, ind_right).astype(int), bls=bls)
                # restore original data / flags/ nsamples
                self.hd.update(data=data_bk, flags=flags_bk, nsamples=nsamples_bk)
                self.data, self.flags, self.nsamples = self.hd.build_datacontainers()
                # set trimmed to True
                trimmed = True
                break

        if not trimmed:
            warnings.warn("no unflagged data so no trimming performed.")

    def write_filtered_data(self, res_outfilename=None, CLEAN_outfilename=None, filled_outfilename=None, filetype='uvh5',
                            partial_write=False, clobber=False, add_to_history='', extra_attrs={}, prefix='clean', **kwargs):
        '''
        Method for writing data products.

        Can write filtered residuals, CLEAN models, and/or original data with flags filled
        by CLEAN models where possible. Uses input_data from DelayFilter.load_data() as a template.

        Arguments:
            res_outfilename: path for writing the filtered visibilities with flags
            CLEAN_outfilename: path for writing the CLEAN model visibilities (with the same flags)
            filled_outfilename: path for writing the original data but with flags unflagged and replaced
                with CLEAN models wherever possible
            filetype: file format of output result. Default 'uvh5.' Also supports 'miriad' and 'uvfits'.
            partial_write: use uvh5 partial writing capability (only works when going from uvh5 to uvh5)
            clobber: if True, overwrites existing file at the outfilename
            add_to_history: string appended to the history of the output file
            extra_attrs : additional UVData/HERAData attributes to update before writing
            prefix : string, the prefix for the datacontainers to write.
            kwargs : extra kwargs to pass to UVData.write_*() call
        '''
        if not hasattr(self, 'data'):
            raise ValueError("Cannot write data without first loading")
        if (res_outfilename is None) and (CLEAN_outfilename is None) and (filled_outfilename is None):
            raise ValueError('You must specifiy at least one outfilename.')
        else:
            # loop over the three output modes if a corresponding outfilename is supplied
            for mode, outfilename in zip(['residual', 'CLEAN', 'filled'],
                                         [res_outfilename, CLEAN_outfilename, filled_outfilename]):
                if outfilename is not None:
                    if mode == 'residual':
                        data_out, flags_out = getattr(self, prefix + '_resid'), getattr(self, prefix + '_resid_flags')
                    elif mode == 'CLEAN':
                        data_out, flags_out = getattr(self, prefix + '_model'), getattr(self, prefix + '_flags')
                    elif mode == 'filled':
                        data_out, flags_out = self.get_filled_data()
                    if partial_write:
                        if not ((filetype == 'uvh5') and (getattr(self.hd, 'filetype', None) == 'uvh5')):
                            raise NotImplementedError('Partial writing requires input and output types to be "uvh5".')
                        self.hd.partial_write(outfilename, data=data_out, flags=flags_out, clobber=clobber,
                                              add_to_history=version.history_string(add_to_history), **kwargs)
                    else:
                        self.write_data(data_out, outfilename, filetype=filetype, overwrite=clobber, flags=flags_out,
                                        add_to_history=add_to_history, extra_attrs=extra_attrs, **kwargs)

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

    def get_filled_data(self, prefix='clean'):
        """Get data with flagged pixels filled with clean_model.
        Parameters
            prefix : string label for data-containers of filtering outputs to get.
        Returns
            filled_data: DataContainer with original data and flags filled with CLEAN model
            filled_flags: DataContainer with flags set to False unless the time is skipped in delay filter
        """
        assert np.all([hasattr(self, n) for n in [prefix + '_model', prefix + '_flags', 'data', 'flags']]), "self.data, self.flags, self.%s_model and self.%s_flags must all exist to get filled data" % (prefix, prefix)
        # construct filled data and filled flags
        filled_data = copy.deepcopy(getattr(self, prefix + '_model'))
        filled_flags = copy.deepcopy(getattr(self, prefix + '_flags'))

        # iterate over filled_data keys
        for k in filled_data.keys():
            # get original data flags
            f = self.flags[k]
            # replace filled_data with original data at f == False
            filled_data[k][~f] = self.data[k][~f]

        return filled_data, filled_flags


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


def trim_model(clean_model, clean_resid, dnu, keys=None, noise_thresh=2.0, delay_cut=3000,
               kernel_size=None, edgecut_low=0, edgecut_hi=0, polyfit_deg=None, verbose=True):
    """
    Truncate CLEAN model components in delay space below some amplitude threshold.

    Estimate the noise in Fourier space by taking median of high delay
    clean residual above delay_cut, and truncate CLEAN model components
    below a multiplier times this level.

    Args:
        clean_model : DataContainer
            Holds clean_model output of self.vis_clean.
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
        polyfit_deg : int
            Degree of polynomial to fit to noise curve w.r.t. time.
            None is no fitting.
        verbose : bool
            Report feedback to stdout

    Returns:
        model : DataContainer
            Truncated clean_model
        noise : DataContainer
            Per integration noise estimate from clean_resid
    """
    # get keys
    if keys is None:
        keys = [k for k in sorted(set(list(clean_model.keys()) + list(clean_resid.keys()))) if k in clean_model and k in clean_resid]

    # estimate noise in Fourier space by taking amplitude of high delay modes
    # above delay_cut
    model = DataContainer({})
    noise = DataContainer({})
    for k in keys:
        # get rfft
        rfft, delays = fft_data(clean_resid[k], dnu, axis=1, window='none', edgecut_low=edgecut_low, edgecut_hi=edgecut_hi, ifft=False, ifftshift=False, fftshift=False)
        delays *= 1e9

        # get NEB of clean_resid: a top-hat window nulled where resid == 0 (i.e. flag pattern)
        w = (~np.isclose(clean_resid[k], 0.0)).astype(np.float)
        neb = noise_eq_bandwidth(w[:, None])

        # get time-dependent noise level in Fourier space from FFT at high delays
        noise[k] = np.median(np.abs((rfft * neb)[:, np.abs(delays) > delay_cut]), axis=1)

        # median filter it
        if kernel_size is not None:
            n = noise[k]
            nlen = len(n)
            n = np.pad(n, nlen, 'reflect', reflect_type='odd')
            noise[k] = signal.medfilt(n, kernel_size=kernel_size)[nlen:-nlen]

        # fit a polynomial if desired
        if polyfit_deg is not None:
            x = np.arange(noise[k].size, dtype=np.float)
            f = ~np.isnan(noise[k]) & ~np.isfinite(noise[k]) & ~np.isclose(noise[k], 0.0)
            # only fit if it is well-conditioned: Ntimes > polyfit_deg + 1
            if f.sum() >= (polyfit_deg + 1):
                fit = np.polyfit(x[f], noise[k][f], deg=polyfit_deg)
                noise[k] = np.polyval(fit, x)
            else:
                # not enough points to fit polynomial
                echo("Need more suitable data points for {} to fit {}-deg polynomial".format(k, polyfit_deg), verbose=verbose)

        # get mfft
        mfft, _ = fft_data(clean_model[k], dnu, axis=1, window='none', edgecut_low=edgecut_low, edgecut_hi=edgecut_hi, ifft=False, ifftshift=False, fftshift=False)

        # set all mfft modes below some threshold to zero
        mfft[np.abs(mfft) < (noise[k][:, None] * noise_thresh)] = 0.0

        # re-fft
        mdl, _ = fft_data(mfft, dnu, axis=1, window='none', edgecut_low=0, edgecut_hi=0, ifft=True, ifftshift=False, fftshift=False)
        model[k] = mdl

    return model, noise


def _trim_status(info_dict, axis, zeropad):
    '''
    Trims the info status dictionary for a zero-padded
    filter so that the status of integrations that were
    in the zero-pad region are deleted

    Parameters
    ----------
    info : dict, info dictionary
    axis : integer, index of axis to trim
    zeropad : integer

    Returns
    -------
    Nothing, modifies the provided dictionary in place.
    '''
    # delete statuses in zero-pad region
    statuses = info_dict['status']['axis_%d' % axis]
    nints = len(statuses)
    for i in range(zeropad):
        del statuses[i]
        del statuses[nints - i - 1]
    # now update keys of the dict elements we wish to keep
    nints = len(statuses)
    for i in range(nints):
        statuses[i] = statuses.pop(i + zeropad)


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

    See https://analog.intgckts.com/equivalent-noise-bandwidth/

    Args:
        window : float ndarray
        axis : int, axis along which to calculate NEB

    Returns
        neb : float or ndarray
            Noise equivalent bandwidth of the window
    """
    return np.sqrt(window.shape[axis] * np.max(window, axis=axis)**2 / np.sum(window**2, dtype=np.float, axis=axis))

# ------------------------------------------
# Here is an argparser with core arguments
# needed for all types of xtalk and delay
# filtering.
# ------------------------------------------


def _filter_argparser(multifile=False):
    """
    Core Arg parser for commandline operation of hera_cal.delay_filter and hera_cal.xtalk_filter
    Parameters:
        multifile, bool: optional. If True, add calfilelist and filelist
                         arguments.
    Returns:
        Argparser with core (but not complete) functionality that is called by _linear_argparser and
        _clean_argparser.
    """
    a = argparse.ArgumentParser(description="Perform delay filter of visibility data.")
    a.add_argument("--filetype_in", type=str, default='uvh5', help='filetype of input data files (default "uvh5")')
    a.add_argument("--filetype_out", type=str, default='uvh5', help='filetype for output data files (default "uvh5")')
    a.add_argument("--res_outfilename", default=None, type=str, help="path for writing the filtered visibilities with flags")
    a.add_argument("--clobber", default=False, action="store_true", help='overwrites existing file at outfile')
    a.add_argument("--spw_range", type=int, default=None, nargs=2, help="spectral window of data to foreground filter.")
    a.add_argument("--tol", type=float, default=1e-9, help='Threshold for foreground and xtalk subtraction (default 1e-9)')
    a.add_argument("infilename", type=str, help="path to visibility data file to delay filter")
    a.add_argument("--partial_load_Nbls", default=None, type=int, help="the number of baselines to load at once (default None means load full data")
    a.add_argument("--skip_wgt", type=float, default=0.1, help='skips filtering and flags times with unflagged fraction ~< skip_wgt (default 0.1)')
    a.add_argument("--factorize_flags", default=False, action="store_true", help="Factorize flags.")
    a.add_argument("--time_thresh", type=float, default=0.05, help="time threshold above which to completely flag channels and below which to flag times with flagged channel.")
    a.add_argument("--trim_edges", default=False, action="store_true", help="If true, trim edge times and frequencies that are comletely flagged.")
    a.add_argument("--skip_flagged_edges", default=False, action="store_true", help="if True, do not filter over flagged edge integrations or channels (depending on filter axis).")
    a.add_argument("--verbose", default=False, action="store_true", help="lots of output.")
    a.add_argument('--a_priori_flag_yaml', default=None, type=str,
                    help=('Path to a priori flagging YAML with frequency, time, and/or '
                          'antenna flagsfor parsable by hera_qm.metrics_io.read_a_priori_*_flags()'))
    a.add_argument("--external_flags", default=None, type=str, nargs="+", help="list of external flags to apply before filtering.")
    a.add_argument("--overwrite_data_flags", default=False, action="store_true", help="overwrite data and calibration flags with external flags.")
    if multifile:
        a.add_argument("--calfilelist", default=None, type=str, nargs="+", help="list of calibration files.")
        a.add_argument("--datafilelist", default=None, type=str, nargs="+", help="list of data files. Used to determine parallelization chunk.")
        a.add_argument("--polarizations", default=None, type=str, nargs="+", help="list of polarizations to filter and write out.")

    else:
        a.add_argument("--calfile", default=None, type=str, help="optional string path to calibration file to apply to data before delay filtering")
    return a


# ------------------------------------------
# Here are arg-parsers clean filtering
# for both xtalk and foregrounds.
# ------------------------------------------


def _clean_argparser(multifile=False):
    '''
    Arg parser for commandline operation of hera_cal.delay_filter in various clean modes.
    Arguments
    ---------
        multifile, bool: optional. If True, add calfilelist and filelist
                         arguments.
    Returns
    -------
        Arg-parser for linear filtering. Still needs domain specific args (delay versus xtalk).
    '''
    a = _filter_argparser(multifile=multifile)
    a.add_argument("--CLEAN_outfilename", default=None, type=str, help="path for writing the filtered model visibilities (with the same flags)")
    a.add_argument("--filled_outfilename", default=None, type=str, help="path for writing the original data but with flags unflagged and replaced with filtered models wherever possible")
    clean_options = a.add_argument_group(title='Options for CLEAN')
    clean_options.add_argument("--window", type=str, default='blackman-harris', help='window function for frequency filtering (default "blackman-harris",\
                              see uvtools.dspec.gen_window for options')
    clean_options.add_argument("--maxiter", type=int, default=100, help='maximum iterations for aipy.deconv.clean to converge (default 100)')
    clean_options.add_argument("--edgecut_low", default=0, type=int, help="Number of channels to flag on lower band edge and exclude from window function.")
    clean_options.add_argument("--edgecut_hi", default=0, type=int, help="Number of channels to flag on upper band edge and exclude from window function.")
    clean_options.add_argument("--gain", type=float, default=0.1, help="Fraction of residual to use in each iteration.")
    clean_options.add_argument("--alpha", type=float, default=.5, help="If window='tukey', use this alpha parameter (default .5).")
    return a

# ------------------------------------------
# Here are are parsers for linear filters.
# for xtalk and foregrounds.
# ------------------------------------------


def _linear_argparser(multifile=False):
    '''
    Arg parser for commandline operation of hera_cal.delay_filter in various linear modes.
    Arguments
    ---------
        multifile, bool: optional. If True, add calfilelist and filelist
                         arguments.
    Returns
    -------
        Arg-parser for linear filtering. Still needs domain specific args (delay versus xtalk)
    '''
    a = _filter_argparser(multifile=multifile)
    cache_options = a.add_argument_group(title='Options for caching')
    a.add_argument("--write_cache", default=False, action="store_true", help="if True, writes newly computed filter matrices to cache.")
    a.add_argument("--cache_dir", type=str, default=None, help="directory to store cached filtering matrices in.")
    a.add_argument("--read_cache", default=False, action="store_true", help="If true, read in cache files in directory specified by cache_dir.")
    a.add_argument("--max_contiguous_edge_flags", type=int, default=1, help="Skip integrations with at least this number of contiguous edge flags.")
    return a

def _dpss_argparser(multifile=False):
    '''
    Arg parser for commandline operation of hera_cal.delay_filter in dpss mode.
    Arguments
    ---------
        multifile, bool: optional. If True, add calfilelist and filelist
                         arguments.
    Returns
    -------
        Arg-parser for dpss filtering.
    '''
    a = _linear_argparser(multifile=multifile)
    a.add_argument("--CLEAN_outfilename", default=None, type=str, help="path for writing the filtered model visibilities (with the same flags)")
    a.add_argument("--filled_outfilename", default=None, type=str, help="path for writing the original data but with flags unflagged and replaced with filtered models wherever possible")
    return a

def reconstitute_files(templatefile, fragments, outfilename, clobber=False, time_bounds=False):
    """Recombine xtalk products into short-time files.

    Construct a new file based on templatefile that combines the files in file_fragments
    over the times in template file.

    Arguments
    ---------
    templatefile : string
        name of the file to use as a template. Will reconstitute the file_fragments over the times in templatefile.
    outfilename : string
        name of the output file
    file_fragments : list of strings
        list of file names to use reconstitute.
    clobber : bool optional.
        If False, don't overwrite outfilename if it already exists. Default is False.
    time_bounds: bool, optional
        If False, then generate new file with exact times from template.
        If True, generate a new file from the file list that keeps times between min/max of
            the times in the template_file. This is helpful if the times dont match in the reconstituted
            data but we want to use the template files to determine time ranges.

    Returns
    -------
        Nothing
    """
    hd_template = io.HERAData(templatefile)
    hd_fragment = io.HERAData(fragments[0])
    times = hd_template.times
    freqs = hd_fragment.freqs
    polarizations = hd_fragment.pols
    # read in the template file, but only include polarizations, frequencies
    # from the fragment files.
    if not time_bounds:
        hd_template.read(times=times, frequencies=freqs, polarizations=polarizations)
        # for each fragment, read in only the times relevant to the templatefile.
        # and update the data, flags, nsamples array of the template file
        # with the fragment data.
        for fragment in fragments:
            hd_fragment = io.HERAData(fragment)
            # find times that are close.
            tload = []
            atol = np.mean(np.diff(hd_fragment.times)) / 10.
            all_times = np.unique(hd_fragment.times)
            for t in all_times:
                if np.any(np.isclose(t, hd_template.times, atol=atol, rtol=0)):
                    tload.append(t)
            d, f, n = hd_fragment.read(times=tload, axis='blt')
            hd_template.update(flags=f, data=d, nsamples=n)
        # now that we've updated everything, we write the output file.
        hd_template.write_uvh5(outfilename, clobber=clobber)
    else:
        tmax = hd_template.times.max()
        tmin = hd_template.times.min()
        hd_combined = io.HERAData(fragments)
        t_select = (hd_fragment.times >= tmin) & (hd_fragment.times <= tmax)
        hd_combined.read(times=hd_fragment.times[t_select], axis='blt')
        hd_combined.write_uvh5(outfilename, clobber=clobber)


def reconstitute_files_argparser():
    """
    Arg parser for file reconstitution.
    """
    a = argparse.ArgumentParser(description="Reconstitute fragmented baseline files.")
    a.add_argument("infilename", type=str, help="name of template file.")
    a.add_argument("--fragmentlist", type=str, nargs="+", help="list of file fragments to reconstitute")
    a.add_argument("--outfilename", type=str, help="Name of output file. Provide the full path string.")
    a.add_argument("--clobber", action="store_true", help="Include to overwrite old files.", default=False)
    a.add_argument("--time_bounds", action="store_true", default=False, help="read times between min and max times of template, regardless of whether they match.")
    return a
