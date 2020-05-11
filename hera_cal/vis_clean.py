# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

import numpy as np
from collections import OrderedDict as odict
import datetime
try:
    from uvtools import dspec
    HAVE_UVTOOLS = True
except ImportError:
    HAVE_UVTOOLS = False

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

    def __init__(self, input_data, filetype='uvh5', input_cal=None, link_data=True, **read_kwargs):
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

    def vis_fourier_filter(self,  keys=None, x=None, data=None, flags=None, wgts=None,
                           ax='freq', horizon=1.0, standoff=0.0,
                           min_dly=0.0, max_frate=None, tol=1e-9,
                           output_prefix='filtered', zeropad=0,
                           cache=None,  skip_wgt=0.1, max_contiguous_edge_flags=10, verbose=False,
                           overwrite=False, mode='dayenu', fitting_options=None,):
        """
        A less flexible but more streamlined wrapper for fourier_filter that filters visibilities based
        on baseline length and standoff.

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
        min_dly: max delay (in nanosec) used for freq filter is never below this.
        max_frate : max fringe rate (in milli-Hz) used for time filtering. See uvtools.dspec.vis_filter for options.
        tol: The fraction of visibilities within the filter region to leave in.
        output_prefix : str, attach output model, resid, etc, to self as output_prefix + '_model' etc.
        zeropad : int, number of bins to zeropad on both sides of FFT axis. Provide 2-tuple if axis='both'
        cache: dict, optional
            dictionary for caching fitting matrices.
        skip_wgt : skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
            Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
            time. Skipped channels are then flagged in self.flags.
            Only works properly when all weights are all between 0 and 1.
        max_contiguous_edge_flags : int, optional
            if the number of contiguous samples at the edge is greater then this
            at either side, skip.
        overwrite : bool, if True, overwrite output modules with the same name
                    if they already exist.
        verbose : Lots of outputs.
        mode: str,
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
        fitting_options: dict
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
        """
        if cache is None:
            cache = {}
        if data is None:
            data = self.data
        if flags is None:
            flags = self.flags
        if keys is None:
            keys = data.keys()
        if wgts is None:
            wgts = DataContainer(dict([(k, np.ones_like(flags[k], dtype=np.float)) for k in keys]))
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
                bl_dly = self.bllens[k[:2]]] * horizon + standoff/1e9
                min_dly /= 1e9
                filter_half_widths_freq = [np.max([bl_dly, min_dly])]
            elif ax == 'time' or ax == 'both':
                filter_centers_time= [0.]
                if max_frate is not None:
                    max_frate = max_frate * 1e-3
                    filter_centers_time = [ 0. ]
                    filter_half_widths_time = [ max_frate ]
            if ax == 'both':
                filter_centers = [filter_centers_time, filter_centers_freq]
                filter_half_widths = [filter_half_widths_time, filter_half_widths_freq]
                filter_centers = [filter_centers_time, filter_centers_freq]
                suppression_factors = [[tol], [tol]]
            else:
                suppression_factors = [tol]
                if ax == 'freq':
                    filter_centers = filter_centers_freq
                    filter_half_widths = filter_half_widths_freq
                elif ax == 'time':
                    filter_centers = filter_centers_time
                    filter_half_widths = filter_half_widths_time

        self.fourier_filter(keys=[k], filter_centers=filter_centers, filter_half_widths=filter_half_widths,
                            suppression_factors=suppression_factors, mode=mode, fitting_options=fitting_options,
                            x=x, data=data, flags=flags, output_prefix=output_prefix, wgts=wgts, zeropad=zeropad,
                            cache=cache, ax=ax, skip_wgt=skip_wgt, max_contiguous_edge_flags=max_contiguous_edge_flags,
                            verbose=verbose, overwrite=overwrite)



    ###
    #TODO: zeropad here will error given its default option if 2d filtering is being used.
    ###
    def fourier_filter(self, keys, filter_centers, filter_half_widths, suppression_factors, mode,
                       fitting_options, x=None, data=None, flags=None, output_prefix='filtered',
                       wgts=None, zeropad=None, cache=None, ax='freq', skip_wgt=0.1,
                       max_contiguous_edge_flags=10, verbose=False, overwrite=False):
        """
        Your one-stop-shop for fourier filtering.
        It can filter 1d or 2d data with x-axis(es) x and wgts in fourier domain
        rectangular windows centered at filter_centers or filter_half_widths
        perform filtering along any of 2 dimensions in 2d or 1d!
        the 'dft' and 'dayenu' modes support irregularly sampled data.

        Parameters
        -----------
        keys : list of bl-pol keys in data to CLEAN
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
        suppression_factors: array-like
            if not 2dfilter: 1d np.ndarray or list of tuples of floats
            specifying the fractional residuals of model to leave in the data.
            For example, 1e-6 means that the filter will leave in 1e-6 of data fitted
            by the model.
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
        fitting_options: dict
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
        data : DataContainer, data to clean. Default is self.data
        flags : Datacontainer, flags to use. Default is self.flags
        wgts : DataContainer, weights to use. Default is None.
        zeropad : int, number of bins to zeropad on both sides of FFT axis. Provide 2-tuple if axis='both'
        cache: dict, optional
            dictionary for caching fitting matrices.
        ax: str, axis to filter, options=['freq', 'time', 'both']
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
                    if they already exist.        """
        if cache is None:
            cache = {}
        if not HAVE_UVTOOLS:
            raise ImportError("uvtools required, install hera_cal[all]")

        # type checks

        if ax == 'both':
            if zeropad is None:
                zeropad = [0, 0]
            filterdim = [0, 1]
            filter2d = True
            if x is None:
                x = [self.times, self.freqs]
        elif ax == 'time':
            filterdim = 0
            filter2d = False
            if x is None:
                x = self.times
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
        containers = ["{}_{}".format(output_prefix, dc) for dc in ['model', 'resid', 'flags', 'data']]
        for i, dc in enumerate(containers):
            if not hasattr(self, dc):
                setattr(self, dc, DataContainer({}))
            containers[i] = getattr(self, dc)
        filtered_model, filtered_resid, filtered_flags, filtered_data = containers
        filtered_info = "{}_{}".format(output_prefix, 'info')
        if not hasattr(self,filtered_info):
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
            wgts = DataContainer(dict([(k, np.ones_like(flags[k], dtype=np.float)) for k in keys]))



        # iterate over keys
        for k in keys:
            if k in filtered_model and overwrite is False:
                echo("{} exists in clean_model and overwrite is False, skipping...".format(k), verbose=verbose)
                continue
            echo("Starting fourier filter of {} at {}".format(k, str(datetime.datetime.now())), verbose=verbose)
            d = data[k]
            f = flags[k]
            fw  = (~f).astype(np.float)
            w = fw * wgts[k]

            if ax == 'freq':
                # zeropad the data
                if zeropad > 0:
                    d, _ = zeropad_array(d, zeropad=zeropad, axis=1)
                    w, _ = zeropad_array(w, zeropad=zeropad, axis=1)
                    x = np.hstack([x.min() - (1+np.arange(zeropad)[::-1]) * np.mean(np.diff(x)), x,
                                   x.max() + (1+np.arange(zeropad)) * np.mean(np.diff(x))])
            elif ax == 'time':
                # zeropad the data
                if zeropad > 0:
                    d, _ = zeropad_array(d, zeropad=zeropad, axis=0)
                    w, _ = zeropad_array(w, zeropad=zeropad, axis=0)
                    x = np.hstack([x.min() - (1+np.arange(zeropad)[::-1]) * np.mean(np.diff(x)),x,
                                   x.max() + (1+np.arange(zeropad)) * np.mean(np.diff(x))])
            elif ax == 'both':
                if not isinstance(zeropad, (list,tuple)) or not len(zeropad) == 2:
                    raise ValueError("zeropad must be a 2-tuple or 2-list of integers")
                if not (isinstance(zeropad[0], (int, np.int)) and isinstance(zeropad[0], (int, np.int))):
                    raise ValueError("zeropad values must all be integers. You provided %s"%(zeropad))
                if zeropad[0] > 0 and zeropad[1] > 0:
                    d, _ = zeropad_array(d, zeropad=zeropad[1], axis=1)
                    w, _ = zeropad_array(w, zeropad=zeropad[1], axis=1)
                    d, _ = zeropad_array(d, zeropad=zeropad[0], axis=0)
                    w, _ = zeropad_array(w, zeropad=zeropad[0], axis=0)
                    x = [np.hstack([x[m].min() - (np.arange(zeropad)[::-1]+1) * np.mean(np.diff(x[m])),x[m],x[m].max() + (1+np.arange(zeropad)) * np.mean(np.diff(x[m]))]) for m in range(2)]
            mdl, res ,info = dspec.fourier_filter(x=x, data=d, wgts=w, filter_centers=filter_centers,
                                                  filter_half_widths=filter_half_widths,
                                                  suppression_factors=suppression_factors, mode=mode,
                                                  filter2d=filter2d, fitting_options=fitting_options,
                                                  filter_dim=filterdim, cache=cache, skip_wgt=skip_wgt,
                                                  max_contiguous_edge_flags=max_contiguous_edge_flags)

            #unzeropad array and put in skip flags.
            if ax == 'freq':
                if mode == 'clean':
                    info={0:{}, 1:info}
                if zeropad > 0:
                    mdl, _ = zeropad_array(mdl, zeropad=zeropad, axis=1, undo=True)
                    res, _ = zeropad_array(res, zeropad=zeropad, axis=1, undo=True)
            elif ax == 'time':
                if mode == 'clean':
                    info = {0:info, 1:{}}
                if zeropad > 0:
                    mdl, _ = zeropad_array(mdl, zeropad=zeropad, axis=0, undo=True)
                    res, _ = zeropad_array(res, zeropad=zeropad, axis=0, undo=True)
                elif ax == 'both':
                    for i in range(2):
                        if zeropad[i] > 0:
                            mdl, _ = zeropad_array(mdl, zeropad=zeropad[i], axis=i, undo=True)
                            res, _ = zeropad_array(res, zeropad=zeropad[i], axis=i, undo=True)

            flgs = np.zeros_like(mdl, dtype=np.bool)
            if not mode == 'clean':
                for dim in range(2):
                    if len(info[dim]) > 0:
                        for i in range(len(info[dim])):
                            if info[dim][i] == 'skipped':
                                if dim == 0:
                                    flgs[:, i] = True
                                elif dim == 1:
                                    flgs[i] = True
            else:
                if not ax=='both':
                    for dim in range(2):
                        if len(info[dim])>0:
                            for inf in info[dim]:
                                if inf['skipped']:
                                    if dim ==  0:
                                        flgs[:,i] = True
                                    elif dim == 1:
                                        flgs[i] = True
                else:
                    if w.max() > 0.0:
                        flgs = np.zeros_like(mdl, type=np.bool)
                    else:
                        info = {'skipped':True}



            filtered_model[k] = mdl
            filtered_resid[k] = res
            filtered_data[k] = mdl + res * fw
            filtered_flags[k] = flgs
            filtered_info[k] = info

        if hasattr(data, 'times'):
            filtered_data.times = data.times
            filtered_model.times = data.times
            filtered_resid.times = data.times
            filtered_flags.times = data.times

    def vis_clean(self, keys=None, data=None, flags=None, wgts=None, ax='freq', horizon=1.0, standoff=0.0,
                  min_dly=0.0, max_frate=None, tol=1e-6, maxiter=100, window='none', zeropad=0,
                  gain=1e-1, skip_wgt=0.1, filt2d_mode='rect', alpha=0.5, edgecut_low=0, edgecut_hi=0,
                  overwrite=False, output_prefix='clean', add_clean_residual=False, dtime=None, dnu=None,
                  verbose=True, mode='clean', cache={}, deconv_dayenu_foregrounds=False,
                  fg_deconv_method='clean', fg_restore_size=None, fg_deconv_fundamental_period=None):
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
            linear : bool,
                 use aipy.deconv.clean if linear == False
                 if True, perform linear delay filtering.
            cache : dict, optional dictionary for storing pre-computed filtering matrices in linear
                cleaning.
            deconv_dayenu_foregrounds : bool, if True, then apply clean to data - residual where
                                              res is the data-vector after applying a linear clean filter.
                                              This allows for in-painting flagged foregrounds without introducing
                                              clean artifacts into EoR window. If False, mdl will still just be the
                                              difference between the original data vector and the residuals after
                                              applying the linear filter.
            fg_deconv_method : string, can be 'leastsq' or 'clean'. If 'leastsq', deconvolve difference between data and linear residual
                                       by performing linear least squares fitting of data - linear resid to dft modes in filter window.
                                       If 'clean', obtain deconv fg model using perform a hogboem clean of difference between data and linear residual.
            fg_restore_size: float, optional, allow user to only restore foregrounds subtracted by linear filter
                             within a region of this size. If None, set to filter_size.
                             This allows us to avoid the problem that if we have RFI flagging and apply a linear filter
                             that is larger then the horizon then the foregrounds that we fit might actually include super
                             -horizon flagging side-lobes and restoring them will introduce spurious structure.
        """
        if linear:
            mode = 'dayenu'
        else:
            mode = 'clean'
        if not HAVE_UVTOOLS:
            raise ImportError("uvtools required, install hera_cal[all]")

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
                                                  gain=gain, skip_wgt=skip_wgt, edgecut_low=edgecut_low, mode=mode,
                                                  edgecut_hi=edgecut_hi, add_clean_residual=add_clean_residual,
                                                  cache=cache, deconv_dayenu_foregrounds=deconv_dayenu_foregrounds,
                                                  fg_deconv_method=fg_deconv_method, fg_restore_size=fg_restore_size, fg_deconv_fundamental_period=None)

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
                                                  edgecut_hi=edgecut_hi, mode=mode, cache=cache, deconv_dayenu_foregrounds=deconv_dayenu_foregrounds,
                                                  fg_deconv_method=fg_deconv_method, fg_restore_size=fg_restore_size)

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

                    mdl, res, info = dspec.vis_filter(d, w, bl_len=self.bllens[k[:2]], sdf=dnu, max_frate=max_frate[k], dt=dtime, mode=mode,
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


    def interleave_products(self, data=None, data2=None, keys=None, assign='interleaved_product', overwrite=False):
        """
        Take products of alternate time samples. Use to compute PS estimates without a
        noise bias.
        Args:
            data : DataContainer
                   Object to get interleaves from.
            data2 : DataContainer
                   Optional second data container to compute interleaved products from
            keys : list of tuples
                 List of keys to compute interleaved products from
            assign : str
                 Name of DataContainer to attach to self. Default is self.iproducts
            overwrite : bool
                 If iproducts[key] already exists, overwrite its contents.
        """
        if not hasattr(self, assign):
            setattr(self, assign, DataContainer({}))
        if data is None:
            data = self.data
        if keys is None:
            keys = data.keys()
        iproducts = getattr(self, assign)
        for k in keys:
        #if data2 is None, interleave time steps
            if data2 is None:
                if self.Ntimes % 2 == 0:
                    iproducts[k] = data[k][::2] * np.conj(data[k][1::2])
                else:
                    iproducts[k] = data[k][:-1:2] * np.conj(data[k][1:-1:2])
            else:
                iproducts[k] = data[k] * np.conj(data2[k])




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
    if not HAVE_UVTOOLS:
        raise ImportError("uvtools required, install hera_cal[all]")

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
