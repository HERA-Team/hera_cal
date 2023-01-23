# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

import numpy as np
from collections import OrderedDict as odict
import operator
import os
import copy
import warnings
from functools import reduce
from collections.abc import Iterable
from pyuvdata import UVCal, UVData
from pyuvdata import utils as uvutils
from astropy import units
from astropy.io import fits
import h5py
import hdf5plugin
import scipy
import pickle
import random
import glob
from pyuvdata.utils import POL_STR2NUM_DICT, POL_NUM2STR_DICT, ENU_from_ECEF, XYZ_from_LatLonAlt
import argparse
from hera_filters.dspec import place_data_on_uniform_grid


try:
    import aipy
    AIPY = True
except ImportError:
    AIPY = False

from . import utils
from . import redcal
from .datacontainer import DataContainer
from .utils import polnum2str, polstr2num, jnum2str, jstr2num, filter_bls, chunk_baselines_by_redundant_groups
from .utils import split_pol, conj_pol, split_bl, LST2JD, JD2LST, HERA_TELESCOPE_LOCATION


def _parse_input_files(inputs, name='input_data'):
    if isinstance(inputs, str):
        filepaths = [inputs]
    elif isinstance(inputs, Iterable):  # List loading
        if np.all([isinstance(i, str) for i in inputs]):  # List of visibility data paths
            filepaths = list(inputs)
        else:
            raise TypeError(f'If {name} is a list, it must be a list of strings.')
    else:
        raise ValueError(f'{name} must be a string or a list of strings.')
    for f in filepaths:
        if not os.path.exists(f):
            raise IOError('Cannot find file ' + f)
    return filepaths


class HERACal(UVCal):
    '''HERACal is a subclass of pyuvdata.UVCal meant to serve as an interface between
    pyuvdata-readable calfits files and dictionaries (the in-memory format for hera_cal)
    that map antennas and polarizations to gains, flags, and qualities. Supports standard
    UVCal functionality, along with read() and update() functionality for going back and
    forth to dictionaires. Upon read(), stores useful metadata internally.

    Does not support partial data loading or writing. Assumes a single spectral window.
    '''

    def __init__(self, input_cal):
        '''Instantiate a HERACal object. Currently only supports calfits files.

        Arguments:
            input_cal: string calfits file path or list of paths
        '''
        super().__init__()

        # parse input_data as filepath(s)
        self.filepaths = _parse_input_files(input_cal, name='input_cal')

    def _extract_metadata(self):
        '''Extract and store useful metadata and array indexing dictionaries.'''
        self.freqs = np.unique(self.freq_array)
        self.times = np.unique(self.time_array)
        self.pols = [jnum2str(j, x_orientation=self.x_orientation) for j in self.jones_array]
        self._jnum_indices = {jnum: i for i, jnum in enumerate(self.jones_array)}
        self.ants = [(ant, pol) for ant in self.ant_array for pol in self.pols]
        self._antnum_indices = {ant: i for i, ant in enumerate(self.ant_array)}

    def build_calcontainers(self):
        '''Turns the calibration information currently loaded into the HERACal object
        into ordered dictionaries that map antenna-pol tuples to calibration waterfalls.
        Computes and stores internally useful metadata in the process.

        Returns:
            gains: dict mapping antenna-pol keys to (Nint, Nfreq) complex gains arrays
            flags: dict mapping antenna-pol keys to (Nint, Nfreq) boolean flag arrays
            quals: dict mapping antenna-pol keys to (Nint, Nfreq) float qual arrays
            total_qual: dict mapping polarization to (Nint, Nfreq) float total quality array
        '''
        self._extract_metadata()
        gains, flags, quals, total_qual = odict(), odict(), odict(), odict()

        # build dict of gains, flags, and quals
        for (ant, pol) in self.ants:
            i, ip = self._antnum_indices[ant], self._jnum_indices[jstr2num(pol, x_orientation=self.x_orientation)]
            gains[(ant, pol)] = np.array(self.gain_array[i, :, :, ip].T)
            flags[(ant, pol)] = np.array(self.flag_array[i, :, :, ip].T)
            quals[(ant, pol)] = np.array(self.quality_array[i, :, :, ip].T)
        # build dict of total_qual if available
        for pol in self.pols:
            ip = self._jnum_indices[jstr2num(pol, x_orientation=self.x_orientation)]
            if self.total_quality_array is not None:
                total_qual[pol] = np.array(self.total_quality_array[:, :, ip].T)
            else:
                total_qual = None

        return gains, flags, quals, total_qual

    def read(self, antenna_nums=None, frequencies=None, freq_chans=None, times=None, pols=None):
        '''Reads calibration information from file, computes useful metadata and returns
        dictionaries that map antenna-pol tuples to calibration waterfalls. Currently, select options
        only perform selection after reading, so they are not true partial I/O. However, when
        initialized with a list of calibration files, non-time selection is done before concantenation,
        potentially saving memory.

        Arguments:
            antenna_nums : array_like of int, optional. Antenna numbers The antennas numbers to keep
                in the object (antenna positions and names for the removed antennas will be retained).
            frequencies : array_like of float, optional. The frequencies to keep in the object, each
                value passed here should exist in the freq_array.
            freq_chans : array_like of int, optional. The frequency channel numbers to keep in the object.
            times : array_like of float, optional. The times to keep in the object, each value passed
                here should exist in the time_array of one of the files in input_cal.
            pols : array_like of str, optional. These strings should be convertable to polarization
                numbers via pyuvdata's jstr2num e.g. ['Jee'].

        Returns:
            gains: dict mapping antenna-pol keys to (Nint, Nfreq) complex gains arrays
            flags: dict mapping antenna-pol keys to (Nint, Nfreq) boolean flag arrays
            quals: dict mapping antenna-pol keys to (Nint, Nfreq) float qual arrays
            total_qual: dict mapping polarization to (Nint, Nfreq) float total quality array
        '''
        # if filepaths is None, this was converted to HERAData
        # from a different pre-loaded object with no history of filepath

        if self.filepaths is not None:
            # load data
            self.read_calfits(self.filepaths[0])
            self.use_future_array_shapes()

            if pols is not None:
                pols = [jstr2num(ap, x_orientation=self.x_orientation) for ap in pols]
            # only read antennas present in the data and raise a warning.
            my_ants = np.unique(self.ant_array)
            if antenna_nums is not None:
                for ant in antenna_nums:
                    if ant not in my_ants:
                        warnings.warn(f"Warning, antenna {ant} not present in calibration solution. Skipping!")
                antenna_nums = np.intersect1d(my_ants, antenna_nums)
            select_dict = {'antenna_nums': antenna_nums, 'frequencies': frequencies,
                           'freq_chans': freq_chans, 'jones': pols}
            if np.any([s is not None for s in select_dict.values()]):
                self.select(inplace=True, **select_dict)

            # If there's more than one file, loop over all files, downselecting and cont
            if len(self.filepaths) > 1:
                for fp in self.filepaths[1:]:
                    uvc = UVCal()
                    uvc.read_calfits(fp)
                    uvc.use_future_array_shapes()

                    if np.any([s is not None for s in select_dict.values()]):
                        uvc.select(inplace=True, **select_dict)
                    self += uvc

        # downselect times at the very end, since this might exclude some files in the original list
        if times is not None:
            self.select(times=times)
        return self.build_calcontainers()

    def update(self, gains=None, flags=None, quals=None, total_qual=None, tSlice=None, fSlice=None):
        '''Update internal calibrations arrays (data_array, flag_array, and nsample_array)
        using DataContainers (if not left as None) in preparation for writing to disk.

        Arguments:
            gains: optional dict mapping antenna-pol to complex gains arrays
            flags: optional dict mapping antenna-pol to boolean flag arrays
            quals: optional dict mapping antenna-pol to float qual arrays
            total_qual: optional dict mapping polarization to float total quality array.
            tSlice: optional slice of indices of the times to update. Must have the same size
                as the 0th dimension of the input gains/flags/quals/total_quals
            fSlice: optional slice of indices of the freqs to update. Must have the same size
                as the 1st dimension of the input gains/flags/quals/total_quals
        '''
        # provide sensible defaults for tSlice and fSlice
        if tSlice is None:
            tSlice = slice(0, self.Ntimes)
        if fSlice is None:
            fSlice = slice(0, self.Nfreqs)

        # loop over and update gains, flags, and quals
        data_arrays = [self.gain_array, self.flag_array, self.quality_array]
        for to_update, array in zip([gains, flags, quals], data_arrays):
            if to_update is not None:
                for (ant, pol) in to_update.keys():
                    i, ip = self._antnum_indices[ant], self._jnum_indices[jstr2num(pol, x_orientation=self.x_orientation)]
                    array[i, fSlice, tSlice, ip] = to_update[(ant, pol)].T

        # update total_qual
        if total_qual is not None:
            if self.total_quality_array is None:
                self.total_quality_array = np.zeros(self.gain_array.shape[1:], dtype=float)
            for pol in total_qual.keys():
                ip = self._jnum_indices[jstr2num(pol, x_orientation=self.x_orientation)]
                self.total_quality_array[fSlice, tSlice, ip] = total_qual[pol].T

    def write(self, filename, spoof_missing_channels=False, **write_kwargs):
        """
        Shallow wrapper for UVCal calfits writer with functionality to spoof missing channels.

        Parameters
        ----------
        filename: str
            name of file to write to.
        fill_in_missing_freqs: bool, optional
            If True, spoof missing channels with flagged gains set equal to unity.
        write_kwargs: kwarg dict
            kwargs for UVCal.write_calfits
        """
        if spoof_missing_channels:
            writer = copy.deepcopy(self)
            # Since calfits do not support frequency discontinunities, we add support here
            # By spoofing frequencies between discontinunities with flagged gains.
            # This line provides freqs_filled -- frequency axis with spoofed frequencies
            # and inserted which is a boolean array that is True at frequencies that are being spoofed.
            freqs_filled, _, _, inserted = place_data_on_uniform_grid(self.freqs, np.ones_like(self.freqs), np.ones_like(self.freqs))
            writer.freq_array = freqs_filled.flatten()
            writer.Nfreqs = len(freqs_filled)
            writer.channel_width = np.median(writer.channel_width) * np.ones_like(writer.freq_array)
            # insert original flags and gains into appropriate channels.
            new_gains = np.ones((writer.Nants_data, writer.Nfreqs, writer.Ntimes, writer.Njones), dtype=complex)
            new_gains[:, ~inserted, :, :] = writer.gain_array
            new_flags = np.ones(new_gains.shape, dtype=bool)
            new_flags[:, ~inserted, :, :] = writer.flag_array
            new_quality = np.zeros(new_gains.shape, dtype=float)
            new_quality[:, ~inserted, :, :] = writer.quality_array

            writer.flag_array = new_flags
            writer.gain_array = new_gains
            writer.quality_array = new_quality

            writer.write_calfits(filename, **write_kwargs)
        else:
            self.write_calfits(filename, **write_kwargs)


def read_hera_calfits(filenames, ants=None, pols=None,
                      read_gains=True, read_flags=False, read_quality=False, read_tot_quality=False,
                      check=False, dtype=np.complex128, verbose=False):
    '''A faster interface to getting data out of HERA calfits files. Only concatenates
    along time axis. Puts times in ascending order,
    but does not check that files are contiguous.

    Arguments:
        filenames: list of files to read
        ants: list of ants or (ant, [polstr]) tuples to read out of files.
              Default (None) is to use the intersection of all antennas
              across files.
        pols: list of pol strings to read out of files
        read_gains: (bool, True): read gains
        read_flags (bool, False): read flags
        read_quality (bool, False): read quality array
        read_tot_quality (bool, False): read total quality array
        check (bool, False): run sanity checks to make sure files match.
        dtype (np.complex128): numpy datatype for output complex-valued arrays
        verbose: print some progress messages.

    Returns:
        rv: dictionary with keys 'info' (metadata), 'gains' (dictionary of waterfalls
            with (ant,pol) keys), 'flags', 'quality', and 'total_quality'. Will omit
            keys according to read_gains, read_flags, and read_quality.
    '''

    info = {}
    times = {}
    inds = {}
    # grab header information from all cal files
    filenames = _parse_input_files(filenames, name='input_cal')
    for cnt, filename in enumerate(filenames):
        with fits.open(filename) as fname:
            hdr = fname[0].header
            _times = uvutils._fits_gethduaxis(fname[0], 3)
            _thash = hash(_times.tobytes())
            if _thash not in times:
                times[_thash] = (_times, [filename])
            else:
                times[_thash][1].append(filename)
            hdunames = uvutils._fits_indexhdus(fname)
            nants = hdr['NAXIS6']
            anthdu = fname[hdunames["ANTENNAS"]]
            antdata = anthdu.data
            _ants = antdata["ANTARR"][:nants].astype(int)
            _ahash = hash(_ants.tobytes())
            if _ahash not in inds:
                inds[_ahash] = {ant: idx for idx, ant in enumerate(_ants)}
                if 'ants' in info:
                    info['ants'].intersection_update(set(inds[_ahash].keys()))
                else:
                    info['ants'] = set(inds[_ahash].keys())
            jones_array = uvutils._fits_gethduaxis(fname[0], 2)
            _jhash = hash(jones_array.tobytes())
            if _jhash not in inds:
                info['x_orientation'] = x_orient = hdr['XORIENT']
                _pols = [uvutils.parse_jpolstr(uvutils.JONES_NUM2STR_DICT[num], x_orientation=x_orient)
                         for num in jones_array]
                if 'pols' in info:
                    info['pols'] = info['pols'].union(set(_pols))
                else:
                    info['pols'] = set(_pols)
                inds[_jhash] = {pol: idx for idx, pol in enumerate(_pols)}
            inds[filename] = (inds[_ahash], inds[_jhash])
            if cnt == 0:
                if 'ANTXYZ' in antdata.names:
                    info['antpos'] = antdata["ANTXYZ"]
                info['freqs'] = uvutils._fits_gethduaxis(fname[0], 4)
                info['gain_convention'] = gain_convention = hdr.pop("GNCONVEN")
                info['cal_type'] = cal_type = hdr.pop("CALTYPE")
            if check:
                assert gain_convention == 'divide'  # HERA standard
                assert cal_type == 'gain'  # delay-style calibration currently unsupported
                assert np.all(info['freqs'] == uvutils._fits_gethduaxis(fname[0], 4))

    if ants is None:
        # generate a set of ants if we didn't have one passed in
        if pols is None:
            pols = info['pols']
        ants = set((ant,) for ant in info['ants'])
        ants = set(ant + (p,) for ant in ants for p in pols)
    else:
        ants = set((ant,) if np.issubdtype(type(ant), np.integer) else ant for ant in ants)
        # if length 1 ants are passed in, add on polarizations
        ants_len1 = set(ant for ant in ants if len(ant) == 1)
        if len(ants_len1) > 0:
            if pols is None:
                pols = info['pols']
            ants = set(ant for ant in ants if len(ant) == 2)
            ants = ants.union([ant + (p,) for ant in ants_len1 for p in pols])
        # record polarizations as total of ones indexed in bls
        pols = set(ant[1] for ant in ants)
    times = list(times.values())
    times.sort(key=lambda x: x[0][0])
    filenames = (v[1] for v in times)
    times = np.concatenate([t[0] for t in times], axis=0)
    info['times'] = times
    tot_times = times.size
    nfreqs = info['freqs'].size

    # preallocate buffers
    def nan_empty(shape, dtype):
        '''Allocate nan-filled buffers, in case file time/pol
        misalignments lead to uninitialized data buffer slots.'''
        buf = np.empty(shape, dtype=dtype)
        buf.fill(np.nan)
        return buf

    rv = {}
    if read_gains:
        rv['gains'] = {ant: nan_empty((tot_times, nfreqs), dtype) for ant in ants}
    if read_flags:
        rv['flags'] = {ant: nan_empty((tot_times, nfreqs), bool) for ant in ants}
    if read_quality:
        rv['quality'] = {ant: nan_empty((tot_times, nfreqs), np.float32) for ant in ants}
    if read_tot_quality:
        rv['total_quality'] = {p: nan_empty((tot_times, nfreqs), np.float32) for p in info['pols']}
    # bail here if all we wanted was the info
    if len(rv) == 0:
        return {'info': info}

    # loop through files and read data
    t = 0
    for cnt, _filenames in enumerate(filenames):
        for filename in _filenames:
            antind, polind = inds[filename]
            with fits.open(filename) as fname:
                hdr = fname[0].header
                ntimes = hdr.pop("NAXIS3")
                if read_gains:
                    data = fname[0].data
                    for (a, p) in rv['gains'].keys():
                        if a not in antind or p not in polind:
                            continue
                        rv['gains'][a, p][t:t + ntimes].real = fname[0].data[antind[a], 0, :, :, polind[p], 0].T
                        rv['gains'][a, p][t:t + ntimes].imag = fname[0].data[antind[a], 0, :, :, polind[p], 1].T
                if read_flags:
                    for (a, p) in rv['flags'].keys():
                        if a not in antind or p not in polind:
                            continue
                        rv['flags'][a, p][t:t + ntimes] = fname[0].data[antind[a], 0, :, :, polind[p], 2].T
                if read_quality:
                    for (a, p) in rv['quality'].keys():
                        if a not in antind or p not in polind:
                            continue
                        rv['quality'][a, p][t:t + ntimes] = fname[0].data[antind[a], 0, :, :, polind[p], 3].T
                if read_tot_quality:
                    tq_hdu = fname[hdunames["TOTQLTY"]]
                    for p in rv['total_quality'].keys():
                        if p not in polind:
                            continue
                        rv['total_quality'][p][t:t + ntimes] = tq_hdu.data[0, :, :, polind[p]].T
        t += ntimes
    rv['info'] = info
    return rv


def get_blt_slices(uvo, tried_to_reorder=False):
    '''For a pyuvdata-style UV object, get the mapping from antenna pair to blt slice.
    If the UV object does not have regular spacing of baselines in its baseline-times,
    this function will try to reorder it using UVData.reorder_blts() to see if that helps.

    Arguments:
        uvo: a "UV-Object" like UVData or baseline-type UVFlag. Blts may get re-ordered internally.
        tried_to_reorder: used internally to prevent infinite recursion

    Returns:
        blt_slices: dictionary mapping anntenna pair tuples to baseline-time slice objects
    '''
    blt_slices = {}
    for ant1, ant2 in uvo.get_antpairs():
        indices = uvo.antpair2ind(ant1, ant2)
        if len(indices) == 1:  # only one blt matches
            blt_slices[(ant1, ant2)] = slice(indices[0], indices[0] + 1, uvo.Nblts)
        elif not (len(set(np.ediff1d(indices))) == 1):  # checks if the consecutive differences are all the same
            if not tried_to_reorder:
                uvo.reorder_blts(order='time')
                return get_blt_slices(uvo, tried_to_reorder=True)
            else:
                raise NotImplementedError('UVData objects with non-regular spacing of '
                                          'baselines in its baseline-times are not supported.')
        else:
            blt_slices[(ant1, ant2)] = slice(indices[0], indices[-1] + 1, indices[1] - indices[0])
    return blt_slices


class HERAData(UVData):
    '''HERAData is a subclass of pyuvdata.UVData meant to serve as an interface between
    pyuvdata-compatible data formats on disk (especially uvh5) and DataContainers,
    the in-memory format for visibilities used in hera_cal. In addition to standard
    UVData functionality, HERAData supports read() and update() functions that interface
    between internal UVData data storage and DataContainers, which contain visibility
    data in a dictionary-like format, along with some useful metadata. read() supports
    partial data loading, though only the most useful subset of selection modes from
    pyuvdata (and not all modes for all data types).

    When using uvh5, HERAData supports additional useful functionality:
    * Upon __init__(), the most useful metadata describing the entire file is loaded into
      the object (everything in HERAData_metas; see get_metadata_dict() for details).
    * Partial writing using partial_write(), which will initialize a new file with the
      same metadata and write to disk using DataContainers by assuming that the user is
      writing to the same part of the data as the most recent read().
    * Generators that enable iterating over baseline, frequency, or time in chunks (see
      iterate_over_bls(), iterate_over_freqs(), and iterate_over_times() for details).

    Assumes a single spectral window. Assumes that data for a given baseline is regularly
    spaced in the underlying data_array.
    '''
    # static list of useful metadata to calculate and save
    HERAData_metas = ['ants', 'data_ants', 'antpos', 'data_antpos', 'freqs', 'times', 'lsts',
                      'pols', 'antpairs', 'bls', 'times_by_bl', 'lsts_by_bl']
    # ants: list of antenna numbers in the array
    # data_ants: list of antenna numbers in the data file
    # antpos: dictionary mapping all antenna numbers in the telescope to np.arrays of position in meters
    # data_antpos: dictionary mapping all antenna numbers in the data to np.arrays of position in meters
    # freqs: np.arrray of frequencies (Hz)
    # times: np.array of unique times in the data file (JD)
    # lsts: np.array of unique LSTs in the data file (radians)
    # pols: list of baseline polarization strings
    # antpairs: list of antenna number pairs in the data as 2-tuples
    # bls: list of baseline-pols in the data as 3-tuples
    # times_by_bl: dictionary mapping antpairs to times (JD). Also includes all reverse pairs.
    # lsts_by_bl: dictionary mapping antpairs to LSTs (radians). Also includes all reverse pairs.

    def __init__(self, input_data, upsample=False, downsample=False, filetype='uvh5', **read_kwargs):
        '''Instantiate a HERAData object. If the filetype is either uvh5 or uvfits, read in and store
        useful metadata (see get_metadata_dict()), either as object attributes or,
        if input_data is a list, as dictionaries mapping string paths to metadata.

        Arguments:
            input_data: string data file path or list of string data file paths
            upsample: bool. If True, will upsample to match the shortest integration time in the file.
                Upsampling will affect the time metadata stored on this object.
            downsample: bool. If True, will downsample to match the longest integration time in the file.
                Downsampling will affect the time metadata stored on this object.
            filetype: supports 'uvh5' (default), 'miriad', 'uvfits'
            read_kwargs : kwargs to pass to UVData.read (e.g. run_check, check_extra and
                run_check_acceptability). Only used for uvh5 filetype
        '''
        # initialize as empty UVData object
        super().__init__()

        # parse input_data as filepath(s)
        self.filepaths = _parse_input_files(input_data, name='input_data')

        # parse arguments into object
        self.upsample = upsample
        self.downsample = downsample
        if self.upsample and self.downsample:
            raise ValueError('upsample and downsample cannot both be True.')
        self.filetype = filetype

        # load metadata from file
        if self.filetype in ['uvh5', 'uvfits']:
            # read all UVData metadata from first file
            temp_paths = copy.deepcopy(self.filepaths)
            self.filepaths = self.filepaths[0]
            self.read(read_data=False, **read_kwargs)
            self.filepaths = temp_paths

            self._attach_metadata(**read_kwargs)

        elif self.filetype == 'miriad':
            for meta in self.HERAData_metas:
                setattr(self, meta, None)  # no pre-loading of metadata
        else:
            raise NotImplementedError('Filetype ' + self.filetype + ' has not been implemented.')

        # save longest and shortest integration times in the file for later use in up/downsampling
        # if available, these will be used instead of the ones in self.integration_time during partial I/O
        self.longest_integration = None
        self.longest_integration = None
        if self.integration_time is not None:
            self.longest_integration = np.max(self.integration_time)
            self.shortest_integration = np.min(self.integration_time)

    def _attach_metadata(self, **read_kwargs):
        """
        Attach metadata.
        """
        if hasattr(self, "filepaths") and self.filepaths is not None and len(self.filepaths) > 1:  # save HERAData_metas in dicts
            for meta in self.HERAData_metas:
                setattr(self, meta, {})
            for f in self.filepaths:
                hd = HERAData(f, filetype='uvh5', **read_kwargs)
                meta_dict = hd.get_metadata_dict()
                for meta in self.HERAData_metas:
                    getattr(self, meta)[f] = meta_dict[meta]
        else:  # save HERAData_metas as attributes
            self._writers = {}
            for key, value in self.get_metadata_dict().items():
                setattr(self, key, value)

    def reset(self):
        '''Resets all standard UVData attributes, potentially freeing memory.'''
        super(HERAData, self).__init__()

    def get_metadata_dict(self):
        ''' Produces a dictionary of the most useful metadata. Used as object
        attributes and as metadata to store in DataContainers.

        Returns:
            metadata_dict: dictionary of all items in self.HERAData_metas
        '''
        antpos, ants = self.get_ENU_antpos(pick_data_ants=False)
        antpos = dict(zip(ants, antpos))
        data_ants = np.unique(np.concatenate((self.ant_1_array, self.ant_2_array)))
        data_antpos = {ant: antpos[ant] for ant in data_ants}

        # get times using the most commonly appearing baseline, presumably the one without BDA
        most_common_bl_num = scipy.stats.mode(self.baseline_array, keepdims=True)[0][0]
        times = self.time_array[self.baseline_array == most_common_bl_num]
        lsts = self.lst_array[self.baseline_array == most_common_bl_num]

        freqs = np.unique(self.freq_array)
        pols = [polnum2str(polnum, x_orientation=self.x_orientation) for polnum in self.polarization_array]
        antpairs = self.get_antpairs()
        bls = [antpair + (pol,) for antpair in antpairs for pol in pols]

        times_by_bl = {antpair: np.array(self.time_array[self._blt_slices[antpair]])
                       for antpair in antpairs}
        times_by_bl.update({(ant1, ant0): times_here for (ant0, ant1), times_here in times_by_bl.items()})
        lsts_by_bl = {antpair: np.array(self.lst_array[self._blt_slices[antpair]])
                      for antpair in antpairs}
        lsts_by_bl.update({(ant1, ant0): lsts_here for (ant0, ant1), lsts_here in lsts_by_bl.items()})

        locs = locals()
        return {meta: locs[meta] for meta in self.HERAData_metas}

    def _determine_blt_slicing(self):
        '''Determine the mapping between antenna pairs and slices of the blt axis of the data_array.'''
        self._blt_slices = get_blt_slices(self)

    def _determine_pol_indexing(self):
        '''Determine the mapping between polnums and indices
        in the polarization axis of the data_array.'''
        self._polnum_indices = {
            polnum: i for i, polnum in enumerate(self.polarization_array)
        }
        pols = [polnum2str(polnum, x_orientation=self.x_orientation) for polnum in self.polarization_array]
        self._polstr_indices = {}
        # Add upper-case indices as well, so we don't need to use .lower() on input
        # keys (for which there can be many tens of thousands).
        for pol in pols:
            indx = self._polnum_indices[polstr2num(pol, x_orientation=self.x_orientation)]
            self._polstr_indices[pol.lower()] = indx
            self._polstr_indices[pol.upper()] = indx

    def _get_slice(self, data_array, key):
        '''Return a copy of the Nint by Nfreq waterfall or waterfalls for a given key. Abstracts
        away both baseline ordering (by applying complex conjugation) and polarization capitalization.

        Arguments:
            data_array: numpy array of shape (Nblts, 1, Nfreq, Npol), i.e. the size of the full data.
                One generally uses this object's own self.data_array, self.flag_array, or self.nsample_array.
            key: if of the form (0,1,'nn'), return anumpy array.
                 if of the form (0,1), return a dict mapping pol strings to waterfalls.
                 if of of the form 'nn', return a dict mapping ant-pair tuples to waterfalls.
        '''
        if isinstance(key, str):  # asking for a pol
            return {antpair: self._get_slice(data_array, antpair + (key,)) for antpair in self.get_antpairs()}
        elif len(key) == 2:  # asking for antpair
            pols = np.array([polnum2str(polnum, x_orientation=self.x_orientation) for polnum in self.polarization_array])
            return {pol: self._get_slice(data_array, key + (pol,)) for pol in pols}
        elif len(key) == 3:  # asking for bl-pol
            try:
                if data_array.ndim == 4: # old shapes
                    return np.array(
                        data_array[
                            self._blt_slices[tuple(key[:2])], 0, :,
                            self._polstr_indices.get(
                                key[2],
                                self._polnum_indices[
                                    polstr2num(key[2], x_orientation=self.x_orientation)
                                ]
                            )
                        ]
                    )
                else:
                    return np.array(
                        data_array[
                            self._blt_slices[tuple(key[:2])], :,
                            self._polstr_indices.get(
                                key[2],
                                self._polnum_indices[
                                    polstr2num(key[2], x_orientation=self.x_orientation)
                                ]
                            )
                        ]
                    )
            except KeyError:
                if data_array.ndim == 4:
                    return np.conj(
                        data_array[
                            self._blt_slices[tuple(key[1::-1])], 0, :,
                            self._polstr_indices.get(
                                conj_pol(key[2]),
                                self._polnum_indices[
                                    polstr2num(conj_pol(key[2]), x_orientation=self.x_orientation)
                                ]
                            )
                        ]
                    )
                else:
                    return np.conj(
                        data_array[
                            self._blt_slices[tuple(key[1::-1])], :,
                            self._polstr_indices.get(
                                conj_pol(key[2]),
                                self._polnum_indices[
                                    polstr2num(conj_pol(key[2]), x_orientation=self.x_orientation)
                                ]
                            )
                        ]
                    )
        else:
            raise KeyError('Unrecognized key type for slicing data.')

    def _set_slice(self, data_array, key, value):
        '''Update data_array with Nint by Nfreq waterfall(s). Abstracts away both baseline
        ordering (by applying complex conjugation) and polarization capitalization.

        Arguments:
            data_array: numpy array of shape (Nblts, 1, Nfreq, Npol), i.e. the size of the full data.
                One generally uses this object's own self.data_array, self.flag_array, or self.nsample_array.
            key: baseline (e.g. (0,1,'nn)), ant-pair tuple (e.g. (0,1)), or pol str (e.g. 'nn')
            value: if key is a baseline, must be an (Nint, Nfreq) numpy array;
                   if key is an ant-pair tuple, must be a dict mapping pol strings to waterfalls;
                   if key is a pol str, must be a dict mapping ant-pair tuples to waterfalls
        '''
        if isinstance(key, str):  # providing pol with all antpairs
            for antpair in value.keys():
                self._set_slice(data_array, (antpair + (key,)), value[antpair])
        elif len(key) == 2:  # providing antpair with all pols
            for pol in value.keys():
                self._set_slice(data_array, (key + (pol,)), value[pol])
        elif len(key) == 3:  # providing bl-pol
            try:
                data_array[self._blt_slices[tuple(key[0:2])], :,
                           self._polnum_indices[polstr2num(key[2], x_orientation=self.x_orientation)]] = value
            except(KeyError):
                data_array[self._blt_slices[tuple(key[1::-1])], :,
                           self._polnum_indices[polstr2num(conj_pol(key[2]), x_orientation=self.x_orientation)]] = np.conj(value)
        else:
            raise KeyError('Unrecognized key type for slicing data.')

    def build_datacontainers(self):
        '''Turns the data currently loaded into the HERAData object into DataContainers.
        Returned DataContainers include useful metadata specific to the data actually
        in the DataContainers (which may be a subset of the total data). This includes
        antenna positions, frequencies, all times, all lsts, and times and lsts by baseline.

        Returns:
            data: DataContainer mapping baseline keys to complex visibility waterfalls
            flags: DataContainer mapping baseline keys to boolean flag waterfalls
            nsamples: DataContainer mapping baseline keys to interger Nsamples waterfalls
        '''
        # build up DataContainers
        data, flags, nsamples = odict(), odict(), odict()
        meta = self.get_metadata_dict()
        for bl in meta['bls']:
            data[bl] = self._get_slice(self.data_array, bl)
            flags[bl] = self._get_slice(self.flag_array, bl)
            nsamples[bl] = self._get_slice(self.nsample_array, bl)
        data = DataContainer(data)
        flags = DataContainer(flags)
        nsamples = DataContainer(nsamples)

        # store useful metadata inside the DataContainers
        for dc in [data, flags, nsamples]:
            for attr in ['ants', 'data_ants', 'antpos', 'data_antpos', 'freqs', 'times', 'lsts', 'times_by_bl', 'lsts_by_bl']:
                setattr(dc, attr, copy.deepcopy(meta[attr]))

        return data, flags, nsamples

    def read(self, bls=None, polarizations=None, times=None, time_range=None, lsts=None, lst_range=None,
             frequencies=None, freq_chans=None, axis=None, read_data=True, return_data=True,
             run_check=True, check_extra=True, run_check_acceptability=True, **kwargs):
        '''Reads data from file. Supports partial data loading. Default: read all data in file.

        Arguments:
            bls: A list of antenna number tuples (e.g. [(0,1), (3,2)]) or a list of
                baseline 3-tuples (e.g. [(0,1,'nn'), (2,3,'ee')]) specifying baselines
                to keep in the object. For length-2 tuples, the  ordering of the numbers
                within the tuple does not matter. For length-3 tuples, the polarization
                string is in the order of the two antennas. If length-3 tuples are provided,
                the polarizations argument below must be None. Ignored if read_data is False.
            polarizations: The polarizations to include when reading data into
                the object.  Ignored if read_data is False.
            times: The times to include when reading data into the object.
                Ignored if read_data is False. Miriad will load then select on this axis.
            time_range : length-2 array-like of float, optional. The time range in Julian Date
                to include. Cannot be used with `times`.
            lsts: The lsts in radians to include when reading data into the object.
                Ignored if read_data is False. Miriad will load then select on this axis.
                Cannot be used with `times` or `time_range`.
            lst_range : length-2 array-like of float, optional. The lst range in radians
                to include when. Cannot be used with `times`, `time_range`, or `lsts`.
                Miriad will load then select on this axis. If the second value is smaller than
                the first, the LSTs are treated as having phase-wrapped around LST = 2*pi = 0
                and the LSTs kept on the object will run from the larger value, through 0, and
                end at the smaller value.
            frequencies: The frequencies to include when reading data. Ignored if read_data
                is False. Miriad will load then select on this axis.
            freq_chans: The frequency channel numbers to include when reading data. Ignored
                if read_data is False. Miriad will load then select on this axis.
            axis: Axis for fast concatenation of files (if len(self.filepaths) > 1).
                Allowed values are: 'blt', 'freq', 'polarization'.
            read_data: Read in the visibility and flag data. If set to false, only the
                basic metadata will be read in and nothing will be returned. Results in an
                incompletely defined object (check will not pass). Default True.
            return_data: bool, if True, return the output of build_datacontainers().
            run_check: Option to check for the existence and proper shapes of
                parameters after reading in the file. Default is True.
            check_extra: Option to check optional parameters as well as required
                ones. Default is True.
            run_check_acceptability: Option to check acceptable range of the values of
                parameters after reading in the file. Default is True.
            kwargs: extra keyword arguments to pass to UVData.read()

        Returns:
            data: DataContainer mapping baseline keys to complex visibility waterfalls
            flags: DataContainer mapping baseline keys to boolean flag waterfalls
            nsamples: DataContainer mapping baseline keys to interger Nsamples waterfalls
        '''
        # save last read parameters
        locs = locals()
        partials = ['bls', 'polarizations', 'times', 'time_range', 'lsts', 'lst_range', 'frequencies', 'freq_chans']
        self.last_read_kwargs = {p: locs[p] for p in partials}

        # if filepaths is None, this was converted to HERAData
        # from a different pre-loaded object with no history of filepath
        if self.filepaths is not None:
            temp_read = self.read  # store self.read while it's being overwritten
            self.read = super().read  # re-define self.read so UVData can call self.read recursively for lists of files
            # load data
            try:
                if self.filetype in ['uvh5', 'uvfits']:
                    super().read(self.filepaths, file_type=self.filetype, axis=axis, bls=bls, polarizations=polarizations,
                                 times=times, time_range=time_range, lsts=lsts, lst_range=lst_range, frequencies=frequencies,
                                 freq_chans=freq_chans, read_data=read_data, run_check=run_check, check_extra=check_extra,
                                 run_check_acceptability=run_check_acceptability, **kwargs)
                    self.use_future_array_shapes()
                    if self.filetype == 'uvfits':
                        self.unphase_to_drift()
                else:
                    if not read_data:
                        raise NotImplementedError('reading only metadata is not implemented for ' + self.filetype)
                    if self.filetype == 'miriad':
                        super().read(self.filepaths, file_type='miriad', axis=axis, bls=bls, polarizations=polarizations,
                                     time_range=time_range, run_check=run_check, check_extra=check_extra,
                                     run_check_acceptability=run_check_acceptability, **kwargs)
                        self.use_future_array_shapes()
                        if any([times is not None, lsts is not None, lst_range is not None,
                                frequencies is not None, freq_chans is not None]):
                            warnings.warn('miriad does not support partial loading for times/lsts (except time_range) and frequencies. '
                                          'Loading the file first and then performing select.')
                            self.select(times=times, lsts=lsts, lst_range=lst_range, frequencies=frequencies, freq_chans=freq_chans)

                # upsample or downsample data, as appropriate, including metadata. Will use self.longest/shortest_integration
                # if not None (which came from whole file metadata) since partial i/o might change the current longest or
                # shortest integration in a way that would create insonsistency between partial reads/writes.
                if self.upsample:
                    if hasattr(self, 'shortest_integration') and self.shortest_integration is not None:
                        self.upsample_in_time(max_int_time=self.shortest_integration)
                    else:
                        self.upsample_in_time(max_int_time=np.min(self.integration_time))
                if self.downsample:
                    if hasattr(self, 'longest_integration') and self.longest_integration is not None:
                        self.downsample_in_time(min_int_time=self.longest_integration)
                    else:
                        self.downsample_in_time(min_int_time=np.max(self.integration_time))

            finally:
                self.read = temp_read  # reset back to this function, regardless of whether the above try excecutes successfully

        # process data into DataContainers
        if read_data or self.filetype in ['uvh5', 'uvfits']:
            self._determine_blt_slicing()
            self._determine_pol_indexing()
        if read_data and return_data:
            return self.build_datacontainers()

    def select(self, inplace=True, **kwargs):
        """
        Select-out parts of a HERAData object.

        Args:
            inplace: Overwrite self, otherwise return a copy.
            kwargs : pyuvdata.UVData select keyword arguments.
        """
        # select
        output = super(HERAData, self).select(inplace=inplace, **kwargs)
        if inplace:
            output = self

        # recompute slices if necessary
        names = ['antenna_nums', 'antenna_names', 'ant_str', 'bls', 'blt_inds',
                 'times', 'time_range', 'lsts', 'lst_range']
        for n in names:
            if n in kwargs and kwargs[n] is not None:
                output._determine_blt_slicing()
                output._determine_pol_indexing()
                break
        if 'polarizations' in kwargs and kwargs['polarizations'] is not None:
            output._determine_pol_indexing()

        if not inplace:
            return output

    def __add__(self, other, inplace=False, **kwargs):
        """
        Combine two HERAData objects.

        Combine along baseline-time, polarization or frequency.
        See pyuvdata.UVData.__add__ for more details.

        Args:
            other : Another HERAData object
            inplace: Overwrite self as we go, otherwise create a third object
                as the sum of the two (default).
            kwargs : UVData.__add__ keyword arguments
        """
        output = super(HERAData, self).__add__(other, inplace=inplace, **kwargs)
        if inplace:
            output = self
        output._determine_blt_slicing()
        output._determine_pol_indexing()
        if not inplace:
            return output

    def __getitem__(self, key):
        """
        Shortcut for reading a single visibility waterfall given a
        baseline tuple. If key exists it will return it using its
        blt_slice, if it does not it will attempt to read it
        from disk.
        """
        try:
            return self._get_slice(self.data_array, key)
        except KeyError:
            return self.read(bls=key)[0][key]

    def update(self, data=None, flags=None, nsamples=None, tSlice=None, fSlice=None):
        '''Update internal data arrays (data_array, flag_array, and nsample_array)
        using DataContainers (if not left as None) in preparation for writing to disk.

        Arguments:
            data: Optional DataContainer mapping baselines to complex visibility waterfalls
            flags: Optional DataContainer mapping baselines to boolean flag waterfalls
            nsamples: Optional DataContainer mapping baselines to interger Nsamples waterfalls
            tSlice: Optional slice of indices of the times to update. Must have the same size
                as the 0th dimension of the input gains/flags/nsamples.
            fSlice: Optional slice of indices of the freqs to update. Must have the same size
                as the 1st dimension of the input gains/flags/nsamples.
        '''
        # provide sensible defaults for tinds and finds
        update_full_waterfall = (tSlice is None) and (fSlice is None)
        if tSlice is None:
            tSlice = slice(0, self.Ntimes)
        if fSlice is None:
            fSlice = slice(0, self.Nfreqs)

        def _set_subslice(data_array, bl, this_waterfall):
            if update_full_waterfall:
                # directly write into relevant data_array
                self._set_slice(data_array, bl, this_waterfall)
            else:
                # copy out full waterfall, update just the relevant slices, and write back to data_array
                full_waterfall = self._get_slice(data_array, bl)
                full_waterfall[tSlice, fSlice] = this_waterfall
                self._set_slice(data_array, bl, full_waterfall)

        if data is not None:
            for bl in data.keys():
                _set_subslice(self.data_array, bl, data[bl])
        if flags is not None:
            for bl in flags.keys():
                _set_subslice(self.flag_array, bl, flags[bl])
        if nsamples is not None:
            for bl in nsamples.keys():
                _set_subslice(self.nsample_array, bl, nsamples[bl])

    def partial_write(self, output_path, data=None, flags=None, nsamples=None,
                      clobber=False, inplace=False, add_to_history='',
                      **kwargs):
        '''Writes part of a uvh5 file using DataContainers whose shape matches the most recent
        call to HERAData.read() in this object. The overall file written matches the shape of the
        input_data file called on __init__. Any data/flags/nsamples left as None will be written
        as was currently stored in the HERAData object. Does not work for other filetypes or when
        the HERAData object is initialized with a list of files.

        Arguments:
            output_path: path to file to write uvh5 file to
            data: Optional DataContainer mapping baselines to complex visibility waterfalls
            flags: Optional DataContainer mapping baselines to boolean flag waterfalls
            nsamples: Optional DataContainer mapping baselines to interger Nsamples waterfalls
            clobber: if True, overwrites existing file at output_path
            inplace: update this object's data_array, flag_array, and nsamples_array.
                This saves memory but alters the HERAData object.
            add_to_history: string to append to history (only used on first call of
                partial_write for a given output_path)
            kwargs: addtional keyword arguments update UVData attributes. (Only used on
                first call of partial write for a given output_path).
        '''
        # Type verifications
        if self.filetype != 'uvh5':
            raise NotImplementedError('Partial writing for filetype ' + self.filetype + ' has not been implemented.')
        if len(self.filepaths) > 1:
            raise NotImplementedError('Partial writing for list-loaded HERAData objects has not been implemented.')

        # get writer or initialize new writer if necessary
        if output_path in self._writers:
            hd_writer = self._writers[output_path]  # This hd_writer has metadata for the entire output file
        else:
            hd_writer = HERAData(self.filepaths[0])
            hd_writer.history += add_to_history
            for attribute, value in kwargs.items():
                hd_writer.__setattr__(attribute, value)
            hd_writer.initialize_uvh5_file(output_path, clobber=clobber)  # Makes an empty file (called only once)
            self._writers[output_path] = hd_writer
        if inplace:  # update this objects's arrays using DataContainers
            this = self
        else:  # make a copy of this object and then update the relevant arrays using DataContainers
            this = copy.deepcopy(self)
        this.update(data=data, flags=flags, nsamples=nsamples)
        hd_writer.write_uvh5_part(output_path, this.data_array, this.flag_array,
                                  this.nsample_array, **self.last_read_kwargs)

    def iterate_over_bls(self, Nbls=1, bls=None, chunk_by_redundant_group=False, reds=None,
                         bl_error_tol=1.0, include_autos=True, frequencies=None):
        '''Produces a generator that iteratively yields successive calls to
        HERAData.read() by baseline or group of baselines.

        Arguments:
            Nbls: number of baselines to load at once.
            bls: optional user-provided list of baselines to iterate over.
                Default: use self.bls (which only works for uvh5).
            chunk_by_redundant_group: bool, optional
                If true, retrieve bls sorted by redundant groups.
                If Nbls is greater then the number of baselines in a redundant group
                then return consecutive redundant groups with total baseline count
                less then or equal to Nbls.
                If Nbls is smaller then the number of baselines in a redundant group
                then still return that group but raise a Warning.
                Default is False
            reds: list, optional
                list of lists; each containing the antpairpols in each redundant group
                must be provided if chunk_by_redundant_group is True.
            bl_error_tol: float, optional
                    the largest allowable difference between baselines in a redundant group in meters.
                    (in the same units as antpos). Normally, this is up to 4x the largest antenna position error.
                    default is 1.0meters
            include_autos: bool, optional
                include autocorrelations in iteration if True.
                Default is True.
            frequencies: array-like, optional
                optional list of float frequencies to load.
                Default (None) loads all frequencies in data.

        Yields:
            data, flags, nsamples: DataContainers (see HERAData.read() for more info).
        '''
        if bls is None:
            if self.filetype != 'uvh5':
                raise NotImplementedError('Baseline iteration without explicitly setting bls for filetype ' + self.filetype
                                          + ' without setting bls has not been implemented.')
            bls = self.bls
            if isinstance(bls, dict):  # multiple files
                bls = list(set([bl for bls in bls.values() for bl in bls]))
            bls = sorted(bls)
        if not chunk_by_redundant_group:
            if not include_autos:
                # filter out autos if include_autos is False.
                bls = [bl for bl in bls if bl[0] != bl[1]]
            baseline_chunks = [bls[i:i + Nbls] for i in range(0, len(bls), Nbls)]
        else:
            if reds is None:
                if self.filetype != 'uvh5':
                    raise NotImplementedError('Redundant group iteration without explicitly setting antpos for filetype ' + self.filetype
                                              + ' without setting antpos has not been implemented.')

                # generate data_antpos dict to feed into get_reds
                # that accounts for possibility that
                # HERAData was initialized from multiple
                # files in which case self.data_antpos is a dict of dicts.
                if len(self.filepaths) > 1:
                    data_antpos = {}
                    for k in self.data_antpos:
                        data_antpos.update(self.data_antpos[k])
                    pols = set({})
                    for k in self.pols:
                        pols.union(set(self.pols[k]))
                    pols = list(pols)
                else:
                    data_antpos = self.data_antpos
                    pols = self.pols

                reds = redcal.get_reds(data_antpos, pols=pols, bl_error_tol=bl_error_tol,
                                       include_autos=include_autos)
            # filter reds by baselines
            reds = redcal.filter_reds(reds, bls=bls)
            # make sure that every baseline is in reds
            baseline_chunks = chunk_baselines_by_redundant_groups(reds=reds, max_chunk_size=Nbls)
        for chunk in baseline_chunks:
            yield self.read(bls=chunk, frequencies=frequencies)

    def iterate_over_freqs(self, Nchans=1, freqs=None):
        '''Produces a generator that iteratively yields successive calls to
        HERAData.read() by frequency channel or group of contiguous channels.

        Arguments:
            Nchans: number of frequencies to load at once.
            freqs: optional user-provided list of frequencies to iterate over.
                Default: use self.freqs (which only works for uvh5).

        Yields:
            data, flags, nsamples: DataContainers (see HERAData.read() for more info).
        '''
        if freqs is None:
            if self.filetype != 'uvh5':
                raise NotImplementedError('Frequency iteration for filetype ' + self.filetype
                                          + ' without setting freqs has not been implemented.')
            freqs = self.freqs
            if isinstance(self.freqs, dict):  # multiple files
                freqs = np.unique(list(self.freqs.values()))
        for i in range(0, len(freqs), Nchans):
            yield self.read(frequencies=freqs[i:i + Nchans])

    def iterate_over_times(self, Nints=1, times=None):
        '''Produces a generator that iteratively yields successive calls to
        HERAData.read() by time or group of contiguous times. N.B. May
        produce unexpected results for BDA data that has not been upsampled
        or downsampled to a common time resolution.

        Arguments:
            Nints: number of integrations to load at once.
            times: optional user-provided list of times to iterate over.
                Default: use self.times (which only works for uvh5).

        Yields:
            data, flags, nsamples: DataContainers (see HERAData.read() for more info).
        '''
        if times is None:
            if self.filetype != 'uvh5':
                raise NotImplementedError('Time iteration for filetype ' + self.filetype
                                          + ' without setting times has not been implemented.')
            times = self.times
            if isinstance(times, dict):  # multiple files
                times = np.unique(list(times.values()))
        for i in range(0, len(times), Nints):
            yield self.read(times=times[i:i + Nints])

    def init_HERACal(self, gain_convention='divide', cal_style='redundant'):
        '''Produces a HERACal object using the metadata in this HERAData object.

        Arguments:
            gain_convention: str indicating whether gains are to calibrated by "multiply"ing or "divide"ing.
            cal_style: str indicating how calibration was done, either "sky" or "redundant".

        Returns:
            HERACal object with gain, flag, quality, and total_quality arrays initialized (to 1, True, 0, and 0)
        '''
        # create UVCal object from self
        uvc = UVCal().initialize_from_uvdata(self, gain_convention='divide', cal_style='redundant')

        # create empty data arrays (using future array shapes, which is default true for initialize_from_uvdata)
        uvc.gain_array = np.ones((uvc.Nants_data, uvc.Nfreqs, uvc.Ntimes, uvc.Njones), dtype=np.complex64)
        uvc.flag_array = np.ones((uvc.Nants_data, uvc.Nfreqs, uvc.Ntimes, uvc.Njones), dtype=bool)
        uvc.quality_array = np.zeros((uvc.Nants_data, uvc.Nfreqs, uvc.Ntimes, uvc.Njones), dtype=np.float32)
        uvc.total_quality_array = np.zeros((uvc.Nfreqs, uvc.Ntimes, uvc.Njones), dtype=np.float32)

        # convert to HERACal and return
        return to_HERACal(uvc)

    def empty_arrays(self):
        '''Sets self.data_array and self.nsample_array to all zeros and self.flag_array to all True (if they are not None).'''
        self.data_array = (np.zeros_like(self.data_array) if self.data_array is not None else None)
        self.flag_array = (np.ones_like(self.flag_array) if self.flag_array is not None else None)
        self.nsample_array = (np.zeros_like(self.nsample_array) if self.nsample_array is not None else None)


def read_hera_hdf5(filenames, bls=None, pols=None, full_read_thresh=0.002,
                   read_data=True, read_flags=False, read_nsamples=False,
                   check=False, dtype=np.complex128, verbose=False):
    '''A potentially faster interface for reading HERA HDF5 files. Only concatenates
    along time axis. Puts times in ascending order, but does not check that
    files are contiguous. Currently not BDA compatible.

    Arguments:
        filenames: list of files to read
        bls: list of (ant_1, ant_2, [polstr]) tuples to read out of files.
             Default: all bls common to all files.
        pols: list of pol strings to read out of files. Default: all, but is
              superceded by any polstrs listed in bls.
        full_read_thresh (0.002): fractional threshold for reading whole file
                                  instead of baseline by baseline.
        read_data (bool, True): read data
        read_flags (bool, False): read flags
        read_nsamples (bool, False): read nsamples
        check (bool, False): run sanity checks to make sure files match.
        dtype (np.complex128): numpy datatype for output complex-valued arrays
        verbose: print some progress messages.

    Returns:
        rv: dict with keys 'info' and optionally 'data', 'flags', and 'nsamples',
            based on whether read_data, read_flags, and read_nsamples are true.
        rv['info']: metadata dict with keys 'freqs' (1D array), 'times' (1D array),
                    'pols' (list), 'ants' (1D array), 'antpos' (dict of antenna: 3D position),
                    'bls' (list of all (ant_1, ant_2) baselines in the file), 'data_ants' (1D array)
                    'latitude' (float in degrees), longitude (float in degrees), altitude (float in m)
        rv['data']: dict of 2D data with (i, j, pol) keys.
        rv['flags']: dict of 2D flags with (i, j, pol) keys.
        rv['nsamples']: dict of 2D nsamples with (i, j, pol) keys.
    '''
    info = {}
    times = []
    bl2ind = {}
    inds = {}
    # Read file metadata to size up arrays and sort times
    filenames = _parse_input_files(filenames, name='input_data')
    for filename in filenames:
        if verbose:
            print(f'Reading header of {filename}')
        with h5py.File(filename, 'r') as f:
            h = f['/Header']
            if check:
                # Check that there aren't extra spectral windows
                assert int(h['Nspws'][()]) == 1  # not a hera file
            if len(times) == 0:
                if len(h['freq_array'].shape) == 2:  # old pyuvdata shapes with spectral windows
                    info['freqs'] = h['freq_array'][0]  # make 1D instead of 2D
                else:
                    info['freqs'] = h['freq_array'][()]  # make 1D instead of 2D
                nfreqs = info['freqs'].size
                pol_array = h['polarization_array'][()]
                npols = pol_array.size
                # the following errors if x_orientation not set in this hdf5
                x_orient = str(h['x_orientation'][()], encoding='utf-8')
                pol_indices = {uvutils.parse_polstr(POL_NUM2STR_DICT[n], x_orientation=x_orient): cnt
                               for cnt, n in enumerate(pol_array)}
                info['pols'] = list(pol_indices.keys())
                info['ants'] = antenna_numbers = h['antenna_numbers'][()]
                info['antpos'] = dict(zip(antenna_numbers, h['antenna_positions'][()]))
                for coord in ['latitude', 'longitude', 'altitude']:
                    info[coord] = h[coord][()]
            elif check:
                # Check that all files have the same number of frequencies
                assert int(h['Nfreqs'][()]) == nfreqs
            # Determine blt ordering (baselines then times, or times then baselines)
            ntimes = int(h['Ntimes'][()])
            _times = h['time_array'][:ntimes]
            time_first = (np.unique(_times).size == ntimes)
            nbls = int(h['Nblts'][()]) // ntimes
            if time_first:
                # time-baseline ordering
                ant1_array = h['ant_1_array'][::ntimes]
                ant2_array = h['ant_2_array'][::ntimes]
            else:
                # baseline-time ordering
                _times = h['time_array'][::nbls]
                ant1_array = h['ant_1_array'][:nbls]
                ant2_array = h['ant_2_array'][:nbls]
            _info = {'time_first': time_first, 'ntimes': ntimes, 'nbls': nbls}
            times.append((_times, filename, _info))
            data_ants = set(ant1_array)
            data_ants.update(set(ant2_array))
            _hash = hash((ant1_array.tobytes(), ant2_array.tobytes(), time_first, ntimes))
            # map baselines to array indices for each unique antenna order
            if _hash not in inds:
                if time_first:
                    inds[_hash] = {(i, j): slice(n * ntimes, (n + 1) * ntimes)
                                   for n, (i, j) in enumerate(zip(ant1_array,
                                                                  ant2_array))}
                else:
                    inds[_hash] = {(i, j): slice(n, None, nbls)
                                   for n, (i, j) in enumerate(zip(ant1_array,
                                                                  ant2_array))}
                if bls is not None:
                    # Make sure our baselines of interest are in this file
                    if not all([bl[:2] in inds[_hash] for bl in bls]):
                        missing_bls = [bl for bl in bls if bl[:2] not in inds[_hash]]
                        raise ValueError(f'File {filename} missing:' + str(missing_bls))
                        assert bl[:2] in inds[_hash]
                if 'bls' not in info:
                    info['bls'] = set(inds[_hash].keys())
                    info['data_ants'] = data_ants
                else:
                    info['bls'].intersection_update(set(inds[_hash].keys()))
                    info['data_ants'].intersection_update(data_ants)
            bl2ind[filename] = inds[_hash]

    if bls is None:
        # generate a set of bls if we didn't have one passed in
        if pols is None:
            pols = list(pol_indices.keys())
        bls = info['bls']
        bls = set(bl + (p,) for bl in bls for p in pols)
    else:
        # if length 2 baselines are passed in, add on polarizations
        bls_len2 = set(bl for bl in bls if len(bl) == 2)
        if len(bls_len2) > 0:
            if pols is None:
                pols = list(pol_indices.keys())
            bls = set(bl for bl in bls if len(bl) == 3)
            bls = bls.union([bl + (p,) for bl in bls_len2 for p in pols])
        # record polarizations as total of ones indexed in bls
        pols = set(bl[2] for bl in bls)
    # sort files by time of first integration
    times.sort(key=lambda x: x[0][0])
    info['times'] = np.concatenate([t[0] for t in times], axis=0)
    tot_times = info['times'].size

    # preallocate buffers
    rv = {}
    if read_data:
        rv['visdata'] = {bl: np.empty((tot_times, nfreqs), dtype=dtype) for bl in bls}
    if read_flags:
        rv['flags'] = {bl: np.empty((tot_times, nfreqs), dtype=bool) for bl in bls}
    if read_nsamples:
        rv['nsamples'] = {bl: np.empty((tot_times, nfreqs), dtype=np.float32) for bl in bls}
    # bail here if all we wanted was the info
    if len(rv) == 0:
        return {'info': info}

    t = 0
    for _times, filename, _info in times:
        inds = bl2ind[filename]
        ntimes = _info['ntimes']
        nbls = _info['nbls']
        if verbose:
            print(f'Reading data from {filename}')
        with h5py.File(filename, 'r') as f:
            if check:
                h = f['/Header']
                assert ntimes == int(h['Ntimes'][()])
                assert nbls == int(h['Nblts'][()]) // ntimes
                # Check that files sorted correctly into time order
                if _info['time_first']:
                    assert np.allclose(h['time_array'][:ntimes], _times)
                else:
                    assert np.allclose(h['time_array'][::nbls], _times)
            # decide whether to read all the data in, or use partial I/O
            full_read = (len(bls) > full_read_thresh * nbls * npols)
            if full_read and verbose:
                print('Reading full file')
            for key, data in rv.items():
                d = f['/Data'][key]  # data not read yet
                if full_read:
                    d = d[()]  # reads data

                # Support old array shapes
                if len(d.shape) == 4:
                    # Support polarization-transposed arrays
                    if d.shape[-1] == nfreqs:
                        def index_exp(i, j, p):
                            return np.index_exp[inds[i, j], 0, pol_indices[p]]
                    else:
                        def index_exp(i, j, p):
                            return np.index_exp[inds[i, j], 0, :, pol_indices[p]]
                # Support new array shapes
                if len(d.shape) == 3:
                    # Support polarization-transposed arrays
                    if d.shape[-1] == nfreqs:
                        def index_exp(i, j, p):
                            return np.index_exp[inds[i, j], pol_indices[p]]
                    else:
                        def index_exp(i, j, p):
                            return np.index_exp[inds[i, j], :, pol_indices[p]]

                # handle HERA's raw (int) and calibrated (complex) file formats
                if key == 'visdata' and not np.iscomplexobj(d):
                    for i, j, p in bls:
                        _d = d[index_exp(i, j, p)]
                        data[i, j, p][t:t + ntimes].real = _d['r']
                        data[i, j, p][t:t + ntimes].imag = _d['i']
                else:
                    for i, j, p in bls:
                        data[i, j, p][t:t + ntimes] = d[index_exp(i, j, p)]

            t += ntimes
    # Quick renaming of data key for niceness
    if 'visdata' in rv:
        rv['data'] = rv.pop('visdata', [])
    info['data_ants'] = np.array(sorted(info['data_ants']))
    rv['info'] = info
    return rv


class HERADataFastReader():
    '''Wrapper class around read_hera_hdf5 meant to mimic the functionality of HERAData for drop-in replacement.'''

    def __init__(self, input_data, read_metadata=True, check=False, skip_lsts=False):
        '''Instantiates a HERADataFastReader object. Only supports reading uvh5 files, not writing them.
        Does not support BDA and only supports patial i/o along baselines and polarization axes.

        Arguments:
            input_data: path or list of paths to uvh5 files.
            read_metadata (bool, True): reads metadata from file and stores it internally to try to match HERAData
            check (bool, False): run sanity checks to make sure files match.
            skip_lsts (bool, False): save time by not computing LSTs from JDs
        '''
        # parse input_data as filepath(s)
        self.filepaths = _parse_input_files(input_data, name='input_data')

        # load metadata only
        rv = {'info': {}}
        if read_metadata:
            rv = read_hera_hdf5(self.filepaths, read_data=False, read_flags=False, read_nsamples=False, check=False)
            self._adapt_metadata(rv['info'], skip_lsts=skip_lsts)

        # update metadata internally
        self.info = rv['info']
        for meta in HERAData.HERAData_metas:
            if meta in rv['info']:
                setattr(self, meta, rv['info'][meta])
            else:
                setattr(self, meta, None)

        # create functions that error informatively when trying to use standard HERAData/UVData methods
        for funcname in list(dir(HERAData)):
            if funcname.startswith('__') and funcname.endswith('__'):
                continue  # don't overwrite things like __class__ and __init__
            if funcname in ['read', '_make_datacontainer', '_HERAData_error']:
                continue  # don't overwrite functions with errors that we actually use
            setattr(self, funcname, self._HERAData_error)

    def _adapt_metadata(self, info_dict, skip_lsts=False):
        '''Updates metadata from read_hera_hdf5 to better match HERAData. Updates info_dict in place.'''
        info_dict['data_ants'] = sorted(info_dict['data_ants'])
        info_dict['antpairs'] = sorted(info_dict['bls'])
        info_dict['bls'] = sorted(set([ap + (pol, ) for ap in info_dict['antpairs'] for pol in info_dict['pols']]))
        XYZ = XYZ_from_LatLonAlt(info_dict['latitude'] * np.pi / 180, info_dict['longitude'] * np.pi / 180, info_dict['altitude'])
        enu_antpos = ENU_from_ECEF(np.array([antpos for ant, antpos in info_dict['antpos'].items()]) + XYZ,
                                   info_dict['latitude'] * np.pi / 180, info_dict['longitude'] * np.pi / 180, info_dict['altitude'])
        info_dict['antpos'] = {ant: enu for enu, ant in zip(enu_antpos, info_dict['antpos'])}
        info_dict['data_antpos'] = {ant: info_dict['antpos'][ant] for ant in info_dict['data_ants']}
        info_dict['times'] = np.unique(info_dict['times'])
        info_dict['times_by_bl'] = {ap: info_dict['times'] for ap in info_dict['antpairs']}
        info_dict['times_by_bl'].update({(a2, a1): info_dict['times'] for (a1, a2) in info_dict['antpairs']})
        if not skip_lsts:
            info_dict['lsts'] = JD2LST(info_dict['times'], info_dict['latitude'], info_dict['longitude'], info_dict['altitude'])
            info_dict['lsts_by_bl'] = {ap: info_dict['lsts'] for ap in info_dict['antpairs']}

    def _HERAData_error(self, *args, **kwargs):
        raise NotImplementedError('HERADataFastReader does not support this method. Try HERAData instead.')

    def read(self, bls=None, pols=None, full_read_thresh=0.002, read_data=True, read_flags=True,
             read_nsamples=True, check=False, dtype=np.complex128, verbose=False, skip_lsts=False):
        '''A faster read that only concatenates along the time axis. Puts times in ascending order, but does not
        check that files are contiguous. Currently not BDA compatible.

        Arguments:
            bls: list of (ant_1, ant_2, [polstr]) tuples to read out of files. Default: all bls common to all files.
            pols: list of pol strings to read out of files. Default: all, but is superceded by any polstrs listed in bls.
            full_read_thresh (0.002): fractional threshold for reading whole file instead of baseline by baseline.
            read_data (bool, True): read data
            read_flags (bool, True): read flags
            read_nsamples (bool, True): read nsamples
            check (bool, False): run sanity checks to make sure files match.
            dtype (np.complex128): numpy datatype for output complex-valued arrays
            verbose: print some progress messages.
            skip_lsts (bool, False): save time by not computing LSTs from JDs

        Returns:
            data: DataContainer mapping baseline keys to complex visibility waterfalls (if read_data is True, else None)
            flags: DataContainer mapping baseline keys to boolean flag waterfalls (if read_flags is True, else None)
            nsamples: DataContainer mapping baseline keys to interger Nsamples waterfalls (if read_nsamples is True, else None)
        '''
        rv = read_hera_hdf5(self.filepaths, bls=bls, pols=pols, full_read_thresh=full_read_thresh,
                            read_data=read_data, read_flags=read_flags, read_nsamples=read_nsamples,
                            check=check, dtype=dtype, verbose=verbose)
        self._adapt_metadata(rv['info'], skip_lsts=skip_lsts)

        # make autocorrleations real by taking the abs, matches UVData._fix_autos()
        if 'data' in rv:
            for bl in rv['data']:
                if split_bl(bl)[0] == split_bl(bl)[1]:
                    rv['data'][bl] = np.abs(rv['data'][bl])

        # construct datacontainers from result
        return self._make_datacontainer(rv, 'data'), self._make_datacontainer(rv, 'flags'), self._make_datacontainer(rv, 'nsamples')

    def _make_datacontainer(self, rv, key='data'):
        '''Converts outputs from read_hera_hdf5 to a more standard HERAData output.'''
        if key not in rv:
            return None

        # construct datacontainer with whatever metadata is available
        dc = DataContainer(rv[key])
        for meta in HERAData.HERAData_metas:
            if meta in rv['info'] and meta not in ['pols', 'antpairs', 'bls']:  # these are functions on datacontainers
                setattr(dc, meta, rv['info'][meta])

        return dc


def read_filter_cache_scratch(cache_dir):
    """
    Load files from a cache specified by cache_dir.
    cache files are intended to serve as common short-term on-disk scratch for filtering matrices
    that can be loaded by multiple compute nodes process a night and save computational time by avoiding
    recomputing filter matrices (that often involve psuedo-inverses).

    A node processing a single chunk will be able to read in any cache matrices that were already
    computed from previous chunks.

    cache files are named with randomly generated strings with the extension ".filter_cache". They
    are not intended for long-term or cross-platform storage and are currently designed to be deleted at the end
    of processing night.

    Parameters
    ----------
    cache_dir, string, path to a folder that is used for the cache
        files in this folder with an extension .filter_cache are assumed
        to be cache files. These files are pickled caches from previous filtering runs.
    """
    # Load up the cache file with the most keys (precomputed filter matrices).
    cache = {}
    cache_files = glob.glob(cache_dir + '/*.filter_cache')
    # loop through cache files, load them.
    # If there are new keys, add them to internal cache.
    # If not, delete the reference matrices from memory.
    for cache_file in cache_files:
        cfile = open(cache_file, 'rb')
        cache_t = pickle.load(cfile)
        for key in cache_t:
            if key not in cache:
                cache[key] = cache_t[key]
    return cache


def write_filter_cache_scratch(filter_cache, cache_dir=None, skip_keys=None):
    """
    write cached cache to a new cache file.

    cache files are intended to serve as common short-term on-disk scratch for filtering matrices
    that can be loaded by multiple compute nodes process a night and save computational time by avoiding
    recomputing filter matrices (that often involve psuedo-inverses).

    A node processing a single chunk will be able to read in any cache matrices that were already
    computed from previous chunks.

    cache files are named with randomly generated strings with the extension ".filter_cache". They
    are not intended for long-term or cross-platform storage and are currently designed to be deleted at the end
    of processing night.

    Parameters
    ----------
    filter_cache, dict, dictionary of values that we wish to cache.
    cache_dir, string, optional, path to a folder that is used for the cache
        files in this folder with an extension .filter_cache are assumed
        to be cache files. These files are pickled caches from previous filtering runs.
        default, current working directory.
    skip_keys, list, list of keys to skip in writing the filter_cache.
    """
    if skip_keys is None:
        skip_keys = []
    # if the keys_before instantiation wasn't a list, then
    # keys_before would just be the current keys of cache and we
    # wouldn't have any new keys.
    new_filters = {k: filter_cache[k] for k in filter_cache if k not in skip_keys}
    if len(new_filters) > 0:
        # generate new file name
        if cache_dir is None:
            cache_dir = os.getcwd()
        cache_file_name = '%032x' % random.getrandbits(128) + '.filter_cache'
        cfile = open(os.path.join(cache_dir, cache_file_name), 'ab')
        pickle.dump(new_filters, cfile)
    else:
        warnings.warn("No new keys provided. No cache file written.")


def load_flags(flagfile, filetype='h5', return_meta=False):
    '''Load flags from a file and returns them as a DataContainer (for per-visibility flags)
    or dictionary (for per-antenna or per-polarization flags). More than one spectral window
    is not supported. Assumes times are evenly-spaced and in order for each baseline.

    Arguments:
        flagfile: path to file containing flags and flagging metadata
        filetype: either 'h5' or 'npz'. 'h5' assumes the file is readable as a hera_qm
            UVFlag object in the 'flag' mode (could be by baseline, by antenna, or by
            polarization). 'npz' provides legacy support for the IDR2.1 flagging npzs,
            but only for per-visibility flags.
        return_meta: if True, return a metadata dictionary with, e.g., 'times', 'freqs', 'history'

    Returns:
        flags: dictionary or DataContainer mapping keys to Ntimes x Nfreqs numpy arrays.
            if 'h5' and 'baseline' mode or 'npz': DataContainer with keys like (0,1,'nn')
            if 'h5' and 'antenna' mode: dictionary with keys like (0,'Jnn')
            if 'h5' and 'waterfall' mode: dictionary with keys like 'Jnn'
        meta: (only returned if return_meta is True)
    '''
    flags = {}
    if filetype not in ['h5', 'npz']:
        raise ValueError("filetype must be 'h5' or 'npz'.")

    elif filetype == 'h5':
        from pyuvdata import UVFlag
        uvf = UVFlag(flagfile)
        assert uvf.mode == 'flag', 'The input h5-based UVFlag object must be in flag mode.'
        assert (np.issubsctype(uvf.polarization_array.dtype, np.signedinteger)
                or np.issubsctype(uvf.polarization_array.dtype, np.str_)), \
            "The input h5-based UVFlag object's polarization_array must be integers or byte strings."
        freqs = np.unique(uvf.freq_array)
        times = np.unique(uvf.time_array)
        history = uvf.history

        if uvf.type == 'baseline':  # one time x freq waterfall per baseline
            blt_slices = get_blt_slices(uvf)
            for ip, pol in enumerate(uvf.polarization_array):
                if np.issubdtype(uvf.polarization_array.dtype, np.signedinteger):
                    pol = polnum2str(pol, x_orientation=uvf.x_orientation)  # convert to string if possible
                else:
                    pol = ','.join([polnum2str(int(p), x_orientation=uvf.x_orientation) for p in pol.split(',')])
                for (ant1, ant2), blt_slice in blt_slices.items():
                    flags[(ant1, ant2, pol)] = uvf.flag_array[blt_slice, 0, :, ip]
            # data container only supports standard polarizations strings
            if np.issubdtype(uvf.polarization_array.dtype, np.signedinteger):
                flags = DataContainer(flags)

        elif uvf.type == 'antenna':  # one time x freq waterfall per antenna
            for i, ant in enumerate(uvf.ant_array):
                for ip, jpol in enumerate(uvf.polarization_array):
                    if np.issubdtype(uvf.polarization_array.dtype, np.signedinteger):
                        jpol = jnum2str(jpol, x_orientation=uvf.x_orientation)  # convert to string if possible
                    else:
                        jpol = ','.join([jnum2str(int(p), x_orientation=uvf.x_orientation) for p in jpol.split(',')])
                    flags[(ant, jpol)] = np.array(uvf.flag_array[i, 0, :, :, ip].T)

        elif uvf.type == 'waterfall':  # one time x freq waterfall (per visibility polarization)
            for ip, jpol in enumerate(uvf.polarization_array):
                if np.issubdtype(uvf.polarization_array.dtype, np.signedinteger):
                    jpol = jnum2str(jpol, x_orientation=uvf.x_orientation)  # convert to string if possible
                else:
                    jpol = ','.join([jnum2str(int(p), x_orientation=uvf.x_orientation) for p in jpol.split(',')])
                flags[jpol] = uvf.flag_array[:, :, ip]

    elif filetype == 'npz':  # legacy support for IDR 2.1 npz format
        npz = np.load(flagfile)
        pols = [polnum2str(p) for p in npz['polarization_array']]
        freqs = np.unique(npz['freq_array'])
        times = np.unique(npz['time_array'])
        history = npz['history']
        nAntpairs = len(npz['antpairs'])
        assert npz['flag_array'].shape[0] == nAntpairs * len(times), \
            'flag_array must have flags for all baselines for all times.'
        for p, pol in enumerate(pols):
            flag_array = np.reshape(npz['flag_array'][:, 0, :, p], (len(times), nAntpairs, len(freqs)))
            for n, (i, j) in enumerate(npz['antpairs']):
                flags[i, j, pol] = flag_array[:, n, :]
        flags = DataContainer(flags)

    if return_meta:
        return flags, {'freqs': freqs, 'times': times, 'history': history}
    else:
        return flags


def get_file_times(filepaths, filetype='uvh5'):
    """
    Get a file's lst_array in radians and time_array in Julian Date.

    Some caveats:
        - Miriad standard is bin start, so a shift by int_time / 2 is performed.
          uvh5 standard is bin center, so times are left untouched.
        - Miriad files do not support baseline-dependent averaging (BDA).
        - With BDA for uvh5 files, the results will correspond to the least-averaged
          baseline in the file.
        - With uvh5 files with a single integration, it is assumed that the integration
          time and dtime are the same. This may not be true in LST-binned files.

    Args:
        filepaths : type=list or str, filepath or list of filepaths
        filetype : str, options=['miriad', 'uvh5']

    Returns:
        dlst : ndarray (or float if filepaths is a string) of lst bin width [radian]
        dtime : ndarray (or float if filepaths is a string) of time bin width [Julian Date]
        file_lst_arrays : list of ndarrays (or list of floats if filepaths is a string)
            of unwrapped lst_array [radians]
        file_time_arrays : list of ndarrays (or list of floats if filepaths is a string)
            of time_array [Julian Date]
    """
    _array = True
    # check filepaths type
    if isinstance(filepaths, str):
        _array = False
        filepaths = [filepaths]

    if filetype not in ['miriad', 'uvh5']:
        raise ValueError("filetype {} not recognized".format(filetype))

    # form empty lists
    dlsts = []
    dtimes = []
    file_lst_arrays = []
    file_time_arrays = []

    # get Nfiles
    Nfiles = len(filepaths)

    # iterate over filepaths and extract time info
    for i, f in enumerate(filepaths):
        if filetype == 'miriad':
            assert AIPY, "you need aipy to use the miriad filetype"
            uv = aipy.miriad.UV(f)
            # get integration time
            int_time = uv['inttime'] / (units.si.day.in_units(units.si.s))
            int_time_rad = uv['inttime'] * 2 * np.pi / (units.si.sday.in_units(units.si.s))
            # get start and stop, add half an integration
            start_lst = uv['lst'] + int_time_rad / 2.0
            start_time = uv['time'] + int_time / 2.0
            # form time arrays
            lst_array = (start_lst + np.arange(uv['ntimes']) * int_time_rad) % (2 * np.pi)
            time_array = start_time + np.arange(uv['ntimes']) * int_time

        elif filetype == 'uvh5':
            # get times directly from uvh5 file's header: faster than loading entire file via HERAData
            with h5py.File(f, mode='r') as _f:
                # pull out time_array and lst_array
                time_array = np.ravel(_f[u'Header'][u'time_array'])
                if u'lst_array' in _f[u'Header']:
                    lst_array = np.ravel(_f[u'Header'][u'lst_array'])
                else:
                    # need to generate lst_array on the fly
                    lst_array = np.ravel(uvutils.get_lst_for_time(_f[u'Header'][u'time_array'],
                                                                  _f[u'Header'][u'latitude'][()],
                                                                  _f[u'Header'][u'longitude'][()],
                                                                  _f[u'Header'][u'altitude'][()]))

                # figure out which baseline has the most times in order to handle BDA appropriately
                baseline_array = uvutils.antnums_to_baseline(np.array(_f[u'Header'][u'ant_1_array']),
                                                             np.array(_f[u'Header'][u'ant_2_array']),
                                                             np.array(_f[u'Header'][u'Nants_telescope']))
                most_common_bl_num = scipy.stats.mode(baseline_array, keepdims=True)[0][0]
                time_array = time_array[baseline_array == most_common_bl_num]
                lst_array = lst_array[baseline_array == most_common_bl_num]

                # figure out dtime and dlst, handling the case where a diff cannot be done.
                if len(time_array) > 1:
                    int_time = np.median(np.diff(time_array))
                    int_time_rad = np.median(np.diff(lst_array))
                else:
                    warnings.warn(f'{f} has only one time, so we assume that dtime is the minimum '
                                  'integration time. This may be incorrect for LST-binned files.')
                    int_time = np.min(_f[u'Header'][u'integration_time']) / units.day.to(units.si.s)
                    int_time_rad = int_time / units.sday.to(units.day) * 2 * np.pi

        dlsts.append(int_time_rad)
        dtimes.append(int_time)
        file_lst_arrays.append(lst_array)
        file_time_arrays.append(time_array)

    dlsts = np.asarray(dlsts)
    dtimes = np.asarray(dtimes)

    if _array is False:
        return dlsts[0], dtimes[0], file_lst_arrays[0], file_time_arrays[0]
    else:
        return dlsts, dtimes, file_lst_arrays, file_time_arrays


def partial_time_io(hd, times=None, time_range=None, lsts=None, lst_range=None, **kwargs):
    '''Perform partial io with a time-select on a HERAData object, even if it is intialized
    using multiple files, some of which do not contain any of the specified times.
    Note: can only use one of times, time_range, lsts, lst_range

    Arguments:
        hd: HERAData object intialized with (usually multiple) uvh5 files
        times: list of times in JD to load
        time_range: length-2 array-like of range of JDs to load
        lsts: list of lsts in radians to load
        lst_range: length-2 array-like of range of lsts in radians to load.
            If the 0th element is greater than the 1st, the range will wrap around 2pi
        kwargs: other partial i/o kwargs (see io.HERAData.read)

    Returns:
        data: DataContainer mapping baseline keys to complex visibility waterfalls
        flags: DataContainer mapping baseline keys to boolean flag waterfalls
        nsamples: DataContainer mapping baseline keys to interger Nsamples waterfalls
    '''
    assert hd.filetype == 'uvh5', 'This function only works for uvh5-based HERAData objects.'
    if np.sum([times is not None, time_range is not None, lsts is not None, lst_range is not None]) > 1:
        raise ValueError('Only one of times, time_range, lsts, and lsts_range can be not None.')

    combined_hd = None
    for f in hd.filepaths:
        hd_here = HERAData(f, upsample=hd.upsample, downsample=hd.downsample)

        # check if any of the selected times are in this particular file
        if times is not None:
            times_here = [time for time in times if time in hd_here.times]
            if len(times_here) == 0:
                continue  # skip this file
        else:
            times_here = None

        # check if any of the selected lsts are in this particular file
        if lsts is not None:
            lsts_here = [lst for lst in lsts if lst in hd_here.lsts]
            if len(lsts_here) == 0:
                continue  # skip this file
        else:
            lsts_here = None

        # attempt to read this file's data
        try:
            hd_here.read(times=times_here, time_range=time_range,
                         lsts=lsts_here, lst_range=lst_range,
                         return_data=False, **kwargs)
        except ValueError as err:
            # check to see if the read failed because of the time range or lst range
            if 'No elements in time range between ' in str(err):
                continue  # no matching times, skip this file
            elif 'No elements in LST range between ' in str(err):
                continue  # no matchings lsts, skip this file
            else:
                raise

        if combined_hd is None:
            combined_hd = hd_here
        else:
            combined_hd += hd_here
    if combined_hd is None:
        raise ValueError('No times or lsts matched any of the files in hd.')
    combined_hd = to_HERAData(combined_hd)  # re-runs the slicing and indexing
    return combined_hd.build_datacontainers()


def save_redcal_meta(meta_filename, fc_meta, omni_meta, freqs, times, lsts, antpos, history, clobber=True):
    '''Saves redcal metadata to a hdf5 file. See also read_redcal_meta.

    Arguments:
        meta_filename: path to hdf5 file to save
        fc_meta: firstcal metadata dictionary, such as that produced by redcal.redcal_iteration()
        omni_meta: omnical metadata dictionary, such as that produced by redcal.redcal_iteration()
        freqs: 1D numpy array of frequencies in the data
        times: 1D numpy array of times in the data
        lsts: 1D numpy array of LSTs in the data
        antpos: dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}
        history: string describing the creation of this file
        clobber: If False and meta_filename exists, raise OSError.
    '''
    if os.path.exists(meta_filename) and not clobber:
        raise OSError(f'{meta_filename} already exists but clobber=False.')

    with h5py.File(meta_filename, "w") as outfile:
        # save the metadata of the metadata
        header = outfile.create_group('header')
        header['freqs'] = freqs
        header['times'] = times
        header['lsts'] = lsts
        antnums = np.array(sorted(list(antpos.keys())))
        header['antpos'] = np.array([antpos[antnum] for antnum in antnums])
        header['antpos'].attrs['antnums'] = antnums
        header['history'] = np.string_(history)

        # save firstcal metadata, saving dictionary keys as attrs
        fc_grp = outfile.create_group('fc_meta')
        ant_keys = sorted(list(fc_meta['dlys'].keys()))
        fc_grp['dlys'] = np.array([fc_meta['dlys'][ant] for ant in ant_keys])
        fc_grp['dlys'].attrs['ants'] = np.string_(ant_keys)
        fc_grp['polarity_flips'] = np.array([fc_meta['polarity_flips'][ant] for ant in ant_keys])
        fc_grp['polarity_flips'].attrs['ants'] = np.string_(ant_keys)

        # save the omnical metadata, saving dictionary keys as attrs
        omni_grp = outfile.create_group('omni_meta')
        pols_keys = sorted(list(omni_meta['chisq'].keys()))
        omni_grp['chisq'] = np.array([omni_meta['chisq'][pols] for pols in pols_keys])
        omni_grp['chisq'].attrs['pols'] = pols_keys
        omni_grp['iter'] = np.array([omni_meta['iter'][pols] for pols in pols_keys])
        omni_grp['iter'].attrs['pols'] = pols_keys
        omni_grp['conv_crit'] = np.array([omni_meta['conv_crit'][pols] for pols in pols_keys])
        omni_grp['conv_crit'].attrs['conv_crit'] = np.string_(pols_keys)


def read_redcal_meta(meta_filename):
    '''Reads redcal metadata to a hdf5 file. See also save_redcal_meta.

    Arguments:
        meta_filename: path to hdf5 file to load

    Returns:
        fc_meta: firstcal metadata dictionary, such as that produced by redcal.redcal_iteration()
        omni_meta: omnical metadata dictionary, such as that produced by redcal.redcal_iteration()
        freqs: 1D numpy array of frequencies in the data
        times: 1D numpy array of times in the data
        lsts: 1D numpy array of LSTs in the data
        antpos: dictionary of antenna positions in the form {ant_index: np.array([x,y,z])}
        history: string describing the creation of this file
    '''
    with h5py.File(meta_filename, "r") as infile:
        # decode metadata of metadata
        freqs = infile['header']['freqs'][:]
        times = infile['header']['times'][:]
        lsts = infile['header']['lsts'][:]
        antpos = {ant: pos for ant, pos in zip(infile['header']['antpos'].attrs['antnums'],
                                               infile['header']['antpos'][:, :])}
        history = infile['header']['history'][()].tobytes().decode('utf8')

        # reconstruct firstcal metadata
        fc_meta = {}
        ants = [(int(num.tobytes().decode('utf8')), pol.tobytes().decode('utf8'))
                for num, pol in infile['fc_meta']['dlys'].attrs['ants']]
        fc_meta['dlys'] = {ant: dly for ant, dly in zip(ants, infile['fc_meta']['dlys'][:, :])}
        fc_meta['polarity_flips'] = {ant: flips for ant, flips in zip(ants, infile['fc_meta']['polarity_flips'][:, :])}

        # reconstruct omnical metadata
        omni_meta = {}
        pols_keys = infile['omni_meta']['chisq'].attrs['pols']
        omni_meta['chisq'] = {pols: chisq for pols, chisq in zip(pols_keys, infile['omni_meta']['chisq'][:, :])}
        omni_meta['iter'] = {pols: itr for pols, itr in zip(pols_keys, infile['omni_meta']['iter'][:, :])}
        omni_meta['conv_crit'] = {pols: cc for pols, cc in zip(pols_keys, infile['omni_meta']['conv_crit'][:, :])}

    return fc_meta, omni_meta, freqs, times, lsts, antpos, history


#######################################################################
#                             LEGACY CODE
#######################################################################


def to_HERAData(input_data, filetype='miriad', **read_kwargs):
    '''Converts a string path, UVData, or HERAData object, or a list of any one of those, to a
    single HERAData object without loading any new data.

    Arguments:
        input_data: data file path, or UVData/HERAData instance, or list of either strings of data
            file paths or list of UVData/HERAData instances to combine into a single HERAData object
        filetype: 'miriad', 'uvfits', or 'uvh5'. Ignored if input_data is UVData/HERAData objects
        read_kwargs : kwargs to pass to UVData.read (e.g. run_check, check_extra and
            run_check_acceptability). Only used for uvh5 filetype

    Returns:
        hd: HERAData object. Will not have data loaded if initialized from string(s).
    '''
    if filetype not in ['miriad', 'uvfits', 'uvh5']:
        raise NotImplementedError("Data filetype must be 'miriad', 'uvfits', or 'uvh5'.")
    if isinstance(input_data, str):  # single visibility data path
        return HERAData(input_data, filetype=filetype, **read_kwargs)
    elif isinstance(input_data, HERAData):  # already a HERAData object
        return input_data
    elif isinstance(input_data, UVData):  # single UVData object
        hd = input_data
        hd.__class__ = HERAData
        hd._determine_blt_slicing()
        hd._determine_pol_indexing()
        if filetype == 'uvh5':
            hd._attach_metadata()
        hd.filepaths = None
        return hd
    elif isinstance(input_data, Iterable):  # List loading
        if np.all([isinstance(i, str) for i in input_data]):  # List of visibility data paths
            return HERAData(input_data, filetype=filetype, **read_kwargs)
        elif np.all([isinstance(i, (UVData, HERAData)) for i in input_data]):  # List of uvdata objects
            hd = reduce(operator.add, input_data)
            hd.__class__ = HERAData
            hd._determine_blt_slicing()
            hd._determine_pol_indexing()
            return hd
        else:
            raise TypeError('If input is a list, it must be only strings or only UVData/HERAData objects.')
    else:
        raise TypeError('Input must be a UVData/HERAData object, a string, or a list of either.')


def load_vis(input_data, return_meta=False, filetype='miriad', pop_autos=False, pick_data_ants=True, nested_dict=False, **read_kwargs):
    '''Load miriad or uvfits files or UVData/HERAData objects into DataContainers, optionally returning
    the most useful metadata. More than one spectral window is not supported. Assumes every baseline
    has the same times present and that the times are in order.
    Arguments:
        input_data: data file path, or UVData/HERAData instance, or list of either strings of data
            file paths or list of UVData/HERAData instances to concatenate into a single dictionary
        return_meta:  boolean, if True: also return antpos, ants, freqs, times, lsts, and pols
        filetype: 'miriad', 'uvfits', or 'uvh5'. Ignored if input_data is UVData/HERAData objects
        pop_autos: boolean, if True: remove autocorrelations
        pick_data_ants: boolean, if True and return_meta=True, return only antennas in data
        nested_dict: boolean, if True replace DataContainers with the legacy nested dictionary filetype
            where visibilities and flags are accessed as data[(0,1)]['nn']
        read_kwargs : keyword arguments to pass to HERAData.read()
    Returns:
        if return_meta is True:
            (data, flags, antpos, ants, freqs, times, lsts, pols)
        else:
            (data, flags)
        data: DataContainer containing baseline-pol complex visibility data with keys
            like (0,1,'nn') and with shape=(Ntimes,Nfreqs)
        flags: DataContainer containing data flags
        antpos: dictionary containing antennas numbers as keys and position vectors
        ants: ndarray containing unique antenna indices
        freqs: ndarray containing frequency channels (Hz)
        times: ndarray containing julian date bins of data
        lsts: ndarray containing LST bins of data (radians)
        pol: ndarray containing list of polarization strings
    '''
    hd = to_HERAData(input_data, filetype=filetype)
    if hd.data_array is not None:
        d, f, n = hd.build_datacontainers()
    else:
        d, f, n = hd.read(**read_kwargs)

    # remove autos if requested
    if pop_autos:
        for k in list(d.keys()):
            if k[0] == k[1]:
                del d[k], f[k], n[k]

    # convert into nested dict if necessary
    if nested_dict:
        data, flags = odict(), odict()
        antpairs = [key[0:2] for key in d.keys()]
        for ap in antpairs:
            data[ap] = d[ap]
            flags[ap] = f[ap]
    else:
        data, flags = d, f

    # get meta
    if return_meta:
        antpos, ants = hd.get_ENU_antpos(center=True, pick_data_ants=pick_data_ants)
        antpos = odict(zip(ants, antpos))
        return data, flags, antpos, ants, d.freqs, d.times, d.lsts, d.pols()
    else:
        return data, flags


def write_vis(fname, data, lst_array, freq_array, antpos, time_array=None, flags=None, nsamples=None,
              filetype='miriad', write_file=True, outdir="./", overwrite=False, verbose=True, history=" ",
              return_uvd=False, start_jd=None, lst_branch_cut=0.0, x_orientation="north", instrument="HERA",
              telescope_name="HERA", object_name='EOR', vis_units='uncalib', dec=-30.72152,
              telescope_location=HERA_TELESCOPE_LOCATION, integration_time=None, **kwargs):
    """
    Take DataContainer dictionary, export to UVData object and write to file. See pyuvdata.UVdata
    documentation for more info on these attributes.

    Parameters:
    -----------
    fname : type=str, output filename of visibliity data

    data : type=DataContainer, holds complex visibility data.

    lst_array : type=float ndarray, contains unique LST time bins [radians] of data (center of integration).

    freq_array : type=ndarray, contains frequency bins of data [Hz].

    antpos : type=dictionary, antenna position dictionary. keys are antenna integers and values
             are position vectors in meters in ENU (TOPO) frame.

    time_array : type=ndarray, contains unique Julian Date time bins of data (center of integration).

    flags : type=DataContainer, holds data flags, matching data in shape.

    nsamples : type=DataContainer, holds number of points averaged into each bin in data (if applicable).

    filetype : type=str, filetype to write-out, options=['miriad'].

    write_file : type=boolean, write UVData to file if True.

    outdir : type=str, output directory for output file.

    overwrite : type=boolean, if True, overwrite output files.

    verbose : type=boolean, if True, report feedback to stdout.

    history : type=str, history string for UVData object

    return_uvd : type=boolean, if True return UVData instance.

    start_jd : type=float, starting integer Julian Date of time_array if time_array is None.

    lst_branch_cut : type=float, LST of data start, ensures that LSTs lower than this are wrapped around
                     and correspond to higher JDs in time_array, but only if time_array is None [radians]

    x_orientation : type=str, orientation of X dipole, options=['east', 'north']

    instrument : type=str, instrument name.

    telescope_name : type=str, telescope name.

    object_name : type=str, observing object name.

    vis_unit : type=str, visibility units.

    dec : type=float, declination of observer in degrees North.

    telescope_location : type=ndarray, telescope location in xyz in ITRF (earth-centered frame).

    integration_time : type=float or ndarray, integration duration in seconds for data_array.
        This does not necessarily have to be equal to the diff(time_array): for the case of
        LST-binning, this is not the duration of the LST-bin but the integration time of the
        pre-binned data. Default is median(diff(time_array)) in seconds. Note: the _total_
        integration time in a visibility is integration_time * nsamples.

    kwargs : type=dictionary, additional parameters to set in UVData object.

    Output:
    -------
    if return_uvd: return UVData instance
    """
    # configure UVData parameters
    # get pols
    pols = np.unique([k[-1] for k in data.keys()])
    Npols = len(pols)
    polarization_array = np.array([polstr2num(p, x_orientation=x_orientation) for p in pols])

    # get telescope ants
    antenna_numbers = np.unique(list(antpos.keys()))
    Nants_telescope = len(antenna_numbers)
    antenna_names = [f"HH{a}" for a in antenna_numbers]

    # get antenna positions in ITRF frame
    tel_lat_lon_alt = uvutils.LatLonAlt_from_XYZ(telescope_location)
    antenna_positions = np.array([antpos[k] for k in antenna_numbers])
    antenna_positions = uvutils.ECEF_from_ENU(antenna_positions, *tel_lat_lon_alt) - telescope_location

    # get times
    if time_array is None:
        if start_jd is None:
            raise AttributeError("if time_array is not fed, start_jd must be fed")
        time_array = LST2JD(lst_array, start_jd, allow_other_jd=True, lst_branch_cut=lst_branch_cut,
                            latitude=(tel_lat_lon_alt[0] * 180 / np.pi),
                            longitude=(tel_lat_lon_alt[1] * 180 / np.pi),
                            altitude=tel_lat_lon_alt[2])
    Ntimes = len(time_array)

    # get freqs
    Nfreqs = len(freq_array)
    channel_width = np.median(np.diff(freq_array))
    freq_array = freq_array.reshape(1, -1)
    spw_array = np.array([0])
    Nspws = 1

    # get baselines keys
    antpairs = sorted(data.antpairs())
    Nbls = len(antpairs)
    Nblts = Nbls * Ntimes

    # reconfigure time_array and lst_array
    time_array = np.repeat(time_array[np.newaxis], Nbls, axis=0).ravel()
    lst_array = np.repeat(lst_array[np.newaxis], Nbls, axis=0).ravel()

    # configure integration time, converting from days (the unit of time_array)
    # to seconds (the unit of integration_time)
    if integration_time is None:
        integration_time = np.ones_like(time_array, dtype=np.float64) * np.median(np.diff(np.unique(time_array))) * 24 * 3600.

    # get data array
    data_array = np.moveaxis(list(map(lambda p: list(map(lambda ap: data[str(p)][ap], antpairs)), pols)), 0, -1)

    # resort time and baseline axes
    data_array = data_array.reshape(Nblts, 1, Nfreqs, Npols)
    if nsamples is None:
        nsample_array = np.ones_like(data_array, float)
    else:
        nsample_array = np.moveaxis(list(map(lambda p: list(map(lambda ap: nsamples[str(p)][ap], antpairs)), pols)), 0, -1)
        nsample_array = nsample_array.reshape(Nblts, 1, Nfreqs, Npols)

    # flags
    if flags is None:
        flag_array = np.zeros_like(data_array, float).astype(bool)
    else:
        flag_array = np.moveaxis(list(map(lambda p: list(map(lambda ap: flags[str(p)][ap].astype(bool), antpairs)), pols)), 0, -1)
        flag_array = flag_array.reshape(Nblts, 1, Nfreqs, Npols)

    # configure baselines
    antpairs = np.repeat(np.array(antpairs), Ntimes, axis=0)

    # get ant_1_array, ant_2_array
    ant_1_array = antpairs[:, 0]
    ant_2_array = antpairs[:, 1]

    # get baseline array
    baseline_array = 2048 * (ant_1_array + 1) + (ant_2_array + 1) + 2**16

    # get antennas in data
    data_ants = np.unique(np.concatenate([ant_1_array, ant_2_array]))
    Nants_data = len(data_ants)

    # set uvw assuming drift phase i.e. phase center is zenith
    uvw_array = np.array([antpos[k[1]] - antpos[k[0]] for k in zip(ant_1_array, ant_2_array)])

    # get zenith location: can only write drift phase
    phase_type = 'drift'

    # instantiate object
    uvd = UVData()

    # assign parameters
    params = ['Nants_data', 'Nants_telescope', 'Nbls', 'Nblts', 'Nfreqs', 'Npols', 'Nspws', 'Ntimes',
              'ant_1_array', 'ant_2_array', 'antenna_names', 'antenna_numbers', 'baseline_array',
              'channel_width', 'data_array', 'flag_array', 'freq_array', 'history', 'x_orientation',
              'instrument', 'integration_time', 'lst_array', 'nsample_array', 'object_name', 'phase_type',
              'polarization_array', 'spw_array', 'telescope_location', 'telescope_name', 'time_array',
              'uvw_array', 'vis_units', 'antenna_positions']
    local_params = locals()

    # overwrite paramters by kwargs
    local_params.update(kwargs)

    # set parameters in uvd
    for p in params:
        uvd.__setattr__(p, local_params[p])

    # write to file
    if write_file:
        # check output
        fname = os.path.join(outdir, fname)
        if os.path.exists(fname) and overwrite is False:
            if verbose:
                print("{} exists, not overwriting".format(fname))
        else:
            if verbose:
                print("saving {}".format(fname))

        if filetype == 'miriad':
            uvd.write_miriad(fname, clobber=True)
        elif filetype == 'uvh5':
            uvd.write_uvh5(fname, clobber=True)
        else:
            raise AttributeError("didn't recognize filetype: {}".format(filetype))

    if return_uvd:
        return uvd


def update_uvdata(uvd, data=None, flags=None, nsamples=None, add_to_history='', **kwargs):
    '''Updates a UVData/HERAData object with data or parameters. Cannot modify the shape of
    data arrays. More than one spectral window is not supported. Assumes every baseline
    has the same times present and that the times are in order.

    Arguments:
        uv: UVData/HERAData object to be updated
        data: dictionary or DataContainer of complex visibility data to update. Keys
            like (0,1,'nn') and shape=(Ntimes,Nfreqs). Default (None) does not update.
        flags: dictionary or DataContainer of data flags to update.
            Default (None) does not update.
        nsamples: dictionary or DataContainer of nsamples to update.
            Default (None) does not update.
        add_to_history: appends a string to the history of the UVData/HERAData object
        kwargs: dictionary mapping updated attributs to their new values.
            See pyuvdata.UVData documentation for more info.
    '''

    # perform update
    original_class = uvd.__class__
    uvd = to_HERAData(uvd)
    uvd.update(data=data, flags=flags, nsamples=nsamples)
    uvd.__class__ = original_class

    # set additional attributes
    uvd.history += add_to_history
    for attribute, value in kwargs.items():
        uvd.__setattr__(attribute, value)
    uvd.check()


def _write_HERAData_to_filetype(hd, outfilename, filetype_out='miriad', clobber=False):
    '''Helper function for update_vis().'''
    if filetype_out == 'miriad':
        hd.write_miriad(outfilename, clobber=clobber)
    elif filetype_out == 'uvfits':
        hd.write_uvfits(outfilename, force_phase=True, spoof_nonessential=True)
    elif filetype_out == 'uvh5':
        hd.write_uvh5(outfilename, clobber=clobber)
    else:
        raise TypeError("Input filetype must be either 'miriad', 'uvfits', or 'uvh5'.")


def update_vis(infilename, outfilename, filetype_in='miriad', filetype_out='miriad',
               data=None, flags=None, nsamples=None, add_to_history='', clobber=False, **kwargs):
    '''Loads an existing file with pyuvdata, modifies some subset of of its parameters, and
    then writes a new file to disk. Cannot modify the shape of data arrays. More than one
    spectral window is not supported. Assumes every baseline has the same times present
    and that the times are in order.

    Arguments:
        infilename: filename of the base visibility file to be updated, or UVData/HERAData object
        outfilename: filename of the new visibility file
        filetype_in: either 'miriad' or 'uvfits' (ignored if infile is a UVData/HERAData object)
        filetype_out: either 'miriad' or 'uvfits'
        data: dictionary or DataContainer of complex visibility data to update. Keys
            like (0,1,'nn') and shape=(Ntimes,Nfreqs). Default (None) does not update.
        flags: dictionary or DataContainer of data flags to update.
            Default (None) does not update.
        nsamples: dictionary or DataContainer of nsamples to update.
            Default (None) does not update.
        add_to_history: appends a string to the history of the output file
        clobber: if True, overwrites existing file at outfilename. Always True for uvfits.
        kwargs: dictionary mapping updated attributs to their new values.
            See pyuvdata.UVData documentation for more info.
    '''

    # Load infile
    if isinstance(infilename, (UVData, HERAData)):
        hd = copy.deepcopy(infilename)
    else:
        hd = HERAData(infilename, filetype=filetype_in)
        hd.read()
    update_uvdata(hd, data=data, flags=flags, nsamples=nsamples, add_to_history=add_to_history, **kwargs)

    # write out results
    _write_HERAData_to_filetype(hd, outfilename, filetype_out=filetype_out, clobber=clobber)


def to_HERACal(input_cal):
    '''Converts a string path, UVCal, or HERACal object, or a list of any one of those, to a
    single HERACal object without loading any new calibration solutions.

    Arguments:
        input_cal: path to calfits file, UVCal/HERACal object, or a list of either to combine
            into a single HERACal object

    Returns:
        hc: HERACal object. Will not have calibration loaded if initialized from string(s).
    '''
    if isinstance(input_cal, str):  # single calfits path
        return HERACal(input_cal)
    if isinstance(input_cal, HERACal):  # single HERACal
        return input_cal
    elif isinstance(input_cal, UVCal):  # single UVCal object
        input_cal.__class__ = HERACal
        input_cal.filepaths = None
        input_cal._extract_metadata()  # initialize metadata vars.
        return input_cal
    elif isinstance(input_cal, Iterable):  # List loading
        if np.all([isinstance(ic, str) for ic in input_cal]):  # List of calfits paths
            return HERACal(input_cal)
        elif np.all([isinstance(ic, (UVCal, HERACal)) for ic in input_cal]):  # List of UVCal/HERACal objects
            hc = reduce(operator.add, input_cal)
            hc.__class__ = HERACal
            return hc
        else:
            raise TypeError('If input is a list, it must be only strings or only UVCal/HERACal objects.')
    else:
        raise TypeError('Input must be a UVCal/HERACal object, a string, or a list of either.')


def load_cal(input_cal, return_meta=False):
    '''Load calfits files or UVCal/HERACal objects into dictionaries, optionally
    returning the most useful metadata. More than one spectral window is not supported.

    Arguments:
        input_cal: path to calfits file, UVCal/HERACal object, or a list of either
        return_meta: if True, returns additional information (see below)

    Returns:
        if return_meta is True:
            (gains, flags, quals, total_qual, ants, freqs, times, pols)
        else:
            (gains, flags)

        gains: Dictionary of complex calibration gains as a function of time
            and frequency with keys in the (1,'x') format
        flags: Dictionary of flags in the same format as the gains
        quals: Dictionary of of qualities of calibration solutions in the same
            format as the gains (e.g. omnical chi^2 per antenna)
        total_qual: ndarray of total calibration quality for the whole array
            (e.g. omnical overall chi^2)
        ants: ndarray containing unique antenna indices
        freqs: ndarray containing frequency channels (Hz)
        times: ndarray containing julian date bins of data
        pols: list of antenna polarization strings
    '''
    # load HERACal object and extract gains, data, etc.
    hc = to_HERACal(input_cal)
    if hc.gain_array is not None:
        gains, flags, quals, total_qual = hc.build_calcontainers()
    else:
        gains, flags, quals, total_qual = hc.read()

    # return quantities
    if return_meta:
        return gains, flags, quals, total_qual, np.array([ant[0] for ant in hc.ants]), hc.freqs, hc.times, hc.pols
    else:
        return gains, flags


def write_cal(fname, gains, freqs, times, flags=None, quality=None, total_qual=None, antnums2antnames=None,
              write_file=True, return_uvc=True, outdir='./', overwrite=False, gain_convention='divide',
              history=' ', x_orientation="north", telescope_name='HERA', cal_style='redundant',
              zero_check=True, **kwargs):
    '''Format gain solution dictionary into pyuvdata.UVCal and write to file

    Arguments:
        fname : type=str, output file basename
        gains : type=dictionary, holds complex gain solutions. keys are antenna + pol
            tuple pairs, e.g. (2, 'x'), and keys are 2D complex ndarrays with time
            along [0] axis and freq along [1] axis.
        freqs : type=ndarray, holds unique frequencies channels in Hz
        times : type=ndarray, holds unique times of integration centers in Julian Date
        flags : type=dictionary, holds boolean flags (True if flagged) for gains.
            Must match shape of gains.
        quality : type=dictionary, holds "quality" of calibration solution. Must match
            shape of gains. See pyuvdata.UVCal doc for more details.
        total_qual : type=dictionary, holds total_quality_array. Key(s) are polarization
            string(s) and values are 2D (Ntimes, Nfreqs) ndarrays.
        antnums2antnames : dict, keys antenna numbers (int), values antenna names (str)
            Default is "ant{}".format(ant_num) for antenna names.
        write_file : type=bool, if True, write UVCal to calfits file
        return_uvc : type=bool, if True, return UVCal object
        outdir : type=str, output file directory
        overwrite : type=bool, if True overwrite output files
        gain_convention : type=str, gain solutions formatted such that they 'multiply' into data
            to get model, or 'divide' into data to get model
            options=['multiply', 'divide']
        history : type=str, history string for UVCal object.
        x_orientation : type=str, orientation of X dipole, options=['east', 'north']
        telescope_name : type=str, name of telescope
        cal_style : type=str, style of calibration solutions, options=['redundant', 'sky']. If
            cal_style == sky, additional params are required. See pyuvdata.UVCal doc.
        zero_check : type=bool, if True, for gain values near zero, set to one and flag them.
        kwargs : additional atrributes to set in pyuvdata.UVCal
    Returns:
        if return_uvc: returns UVCal object
        else: returns None
    '''
    # get antenna info
    ant_array = np.unique([k[0] for k in gains]).astype(int)
    antenna_numbers = copy.copy(ant_array)
    if antnums2antnames is None:
        antenna_names = np.array(["ant{}".format(ant_num) for ant_num in antenna_numbers])
    else:
        antenna_names = np.array([antnums2antnames[ant_num] for ant_num in antenna_numbers])
    Nants_data = len(ant_array)
    Nants_telescope = len(antenna_numbers)

    # get polarization info: ordering must be monotonic in Jones number
    jones_array = np.array(list(set([jstr2num(k[1], x_orientation=x_orientation) for k in gains.keys()])))
    jones_array = jones_array[np.argsort(np.abs(jones_array))]
    pol_array = np.array([jnum2str(j, x_orientation=x_orientation) for j in jones_array])
    Njones = len(jones_array)

    # get time info
    time_array = np.array(times, float)
    Ntimes = len(time_array)
    time_range = np.array([time_array.min(), time_array.max()], float)
    if len(time_array) > 1:
        integration_time = np.median(np.diff(time_array)) * 24. * 3600.
    else:
        integration_time = 0.0

    # get frequency info
    freq_array = np.array(freqs, float)
    Nfreqs = len(freq_array)
    Nspws = 1
    freq_array = freq_array[None, :]
    spw_array = np.arange(Nspws)
    channel_width = np.median(np.diff(freq_array))

    # form gain, flags and qualities
    gain_array = np.empty((Nants_data, Nspws, Nfreqs, Ntimes, Njones), complex)
    flag_array = np.empty((Nants_data, Nspws, Nfreqs, Ntimes, Njones), bool)
    quality_array = np.empty((Nants_data, Nspws, Nfreqs, Ntimes, Njones), float)
    total_quality_array = np.empty((Nspws, Nfreqs, Ntimes, Njones), float)
    for i, p in enumerate(pol_array):
        if total_qual is not None:
            total_quality_array[0, :, :, i] = total_qual[p].T[None, :, :]
        for j, a in enumerate(ant_array):
            # ensure (a, p) is in gains
            if (a, p) in gains:
                gain_array[j, :, :, :, i] = gains[(a, p)].T[None, :, :]
                if flags is not None:
                    flag_array[j, :, :, :, i] = flags[(a, p)].T[None, :, :]
                else:
                    flag_array[j, :, :, :, i] = np.zeros((Nspws, Nfreqs, Ntimes), bool)
                if quality is not None:
                    quality_array[j, :, :, :, i] = quality[(a, p)].T[None, :, :]
                else:
                    quality_array[j, :, :, :, i] = np.ones((Nspws, Nfreqs, Ntimes), float)
            else:
                gain_array[j, :, :, :, i] = np.ones((Nspws, Nfreqs, Ntimes), complex)
                flag_array[j, :, :, :, i] = np.ones((Nspws, Nfreqs, Ntimes), bool)
                quality_array[j, :, :, :, i] = np.ones((Nspws, Nfreqs, Ntimes), float)

    if total_qual is None:
        total_quality_array = None

    if zero_check:
        # Check gain_array for values close to zero, if so, set to 1
        zero_check_arr = np.isclose(gain_array, 0, rtol=1e-10, atol=1e-10)
        # copy arrays b/c they are still references to the input gain dictionaries
        gain_array = gain_array.copy()
        flag_array = flag_array.copy()
        gain_array[zero_check_arr] = 1.0 + 0j
        flag_array[zero_check_arr] += True
        if zero_check_arr.max() is True:
            warnings.warn("Some of values in self.gain_array were zero and are flagged and set to 1.")

    # instantiate UVCal
    uvc = UVCal()

    # enforce 'gain' cal_type
    uvc.cal_type = "gain"

    # optional calfits parameters to get overwritten via kwargs
    telescope_location = None
    antenna_positions = None
    lst_array = None

    # create parameter list
    params = ["Nants_data", "Nants_telescope", "Nfreqs", "Ntimes", "Nspws", "Njones",
              "ant_array", "antenna_numbers", "antenna_names", "cal_style", "history",
              "channel_width", "flag_array", "gain_array", "quality_array", "jones_array",
              "time_array", "spw_array", "freq_array", "history", "integration_time",
              "time_range", "x_orientation", "telescope_name", "gain_convention", "total_quality_array",
              "telescope_location", "antenna_positions", "lst_array"]

    # create local parameter dict
    local_params = locals()

    # overwrite with kwarg parameters
    local_params.update(kwargs)

    # set parameters
    for p in params:
        uvc.__setattr__(p, local_params[p])

    # run check
    uvc.check()

    # write to file
    if write_file:
        # check output
        fname = os.path.join(outdir, fname)
        if os.path.exists(fname) and overwrite is False:
            print("{} exists, not overwriting...".format(fname))
        else:
            uvc.write_calfits(fname, clobber=True)

    # return object
    if return_uvc:
        return uvc


def update_uvcal(cal, gains=None, flags=None, quals=None, add_to_history='', **kwargs):
    '''LEGACY CODE TO BE DEPRECATED!
    Update UVCal object with gains, flags, quals, history, and/or other parameters
    Cannot modify the shape of gain arrays. More than one spectral window is not supported.

    Arguments:
        cal: UVCal/HERACal object to be updated
        gains: Dictionary of complex calibration gains with shape=(Ntimes,Nfreqs)
            with keys in the (1,'x') format. Default (None) leaves unchanged.
        flags: Dictionary like gains but of flags. Default (None) leaves unchanged.
        quals: Dictionary like gains but of per-antenna quality. Default (None) leaves unchanged.
        add_to_history: appends a string to the history of the output file
        overwrite: if True, overwrites existing file at outfilename
        kwargs: dictionary mapping updated attributs to their new values.
            See pyuvdata.UVCal documentation for more info.
    '''
    original_class = cal.__class__
    cal.__class__ = HERACal
    cal._extract_metadata()
    cal.update(gains=gains, flags=flags, quals=quals)

    # Check gain_array for values close to zero, if so, set to 1
    zero_check = np.isclose(cal.gain_array, 0, rtol=1e-10, atol=1e-10)
    cal.gain_array[zero_check] = 1.0 + 0j
    cal.flag_array[zero_check] += True
    if zero_check.max() is True:
        warnings.warn("Some of values in self.gain_array were zero and are flagged and set to 1.")

    # Set additional attributes
    cal.history += add_to_history
    for attribute, value in kwargs.items():
        cal.__setattr__(attribute, value)
    cal.check()
    cal.__class__ = original_class


def update_cal(infilename, outfilename, gains=None, flags=None, quals=None, add_to_history='', clobber=False, **kwargs):
    '''Loads an existing calfits file with pyuvdata, modifies some subset of of its parameters,
    and then writes a new calfits file to disk. Cannot modify the shape of gain arrays.
    More than one spectral window is not supported.

    Arguments:
        infilename: filename of the base calfits file to be updated, or UVCal object
        outfilename: filename of the new calfits file
        gains: Dictionary of complex calibration gains with shape=(Ntimes,Nfreqs)
            with keys in the (1,'x') format. Default (None) leaves unchanged.
        flags: Dictionary like gains but of flags. Default (None) leaves unchanged.
        quals: Dictionary like gains but of per-antenna quality. Default (None) leaves unchanged.
        add_to_history: appends a string to the history of the output file
        clobber: if True, overwrites existing file at outfilename
        kwargs: dictionary mapping updated attributs to their new values.
            See pyuvdata.UVCal documentation for more info.
    '''
    # Load infile
    if isinstance(infilename, (UVCal, HERACal)):
        cal = copy.deepcopy(infilename)
    else:
        cal = HERACal(infilename)
        cal.read()

    update_uvcal(cal, gains=gains, flags=flags, quals=quals, add_to_history=add_to_history, **kwargs)

    # Write to calfits file
    cal.write_calfits(outfilename, clobber=clobber)


def baselines_from_filelist_position(filename, filelist):
    """Determine indices of baselines to process.


    This function determines antpairs to process given the position of a filename
    in a list of files.


    Parameters
    ----------
    filename : string
        name of the file being processed.
    filelist : list of strings
        name of all files over which computations are being parallelized.
    Returns
    -------
    list
        list of antpairs to process based on the position of the filename in the list of files.
    """
    # The reason this function is not in utils is that it needs to use HERAData
    hd = HERAData(filename)
    bls = list(set([bl[:2] for bl in hd.bls]))
    file_index = filelist.index(filename)
    nfiles = len(filelist)
    # Determine chunk size
    nbls = len(bls)
    chunk_size = nbls // nfiles + 1
    lower_index = file_index * chunk_size
    upper_index = np.min([(file_index + 1) * chunk_size, nbls])
    output = bls[lower_index:upper_index]
    return output


def throw_away_flagged_ants(infilename, outfilename, yaml_file=None, throw_away_fully_flagged_data_baselines=False, clobber=False):
    """Throw away completely flagged data.

    Parameters
    ----------
        infilename: str
            path to a UVData file in uvh5 format.
        outfilename: str
            path to file to output trimmed data file.
        yaml_file: str
            path to a yaml flagging file with a list of antennas to flag.
            Default is None.
        throw_away_flagged_ants: bool, optional
            if True, also throw away baselines where all data is flagged.
            Warning: Don't use this for files with a small number of time integrations.
                     since this can easily happen by chance in files like this.
            Default is False
        clobber: bool, optional
            overwrite output file if it already exists.
            Default is False.
    Returns
    -------
        hd: HERAData object
            HERAData object containing data from infilename with baselines thrown out.

    """
    hd = HERAData(infilename)
    hd.read()

    # throw away flagged antennas in yaml file.
    if yaml_file is not None:
        from hera_qm import utils as qm_utils
        qm_utils.apply_yaml_flags(uv=hd, a_priori_flag_yaml=yaml_file,
                                  ant_indices_only=True, flag_ants=True, flag_freqs=False,
                                  flag_times=False, throw_away_flagged_ants=True)

    # Write data
    if throw_away_fully_flagged_data_baselines:
        antpairs_to_keep = []
        antpairs_not_to_keep = []
        for antpair in hd.get_antpairs():
            fully_flagged = True
            for pol in hd.pols:
                fully_flagged = fully_flagged & np.all(hd.get_flags(antpair + (pol, )))
            if not fully_flagged:
                antpairs_to_keep.append(antpair)
            else:
                antpairs_not_to_keep.append(antpair)
        hd.select(bls=antpairs_to_keep)
    else:
        antpairs_not_to_keep = None
    # wite to history.
    history_string = f"Threw away flagged antennas from yaml_file={yaml_file} using throw_away_flagged_ants.\n"
    history_string += f"Also threw out {antpairs_not_to_keep} because data was fully flagged.\n"
    hd.history += utils.history_string(notes=history_string)
    hd.write_uvh5(outfilename, clobber=clobber)
    return hd


def throw_away_flagged_ants_parser():
    # Parse arguments
    ap = argparse.ArgumentParser(description="Throw away baselines whose antennas are flagged in a yaml file or which have all integrations/chans flagged.")
    ap.add_argument("infilename", type=str, help="path to visibility data throw out flagged baselines..")
    ap.add_argument("outfilename", type=str, help="path to new visibility file to write data with thrown out baselines..")
    ap.add_argument("--yaml_file", default=None, type=str, help='yaml file with list of antennas to throw away.')
    ap.add_argument("--throw_away_fully_flagged_data_baselines", default=False, action="store_true",
                    help="Also throw away baselines that have all channels and integrations flagged.")
    ap.add_argument("--clobber", default=False, action="store_true", help='overwrites existing file at outfile')
    return ap
