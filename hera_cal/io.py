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
import collections
from pyuvdata import UVCal, UVData
from pyuvdata import utils as uvutils
from astropy import units
import h5py

try:
    import aipy
    AIPY = True
except ImportError:
    AIPY = False

from .datacontainer import DataContainer
from .utils import polnum2str, polstr2num, jnum2str, jstr2num
from .utils import split_pol, conj_pol, LST2JD


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
        if isinstance(input_cal, str):
            assert os.path.exists(input_cal), '{} does not exist.'.format(input_cal)
            self.filepaths = [input_cal]
        elif isinstance(input_cal, collections.Iterable):  # List loading
            if np.all([isinstance(i, str) for i in input_cal]):  # List of visibility data paths
                for ic in input_cal:
                    assert os.path.exists(ic), '{} does not exist.'.format(ic)
                self.filepaths = list(input_cal)
            else:
                raise TypeError('If input_cal is a list, it must be a list of strings.')
        else:
            raise ValueError('input_cal must be a string or a list of strings.')

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
            gains[(ant, pol)] = np.array(self.gain_array[i, 0, :, :, ip].T)
            flags[(ant, pol)] = np.array(self.flag_array[i, 0, :, :, ip].T)
            quals[(ant, pol)] = np.array(self.quality_array[i, 0, :, :, ip].T)

        # build dict of total_qual if available
        for pol in self.pols:
            ip = self._jnum_indices[jstr2num(pol, x_orientation=self.x_orientation)]
            if self.total_quality_array is not None:
                total_qual[pol] = np.array(self.total_quality_array[0, :, :, ip].T)
            else:
                total_qual = None

        return gains, flags, quals, total_qual

    def read(self):
        '''Reads calibration information from file, computes useful metadata and returns
        dictionaries that map antenna-pol tuples to calibration waterfalls.

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
            self.read_calfits(self.filepaths)

        return self.build_calcontainers()

    def update(self, gains=None, flags=None, quals=None, total_qual=None):
        '''Update internal calibrations arrays (data_array, flag_array, and nsample_array)
        using DataContainers (if not left as None) in preparation for writing to disk.

        Arguments:
            gains: optional dict mapping antenna-pol to complex gains arrays
            flags: optional dict mapping antenna-pol to boolean flag arrays
            quals: optional dict mapping antenna-pol to float qual arrays
            total_qual: optional dict mapping polarization to float total quality array
        '''
        # loop over and update gains, flags, and quals
        data_arrays = [self.gain_array, self.flag_array, self.quality_array]
        for to_update, array in zip([gains, flags, quals], data_arrays):
            if to_update is not None:
                for (ant, pol) in to_update.keys():
                    i, ip = self._antnum_indices[ant], self._jnum_indices[jstr2num(pol, x_orientation=self.x_orientation)]
                    array[i, 0, :, :, ip] = to_update[(ant, pol)].T

        # update total_qual
        if total_qual is not None:
            for pol in total_qual.keys():
                ip = self._jnum_indices[jstr2num(pol, x_orientation=self.x_orientation)]
                self.total_quality_array[0, :, :, ip] = total_qual[pol].T


def get_blt_slices(uvo):
    '''For a pyuvdata-style UV object, get the mapping from antenna pair to blt slice.

    Arguments:
        uvo: a "UV-Object" like UVData or baseline-type UVFlag

    Returns:
        blt_slices: dictionary mapping anntenna pair tuples to baseline-time slice objects
    '''
    blt_slices = {}
    for ant1, ant2 in uvo.get_antpairs():
        indices = uvo.antpair2ind(ant1, ant2)
        if len(indices) == 1:  # only one blt matches
            blt_slices[(ant1, ant2)] = slice(indices[0], indices[0] + 1, uvo.Nblts)
        elif not (len(set(np.ediff1d(indices))) == 1):  # checks if the consecutive differences are all the same
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
    HERAData_metas = ['ants', 'antpos', 'freqs', 'times', 'lsts', 'pols',
                      'antpairs', 'bls', 'times_by_bl', 'lsts_by_bl']
    # ants: list of antenna numbers
    # antpos: dictionary mapping antenna numbers to np.arrays of position in meters
    # freqs: np.arrray of frequencies (Hz)
    # times: np.array of unique times in the data file (JD)
    # lsts: np.array of unique LSTs in the data file (radians)
    # pols: list of baseline polarization strings
    # antpairs: list of antenna number pairs in the data as 2-tuples
    # bls: list of baseline-pols in the data as 3-tuples
    # times_by+bl: dictionary mapping antpairs to times (JD). Also includes all reverse pairs.
    # times_by+bl: dictionary mapping antpairs to LSTs (radians). Also includes all reverse pairs.

    def __init__(self, input_data, filetype='uvh5', **check_kwargs):
        '''Instantiate a HERAData object. If the filetype == uvh5, read in and store
        useful metadata (see get_metadata_dict()), either as object attributes or,
        if input_data is a list, as dictionaries mapping string paths to metadata.

        Arguments:
            input_data: string data file path or list of string data file paths
            filetype: supports 'uvh5' (defualt), 'miriad', 'uvfits'
            check_kwargs : run_check, check_extra and run_check_acceptability
                See UVData.read for more details.
        '''
        # initialize as empty UVData object
        super().__init__()

        # parse input_data as filepath(s)
        if isinstance(input_data, str):
            self.filepaths = [input_data]
        elif isinstance(input_data, collections.Iterable):  # List loading
            if np.all([isinstance(i, str) for i in input_data]):  # List of visibility data paths
                self.filepaths = list(input_data)
            else:
                raise TypeError('If input_data is a list, it must be a list of strings.')
        else:
            raise ValueError('input_data must be a string or a list of strings.')
        for f in self.filepaths:
            if not os.path.exists(f):
                raise IOError('Cannot find file ' + f)

        # load metadata from file
        self.filetype = filetype
        if self.filetype == 'uvh5':
            # read all UVData metadata from first file
            temp_paths = copy.deepcopy(self.filepaths)
            self.filepaths = self.filepaths[0]
            self.read(read_data=False, **check_kwargs)
            self.filepaths = temp_paths

            if len(self.filepaths) > 1:  # save HERAData_metas in dicts
                for meta in self.HERAData_metas:
                    setattr(self, meta, {})
                for f in self.filepaths:
                    hd = HERAData(f, filetype='uvh5', **check_kwargs)
                    meta_dict = hd.get_metadata_dict()
                    for meta in self.HERAData_metas:
                        getattr(self, meta)[f] = meta_dict[meta]
            else:  # save HERAData_metas as attributes
                self._writers = {}
                for key, value in self.get_metadata_dict().items():
                    setattr(self, key, value)

        elif self.filetype in ['miriad', 'uvfits']:
            for meta in self.HERAData_metas:
                setattr(self, meta, None)  # no pre-loading of metadata
        else:
            raise NotImplementedError('Filetype ' + self.filetype + ' has not been implemented.')

    def reset(self):
        '''Resets all standard UVData attributes, potentially freeing memory.'''
        super(HERAData, self).__init__()

    def get_metadata_dict(self):
        ''' Produces a dictionary of the most useful metadata. Used as object
        attributes and as metadata to store in DataContainers.

        Returns:
            metadata_dict: dictionary of all items in self.HERAData_metas
        '''
        antpos, ants = self.get_ENU_antpos()
        antpos = odict(zip(ants, antpos))

        freqs = np.unique(self.freq_array)
        times = np.unique(self.time_array)
        lst_indices = np.unique(self.lst_array.ravel(), return_index=True)[1]
        lsts = self.lst_array.ravel()[np.sort(lst_indices)]
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
        self._polnum_indices = {}
        for i, polnum in enumerate(self.polarization_array):
            self._polnum_indices[polnum] = i

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
                return np.array(data_array[self._blt_slices[tuple(key[0:2])], 0, :,
                                           self._polnum_indices[polstr2num(key[2], x_orientation=self.x_orientation)]])
            except KeyError:
                return np.conj(data_array[self._blt_slices[tuple(key[1::-1])], 0, :,
                                          self._polnum_indices[polstr2num(conj_pol(key[2]), x_orientation=self.x_orientation)]])
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
                data_array[self._blt_slices[tuple(key[0:2])], 0, :,
                           self._polnum_indices[polstr2num(key[2], x_orientation=self.x_orientation)]] = value
            except(KeyError):
                data_array[self._blt_slices[tuple(key[1::-1])], 0, :,
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
            for attr in ['antpos', 'freqs', 'times', 'lsts', 'times_by_bl', 'lsts_by_bl']:
                setattr(dc, attr, meta[attr])

        return data, flags, nsamples

    def read(self, bls=None, polarizations=None, times=None, frequencies=None,
             freq_chans=None, axis=None, read_data=True, return_data=True,
             run_check=True, check_extra=True, run_check_acceptability=True):
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

        Returns:
            data: DataContainer mapping baseline keys to complex visibility waterfalls
            flags: DataContainer mapping baseline keys to boolean flag waterfalls
            nsamples: DataContainer mapping baseline keys to interger Nsamples waterfalls
        '''
        # save last read parameters
        locs = locals()
        partials = ['bls', 'polarizations', 'times', 'frequencies', 'freq_chans']
        self.last_read_kwargs = {p: locs[p] for p in partials}

        # if filepaths is None, this was converted to HERAData
        # from a different pre-loaded object with no history of filepath
        if self.filepaths is not None:
            temp_read = self.read  # store self.read while it's being overwritten
            self.read = super().read  # re-define self.read so UVData can call self.read recursively for lists of files
            # load data
            try:
                if self.filetype == 'uvh5':
                    super().read(self.filepaths, file_type='uvh5', axis=axis, bls=bls, polarizations=polarizations,
                                 times=times, frequencies=frequencies, freq_chans=freq_chans, read_data=read_data,
                                 run_check=run_check, check_extra=check_extra, run_check_acceptability=run_check_acceptability)
                else:
                    if not read_data:
                        raise NotImplementedError('reading only metadata is not implemented for ' + self.filetype)
                    if self.filetype == 'miriad':
                        super().read(self.filepaths, file_type='miriad', axis=axis, bls=bls, polarizations=polarizations,
                                     run_check=run_check, check_extra=check_extra, run_check_acceptability=run_check_acceptability)
                        if any([times is not None, frequencies is not None, freq_chans is not None]):
                            warnings.warn('miriad does not support partial loading for times and frequencies. '
                                          'Loading the file first and then performing select.')
                            self.select(times=times, frequencies=frequencies, freq_chans=freq_chans)
                    elif self.filetype == 'uvfits':
                        super().read(self.filepaths, file_type='uvfits', axis=axis, bls=bls, polarizations=polarizations,
                                     times=times, frequencies=frequencies, freq_chans=freq_chans, run_check=run_check,
                                     check_extra=check_extra, run_check_acceptability=run_check_acceptability)
                        self.unphase_to_drift()
            finally:
                self.read = temp_read  # reset back to this function, regardless of whether the above try excecutes successfully

        # process data into DataContainers
        if read_data or self.filetype == 'uvh5':
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
        names = ['antenna_nums', 'antenna_names', 'ant_str',
                 'bls', 'times', 'blt_inds']
        for n in names:
            if n in kwargs and kwargs[n] is not None:
                output._determine_blt_slicing()
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

    def update(self, data=None, flags=None, nsamples=None):
        '''Update internal data arrays (data_array, flag_array, and nsample_array)
        using DataContainers (if not left as None) in preparation for writing to disk.

        Arguments:
            data: Optional DataContainer mapping baselines to complex visibility waterfalls
            flags: Optional DataContainer mapping baselines to boolean flag waterfalls
            nsamples: Optional DataContainer mapping baselines to interger Nsamples waterfalls
        '''
        if data is not None:
            for bl in data.keys():
                self._set_slice(self.data_array, bl, data[bl])
        if flags is not None:
            for bl in flags.keys():
                self._set_slice(self.flag_array, bl, flags[bl])
        if nsamples is not None:
            for bl in nsamples.keys():
                self._set_slice(self.nsample_array, bl, nsamples[bl])

    def partial_write(self, output_path, data=None, flags=None, nsamples=None, clobber=False, inplace=False, add_to_history='', **kwargs):
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

    def iterate_over_bls(self, Nbls=1, bls=None):
        '''Produces a generator that iteratively yields successive calls to
        HERAData.read() by baseline or group of baselines.

        Arguments:
            Nbls: number of baselines to load at once.
            bls: optional user-provided list of baselines to iterate over.
                Default: use self.bls (which only works for uvh5).

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
        for i in range(0, len(bls), Nbls):
            yield self.read(bls=bls[i:i + Nbls])

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
        HERAData.read() by time or group of contiguous times.

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
                or np.issubsctype(uvf.polarization_array.dtype, np.bytes_)), \
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
                    pol = ','.join([polnum2str(int(p), x_orientation=uvf.x_orientation) for p in pol.split(b',')])
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
                        jpol = ','.join([jnum2str(int(p), x_orientation=uvf.x_orientation) for p in jpol.split(b',')])
                    flags[(ant, jpol)] = np.array(uvf.flag_array[i, 0, :, :, ip].T)

        elif uvf.type == 'waterfall':  # one time x freq waterfall (per visibility polarization)
            for ip, jpol in enumerate(uvf.polarization_array):
                if np.issubdtype(uvf.polarization_array.dtype, np.signedinteger):
                    jpol = jnum2str(jpol, x_orientation=uvf.x_orientation)  # convert to string if possible
                else:
                    jpol = ','.join([jnum2str(int(p), x_orientation=uvf.x_orientation) for p in jpol.split(b',')])
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

    Miriad standard is bin start, so a shift by int_time / 2 is performed.
    UVH5 standard is bin center, so times are left untouched.

    Note: this is not currently compatible with Baseline Dependent Averaged data.

    Args:
        filepaths : type=list or str, filepath or list of filepaths
        filetype : str, options=['miriad', 'uvh5']

    Returns:
        If input is a string, output are floats, otherwise outputs are ndarrays.
        dlst : ndarray of lst bin width [radian]
        dtime : ndarray of time bin width [Julian Date]
        file_lst_arrays : ndarrays of unwrapped lst_array [radians]
        file_time_arrays : ndarrays of time_array [Julian Date]
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
                time_array = np.unique(_f[u'Header'][u'time_array'])
                if u'lst_array' in _f[u'Header']:
                    lst_array = np.ravel(_f[u'Header'][u'lst_array'])
                else:
                    lst_array = np.ravel(uvutils.get_lst_for_time(_f[u'Header'][u'time_array'],
                                                                  _f[u'Header'][u'latitude'][()],
                                                                  _f[u'Header'][u'longitude'][()],
                                                                  _f[u'Header'][u'altitude'][()]))
            lst_indices = np.unique(lst_array, return_index=True)[1]
            # resort by their appearance in lst_array, then unwrap
            lst_array = np.unwrap(lst_array[np.sort(lst_indices)])
            int_time_rad = np.median(np.diff(lst_array))
            int_time = np.median(np.diff(time_array))
            
        dlsts.append(int_time_rad)
        dtimes.append(int_time)
        file_lst_arrays.append(lst_array)
        file_time_arrays.append(time_array)

    dlsts = np.asarray(dlsts)
    dtimes = np.asarray(dtimes)
    file_lst_arrays = np.asarray(file_lst_arrays)
    file_time_arrays = np.asarray(file_time_arrays)

    if _array is False:
        return dlsts[0], dtimes[0], file_lst_arrays[0], file_time_arrays[0]
    else:
        return dlsts, dtimes, file_lst_arrays, file_time_arrays


def partial_time_io(hd, times, **kwargs):
    '''Perform partial io with a time-select on a HERAData object, even if it is intialized
    using multiple files, some of which do not contain any of the specified times.

    Arguments:
        hd: HERAData object intialized with (usually multiple) uvh5 files
        times: list of times in JD to load
        kwargs: other partial i/o kwargs (see io.HERAData.read)

    Returns:
        data: DataContainer mapping baseline keys to complex visibility waterfalls
        flags: DataContainer mapping baseline keys to boolean flag waterfalls
        nsamples: DataContainer mapping baseline keys to interger Nsamples waterfalls
        '''
    assert hd.filetype == 'uvh5', 'This function only works for uvh5-based HERAData objects.'
    combined_hd = None
    for f in hd.filepaths:
        hd_here = HERAData(f)
        times_here = [t for t in times if t in hd_here.times]
        if len(times_here) > 0:
            hd_here.read(times=times_here, return_data=False, **kwargs)
            if combined_hd is None:
                combined_hd = hd_here
            else:
                combined_hd += hd_here
    combined_hd = to_HERAData(combined_hd)  # re-runs the slicing and indexing
    return combined_hd.build_datacontainers()


def save_redcal_meta(meta_filename, fc_meta, omni_meta, freqs, times, lsts, antpos, history):
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
    '''
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
        fc_grp['polarity_flips'].attrs['ants'] = ant_keys

        # save the omnical metadata, saving dictionary keys as attrs
        omni_grp = outfile.create_group('omni_meta')
        pols_keys = sorted(list(omni_meta['chisq'].keys()))
        omni_grp['chisq'] = np.array([omni_meta['chisq'][pols] for pols in pols_keys])
        omni_grp['chisq'].attrs['pols'] = pols_keys
        omni_grp['iter'] = np.array([omni_meta['iter'][pols] for pols in pols_keys])
        omni_grp['iter'].attrs['pols'] = pols_keys
        omni_grp['conv_crit'] = np.array([omni_meta['conv_crit'][pols] for pols in pols_keys])
        omni_grp['conv_crit'].attrs['conv_crit'] = pols_keys


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
        history = infile['header']['history'][()].tostring().decode('utf8')

        # reconstruct firstcal metadata
        fc_meta = {}
        ants = [(int(num.tostring().decode('utf8')), pol.tostring().decode('utf8')) 
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


def to_HERAData(input_data, filetype='miriad'):
    '''Converts a string path, UVData, or HERAData object, or a list of any one of those, to a
    single HERAData object without loading any new data.

    Arguments:
        input_data: data file path, or UVData/HERAData instance, or list of either strings of data
            file paths or list of UVData/HERAData instances to combine into a single HERAData object
        filetype: 'miriad', 'uvfits', or 'uvh5'. Ignored if input_data is UVData/HERAData objects

    Returns:
        hd: HERAData object. Will not have data loaded if initialized from string(s).
    '''
    if filetype not in ['miriad', 'uvfits', 'uvh5']:
        raise NotImplementedError("Data filetype must be 'miriad', 'uvfits', or 'uvh5'.")
    if isinstance(input_data, str):  # single visibility data path
        return HERAData(input_data, filetype=filetype)
    elif isinstance(input_data, HERAData):  # already a HERAData object
        return input_data
    elif isinstance(input_data, UVData):  # single UVData object
        hd = input_data
        hd.__class__ = HERAData
        hd._determine_blt_slicing()
        hd._determine_pol_indexing()
        hd.filepaths = None
        return hd
    elif isinstance(input_data, collections.Iterable):  # List loading
        if np.all([isinstance(i, str) for i in input_data]):  # List of visibility data paths
            return HERAData(input_data, filetype=filetype)
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
              return_uvd=False, longitude=21.42830, start_jd=None, x_orientation="north", instrument="HERA",
              telescope_name="HERA", object_name='EOR', vis_units='uncalib', dec=-30.72152,
              telescope_location=np.array([5109325.85521063, 2005235.09142983, -3239928.42475395]),
              integration_time=None, **kwargs):
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

    longitude : type=float, longitude of observer in degrees East

    start_jd : type=float, starting integer Julian Date of time_array if time_array is None.

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
    pols = np.unique(list(map(lambda k: k[-1], data.keys())))
    Npols = len(pols)
    polarization_array = np.array(list(map(lambda p: polstr2num(p, x_orientation=x_orientation), pols)))

    # get times
    if time_array is None:
        if start_jd is None:
            raise AttributeError("if time_array is not fed, start_jd must be fed")
        time_array = LST2JD(lst_array, start_jd, longitude=longitude)
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
        nsample_array = np.ones_like(data_array, np.float)
    else:
        nsample_array = np.moveaxis(list(map(lambda p: list(map(lambda ap: nsamples[str(p)][ap], antpairs)), pols)), 0, -1)
        nsample_array = nsample_array.reshape(Nblts, 1, Nfreqs, Npols)

    # flags
    if flags is None:
        flag_array = np.zeros_like(data_array, np.float).astype(np.bool)
    else:
        flag_array = np.moveaxis(list(map(lambda p: list(map(lambda ap: flags[str(p)][ap].astype(np.bool), antpairs)), pols)), 0, -1)
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

    # get telescope ants
    antenna_numbers = np.unique(list(antpos.keys()))
    Nants_telescope = len(antenna_numbers)
    antenna_names = list(map(lambda a: "HH{}".format(a), antenna_numbers))

    # set uvw assuming drift phase i.e. phase center is zenith
    uvw_array = np.array([antpos[k[1]] - antpos[k[0]] for k in zip(ant_1_array, ant_2_array)])

    # get antenna positions in ITRF frame
    tel_lat_lon_alt = uvutils.LatLonAlt_from_XYZ(telescope_location)
    antenna_positions = np.array(list(map(lambda k: antpos[k], antenna_numbers)))
    antenna_positions = uvutils.ECEF_from_ENU(antenna_positions, *tel_lat_lon_alt) - telescope_location

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
    if filetype_out == 'miriad':
        hd.write_miriad(outfilename, clobber=clobber)
    elif filetype_out == 'uvfits':
        hd.write_uvfits(outfilename, force_phase=True, spoof_nonessential=True)
    elif filetype_out == 'uvh5':
        hd.write_uvh5(outfilename, clobber=clobber)
    else:
        raise TypeError("Input filetype must be either 'miriad', 'uvfits', or 'uvh5'.")


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
        return input_cal
    elif isinstance(input_cal, collections.Iterable):  # List loading
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


def write_cal(fname, gains, freqs, times, flags=None, quality=None, total_qual=None, write_file=True,
              return_uvc=True, outdir='./', overwrite=False, gain_convention='divide',
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
    ant_array = np.unique(list(map(lambda k: k[0], gains.keys()))).astype(np.int)
    antenna_numbers = copy.copy(ant_array)
    antenna_names = np.array(list(map(lambda a: "ant{}".format(a), antenna_numbers)))
    Nants_data = len(ant_array)
    Nants_telescope = len(antenna_numbers)

    # get polarization info: ordering must be monotonic in Jones number
    jones_array = np.array(list(set([jstr2num(k[1], x_orientation=x_orientation) for k in gains.keys()])))
    jones_array = jones_array[np.argsort(np.abs(jones_array))]
    pol_array = np.array([jnum2str(j, x_orientation=x_orientation) for j in jones_array])
    Njones = len(jones_array)

    # get time info
    time_array = np.array(times, np.float)
    Ntimes = len(time_array)
    time_range = np.array([time_array.min(), time_array.max()], np.float)
    if len(time_array) > 1:
        integration_time = np.median(np.diff(time_array)) * 24. * 3600.
    else:
        integration_time = 0.0

    # get frequency info
    freq_array = np.array(freqs, np.float)
    Nfreqs = len(freq_array)
    Nspws = 1
    freq_array = freq_array[None, :]
    spw_array = np.arange(Nspws)
    channel_width = np.median(np.diff(freq_array))

    # form gain, flags and qualities
    gain_array = np.empty((Nants_data, Nspws, Nfreqs, Ntimes, Njones), np.complex)
    flag_array = np.empty((Nants_data, Nspws, Nfreqs, Ntimes, Njones), np.bool)
    quality_array = np.empty((Nants_data, Nspws, Nfreqs, Ntimes, Njones), np.float)
    total_quality_array = np.empty((Nspws, Nfreqs, Ntimes, Njones), np.float)
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
                    flag_array[j, :, :, :, i] = np.zeros((Nspws, Nfreqs, Ntimes), np.bool)
                if quality is not None:
                    quality_array[j, :, :, :, i] = quality[(a, p)].T[None, :, :]
                else:
                    quality_array[j, :, :, :, i] = np.ones((Nspws, Nfreqs, Ntimes), np.float)
            else:
                gain_array[j, :, :, :, i] = np.ones((Nspws, Nfreqs, Ntimes), np.complex)
                flag_array[j, :, :, :, i] = np.ones((Nspws, Nfreqs, Ntimes), np.bool)
                quality_array[j, :, :, :, i] = np.ones((Nspws, Nfreqs, Ntimes), np.float)

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

    # create parameter list
    params = ["Nants_data", "Nants_telescope", "Nfreqs", "Ntimes", "Nspws", "Njones",
              "ant_array", "antenna_numbers", "antenna_names", "cal_style", "history",
              "channel_width", "flag_array", "gain_array", "quality_array", "jones_array",
              "time_array", "spw_array", "freq_array", "history", "integration_time",
              "time_range", "x_orientation", "telescope_name", "gain_convention", "total_quality_array"]

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
