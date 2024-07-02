# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License
from __future__ import annotations

import numpy as np
from collections import OrderedDict as odict
import copy
import warnings
from functools import cache

from typing import Sequence
from .utils import conj_pol, comply_pol, make_bl, comply_bl, reverse_bl
from .red_groups import RedundantGroups, Baseline, AntPair


class DataContainer:
    """Dictionary-like object that abstracts away the pol/ant pair ordering of data
    dictionaries and the the polarization case (i.e. 'nn' vs. 'NN'). Keys are in
    the form (0,1,'nn'). Does not know about x_orientation, which means that
    'xx' and 'nn' are always treated as different polarizations.

    Supports much of the same functionality as dictionaries, including:
        * __getitem__
        * __setitem__
        * __delitem__
        * __len__
        * .has_key()
        * .keys()
        * .items()
        * .values()

    DataContainer knows to return a complex-conjugated visibility if asked for
    an antenna-reversed baseline. It also makes sure that a given polarization is
    maintained internally in a consistent way so that the user can ask for either.
    Data container keeps track a set of unique baselines and polarizations and
    supports adding or multiplying two data containers with identical keys and
    data shapes via the overloaded + and * operators."""

    def __init__(self, data):
        """Create a DataContainer object from a dictionary of data. Assumes that all
        keys have the same format and that polarization case is internally consistent.

        Arguments:
            data: dictionary of visibilities with keywords of pol/ant pair
                in any order. Supports both three element keys, e.g. data[(i,j,pol)],
                or nested dictions, e.g. data[(i,j)][pol] or data[pol][(i,j)].
        """
        if isinstance(data, DataContainer):
            self.__dict__.update(data.__dict__)
        else:
            self._data = odict()
            if np.all([isinstance(k, str) for k in data.keys()]):  # Nested POL:{antpairs}
                for pol in data.keys():
                    for antpair in data[pol]:
                        self._data[make_bl(antpair, pol)] = data[pol][antpair]
            elif np.all([len(k) == 2 for k in data.keys()]):  # Nested antpair:{POL}
                for antpair in data.keys():
                    for pol in data[antpair]:
                        self._data[make_bl(antpair, pol)] = data[antpair][pol]
            elif np.all([len(k) == 3 for k in data.keys()]):
                self._data = odict([(comply_bl(k), data[k]) for k in sorted(data.keys())])
            else:
                raise KeyError('Unrecognized key type or mix of key types in data dictionary.')
            self._antpairs = set([k[:2] for k in self._data])
            self._pols = set([k[-1] for k in self._data])

            # placeholders for metadata (or get them from data, if possible)
            for attr in ['ants', 'data_ants', 'antpos', 'data_antpos',
                         'freqs', 'times', 'lsts', 'times_by_bl', 'lsts_by_bl']:
                if hasattr(data, attr):
                    setattr(self, attr, getattr(data, attr))
                else:
                    setattr(self, attr, None)

    @property
    def dtype(self):
        """The dtype of the underlying data."""
        if self.keys():
            try:
                return self._data[next(iter(self.keys()))].dtype
            except AttributeError:
                return None
        else:
            return None

    def antpairs(self, pol=None):
        '''Return a set of antenna pairs (with a specific pol or more generally).'''
        if pol is None:
            return self._antpairs.copy()
        else:
            return set([ap for ap in self._antpairs if self.has_key(ap, pol)])

    def pols(self, antpair=None):
        '''Return a set of polarizations (with a specific antpair or more generally).'''
        if antpair is None:
            return self._pols.copy()
        else:
            return set([pol for pol in self._pols if self.has_key(antpair, pol)])

    def bls(self):
        '''Returns a set of baseline-pol tuples'''
        return set(self._data.keys())

    def keys(self):
        '''Returns the keys of the data as a list.'''
        return self._data.keys()

    def values(self):
        '''Returns the values of the data as a list.'''
        return self._data.values()

    def items(self):
        '''Returns the keys and values of the data as a list of tuples.'''
        return self._data.items()

    def __len__(self):
        '''Returns the number of keys in the data.'''
        return len(self._data)

    def __getitem__(self, key):
        '''Returns the data corresponding to the key. If the key is just a polarization,
        returns all baselines for that polarization. If the key is just a baseline,
        returns all polarizations for that baseline. If the key is of the form (0,1,'nn'),
        returns the associated entry. Abstracts away both baseline ordering (applying the
        complex conjugate when appropriate) and polarization capitalization.'''
        try:  # just see if the key works first
            return self._data[key]
        except (KeyError):
            if isinstance(key, str):  # asking for a pol
                return dict(zip(self._antpairs, [self[make_bl(bl, key)] for bl in self._antpairs]))
            elif len(key) == 2:  # asking for a bl
                return dict(zip(self._pols, [self[make_bl(key, pol)] for pol in self._pols]))
            else:
                bl = comply_bl(key)
                try:
                    return self._data[bl]
                except (KeyError):
                    try:
                        if np.iscomplexobj(self._data[reverse_bl(bl)]):
                            return np.conj(self._data[reverse_bl(bl)])
                        else:
                            return self._data[reverse_bl(bl)]
                    except KeyError:
                        raise KeyError(f'Cannot find either {key} or {reverse_bl(key)} in this DataContainer.')

    def __setitem__(self, key, value):
        '''Sets the data corresponding to the key. Only supports the form (0,1,'nn').
        Abstracts away both baseline ordering (applying the complex conjugate when
        appropriate) and polarization capitalization.'''
        if len(key) == 3:
            key = comply_bl(key)
            # check given bl ordering
            if key in self.keys() or reverse_bl(key) in self.keys():
                # key already exists
                if key in self.keys():
                    self._data[key] = value
                else:
                    if np.iscomplexobj(value):
                        self._data[reverse_bl(key)] = np.conj(value)
                    else:
                        self._data[reverse_bl(key)] = value
            else:
                self._data[key] = value
                self._antpairs.update({tuple(key[:2])})
                self._pols.update({key[2]})
        else:
            raise ValueError('only supports setting (ant1, ant2, pol) keys')

    def __delitem__(self, key):
        '''Deletes the input key(s) and corresponding data. Only supports tuples of form (0,1,'nn')
        or lists or ndarrays of tuples of that form. Abstracts away both baseline ordering and
        polarization capitalization.'''
        if isinstance(key, tuple):
            key = [key]
        for k in key:
            if not isinstance(k, tuple) or len(k) != 3:
                raise ValueError(
                    'Tuple keys to delete must be in the format (ant1, ant2, pol), '
                    f'{k} is not.'
                )
            k = comply_bl(k)
            del self._data[k]
        self._antpairs = {k[:2] for k in self._data.keys()}
        self._pols = {k[-1] for k in self._data.keys()}

    @property
    def shape(self) -> tuple[int]:
        return self[next(iter(self.keys()))].shape

    def concatenate(self, D, axis=0):
        '''Concatenates D, a DataContainer or a list of DCs, with self along an axis'''
        # check type of D
        if isinstance(D, DataContainer):
            # turn into list if not already a list
            D = [D]
        if axis == 0:
            # check 1 axis is identical for all D
            for d in D:
                if d.shape[1] != self.shape[1]:
                    raise ValueError("[1] axis of dictionary values aren't identical in length")

        if axis == 1:
            # check 0 axis is identical for all D
            for d in D:
                if d.shape[0] != self.shape[0]:
                    raise ValueError("[0] axis of dictionary values aren't identical in length")

        # get shared keys
        keys = set(sum((list(d.keys()) for d in D), []))
        keys = [k for k in keys if k in self]

        # iterate over D keys
        newD = {k: np.concatenate([self[k]] + [d[k] for d in D], axis=axis) for k in keys}

        return DataContainer(newD)

    def __add__(self, D):
        '''
        Addition operator overload.

        Add the values of this DataContainer with
        the value of D. If D is another DataContainer, add
        their values together and form a new container,
        otherwise add D to each ndarray in self.
        '''
        # check type of D
        if isinstance(D, DataContainer):
            # check time and frequency structure matches
            if D[list(D.keys())[0]].shape[0] != self.__getitem__(list(self.keys())[0]).shape[0]:
                raise ValueError("[0] axis of dictionary values don't match")
            if D[list(D.keys())[0]].shape[1] != self.__getitem__(list(self.keys())[0]).shape[1]:
                raise ValueError("[1] axis of dictionary values don't match")

            # start new object
            newD = odict()

            # iterate over D keys
            for i, k in enumerate(D.keys()):
                if self.__contains__(k):
                    newD[k] = self.__getitem__(k) + D[k]

            return DataContainer(newD)

        else:
            newD = copy.deepcopy(self)
            for k in newD.keys():
                newD[k] = newD[k] + D

            return newD

    def __sub__(self, D):
        '''
        Subtraction operator overload.

        Subtract D with the values of this DataContainer.
        If D is another DataContainer, subtract
        their values and form a new container,
        otherwise subtract D from each ndarray in self.
        '''
        # check type of D
        if isinstance(D, DataContainer):
            # check time and frequency structure matches
            if D[list(D.keys())[0]].shape[0] != self.__getitem__(list(self.keys())[0]).shape[0]:
                raise ValueError("[0] axis of dictionary values don't match")
            if D[list(D.keys())[0]].shape[1] != self.__getitem__(list(self.keys())[0]).shape[1]:
                raise ValueError("[1] axis of dictionary values don't match")

            # start new object
            newD = odict()

            # iterate over D keys
            for i, k in enumerate(D.keys()):
                if self.__contains__(k):
                    newD[k] = self.__getitem__(k) - D[k]

            return DataContainer(newD)

        else:
            newD = copy.deepcopy(self)
            for k in newD.keys():
                newD[k] = newD[k] - D

            return newD

    def __mul__(self, D):
        '''
        Multiplication operator overload.

        Multiply D with the values of this DataContainer.
        If D is another DataContainer, multiply
        their values together and form a new container,
        otherwise multiply D with each ndarray in self.
        '''
        # check type of D
        if isinstance(D, DataContainer):
            # check time and frequency structure matches
            if D[list(D.keys())[0]].shape[0] != self.__getitem__(list(self.keys())[0]).shape[0]:
                raise ValueError("[0] axis of dictionary values don't match")
            if D[list(D.keys())[0]].shape[1] != self.__getitem__(list(self.keys())[0]).shape[1]:
                raise ValueError("[1] axis of dictionary values don't match")

            # start new object
            newD = odict()

            # iterate over D keys
            for i, k in enumerate(D.keys()):
                if self.__contains__(k):
                    newD[k] = self.__getitem__(k) * D[k]

            return DataContainer(newD)

        else:
            newD = copy.deepcopy(self)
            for k in newD.keys():
                newD[k] = newD[k] * D

            return newD

    def __floordiv__(self, D):
        '''
        Floor division operator overload, i.e. //.

        Floor divide the values of this DataContainer by D.
        If D is another DataContainer, floor divide
        their values and form a new container,
        otherwise floor divide D from each ndarray in self.
        '''
        # check type of D
        if isinstance(D, DataContainer):
            # check time and frequency structure matches
            if D[list(D.keys())[0]].shape[0] != self.__getitem__(list(self.keys())[0]).shape[0]:
                raise ValueError("[0] axis of dictionary values don't match")
            if D[list(D.keys())[0]].shape[1] != self.__getitem__(list(self.keys())[0]).shape[1]:
                raise ValueError("[1] axis of dictionary values don't match")

            # start new object
            newD = odict()

            # iterate over D keys
            for i, k in enumerate(D.keys()):
                if self.__contains__(k):
                    if not (np.iscomplexobj(self.__getitem__(k)) or np.iscomplexobj(D[k])):
                        newD[k] = self.__getitem__(k) // D[k]
                    else:
                        div = self.__getitem__(k) / D[k]
                        newD[k] = np.real(div).astype(int) + 1j * np.imag(div).astype(int)

            return DataContainer(newD)

        else:
            newD = copy.deepcopy(self)
            for k in newD.keys():
                if not (np.iscomplexobj(newD[k]) or np.iscomplexobj(D)):
                    newD[k] = newD[k] // D
                else:
                    div = newD[k] / D
                    newD[k] = np.real(div).astype(int) + 1j * np.imag(div).astype(int)

            return newD

    def __truediv__(self, D):
        '''
        True division operator overload, i.e. /.

        True divide the values of this DataContainer by D.
        If D is another DataContainer, true divide
        their values and form a new container,
        otherwise true divide D from each ndarray in self.
        '''
        # check type of D
        if isinstance(D, DataContainer):
            # check time and frequency structure matches
            if D[list(D.keys())[0]].shape[0] != self.__getitem__(list(self.keys())[0]).shape[0]:
                raise ValueError("[0] axis of dictionary values don't match")
            if D[list(D.keys())[0]].shape[1] != self.__getitem__(list(self.keys())[0]).shape[1]:
                raise ValueError("[1] axis of dictionary values don't match")

            # start new object
            newD = odict()

            # iterate over D keys
            for i, k in enumerate(D.keys()):
                if self.__contains__(k):
                    newD[k] = self.__getitem__(k) / D[k]

            return DataContainer(newD)

        else:
            newD = copy.deepcopy(self)
            for k in newD.keys():
                newD[k] = newD[k] / D

            return newD

    def __invert__(self):
        '''Inverts the values of the DataContainer via logical not.'''
        # start new object
        newD = copy.deepcopy(self)

        # iterate over keys
        for i, k in enumerate(newD.keys()):
            newD[k] = ~newD[k]

        return newD

    def __neg__(self):
        '''Negates the values of the DataContainer.'''
        # start new object
        newD = copy.deepcopy(self)

        # iterate over keys
        for i, k in enumerate(newD.keys()):
            newD[k] = -newD[k]

        return newD

    def __transpose__(self):
        '''Tranposes the values of the DataContainer'''
        # start new object
        newD = copy.deepcopy(self)

        # iterate over keys
        for i, k in enumerate(newD.keys()):
            newD[k] = newD[k].T

        return newD

    @property
    def T(self):
        return self.__transpose__()

    def __contains__(self, key):
        '''Returns True if the key is in the data, abstracting away case and baseline order.'''
        try:
            bl = comply_bl(key)
            return (bl in self.keys() or reverse_bl(bl) in self.keys())
        except (BaseException):  # if key is unparsable by comply_bl or reverse_bl, then it's not in self.keys()
            return False

    def __iter__(self):
        '''Iterates over keys, just like a standard dictionary.'''
        return iter(self._data.keys())

    def get_data(self, *args):
        '''Interface to DataContainer.__getitem__(key).'''
        if len(args) > 1:
            return self.__getitem__(tuple(args))
        else:
            return self.__getitem__(*args)

    def has_key(self, *args):
        '''Interface to DataContainer.__contains__(key).'''
        if len(args) == 1:
            bl = comply_bl(args[0])
            return (bl in self._data or reverse_bl(bl) in self._data)
        else:
            return make_bl(args[0], args[1]) in self

    def has_antpair(self, antpair):
        '''Returns True if baseline or its complex conjugate is in the data.'''
        return (antpair in self._antpairs or reverse_bl(antpair) in self._antpairs)

    def has_pol(self, pol):
        '''Returns True if polarization (with some capitalization) is in the data.'''
        return comply_pol(pol) in self._pols

    def get(self, key, val=None):
        '''Allows for getting values with fallback if not found. Default None.'''
        return (self[key] if key in self else val)

    def select_or_expand_times(self, new_times: Sequence[float] | None = None, in_place=True, skip_bda_check=False, *, indices: np.ndarray | None = None):
        '''Update self.times with new times, updating data and metadata to be consistent. Data and
        metadata will be deleted, rearranged, or duplicated as necessary using numpy's fancy indexing.
        Assumes that the 0th data axis is time. Does not support baseline-dependent averaging.

        Parameters
        ----------
        new_times : list or numpy array or None
            Times to use to index into this object. If given, these must all be in
            self.times, but they can be a subset in any order with any number of
            duplicates. If not given, ``indices`` must be given.
        in_place : bool
            If True, this DataContainer is modified. Otherwise, a modified copy is returned.
        skip_bda_check : bool
            If True, do not check that the object is sensible for this operation.
            This is useful for performance reasons when you know the object is sensible.
        indices : integer array
            If given, these are the indices to use to index the time axis.
            If given, new_times must be None.

        '''
        if new_times is None and indices is None:
            raise ValueError('Either new_times or indices must be given.')
        if new_times is not None and indices is not None:
            raise ValueError('Cannot specify both new_times and indices.')

        if in_place:
            dc = self
        else:
            dc = copy.deepcopy(self)

        # make sure this is a sensible object for performing this operation
        assert dc.times is not None

        if new_times is not None and not np.all([nt in dc.times for nt in new_times]):
            raise ValueError('All new_times must be in self.times.')

        if not skip_bda_check:
            if dc.times_by_bl is not None:
                for tbbl in dc.times_by_bl.values():
                    assert np.all(tbbl == np.asarray(dc.times)), 'select_or_expand_times does not support baseline dependent averaging.'
            if dc.lsts_by_bl is not None:
                for lbbl in dc.lsts_by_bl.values():
                    assert np.all(lbbl == np.asarray(dc.lsts)), 'select_or_expand_times does not support baseline dependent averaging.'

        # update data
        if indices is not None:
            nt_inds = indices
            new_times = np.array(dc.times)[nt_inds]
        else:
            nt_inds = np.searchsorted(np.array(dc.times), np.array(new_times))

        for bl in dc:
            assert dc[bl].shape[0] == len(dc.times), 'select_or_expand_times assume that time is the 0th data dimension.'
            dc[bl] = dc[bl][nt_inds]

        # update metadata
        dc.times = new_times
        if dc.lsts is not None:
            dc.lsts = np.array(dc.lsts)[nt_inds]
        if dc.times_by_bl is not None:
            for bl in dc.times_by_bl:
                dc.times_by_bl[bl] = dc.times_by_bl[bl][nt_inds]
        if dc.lsts_by_bl is not None:
            for bl in dc.lsts_by_bl:
                dc.lsts_by_bl[bl] = dc.lsts_by_bl[bl][nt_inds]

        if not in_place:
            return dc

    def select_freqs(
        self,
        freqs: np.ndarray | None = None,
        channels: np.ndarray | slice | None = None,
        in_place: bool = True
    ):
        """Update the object with a subset of frequencies (which may be repeated).

        While typically this will be used to down-select frequencies, one can
        'expand' the frequencies by duplicating channels.

        Parameters
        ----------
        freqs : np.ndarray, optional
            Frequencies to select. If given, all frequencies must be in the datacontainer.
        channels : np.ndarray, slice, optional
            Channels to select. If given, all channels must be in the datacontainer.
            Only one of freqs or channels can be given.
        in_place : bool, optional
            If True, modify the object in place. Otherwise, return a modified copy.
            Even if `in_place` is True, the object is still returned for convenience.

        Returns
        -------
        DataContainer
            The modified object. If `in_place` is True, this is the same object.
        """
        obj = self if in_place else copy.deepcopy(self)
        if freqs is None and channels is None:
            return obj
        elif freqs is not None and channels is not None:
            raise ValueError('Cannot specify both freqs and channels.')

        if freqs is not None:
            if obj.freqs is None:
                raise ValueError('Cannot select frequencies if self.freqs is None.')

            if not np.all([fq in obj.freqs for fq in freqs]):
                raise ValueError('All freqs must be in self.freqs.')
            channels = np.searchsorted(obj.freqs, freqs)

        if obj.freqs is None:
            warnings.warn("It is impossible to automatically detect which axis is frequency. Trying last axis.")
            axis = -1
        else:
            axis = obj[next(iter(obj.keys()))].shape.index(len(obj.freqs))
        for bl in obj:
            obj[bl] = obj[bl].take(channels, axis=axis)

        # update metadata
        if obj.freqs is not None:
            obj.freqs = obj.freqs[channels]

        return obj


class RedDataContainer(DataContainer):
    '''Structure for containing redundant visibilities that can be accessed by any
        one of the redundant baseline keys (or their conjugate).'''

    def __init__(
        self,
        data: DataContainer | dict[Baseline, np.ndarray],
        reds: RedundantGroups | Sequence[Sequence[Baseline | AntPair]] | None = None,
        antpos: dict[int, np.ndarray] | None = None,
        bl_error_tol: float = 1.0
    ):
        '''Creates a RedDataContainer.

        Parameters
        ----------
        data : DataContainer or dictionary of visibilities, just as one would pass into DataContainer().
            Will error if multiple baselines are part of the same redundant group.
        reds : :class:`RedundantGroups` object, or list of lists of redundant baseline tuples, e.g. (ind1, ind2, pol).
            These are the redundant groups of baselines. If not provided, will try to
            infer them from antpos.
        antpos: dictionary of antenna positions in the form {ant_index: np.array([x, y, z])}.
            Will error if one tries to provide both reds and antpos. If neither is provided,
            will try to to use data.antpos (which it might have if its is a DataContainer).
        bl_error_tol : float
            the largest allowable difference between baselines in a redundant group
            (in the same units as antpos). Normally, this is up to 4x the largest antenna position
            error. Will only be used if reds is inferred from antpos.

        Attributes
        ----------
        reds
            A :class:`RedundantGroups` object that contains the redundant groups for
            the entire array, and methods to manipulate them.

        '''
        if reds is not None and antpos is not None:
            raise ValueError('Can only provide reds or antpos, not both.')

        super().__init__(data)

        # Figure out reds
        if reds is None:
            from .redcal import get_reds
            if antpos is not None:
                reds = RedundantGroups.from_antpos(
                    antpos=antpos, pols=self.pols(), bl_error_tol=bl_error_tol, include_autos=False
                )
            elif hasattr(self, 'antpos') and self.antpos is not None:
                reds = RedundantGroups.from_antpos(
                    antpos=self.antpos, pols=self.pols(), bl_error_tol=bl_error_tol, include_autos=False
                )
            else:
                raise ValueError('Must provide reds, antpos, or have antpos available at data.antpos')

        if not isinstance(reds, RedundantGroups):
            reds = RedundantGroups(red_list=reds, antpos=self.antpos)

        self.build_red_keys(reds)

    def build_red_keys(self, reds: RedundantGroups | list[list[Baseline]]):
        '''Build the dictionaries that map baselines to redundant keys.

        Arguments:
            reds: list of lists of redundant baseline tuples, e.g. (ind1, ind2, pol).
        '''

        if isinstance(reds, RedundantGroups):
            self.reds = reds
        else:
            self.reds = RedundantGroups(red_list=reds, antpos=getattr(self, 'antpos', None))

        # delete unused data to avoid leaking memory
        del self[[k for k in self._data if k not in self.reds]]

        # Check that the data only has one baseline per redundant group
        redkeys = {}
        for bl in self.bls():
            ubl = self.reds.get_ubl_key(bl)
            if ubl in redkeys:
                raise ValueError(
                    'RedDataContainer can only be constructed with (at most) one baseline per group, '
                    f'but {bl} is redundant with {redkeys[ubl]}.'
                )
            else:
                redkeys[ubl] = bl

    def get_ubl_key(self, bl: Baseline | AntPair):
        '''Returns the blkey used to internally denote the data stored.

        If this bl is in a redundant group present in the data, this will return the
        blkey that exists in the data. Otherwise, it will return the array-wide blkey
        representing this group.
        '''
        # return self._reds_keyed_on_data.get_ubl_key(bl)
        out = self.reds.get_reds_in_bl_set(bl, self.keys(), include_conj=False)
        if len(out) == 1:
            return next(iter(out))
        elif len(out) == 0:
            return self.reds.get_ubl_key(bl)
        else:
            raise ValueError(
                f'Baseline {bl} corresponds to multiple baselines in the data: {out}.'
            )

    def get_red(self, key):
        '''Returns the list of baselines in the array redundant with this key.

        Note: this is not just baselines existing in the data itself, but in the
              entire array.
        '''
        return self.reds[key]

    def __getitem__(self, key):
        '''Returns data corresponding to the unique baseline that key is a member of.'''
        return super().__getitem__(self.get_ubl_key(key))

    def __setitem__(self, key, value):
        '''Sets data for unique baseline that the key is a member of.'''
        if key in self.reds:
            ubl_key = self.get_ubl_key(key)
        else:
            # treat this as a new baseline not redundant with anything
            self.reds.append([key])
            ubl_key = key

        super().__setitem__(ubl_key, value)

    def __contains__(self, key):
        '''Returns true if the baseline redundant with the key is in the data.'''
        return (key in self.reds) and (super().__contains__(self.get_ubl_key(key)))
