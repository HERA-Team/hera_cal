# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

import numpy as np
from collections import OrderedDict as odict
import copy

from .utils import conj_pol, comply_pol, make_bl, comply_bl, reverse_bl


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
        except(KeyError):
            if isinstance(key, str):  # asking for a pol
                return dict(zip(self._antpairs, [self[make_bl(bl, key)] for bl in self._antpairs]))
            elif len(key) == 2:  # asking for a bl
                return dict(zip(self._pols, [self[make_bl(key, pol)] for pol in self._pols]))
            else:
                bl = comply_bl(key)
                try:
                    return self._data[bl]
                except(KeyError):
                    try:
                        if np.iscomplexobj(self._data[reverse_bl(bl)]):
                            return np.conj(self._data[reverse_bl(bl)])
                        else:
                            return self._data[reverse_bl(bl)]
                    except(KeyError):
                        raise KeyError('Cannot find either {} or {} in this DataContainer.'.format(key, reverse_bl(key)))

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
            if isinstance(k, tuple) and (len(k) == 3):
                k = comply_bl(k)
                del self._data[k]
            else:
                raise ValueError(f'Tuple keys to delete must be in the format (ant1, ant2, pol), {k} is not.')
        self._antpairs = set([k[:2] for k in self._data.keys()])
        self._pols = set([k[-1] for k in self._data.keys()])

    def concatenate(self, D, axis=0):
        '''Concatenates D, a DataContainer or a list of DCs, with self along an axis'''
        # check type of D
        if isinstance(D, DataContainer):
            # turn into list if not already a list
            D = [D]
        if axis == 0:
            # check 1 axis is identical for all D
            for d in D:
                if d[list(d.keys())[0]].shape[1] != self.__getitem__(list(self.keys())[0]).shape[1]:
                    raise ValueError("[1] axis of dictionary values aren't identical in length")

        if axis == 1:
            # check 0 axis is identical for all D
            for d in D:
                if d[list(d.keys())[0]].shape[0] != self.__getitem__(list(self.keys())[0]).shape[0]:
                    raise ValueError("[0] axis of dictionary values aren't identical in length")

        # start new object
        newD = odict()

        # get shared keys
        keys = set()
        for d in D:
            keys.update(d.keys())

        # iterate over D keys
        for i, k in enumerate(keys):
            if self.__contains__(k):
                newD[k] = np.concatenate([self.__getitem__(k)] + [d[k] for d in D], axis=axis)

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
        except(BaseException):  # if key is unparsable by comply_bl or reverse_bl, then it's not in self.keys()
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

    def select_or_expand_times(self, new_times, in_place=True):
        '''Update self.times with new times, updating data and metadata to be consistent. Data and
        metadata will be deleted, rearranged, or duplicated as necessary using numpy's fancy indexing.
        Assumes that the 0th data axis is time. Does not support baseline-dependent averaging.

        Arguments:
            new_times: list or numpy array of times to use to index into this object. These must all be in
                self.times, but they can be a subset in any order with any number of duplicates.
            in_place: if True, this DataContainer is modified. Otherwise, a modified copy is returned.
        '''
        if in_place:
            dc = self
        else:
            dc = copy.deepcopy(self)

        # make sure this is a sensible object for performing this operation
        assert dc.times is not None
        if not np.all([nt in dc.times for nt in new_times]):
            raise ValueError('All new_times must be in self.times.')
        if dc.times_by_bl is not None:
            for tbbl in dc.times_by_bl.values():
                assert np.all(tbbl == dc.times), 'select_or_expand_times does not support baseline dependent averaging.'
        if dc.lsts_by_bl is not None:
            for lbbl in dc.lsts_by_bl.values():
                assert np.all(lbbl == dc.lsts), 'select_or_expand_times does not support baseline dependent averaging.'

        # update data
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


class RedDataContainer(DataContainer):
    '''Structure for containing redundant visibilities that can be accessed by any
        one of the redundant baseline keys (or their conjugate).'''

    def __init__(self, data, reds=None, antpos=None, bl_error_tol=1.0):
        '''Creates a RedDataContainer.

        Arguments:
            data: DataContainer or dictionary of visibilities, just as one would pass into DataContainer().
                Will error if multiple baselines are part of the same redundant group.
            reds: list of lists of redundant baseline tuples, e.g. (ind1, ind2, pol).
            antpos: dictionary of antenna positions in the form {ant_index: np.array([x, y, z])}.
                Will error if one tries to provide both reds and antpos. If neither is provided,
                will try to to use data.antpos (which it might have if its is a DataContainer).
            bl_error_tol: the largest allowable difference between baselines in a redundant group
                (in the same units as antpos). Normally, this is up to 4x the largest antenna position
                error. Will only be used if reds is inferred from antpos.
        '''
        if reds is not None and antpos is not None:
            raise ValueError('Can only provide reds or antpos, not both.')

        super().__init__(data)

        # Figure out reds
        if reds is None:
            from .redcal import get_reds
            if antpos is not None:
                reds = get_reds(antpos, pols=self.pols(), bl_error_tol=bl_error_tol)
            elif hasattr(self, 'antpos') and self.antpos is not None:
                reds = get_reds(self.antpos, pols=self.pols(), bl_error_tol=bl_error_tol)
            else:
                raise ValueError('Must provide reds, antpos, or have antpos available at data.antpos')
        self.build_red_keys(reds)

    def _add_red(self, ubl_key, red):
        '''Updates internal dictionaries with a new redundant group.'''
        self.reds.append(red)
        self._red_key_to_bls[ubl_key] = []
        self._red_key_to_bls[reverse_bl(ubl_key)] = []
        for bl in red:
            self._bl_to_red_key[bl] = ubl_key
            self._bl_to_red_key[reverse_bl(bl)] = reverse_bl(ubl_key)
            self._red_key_to_bls[ubl_key].append(bl)
            self._red_key_to_bls[reverse_bl(ubl_key)].append(reverse_bl(bl))

    def build_red_keys(self, reds):
        '''Build the dictionaries that map baselines to redundant keys.

        Arguments:
            reds: list of lists of redundant baseline tuples, e.g. (ind1, ind2, pol).
        '''
        # Map all redundant keys to the same underlying data
        self.reds = []
        self._bl_to_red_key = {}
        self._red_key_to_bls = {}
        for red in copy.deepcopy(reds):
            bls_in_data = [bl for bl in red if self.has_key(bl)]
            if len(bls_in_data) > 1:
                raise ValueError('RedDataContainer can only be constructed with (at most) one baseline per group, '
                                 + f'but this data has the following redundant baselines: {bls_in_data}')
            if len(bls_in_data) == 0:
                self._add_red(red[0], red)
            elif len(bls_in_data) > 0:
                self._add_red(bls_in_data[0], red)

        # delete unused data to avoid leaking memory
        del self[[k for k in self._data if k not in self._bl_to_red_key]]

    def get_ubl_key(self, key):
        '''Returns the key used interally denote the data stored. Useful for del'''
        return self._bl_to_red_key[key]

    def get_red(self, key):
        '''Returns the list of baselines redundant with this key.'''
        return self._red_key_to_bls[self._bl_to_red_key[key]]

    def __getitem__(self, key):
        '''Returns data corresponding to the unique baseline that key is a member of.'''
        return super().__getitem__(self._bl_to_red_key[key])

    def __setitem__(self, key, value):
        '''Sets data for to unique baseline that the key is a member of.'''
        ubl_key = self._bl_to_red_key.get(key, None)
        if ubl_key is None:
            self._add_red(key, [key])  # treat this as a new baseline not redundant with anything
            ubl_key = key
        super().__setitem__(ubl_key, value)

    def __contains__(self, key):
        '''Returns true if the baseline redundant with the key is in the data.'''
        return (key in self._bl_to_red_key) and (super().__contains__(self._bl_to_red_key[key]))
