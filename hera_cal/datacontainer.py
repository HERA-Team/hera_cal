# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

import numpy as np
from collections import OrderedDict as odict
from hera_cal.utils import conj_pol, comply_pol, make_bl, comply_bl, reverse_bl


class DataContainer:
    """Dictionary-like object that abstracts away the pol/ant pair ordering of data
    dictionaries and the the polarization case (i.e. 'xx' vs. 'XX'). Keys are in
    the form (0,1,'xx').

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
        self._data = odict()
        if np.all([isinstance(k, (str, np.str)) for k in data.keys()]):  # Nested POL:{antpairs}
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
        self._antpairs = set([k[:2] for k in self._data.keys()])
        self._pols = set([k[-1] for k in self._data.keys()])

        # placeholders for metadata
        self.antpos = None
        self.freqs = None
        self.times = None
        self.lsts = None
        self.times_by_bl = None
        self.lsts_by_bl = None

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
        returns all polarizations for that baseline. If the key is of the form (0,1,'xx'),
        returns the associated entry. Abstracts away both baseline ordering (applying the
        complex conjugate when appropriate) and polarization capitalization.'''
        if isinstance(key, str):  # asking for a pol
            return dict(zip(self._antpairs, [self[make_bl(bl, key)] for bl in self._antpairs]))
        elif len(key) == 2:  # asking for a bl
            return dict(zip(self._pols, [self[make_bl(key, pol)] for pol in self._pols]))
        else:
            try:
                return self._data[comply_bl(key)]
            except(KeyError):
                return np.conj(self._data[reverse_bl(key)])

    def __setitem__(self, key, value):
        '''Sets the data corresponding to the key. Only supports the form (0,1,'xx').
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
                    self._data[reverse_bl(key)] = np.conj(value)
            else:
                self._data[key] = value
                self._antpairs.update({tuple(key[:2])})
                self._pols.update({key[2]})
        else:
            raise ValueError('only supports setting (ant1, ant2, pol) keys')

    def __delitem__(self, key):
        '''Deletes the input key and corresponding data. Only supports the form (0,1,'xx').
        Abstracts away both baseline ordering and polarization capitalization.'''
        if len(key) == 3:
            key = comply_bl(key)
            del self._data[key]
            self._antpairs = set([k[:2] for k in self._data.keys()])
            self._pols = set([k[-1] for k in self._data.keys()])
        else:
            raise ValueError('only supports setting (ant1, ant2, pol) keys')

    def concatenate(self, D, axis=0):
        '''Concatenates D, a DataContainer or a list of DCs, with self along an axis'''
        # check type of D
        if isinstance(D, DataContainer):
            # turn into list if not already a list
            D = [D]
        if axis == 0:
            # check 1 axis is identical for all D
            for d in D:
                if d[d.keys()[0]].shape[1] != self.__getitem__(self.keys()[0]).shape[1]:
                    raise ValueError("[1] axis of dictionary values aren't identical in length")

        if axis == 1:
            # check 0 axis is identical for all D
            for d in D:
                if d[d.keys()[0]].shape[0] != self.__getitem__(self.keys()[0]).shape[0]:
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
                newD[k] = np.concatenate([self.__getitem__(k)] + map(lambda d: d[k], D), axis=axis)

        return DataContainer(newD)

    def __add__(self, D):
        '''Adds values of two DataContainers together.'''
        # check time and frequency structure matches
        if D[D.keys()[0]].shape[0] != self.__getitem__(self.keys()[0]).shape[0]:
            raise ValueError("[0] axis of dictionary values don't match")
        if D[D.keys()[0]].shape[1] != self.__getitem__(self.keys()[0]).shape[1]:
            raise ValueError("[1] axis of dictionary values don't match")

        # start new object
        newD = odict()

        # iterate over D keys
        for i, k in enumerate(D.keys()):
            if self.__contains__(k):
                newD[k] = self.__getitem__(k) + D[k]

        return DataContainer(newD)

    def __mul__(self, D):
        '''Multiplies the values of two DataContainers together.'''
        # check time and frequency structure matches
        if D[D.keys()[0]].shape[0] != self.__getitem__(self.keys()[0]).shape[0]:
            raise ValueError("[0] axis of dictionary values don't match")
        if D[D.keys()[0]].shape[1] != self.__getitem__(self.keys()[0]).shape[1]:
            raise ValueError("[1] axis of dictionary values don't match")

        # start new object
        newD = odict()

        # iterate over D keys
        for i, k in enumerate(D.keys()):
            if self.__contains__(k):
                newD[k] = self.__getitem__(k) * D[k]

        return DataContainer(newD)

    def __contains__(self, key):
        '''Returns True if the key is in the data, abstracting away case and baseline order.'''
        try:
            return comply_bl(key) in self.keys() or reverse_bl(key) in self.keys()
        except(BaseException):  # if key is unparsable by comply_bl or reverse_bl, then it's not in self.keys()
            return False

    def get_data(self, *args):
        '''Interface to DataContainer.__getitem__(key).'''
        if len(args) > 1:
            return self.__getitem__(tuple(args))
        else:
            return self.__getitem__(*args)

    def has_key(self, *args):
        '''Interface to DataContainer.__contains__(key).'''
        if len(args) == 1:
            return (comply_bl(args[0]) in self._data
                    or reverse_bl(args[0]) in self._data)
        else:
            return make_bl(args[0], args[1]) in self

    def has_antpair(self, antpair):
        '''Returns True if baseline or its complex conjugate is in the data.'''
        return antpair in self._antpairs or reverse_bl(antpair) in self._antpairs

    def has_pol(self, pol):
        '''Returns True if polarization (with some capitalization) is in the data.'''
        return comply_pol(pol) in self._pols

    def get(self, antpair, pol):
        '''Interface to DataContainer.__getitem__(bl + (pol,)).'''
        return self[make_bl(antpair, pol)]
