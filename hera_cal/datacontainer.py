import numpy as np
from collections import OrderedDict as odict


class DataContainer:
    """
    Object that abstracts away the pol/ant pair ordering of data dict's.
    Also abstracts away polarization case (i.e. 'xx' vs. 'XX') for 
    __getitem__, __setitem__, __delitem__, and has_key()

    adding two DataContainers adds their values: DC1 + DC2
    multiplying two DataContainers multiplies their values: DC1 * DC2
    """

    def __init__(self, data):
        """
        Args:
            data (dict): dictionary of visibilities with keywords of pol/ant pair
                in any order.
        """
        self._data = odict()
        if type(data.keys()[0]) is str:  # Nested POL:{bls}
            for pol in data.keys():
                for bl in data[pol]:
                    self._data[self.mk_key(bl, pol)] = data[pol][bl]
        elif len(data.keys()[0]) == 2:  # Nested bl:{POL}
            for bl in data.keys():
                for pol in data[bl]:
                    self._data[self.mk_key(bl, pol)] = data[bl][pol]
        else:
            assert(len(data.keys()[0]) == 3)
            self._data = odict(map(lambda k: (k, data[k]), sorted(data.keys())))
        self._bls = set([k[:2] for k in self._data.keys()])
        self._pols = set([k[-1] for k in self._data.keys()])

    def mk_key(self, bl, pol):
        return bl + (pol,)

    def _switch_bl(self, key):
        if len(key) == 3:
            return (key[1], key[0], key[2][::-1])
        else:
            return (key[1], key[0])

    def _convert_case(self, key):
        if isinstance(key, str):
            pol = key
        elif len(key) == 3:
            pol = key[-1]
        else:
            return key

        #convert the key's pol to whatever matches the current set of pols
        if pol.lower() in self._pols:
            pol = pol.lower()
        elif pol.upper() in self._pols:
            pol = pol.upper()

        if isinstance(key, str):
            return pol
        elif len(key) == 3:
            return key[0:2] + (pol,)


    def bls(self, pol=None):
        if pol is None:
            return self._bls.copy()
        else:
            return set([bl for bl in self._bls if self.has_key(bl, pol)])

    def pols(self, bl=None):
        if bl is None:
            return self._pols.copy()
        else:
            return set([pol for pol in self._pols if self.has_key(bl, pol)])

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        key = self._convert_case(key)
        if type(key) is str:  # asking for a pol
            return dict(zip(self._bls, [self[self.mk_key(bl, key)] for bl in self._bls]))
        elif len(key) == 2:  # asking for a bl
            return dict(zip(self._pols, [self[self.mk_key(key, pol)] for pol in self._pols]))
        else:
            try:
                return self._data[key]
            except(KeyError):
                return np.conj(self._data[self._switch_bl(key)])

    def __setitem__(self, key, value):
        key = self._convert_case(key)
        if len(key) == 3:
            # check given bl ordering
            if key in self.keys() or self._switch_bl(key) in self.keys():
                # key already exists
                if key in self.keys():
                    self._data[key] = value
                else:
                    self._data[self._switch_bl(key)] = np.conj(value)
            else:
                self._data[key] = value
                self._bls.update({tuple(key[:2])})
                self._pols.update({key[2]})
        else:
            raise ValueError('only supports setting (ant1, ant2, pol) keys')

    def __delitem__(self, key):
        key = self._convert_case(key)
        if len(key) == 3:
            del self._data[key]
            self._bls = set([k[:2] for k in self._data.keys()])
            self._pols = set([k[-1] for k in self._data.keys()])
        else:
            raise ValueError('only supports setting (ant1, ant2, pol) keys')

    def concatenate(self, D, axis=0):
        """
        Concatenates D, a DataContainer or a list of DCs, with self along an axis
        """
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
        for d in D: keys.update(d.keys())

        # iterate over D keys
        for i, k in enumerate(keys):
            if self.__contains__(k):
                newD[k] = np.concatenate([self.__getitem__(k)] + map(lambda d: d[k], D), axis=axis)

        return DataContainer(newD)
  
    def __add__(self, D):
        """ adds values of two DataContainers together """
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
        """ multiplies the values of two DataContainers together """
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
        return key in self.keys() or self._switch_bl(key) in self.keys()

    def get_data(self, *args):
        if len(args) > 1:
            return self.__getitem__(tuple(args))
        else:
            return self.__getitem__(*args)

    def has_key(self, *args):
        if len(args) == 1:
            return (self._data.has_key(self._convert_case(args[0])) or 
                    self._data.has_key(self._convert_case(self._switch_bl(args[0]))))
        else:
            return self.has_key(self._convert_case(self.mk_key(args[0], args[1])))

    def has_bl(self, bl):
        return bl in self._bls

    def has_pol(self, pol):
        return pol in self._pols

    def get(self, bl, pol):
        return self[self.mk_key(bl, pol)]


