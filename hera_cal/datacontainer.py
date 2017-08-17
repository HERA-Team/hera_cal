import numpy as np


class DataContainer:
    """Object that abstracts away the pol/ant pair ordering of data dict's."""

    def __init__(self, data):
        """
        Args:
            data (dict): dictionary of visibilities with keywords of pol/ant pair
                in any order.
        """
        self._data = {}
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
            self._data = data
        self._bls = set([k[:2] for k in self._data.keys()])
        self._pols = set([k[-1] for k in self._data.keys()])

    def mk_key(self, bl, pol):
        return bl + (pol,)

    def _switch_bl(self, key):
        if len(key) == 3:
            return (key[1], key[0], key[2])
        else:
            return (key[1], key[0])

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

    def __getitem__(self, key):
        if type(key) is str:  # asking for a pol
            return dict(zip(self._bls, [self[self.mk_key(bl, key)] for bl in self._bls]))
        elif len(key) == 2:  # asking for a bl
            return dict(zip(self._pols, [self[self.mk_key(key, pol)] for pol in self._pols]))
        else:
            try:
                return self._data[key]
            except(KeyError):
                return np.conj(self._data[self._switch_bl(key)])

    def get_data(self, *args):
        if len(args) > 1:
            return self.__getitem__(tuple(args))
        else:
            return self.__getitem__(*args)

    def has_key(self, *args):
        if len(args) == 1:
            return self._data.has_key(args[0]) or self._data.has_key(self._switch_bl(args[0]))
        else:
            return self.has_key(self.mk_key(args[0], args[1]))

    def has_bl(self, bl):
        return bl in self._bls

    def has_pol(self, pol):
        return pol in self._pols

    def get(self, bl, pol):
        return self[self.mk_key(bl, pol)]
