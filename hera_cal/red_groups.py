"""Module with classes and functions for dealing with redundant baseline groups."""
from __future__ import annotations

import attrs
from .utils import reverse_bl
from . import redcal
import numpy as np
import copy
from functools import cached_property, wraps, partial
from frozendict import frozendict

from typing import Sequence

AntPair = tuple[int, int]
Baseline = tuple[int, int, str]

def _convert_red_list(red_list: Sequence[Sequence[AntPair | Baseline]]) -> list[list[AntPair | Baseline]]:
    """Convert a list of redundant baseline groups to a list of lists of antenna pairs."""
    return [list(bls) for bls in red_list]

@attrs.define(frozen=False, slots=False)
class RedundantGroups:
    """Redundant baseline group manager.

    This class is used to manage redundant baseline groups. It is (roughly)
    API-compatible with the list-of-lists-of-tuples format returned by, e.g., 
    :func:`hera_cal.redcal.get_reds`, and in the future that may simply return
    one of these objects.

    To keep this compatibility, this object is mutable, but we absolutely discourage
    directly mutating it. Instead, use the methods provided, including the `.append`
    method, which is able to add a new redundant group to the object. 

    None of the mutating methods on the object will ever mutate the antpos, only the
    redundant baseline group lists. The idea here is that the antpos are meant to be
    *fixed* to the full array antpos, but you may consider subsets of the baselines.


    Parameters
    ----------
    antpos : dict[int, np.ndarray]
        Dictionary mapping antenna number to antenna position in ENU frame.
        Not required if ``red_list`` is provided, but highly encouraged even if so.
    bl_error_tol : float
        Baseline length error tolerance in meters.
    include_autos : bool
        Whether to include autocorrelations in the redundant baseline groups.
    red_list : list[list[tuple[int, int]]]
        List of redundant baseline groups. Each group is a list of antenna pairs.

    Examples
    --------
    You can create a RedundantGroups object from a list of redundant baseline groups::

        >>> from hera_cal.red_groups import RedundantGroups
        >>> reds = [[(0, 1), (1, 2), (2, 3)], [(0, 2), (1, 3), (2, 4)]]
        >>> rg = RedundantGroups(red_list=reds)

    You can also create a RedundantGroups object from a dictionary of antenna positions
    (in fact, we encourage you to do it this way if possible)::

        >>> import numpy as np
        >>> antpos = {
                0: np.array([0, 0, 0]), 
                1: np.array([0, 0, 1]), 
                2: np.array([0, 0, 2]), 
                3: np.array([0, 0, 3]), 
                4: np.array([0, 0, 4])
            }
        >>> rg = RedundantGroups.from_antpos(antpos=antpos)

    The RedundantGroups object "feels" like a list-of-lists, so it can be iterated over::

        >>> for red in rg:
        ...     for bl in red:
        ...         print(bl)

    You can also index into it and get its length::

        >>> print(rg[0])
        >>> print(len(rg))

    You can also "append" to it, which will add a new redundant group to the object::

        >>> rg.append([(0, 4), (1, 5), (2, 6)])

    The RedundantGroups object also "feels" like a dictionary. You can index into it::

        >>> print(rg[(0, 1)])

    Which will return the list of baselines redundant with the antpair provided. You can
    also retrieve the "primary baseline" (the first in the list) for a given baseline::

        >>> print(rg.get_ubl_key((0, 1, 'nn')))


    You may add together two RedundantGroups objects, which will combine their redundant
    baseline groups and their antennas (as found in ``antpos``)::

        >>> rg2 = rg + rg
        >>> rg2.data_bls == rg.data_bls
        True
    
    To filter out baselines for a specific purpose (but keeping the full antpos array)::

        >>> filtered = rg.filtered(bls=[(0, 1), (1, 2), (2, 3)])
    
    The "unique key" returned by the `.get_ubl_key` method can be changed by setting the
    `key_chooser` attribute. This is a callable that takes a list of baselines and returns
    a single baseline from that list. By default, it returns the first baseline in the 
    list. Examples of other useful choices might be the last baseline::

        >>> rg_reverse = attrs.evolve(rg, key_chooser=lambda x: x[-1])
    
    A particularly useful example is to key on the first baseline that appears in a 
    given set of baselines (eg. if you have a data file that doesn't have all the 
    possible baselines in it, and want to be able to key only to baselines in that file).
    Since this is a common use case, we provide a convenience function for it::

        >>> rg_data_keys = rg.keyed_on_bls([(1, 2)])
    
    If you really need to a new object with extra antennas, you can do so by 
    adding two objects, where the second object has an empty `red_list`::

        >>> rg2 = rg + RedundantGroups(
                antpos={5: np.array([0, 0, 5]), 6: np.array([0, 0, 6])},
                red_list=[]
            )
    """

    _red_list: list[list[AntPair | Baseline]] = attrs.field(converter=_convert_red_list)
    _antpos: frozendict[int, np.ndarray] | None = attrs.field(
        default=None, 
        converter=attrs.converters.optional(frozendict)
    )
    key_chooser: callable = attrs.field(default=lambda x: x[0])

    def _mutator(meth):
        """Decorator to clear the cache when a mutator method is called."""
        @wraps(meth)
        def wrapper(self, *args, **kwargs):
            out = meth(self, *args, **kwargs)
            self.clear_cache()

        return wrapper
     
    @property
    def antpos(self) -> frozendict | None:
        """The antenna position dictionary, if it exists."""
        # We make this a property with the actual data as a private attribute
        # to make it less likely that a user will accidentally mutate it.
        return self._antpos
         
    @classmethod
    def from_antpos(
        cls, 
        antpos: dict[int, np.ndarray], 
        pols: Sequence[str] = (), 
        include_autos: bool = True, 
        bl_error_tol: float = 1.0,
        pol_mode: str='1pol',
        **kwargs
    ) -> RedundantGroups:
        """Create a RedundantGroups object from an antpos dictionary."""
        if pols:
            reds = redcal.get_reds(
                antpos, 
                pols=pols, # we end up removing the pols anyway.
                bl_error_tol=bl_error_tol, 
                include_autos=include_autos,
                pol_mode=pol_mode,
            )
        else:
            reds = redcal.get_pos_reds(
                antpos, 
                bl_error_tol=bl_error_tol, 
                include_autos=include_autos
            )
        return cls(antpos=antpos, red_list=reds, **kwargs)
    
    @cached_property
    def data_ants(self) -> frozenset[int]:
        """The list of antennas that have are in the baseline groups."""    
        out = set()
        for red in self._red_list:
            for ap in red:
                out.add(ap[0])
                out.add(ap[1])
        return out

    @cached_property
    def data_bls(self) -> frozenset[Baseline | AntPair]:
        """The list of baselines in the baseline groups."""
        return set(self._bl_to_red_map.keys())

    @cached_property
    def _red_key_to_bls_map(self) -> dict[AntPair, tuple[AntPair]]:
        out = {}
        for red in self._red_list:
            out[self.key_chooser(red)] = red
            out[reverse_bl(self.key_chooser(red))] = tuple(reverse_bl(r) for r in red)
        return out
    
    @cached_property
    def _bl_to_red_map(self):
        out = {}
        for red in self._red_list:
            rev = reverse_bl(self.key_chooser(red))
            for bl in red:
                out[bl] = self.key_chooser(red)
                out[reverse_bl(bl)] = rev 
        return out
    
    def get_ubl_key(self, key: AntPair | Baseline) -> AntPair | Baseline:
        '''Returns the key used interally denote the data stored. Useful for del'''
        return self._bl_to_red_map[key]
        
    def get_red(self, key: AntPair | Baseline) -> tuple[AntPair | Baseline]:
        '''Returns the tuple of baselines redundant with this key.'''
        return self._red_key_to_bls_map[self.get_ubl_key(key)]
        
    def __contains__(self, key: AntPair | Baseline):
        '''Returns true if the baseline redundant with the key is in the data.'''
        return key in self._bl_to_red_map
    
    def __add__(self, other: RedundantGroups):
        """Add two RedundantGroups objects together.
        
        This will return a *new* RedundantGroups object that contains all the baselines
        from both objects. This cannot be done if one of the two doesn't have antpos.
        """
        if not isinstance(other, RedundantGroups):
            raise TypeError("can only add RedundantGroups object to RedundantGroups object")
        
        if (self.antpos is None) != (other.antpos is None):
            raise ValueError(
                "can't add RedundantGroups objects with one having antpos and the other not"
            )
        
        if self.antpos is None:
            return self.append(other._red_list, inplace=False)
        else:
            new_antpos = {**self.antpos, **other.antpos}
            new = attrs.evolve(self, antpos=new_antpos)
            return new.append(other._red_list, inplace=False)            
        
    def append(self, red: Sequence[AntPair | Baseline], inplace: bool=True, pos: int | None = None) -> None:
        """In-place append a new redundant group to the list of redundant groups.
        
        This maintains the list-of-lists duck-typing of the redundant groups.
        """
        obj = self if inplace else copy.deepcopy(self)

        if self.antpos is not None:
            # Ensure that all the ants in the new reds exist in the antpos
            for bl in red:
                if bl[0] not in self.antpos:
                    raise ValueError(f"Antenna {bl[0]} not in antpos (valid ants: {self._bl_to_red_map.keys()}).")
                if bl[1] not in self.antpos:
                    raise ValueError(f"Antenna {bl[1]} not in antpos (valid ants: {self._bl_to_red_map.keys()}).")

        ubls = {r: self.get_ubl_key(r) for r in red if r in self}

        if ubls:

            if len(set(ubls.values())) > 1:
                invubl = {v: k for k, v in ubls.items()}
                raise ValueError(
                    "Attempting to add a new redundant group where some baselines "
                    "already exist in differing groups. Got the following baselines "
                    f"mapping to different groups: {dict(zip(invubl.values(), invubl.keys()))}"
                )
            if len(ubls) == len(red):
                if inplace:
                    return
                else:
                    return self
                
            # Now we're gaurenteed that we have some baselines that are new, and that
            # they all map to the same already-existing group.
            ublkey = next(iter(ubls.values()))
            obj[ublkey] = sorted(set(self[ublkey] + red))

        else:
            # We're adding a completely new group
            if pos is None:
                obj._red_list.append(red)
            else:
                obj._red_list.insert(pos, red)

        # Reset the maps
        obj.clear_cache()

        if not inplace:
            return obj

    def insert(self, pos: int, red: Sequence[AntPair | Baseline], inplace: bool=True) -> None:
        """In-place insert a new redundant group to the list of redundant groups.
        
        This maintains the list-of-lists duck-typing of the redundant groups.
        """
        return self.append(red, inplace=inplace, pos=pos)
    
    def __len__(self):
        return len(self._red_list)

    def __iter__(self):
        return iter(self._red_list)
    
    def __getitem__(self, key: int | AntPair | Baseline) -> list[AntPair | Baseline]:
        if isinstance(key, int):
            return self._red_list[key]
        else:
            return self.get_red(key)
        
    def __setitem__(self, key: int | AntPair | Baseline, value: list[AntPair | Baseline]):
        if isinstance(key, int):
            self._red_list[key] = value
        elif key in self:
            ukey = self.get_ubl_key(key)
            self._red_list = [value if red[0]==ukey else red for red in self._red_list]
        else:
            # We're setting a new redundant group
            self.append(value)

        # Reset the maps
        self.clear_cache()

    def index(self, key: AntPair | Baseline) -> int:
        """Return the index of a redundant group."""
        return list(self._red_key_to_bls_map.keys()).index(self.get_ubl_key(key))
    
    def filtered(self, **kwargs) -> RedundantGroups:
        """Return a new RedundantGroups object with baselines filtered out.

        """
        new_reds = redcal.filter_reds(copy.deepcopy(self), **kwargs)
        print("OLD REDS: ", self._red_list)
        print(kwargs)
        print("NEW REDS: ", new_reds)
        return attrs.evolve(self, red_list=new_reds)
    
    def extend(self, reds: Sequence[Sequence[AntPair | Baseline]], inplace: bool = True) -> None:
        """In-place extend the list of redundant groups with another list of redundant groups.
        
        This maintains the list-of-lists duck-typing of the redundant groups.
        """
        for red in reds:
            self.append(red, inplace)
    
    def sort(self):
        """Sort the redundant groups in-place."""
        self._red_list.sort()
        self.clear_cache()

    def clear_cache(self):
        """Clear the cached maps."""
        for att in (
            self._bl_to_red_map,
            self._red_key_to_bls_map,
            self.data_ants,
            self.data_bls,
        ):
            try:
                del att
            except AttributeError:
                pass

    def get_full_redundancies(
        self, pols: Sequence[str] | None = None, 
        include_autos: bool = True, 
        bl_error_tol: float = 1.0,
        pol_mode: str='1pol',
    ) -> RedundantGroups:
        """Create a RedundantGroups object from an antpos dictionary."""
        if pols is None:
            if len(next(iter(self._bl_to_red_map.values()))) == 2:
                pols = ()
            else:
                pols = tuple({bl[2] for bl in self._red_key_to_bls_map.keys()})

        if pols:
            reds = redcal.get_reds(
                self.antpos, 
                pols=pols, # we end up removing the pols anyway.
                bl_error_tol=bl_error_tol, 
                include_autos=include_autos,
                pol_mode=pol_mode,
            )
        else:
            reds = redcal.get_pos_reds(
                self.antpos, 
                bl_error_tol=bl_error_tol, 
                include_autos=include_autos
            )

        return attrs.evolve(self, red_list=reds)
    
    def keyed_on_bls(self, bls: Sequence[AntPair | Baseline]) -> RedundantGroups:
        """Return a new RedundantGroups object keyed on the given baselines."""
        out = attrs.evolve(self, key_chooser=partial(_choose_key, bls=bls, key_chooser=self.key_chooser))
        print("_bl_to_red_map:", "_bl_to_red_map" in out.__dict__)
        return out

def _choose_key(red, bls, key_chooser):
            
    filtered_red = [bl for bl in red if bl in bls]
    #print("in key:", red, bls, filtered_red)
    if filtered_red:
        return key_chooser(filtered_red)
    else:
        return key_chooser(red)
