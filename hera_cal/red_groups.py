"""Module with classes and functions for dealing with redundant baseline groups."""
from __future__ import annotations

import attrs
from .utils import reverse_bl
from . import redcal
import numpy as np
from functools import cached_property

from typing import Sequence

AntPair = tuple[int, int]
Baseline = tuple[int, int, str]

def _convert_red_list(red_list: Sequence[Sequence[AntPair | Baseline]]) -> tuple[tuple[AntPair]]:
    return [[r[:2] for r in red] for red in red_list]

@attrs.define(frozen=False)
class RedundantGroups:
    """Redundant baseline group manager.

    This class is used to manage redundant baseline groups. It is (roughly)
    API-compatible with the list-of-lists-of-tuples format returned by, e.g., 
    :func:`hera_cal.redcal.get_reds`, and in the future that may simply return
    one of these objects.

    To keep this compatibility, this object is mutable, but we absolutely discourage
    directly mutating it. Instead, use the methods provided, including the `.append`
    method, which is able to add a new redundant group to the object. 

    In the future, we may disable direct mutation of this object.

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
        >>> rg = RedundantGroups(antpos=antpos)

    The RedundantGroups object "feels" like a list-of-lists, so it can be iterated over::

        >>> for red in rg:
        ...     for antpair in red:
        ...         print(antpair)

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

    Both the dictionary-like indexing and the `.get_ubl_key` method are able to handle
    either antenna pairs or baselines, and will return the same type of key that was
    provided.

    Finally, you can form new RedundantGroups objects from existing ones, appending
    new antenna position::

        >>> rg2 = rg.with_new_antpos({5: np.array([0, 0, 5]), 6: np.array([0, 0, 6])})

    """

    antpos: dict[int, np.ndarray] | None = attrs.field(default=None)
    bl_error_tol: float = attrs.field(default=1.0)
    include_autos: bool = attrs.field(default=True, converter=bool)

    _red_list: list[list[AntPair]] = attrs.field(repr=False, converter=_convert_red_list)
            
    @_red_list.default
    def _red_list_default(self) -> list[list[AntPair]]:
        if self.antpos is None:
            raise ValueError("either antpos or red_list must be provided")
        reds = redcal.get_reds(
            self.antpos, 
            pols=('nn',), # we end up removing the pols anyway.
            bl_error_tol=self.bl_error_tol, 
            include_autos=self.include_autos
        )
        return [[r[:2] for r in red] for red in reds]
    
    @cached_property
    def _red_key_to_bls_map(self) -> dict[AntPair, tuple[AntPair]]:
        out = {}
        for red in self._red_list:
            out[red[0]] = red 
            out[reverse_bl(red[0])] = tuple(reverse_bl(r) for r in red)
        return out
    
    @cached_property
    def _bl_to_red_map(self):
        out = {}
        for red in self._red_list:
            rev = reverse_bl(red[0])
            for bl in red:
                out[bl] = red[0]
                out[reverse_bl(bl)] = rev 
        return out
    
    def get_ubl_key(self, key: AntPair | Baseline) -> AntPair | Baseline:
        '''Returns the key used interally denote the data stored. Useful for del'''
        if len(key) == 2:
            return self._bl_to_red_map[key]
        elif len(key) == 3:
            return self._bl_to_red_map[key[:2]] + (key[2],)
        else:
            raise ValueError("key must be a length 2 or 3 tuple")
        
    def get_red(self, key: AntPair | Baseline) -> tuple[AntPair | Baseline]:
        '''Returns the tuple of baselines redundant with this key.'''
        ukey = self.get_ubl_key(key)
        if len(ukey) == 2:
            return self._red_key_to_bls_map[self.get_ubl_key(key)]
        elif len(ukey) == 3:
            return tuple(b + (ukey[2],) for b in self._red_key_to_bls_map[ukey[:2]])
        else:
            raise ValueError("key must be a length 2 or 3 tuple")
        
    def __contains__(self, key: AntPair | Baseline):
        '''Returns true if the baseline redundant with the key is in the data.'''
        if len(key) == 2:
            return key in self._bl_to_red_map
        elif len(key) == 3:
            return key[:2] in self._bl_to_red_map
        else:
            return False
    
    def merge(self, other: RedundantGroups):
        """Merge another RedundantGroups object with this one."""
        if self.bl_error_tol != other.bl_error_tol:
            raise ValueError(
                "the two redundant groups can't be merged as they have different error tols"
            )
        if self.antpos is None or other.antpos is None:
            raise ValueError("can't merge RedundantGroups objects without antpos")
                        
        # TODO: there's almost certainly a quicker way to do this where we use all
        #       the dict info that we have. But this will always work.
        new_antpos = {**self.antpos, **other.antpos}
        return RedundantGroups(
            antpos = new_antpos,
            bl_error_tol = self.bl_error_tol
        )
    
    def with_additional_ants(self, antpos: dict[int, np.ndarray]):
        # First check if we're actually adding anything.
        if all(k in self.antpos for k in antpos):
            return self
        if self.antpos is None:
            raise ValueError("can't add antennas to a RedundantGroups object without antpos")
        
        new_antpos = {**self.antpos, **antpos}
        return RedundantGroups(
            antpos = new_antpos,
            bl_error_tol = self.bl_error_tol
        )
    
    def append(self, red: Sequence[AntPair | Baseline]) -> None:
        """In-place append a new redundant group to the list of redundant groups.
        
        This maintains the list-of-lists duck-typing of the redundant groups.
        """
        # First check if we're actually adding anything new
        red = [r[:2] for r in red]

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
                # already have all these!
                # Note, we have to do this *after* the above check, because although
                # we have all the baselines, they might not be in the same group, and 
                # we want to raise an error in that case.
                return

            # Now we're gaurenteed that we have some baselines that are new, and that
            # they all map to the same already-existing group.
            ublkey = next(iter(ubls.values()))
            self[ublkey] = sorted(set(self[ublkey] + red))
            
        else:
            # We're adding a completely new group
            self._red_list.append(red)
            
        # Reset the maps
        del self._bl_to_red_map
        del self._red_key_to_bls_map

    def __len__(self):
        return len(self._red_list)

    def __iter__(self):
        return iter(self._red_list)
    
    def __getitem__(self, key: int | AntPair | Baseline) -> list[AntPair | Baseline]:
        if isinstance(key, int):
            return self._red_list[key]
        else:
            return self.get_red(key)
        