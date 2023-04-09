"""Module with classes and functions for dealing with redundant baseline groups."""
from __future__ import annotations

import attrs
from .utils import reverse_bl
from . import redcal
import numpy as np

from typing import Sequence

AntPair = tuple[int, int]
Baseline = tuple[int, int, str]

def _convert_red_list(red_list: Sequence[Sequence[AntPair | Baseline]]) -> tuple[tuple[AntPair]]:
    return tuple(tuple(r[:2] for r in red) for red in red_list)

@attrs.define
class RedundantGroups:
    """Redundant baseline group manager.

    Parameters
    ----------
    antpos : dict[int, np.ndarray]
    """

    antpos: dict[int, np.ndarray] | None = attrs.field(default=None)
    bl_error_tol: float = attrs.field(default=1.0)

    _red_list: list[list[AntPair]] = attrs.field(repr=False, converter=_convert_red_list)
    _red_key_to_bls_map: dict[AntPair, tuple[AntPair]] = attrs.field(repr=False)
    _bl_to_red_map: dict[AntPair, AntPair] = attrs.field(repr=False)
            
    @_red_list.default
    def _red_list_default(self) -> tuple[tuple[AntPair]]:
        if self.antpos is None:
            raise ValueError("either antpos or red_list must be provided")
        reds = redcal.get_reds(self.antpos, pols=('nn',), bl_error_tol=self.bl_error_tol, include_autos=True)
        return tuple(tuple(r[:2] for r in red) for red in reds)
    
    @_red_key_to_bls_map.default
    def _red_key_to_bls_map_default(self) -> dict[AntPair, tuple[AntPair]]:
        out = {}
        for red in self._red_list:
            out[red[0]] = red 
            out[reverse_bl(red[0])] = tuple(reverse_bl(r) for r in red)
        return out
    
    @_bl_to_red_map.default
    def _bl_to_red_map_default(self):
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
    
    def with_new_red(self, red: Sequence[AntPair | Baseline]):
        # First check if we're actually adding anything new
        red = tuple(r[:2] for r in red)

        if any(r in self for r in red):
            ubls = {self.get_ubl_key(r): r for r in red if r in self}
            if len(ubls) > 1:
                raise ValueError(
                    "Attempting to add a new redundant group where some baselines "
                    "already exist in differing groups. Got the following baselines "
                    f"mapping to different groups: {dict(zip(ubls.values(), ubls.keys()))}"
                )
            if all(r in self for r in red):
                return self
            else:
                for r in red:
                    if r in self:
                        this_ublk = self.get_ubl_key(r)
                        break
                newreds = [
                    r if self.get_ubl_key(r[0]) != this_ublk else self.get_red(r) + tuple(red) 
                    for r in self._red_list if r not in self
                ]
                return RedundantGroups(
                    antpos = self.antpos,
                    red_list = newreds,
                    bl_error_tol=self.bl_error_tol
                )
        else:
            return RedundantGroups(
                antpos = self.antpos,
                bl_error_tol = self.bl_error_tol,
                red_list = self._red_list + (red,)
            )

    def __len__(self):
        return len(self._red_list)

    def __iter__(self):
        return iter(self._red_list)