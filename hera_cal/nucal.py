from . import utils
from . import redcal

import numpy as np
from copy import deepcopy
import astropy.constants as const
from collections import defaultdict
from scipy.cluster.hierarchy import fclusterdata

SPEED_OF_LIGHT = const.c.si.value

class RadialRedundantGroup:
    """
    """
    def __init__(self, baselines, antpos, blvec=None, pol=None):
        """
        """
        _baselines = deepcopy(baselines)

        # Attach polarization and normalized vector to radial redundant group
        if pol is None:
            pols = list(set([bl[2] for bl in baselines]))
            if len(pols) > 1:
                raise ValueError(
                    f"Multiple polarizations are in your radially redundant group: {pols}"
                )
            else:
                self.pol = pols[0]
        else:
            self.pol = pol
            
        if blvec is None:
            ant1, ant2, pol = baselines[0]
            self.blvec = (antpos[ant2] - antpos[ant1]) / np.linalg.norm(antpos[ant2] - antpos[ant1])
        else:
            self.blvec = blvec
        
        # Store baseline lengths
        baseline_lengths = []
        for baseline in baselines:
            ant1, ant2, pol = baseline
            baseline_lengths.append(np.linalg.norm(antpos[ant2] - antpos[ant1]))

        # Sort baselines list by baseline length
        self._baselines = [_baselines[idx] for idx in np.argsort(baseline_lengths)]
        self.baseline_lengths = [
            baseline_lengths[idx] for idx in np.argsort(baseline_lengths)
        ]

    def get_u_bounds(self, freqs):
        """
        Calculates the magnitude of the minimum and maximum u-modes values of the radial redundant group 
        given an array of frequency values

        Parameters:
        ----------
            freqs: np.ndarray
                Array of frequencies found in the data in units of Hz

        Returns:
            ubounds: tuple
                Tuple of the magnitude minimum and maximum u-modes sampled by this baseline group
        """
        umin = freqs.min() / 2.998e8 * np.min(self.baseline_lengths)
        umax = freqs.max() / 2.998e8 * np.max(self.baseline_lengths)
        return (umin, umax)

    def filter_group(
        self,
        bls=None,
        ex_bls=None,
        ants=None,
        ex_ants=None,
        ubls=None,
        ex_ubls=None,
        pols=None,
        ex_pols=None,
        antpos=None,
        min_bl_cut=None,
        max_bl_cut=None,
    ):
        """
        """
        _baselines = redcal.filter_reds(
            [self._baselines],
            bls=bls,
            ex_bls=ex_bls,
            ants=ants,
            ex_ants=ex_ants,
            ubls=ubls,
            ex_ubls=ex_ubls,
            pols=pols,
            ex_pols=ex_pols,
        )
        if len(_baselines) == 0:
            self._baselines = []
            self.baseline_lengths = []
        else:
            new_bls = []
            new_bls_lengths = []
            for bls in _baselines[0]:
                index = self._baselines.index(bls)
                if min_bl_cut is not None and self.baseline_lengths[index] < min_bl_cut:
                    continue
                if max_bl_cut is not None and self.baseline_lengths[index] > max_bl_cut:
                    continue
                new_bls.append(bls)
                new_bls_lengths.append(self.baseline_lengths[index])

            self._baselines = new_bls
            self.baseline_lengths = new_bls_lengths

    def __iter__(self):
        """Iterate through baselines in the radially redundant group
        """
        return iter(self._baselines)

    def __len__(self):
        """Return the length of the baselines list
        """
        return len(self._baselines)

    def __getitem__(self, index):
        """Get the baseline at the chosen index
        """
        return self._baselines[index]

def is_frequency_redundant(bl1, bl2, freqs, antpos, blvec_error_tol=1e-9):
    """
    Determine whether or not two baselines are frequency redundant. Checks that
    both baselines have the same heading, polarization, and have overlapping uv-modes

    Parameters:
    ----------
    bl1 : tuple
        pass
    bl2 : tuple
        pass
    freqs : np.ndarray
        pass
    antpos : dict
        pass
    blvec_error_tol : float, default=1e-9
        pass

    Returns:
        Boolean value determining whether or not the baselines are frequency
        redundant

    """
    # Split baselines in component antennas
    _ant1, _ant2, _pol = bl1
    ant1, ant2, pol = bl2

    # Get baseline vectors
    blvec1 = antpos[_ant2] - antpos[_ant1]
    blvec2 = antpos[ant2] - antpos[ant1]

    # Check umode overlap
    blmag1 = np.linalg.norm(blvec1)
    blmag2 = np.linalg.norm(blvec2)
    cond1 = (
        blmag1 * freqs.min() <= blmag2 * freqs.max()
        and blmag1 * freqs.max() >= blmag2 * freqs.max()
    )
    cond2 = (
        blmag1 * freqs.min() <= blmag2 * freqs.min()
        and blmag1 * freqs.max() >= blmag2 * freqs.min()
    )

    # Check polarization match
    cond3 = _pol == pol

    # Check headings
    norm_vec1 = blvec1 / np.linalg.norm(blvec1)
    norm_vec2 = blvec2 / np.linalg.norm(blvec2)
    clusters = fclusterdata(
        np.array([norm_vec1, norm_vec2]), blvec_error_tol, criterion="distance"
    )
    cond4 = clusters[0] == clusters[1]

    return (cond1 or cond2) and cond3 and cond4