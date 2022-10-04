from . import utils
from . import redcal

import numpy as np
from copy import deepcopy
import astropy.constants as const
from collections import defaultdict
from scipy.cluster.hierarchy import fclusterdata

SPEED_OF_LIGHT = const.c.si.value

def is_same_orientation(bl1, bl2, antpos, blvec_error_tol=1e-3):
    """
    Determine whether or not two baselines have the same orientation

    Parameters:
    ----------
    bl1 : tuple
        Tuple of antenna indices and polarizations of the first baseline
    bl2 : tuple
        Tuple of antenna indices and polarizations of the second baseline
    freqs : np.ndarray
        Array of frequencies found in the data in units of Hz
    antpos : dict
        Antenna positions in the form {ant_index: np.array([x,y,z])}.
    blvec_error_tol : float, default=1e-3
        Largest allowable euclidean distance the first unit baseline vector can be away from
        the second

    Returns:
        Boolean value determining whether or not the baselines are frequency
        redundant
    """
    # Split baselines in component antennas
    ant1, ant2, _ = bl1
    ant3, ant4, _ = bl2

    # Get baseline vectors
    blvec1 = antpos[ant1] - antpos[ant2]
    blvec2 = antpos[ant3] - antpos[ant4]

    # Check headings
    norm_vec1 = blvec1 / np.linalg.norm(blvec1)
    norm_vec2 = blvec2 / np.linalg.norm(blvec2)

    return np.isclose(np.linalg.norm(norm_vec1 - norm_vec2), 0, rtol=blvec_error_tol)

def is_frequency_redundant(bl1, bl2, freqs, antpos, blvec_error_tol=1e-3):
    """
    Determine whether or not two baselines are frequency redundant. Checks that
    both baselines have the same heading, polarization, and have overlapping uv-modes

    Parameters:
    ----------
    bl1 : tuple
        Tuple of antenna indices and polarizations of the first baseline
    bl2 : tuple
        Tuple of antenna indices and polarizations of the second baseline
    freqs : np.ndarray
        Array of frequencies found in the data in units of Hz
    antpos : dict
        Antenna positions in the form {ant_index: np.array([x,y,z])}.
    blvec_error_tol : float, default=1e-3
        Largest allowable euclidean distance the first unit baseline vector can be away from
        the second

    Returns:
        Boolean value determining whether or not the baselines are frequency
        redundant

    """
    # Split baselines in component antennas
    ant1, ant2, pol1 = bl1
    ant3, ant4, pol2 = bl2

    # Check polarization match
    if pol1 != pol2:
        return False

    # Get baseline vectors
    blvec1 = antpos[ant1] - antpos[ant2]
    blvec2 = antpos[ant3] - antpos[ant4]

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
    if not (cond1 or cond2):
        return False

    # Last step - return whether or not baselines are in the same orientation
    return is_same_orientation(bl1, bl2, antpos, blvec_error_tol=blvec_error_tol)

def get_u_bounds(radial_reds, antpos, freqs):
    """
    Calculates the magnitude of the minimum and maximum u-modes values of the radial redundant group
    given an array of frequency values

    Parameters:
    ----------
    radial_reds: list of lists of baseline tuples (or FrequencyRedundancy)
        List of lists of radially redundant groups of baselines
    antpos: dict
        Antenna positions in the form {ant_index: np.array([x,y,z])}.
    freqs: np.ndarray
        Array of frequencies found in the data in units of Hz

    Returns:
    -------
    ubounds: tuple
        Tuple of the magnitude minimum and maximum u-modes sampled by this baseline group
    """
    ubounds = []
    for group in radial_reds:
        umodes = [np.linalg.norm(antpos[ant1] - antpos[ant2]) for (ant1, ant2, pol) in group]
        umin = np.min(umodes) * freqs.min() / SPEED_OF_LIGHT
        umax = np.max(umodes) * freqs.max() / SPEED_OF_LIGHT
        ubounds.append((umin, umax))

    return ubounds
            

def get_unique_orientations(
    antpos, reds, min_ubl_per_orient=1, blvec_error_tol=1e-3, bl_error_tol=1.0,
):
    """
    Sort baselines into groups with the same radial heading. These groups of baselines are potentially
    frequency redundant in a similar way to redcal.get_reds does. Returns a list of RadialRedundantGroup objects

    Parameters:
    ----------
    antpos : dict
        Antenna positions in the form {ant_index: np.array([x,y,z])}.
    reds : list of lists
        List of lists of spatially redundant baselines in the array. Can be found using redcal.get_reds
    min_ubl_per_orient : int, default=1
        Minimum number of baselines per unique orientation
    blvec_error_tol : float, default=1e-3
        Largest allowable euclidean distance a unit baseline vector can be away from an existing
        cluster to be considered a unique orientation. See "fclusterdata" for more details.
    bl_error_tol: float, default=1.0
        The largest allowable difference between baselines in a redundant group
        (in the same units as antpos). Normally, this is up to 4x the largest antenna position error.

    Returns:
    -------
    uors : list of lists of tuples
        List of list of tuples that are considered to be radially redundant
    """
    # Get polarizations from reds
    pols = list(set([rdgrp[0][-1] for rdgrp in reds]))

    _uors = {}
    for pol in pols:
        ubl_pairs = [red[0] for red in reds if red[0][-1] == pol]

        # Compute normalized baseline vectors
        normalized_vecs = []
        for (ant1, ant2, pol) in ubl_pairs:
            normalized_vecs.append(
                (antpos[ant2] - antpos[ant1])
                / np.linalg.norm(antpos[ant2] - antpos[ant1])
            )

        # Cluster orientations
        clusters = fclusterdata(normalized_vecs, blvec_error_tol, criterion="distance")
        uors = [[] for i in range(np.max(clusters))]

        for cluster, bl in zip(clusters, ubl_pairs):
            uors[cluster - 1].append(bl)

        uors = sorted(uors, key=len, reverse=True)

        # Find clusters with headings anti-parallel to others
        for group in uors:
            ant1, ant2, pol = group[0]
            vec = (antpos[ant2] - antpos[ant1]) / np.linalg.norm(
                antpos[ant2] - antpos[ant1]
            )
            vec = np.array(vec / blvec_error_tol, dtype=int)
            if tuple(-vec) + (pol,) in _uors:
                _uors[tuple(-vec) + (pol,)] += [utils.reverse_bl(bls) for bls in group]
            else:
                _uors[tuple(vec) + (pol,)] = group

    # Convert lists to RadialRedundantGroup objects
    uors = [_uors[key] for key in _uors if len(_uors[key]) >= min_ubl_per_orient]
    uors = sorted(uors, key=len, reverse=True)
    return uors


class RadialRedundantGroup:
    """List-like object that holds tuples of baselines that are assumed to be
    radially redundant (have the same heading). In addition to supporting list
    like behavior, this object also gets the minimum and maximum u-mode magnitude
    of the radially redundant group and filter the group based on a number of factors.
    """

    def __init__(self, baselines, antpos, bl_unit_vec=None, pol=None):
        """
        Create a RadialRedundantGroup object from a list baselines assumed
        to be radially redundant

        Parameters:
        ----------
        baselines : list of tuples
            List of baseline tuples
        antpos : dict
            Antenna positions in the form {ant_index: np.array([x,y,z])}.
        bl_unit_vec : np.ndarray
            Normalized baseline vector for the radially redundant group. If one is not
            provided, it will be estimated from the antenna positions
        pol : str
            Polarization of the baseline group
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

        if bl_unit_vec is None:
            ant1, ant2, pol = baselines[0]
            self.blvec = (antpos[ant2] - antpos[ant1]) / np.linalg.norm(
                antpos[ant2] - antpos[ant1]
            )
        else:
            self.bl_unit_vec = bl_unit_vec

        # Store baseline lengths
        baseline_lengths = []
        for baseline in baselines:
            ant1, ant2, pol = baseline
            baseline_lengths.append(np.linalg.norm(antpos[ant2] - antpos[ant1]))

    def filter_group(
        self,
        bls=None,
        ex_bls=None,
        ants=None,
        ex_ants=None,
        min_bl_cut=None,
        max_bl_cut=None,
    ):
        """
        Filter radially redundant group to include/exclude the specified bls, antennas. and polarizations.
        Arguments are evaluated, in order of increasing precedence: (pols, ex_pols, bls, ex_bls, ants, ex_ants,
        min_bl_cut, max_bl_cut).

        Parameters:
        ----------
        bls : list of tuples, default=None
            baselines to include. Baselines of the form (i,j,pol) include that specific
            visibility.  Baselines of the form (i,j) are broadcast across all polarizations present in reds.
        ex_bls : list of tuples, default=None
            same as bls, but excludes baselines.
        ants : list of tuples, default=None
            antennas to include. Only baselines where both antenna indices are in ants
            are included.  Antennas of the form (i,pol) include that antenna/pol. Antennas of the form i are
            broadcast across all polarizations present in reds.
        ex_ants : list of tuples, default=None
            same as ants, but excludes antennas.
        min_bl_cut:
            Cut baselines in the radially redundant group with lengths less than min_bl_cut
        max_bl_cut:
            Cut baselines in the radially redundant group with lengths less than min_bl_cut
        """
        _baselines = redcal.filter_reds(
            [self._baselines],
            bls=bls,
            ex_bls=ex_bls,
            ants=ants,
            ex_ants=ex_ants,
        )
        if len(_baselines) == 0:
            self._baselines = []
            self.baseline_lengths = []
        else:
            new_bls = []
            new_bls_lengths = []
            for bl in _baselines[0]:
                index = self._baselines.index(bl)
                if min_bl_cut is not None and self.baseline_lengths[index] < min_bl_cut:
                    continue
                if max_bl_cut is not None and self.baseline_lengths[index] > max_bl_cut:
                    continue
                new_bls.append(bl)
                new_bls_lengths.append(self.baseline_lengths[index])

            self._baselines = new_bls
            self.baseline_lengths = new_bls_lengths

    def __iter__(self):
        """Iterate through baselines in the radially redundant group"""
        return iter(self._baselines)

    def __len__(self):
        """Return the length of the baselines list"""
        return len(self._baselines)

    def __getitem__(self, index):
        """Get the baseline at the chosen index"""
        return self._baselines[index]

    def sort(self):
        """Sort baselines list by baseline length
        """
        self._baselines = [self._baselines[idx] for idx in np.argsort(self.baseline_lengths)]
        self.baseline_lengths = [
            self.baseline_lengths[idx] for idx in np.argsort(self.baseline_lengths)
        ]


class FrequencyRedundancy:
    """List-like object that contains groups RadialRedundantGroup objects.
    Functions similarly to the output of redcal.get_reds for frequency redundant
    calibration. In addition to mimicking list functionality, this object also filters
    radially redundant groups based on a number of factors, can get specific polarizations, and
    radially redundant and spatially redundant groups by baseline key.
    """

    def __init__(
        self, antpos, reds=None, blvec_error_tol=1e-3, pols=["nn"], bl_error_tol=1.0
    ):
        """
        Parameters:
        ----------
        antpos : dict
            Antenna positions in the form {ant_index: np.array([x,y,z])}.
        reds : list of list
            List of lists of baseline keys. Can be determined using redcal.get_reds
        pols : list of strs
            List of polarization strings to be used in the frequency redundant group
        """
        self.antpos = antpos

        if reds is None:
            reds = redcal.get_reds(antpos, pols=pols, bl_error_tol=bl_error_tol)

        self._radial_groups = get_unique_orientations(
            antpos, reds=reds, pols=pols, blvec_error_tol=blvec_error_tol
        )

        # Map baseline key to baseline length
        self.baseline_lengths = {}
        for group in self._radial_groups:
            for bl in group:
                ant1, ant2, _ = bl
                blmag = np.linalg.norm(self.antpos[ant2] - self.antpos[ant1])
                self.baseline_lengths[bl] = blmag

        # Spatial reds
        self._mapped_reds = {red[0]: red for red in reds}
        self._baseline_to_red_key = {}
        for red in reds:
            for bl in red:
                self._baseline_to_red_key[bl] = red[0]

        # Spectral reds
        

    def get_radial_group(self, key):
        """
        Get baselines with the same radial heading as a given baseline

        Parameters:
        ----------
        key : tuple
            Baseline key of type (ant1, ant2, pol)

        Returns:
        -------
        group : list of tuples
            List of baseline tuples that have the same radial headings

        """
        # Identify headings
        for group_key in self._mapped_reds:
            if key == group_key or key in self._mapped_reds[group_key]:
                key = group_key
                break

        for group in self._radial_groups:
            if key in group:
                return group

    def get_redundant_group(self, key):
        """
        Get a list of baseline that are spatially redundant with the input baseline

        Parameters:
        ----------
        key: tuple
            Baseline key with of type (ant1, ant2, pol)

        Returns:
        -------
        group: list of tuples
            Return baseline tuples that are spatially redundant
        """
        if key in self._baseline_to_red_key:
            group_key = self._baseline_to_red_key[key]
        elif utils.reverse_bl(key) in self._baseline_to_red_key:
            group_key = utils.reverse_bl(self._baseline_to_red_key[utils.reverse_bl(key)])
        else:
            raise KeyError(
                f"Baseline {key} is not in the group of spatial redundancies"
            )

        if group_key in self._mapped_reds:
            return self._mapped_reds[group_key]
        else:
            return [utils.reverse_bl(bl) for bl in self._mapped_reds[utils.reverse_bl(group_key)]]

    def get_pol(self, pol):
        """Get all radially redundant groups with a given polarization"""
        return [group for group in self if group[0][-1] == pol]

    def filter_radial_groups(
        self,
        min_nbls=1,
        min_bl_cut=None,
        max_bl_cut=None,
    ):
        """
        Filter each radially redundant group to include/exclude the specified bls, antennas. and polarizations.
        Arguments are evaluated, in order of increasing precedence: (pols, ex_pols, bls, ex_bls, ants, ex_ants,
        min_bl_cut, max_bl_cut, min_nbls).

        Parameters:
        ----------
        min_nbls : int, default=1
            Minimum number of baselines allowed in a radially redundant group
        pols : list of strings
            polarizations to include in reds. e.g. ['nn','ee','ne','en']
        ex_pols : list of strings
            same as pols, but excludes polarizations.
        min_bl_cut:
            Cut baselines in the radially redundant group with lengths less than min_bl_cut
        max_bl_cut:
            Cut baselines in the radially redundant group with lengths less than min_bl_cut
        """
        # Filter radially redundant group
        radial_reds = []
        for group in self._radial_groups:
            filtered_group = []
            for bl in group:
                if (max_bl_cut is None or self.baseline_lengths[bl] < max_bl_cut) and (min_bl_cut is None or self._baseline_lengths[bl] > min_bl_cut):
                    filtered_group.append(bl)
                else:
                    self.baseline_lengths.pop(bl)

            # Identify groups with fewer than min_nbls baselines
            if len(filtered_group) > min_nbls:
                radial_reds.append(filtered_group)

        # Remove filtered groups from baseline lengths and reds dictionaries
        self._radial_groups = radial_reds

    def __len__(self):
        """Get number of frequency redundant groups"""
        return len(self._radial_groups)

    def __getitem__(self, index):
        """Get RadialRedundantGroup object from list of unique orientations"""
        return self._radial_groups[index]

    def __setitem__(self, index, value):
        """Set value of index in _radial_groups"""
        self._radial_groups[index] = value
    
    def __iter__(self):
        """Iterates through the list of redundant groups"""
        return iter(self._radial_groups)

    
