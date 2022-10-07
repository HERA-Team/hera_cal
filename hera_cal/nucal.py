from . import utils
from . import redcal

import numpy as np
from copy import deepcopy
import astropy.constants as const
from scipy.cluster.hierarchy import fclusterdata

SPEED_OF_LIGHT = const.c.si.value

def is_same_orientation(bl1, bl2, antpos, blvec_error_tol=1e-4):
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
    blvec_error_tol : float, default=1e-4
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

def is_frequency_redundant(bl1, bl2, freqs, antpos, blvec_error_tol=1e-4):
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
    blvec_error_tol : float, default=1e-4
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
            

def get_unique_orientations(antpos, reds, min_ubl_per_orient=1, blvec_error_tol=1e-4):
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
    blvec_error_tol : float, default=1e-4
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
        ubl_pairs = []

        # Compute normalized baseline vectors
        normalized_vecs = []
        ubl_pairs = []
        for red in reds:
            ant1, ant2, antpol = red[0]
            if antpol == pol:
                vector = (antpos[ant2] - antpos[ant1]) / np.linalg.norm(antpos[ant2] - antpos[ant1])
                # If vector has an EW component less than 0, flip it
                if vector[0] <= 0:
                    normalized_vecs.append(-vector)
                    ubl_pairs.append((ant2, ant1, antpol))

                else:
                    normalized_vecs.append(vector)
                    ubl_pairs.append((ant1, ant2, antpol))

        # Cluster orientations
        clusters = fclusterdata(normalized_vecs, blvec_error_tol, criterion="distance")
        uors = [[] for i in range(np.max(clusters))]

        for cluster, bl in zip(clusters, ubl_pairs):
            uors[cluster - 1].append(bl)

        for group in uors:
            _uors[group[0]] = group

    # Convert lists to RadialRedundantGroup objects
    uors = [_uors[key] for key in _uors if len(_uors[key]) >= min_ubl_per_orient]
    return sorted(uors, key=len, reverse=True)


class FrequencyRedundancy:
    """List-like object that contains lists of baselines that are radially redundant. 
    Functions similarly to the output of redcal.get_reds for frequency redundant
    calibration. In addition to mimicking list functionality, this object also filters
    radially redundant groups by baseline length and number of baselines in a radially redundant
    group radially redundant and spatially redundant groups by baseline key.
    """
    def __init__(
        self, antpos, reds=None, blvec_error_tol=1e-4, pols=["nn"], bl_error_tol=1.0
    ):
        """
        Parameters:
        ----------
        antpos : dict
            Antenna positions in the form {ant_index: np.array([x,y,z])}.
        reds : list of list
            List of lists of baseline keys. Can be determined using redcal.get_reds
        blvec_error_tol : float, default=1e-4
            Largest allowable euclidean distance a unit baseline vector can be away from an existing
            cluster to be considered a unique orientation. See "fclusterdata" for more details.
        pols : list, default=['nn']
            A list of polarizations e.g. ['nn', 'ne', 'en', 'ee']
        bl_error_tol : float, default=1.0
            The largest allowable difference between baselines in a redundant group
            (in the same units as antpos). Normally, this is up to 4x the largest antenna position error.
        """
        self.antpos = antpos
        self.blvec_error_tol = blvec_error_tol

        if reds is None:
            reds = redcal.get_reds(antpos, pols=pols, bl_error_tol=bl_error_tol)

        # Get unique orientations
        self._radial_groups = get_unique_orientations(antpos, reds=reds, blvec_error_tol=blvec_error_tol)

        # Map baseline key to baseline length
        self.baseline_lengths = {}
        for group in self._radial_groups:
            for bl in group:
                ant1, ant2, _ = bl
                blmag = np.linalg.norm(self.antpos[ant2] - self.antpos[ant1])
                self.baseline_lengths[bl] = blmag

        # Map baselines to spatially redundant groups
        self._mapped_reds = {red[0]: red for red in reds}
        self._bl_to_red_key = {}
        for red in reds:
            for bl in red:
                self._bl_to_red_key[bl] = red[0]

        # Map baselines to spectrally redundant groups
        self._reset_mapping_dictionaries()

    def _reset_mapping_dictionaries(self):
        """Map baselines to spectrally redundant groups"""
        self._mapped_spectral_reds = {group[0]: group for group in self._radial_groups}
        self._bl_to_spec_red_key = {}
        for group in self._radial_groups:
            for bl in group:
                self._bl_to_spec_red_key[bl] = group[0]

    def _check_new_group(self, group):
        """Check to make sure a list of baseline tuples is actually radially redundant"""
        if not isinstance(group, list) and not isinstance(group[0], tuple):
            raise TypeError("Input value not list of tuples")

        # Check to see if baselines are in the same orientation and have the same polarization
        if len(group) > 1:
            for bi in range(1, len(group)):
                if not is_same_orientation(group[0], group[bi], self.antpos, blvec_error_tol=self.blvec_error_tol):
                    raise ValueError(f'Baselines {group[0]} and {group[bi]} are not in the same orientation')
                if group[0][-1] != group[bi][-1]:
                    raise ValueError(f'Baselines {group[0]} and {group[bi]} do not have the same polarization')
                
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
        if key in self._bl_to_red_key:
            group_key = self._bl_to_red_key[key]
        elif utils.reverse_bl(key) in self._bl_to_red_key:
            group_key = utils.reverse_bl(self._bl_to_red_key[utils.reverse_bl(key)])
        else:
            raise KeyError(
                f"Baseline {key} is not in the group of spatial redundancies"
            )

        if group_key in self._bl_to_spec_red_key:
            group_key = self._bl_to_spec_red_key[group_key]
        else:
            group_key = utils.reverse_bl(self._bl_to_spec_red_key[utils.reverse_bl(group_key)])
        
        if group_key in self._mapped_spectral_reds:
            return self._mapped_spectral_reds[group_key]
        else:
            return [utils.reverse_bl(bl) for bl in self._mapped_spectral_reds[utils.reverse_bl(group_key)]]


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
        if key in self._bl_to_red_key:
            group_key = self._bl_to_red_key[key]
        elif utils.reverse_bl(key) in self._bl_to_red_key:
            group_key = utils.reverse_bl(self._bl_to_red_key[utils.reverse_bl(key)])
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

    def filter_radial_groups(self, min_nbls=1, min_bl_cut=None, max_bl_cut=None):
        """
        Filter each radially redundant group to include/exclude the baselines based on baseline length.
        Radially redundant groups can also be completely filtered based on the number of baselines in 
        the group.

        Parameters:
        ----------
        min_nbls : int, default=1
            Minimum number of baselines allowed in a radially redundant group
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
                if (max_bl_cut is None or self.baseline_lengths[bl] < max_bl_cut) and (min_bl_cut is None or self.baseline_lengths[bl] > min_bl_cut):
                    filtered_group.append(bl)
                
            # Identify groups with fewer than min_nbls baselines
            if len(filtered_group) > min_nbls:
                radial_reds.append(filtered_group)

        # Remove filtered groups from baseline lengths and reds dictionaries
        self._radial_groups = radial_reds

        # Reset baseline mapping to spectrally redundant groups
        self._reset_mapping_dictionaries()

    def add_radial_group(self, group):
        """Adds a radially redundant group to the list of radially redundant groups stored in this object.
        First checks to if the group is radially redundant, then adds the group to an existing group if a
        group with the same heading already exists or appends the group if the heading is new.

        Parameters:
        ----------
        group : list
            List of baseline tuples to be added to the list of radially redundant groups
        """
        # Check to make sure the new group is radially redundant
        self._check_new_group(group)
            
        # If group with same heading already exists, add it to that group. Otherwise, append the group to the list
        for bl in self._mapped_spectral_reds:
            if is_same_orientation(group[0], bl, self.antpos, blvec_error_tol=self.blvec_error_tol) and bl[-1] == group[0][-1]:
                index = self._radial_groups.index(self._mapped_spectral_reds[bl])
                self._radial_groups[index] += group
                self._radial_groups[index] = list(set(self._radial_groups[index]))
                break
        else:
            self._radial_groups.append(group)

        # Add baseline lengths to length dictionary
        for bl in group:
            ant1, ant2, _ = bl
            blmag = np.linalg.norm(self.antpos[ant2] - self.antpos[ant1])
            self.baseline_lengths[bl] = blmag

        # Reset the group now that radially redundant groups have changed
        self._reset_mapping_dictionaries()

    def __len__(self):
        """Get number of frequency redundant groups"""
        return len(self._radial_groups)

    def __getitem__(self, index):
        """Get RadialRedundantGroup object from list of unique orientations"""
        return self._radial_groups[index]

    def __setitem__(self, index, value):
        """
        Set a list of baseline tuples that are radially redundant at index in _radial_groups. 
        Also raises an error if a baseline with the same heading is already in the list of 
        radially redundant groups
        """
        # Check to make sure the new group is radially redundant
        self._check_new_group(value)
            
        for bl in self._mapped_spectral_reds:
            if is_same_orientation(value[0], bl, self.antpos, blvec_error_tol=self.blvec_error_tol) and bl[-1] == value[0][-1]:
                raise ValueError('Radially redundant group with same orientation and polarization already exists in the data')
                
        # Add group at index
        self._radial_groups[index] = value

        # Add baseline lengths to length dictionary
        for bl in value:
            ant1, ant2, _ = bl
            blmag = np.linalg.norm(self.antpos[ant2] - self.antpos[ant1])
            self.baseline_lengths[bl] = blmag

        # Add baseline group to mapped spectrally redundant groups
        self._mapped_spectral_reds[value[0]] = value
        for bl in value:
            self._bl_to_spec_red_key[bl] = value[0]

    def append(self, value):
        """
        Append a list of baseline tuples that are radially redundant to the end of _radial_groups. 
        Also raises an error if a baseline with the same heading is already in the list of 
        radially redundant groups
        """
        # Check to make sure the new group is radially redundant
        self._check_new_group(value)
        
        for bl in self._mapped_spectral_reds:
            if is_same_orientation(value[0], bl, self.antpos, blvec_error_tol=self.blvec_error_tol) and bl[-1] == value[0][-1]:
                raise ValueError('Radially redundant group with same orientation and polarization already exists in the data')

        # Append new group
        self._radial_groups.append(value)

        # Add baseline lengths to length dictionary
        for bl in value:
            ant1, ant2, _ = bl
            blmag = np.linalg.norm(self.antpos[ant2] - self.antpos[ant1])
            self.baseline_lengths[bl] = blmag

        # Add baseline group to mapped spectrally redundant groups
        self._mapped_spectral_reds[value[0]] = value
        for bl in value:
            self._bl_to_spec_red_key[bl] = value[0]
    
    def __iter__(self):
        """Iterates through the list of redundant groups"""
        return iter(self._radial_groups)

    def sort(self, key=None, reverse=True):
        """Sorts list by length of the radial groups"""
        self._radial_groups.sort(key=(len if key is None else key), reverse=reverse)
