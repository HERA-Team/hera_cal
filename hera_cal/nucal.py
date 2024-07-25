from . import utils
from . import redcal
from . import abscal
from .datacontainer import DataContainer, RedDataContainer

import inspect
import warnings
import numpy as np
from scipy import linalg
from hera_filters import dspec
import astropy.constants as const
from scipy.cluster.hierarchy import fclusterdata

import jax
import optax
from jax import numpy as jnp
jax.config.update("jax_enable_x64", True)


# Approved Optax Optimizers
OPTIMIZERS = {
    'adabelief': optax.adabelief, 'adafactor': optax.adafactor, 'adagrad': optax.adagrad, 'adam': optax.adam,
    'adamax': optax.adamax, 'adamaxw': optax.adamaxw, 'amsgrad': optax.amsgrad, 'adamw': optax.adamw,
    'fromage': optax.fromage, 'lamb': optax.lamb, 'lars': optax.lars, 'lion': optax.lion, 'novograd': optax.novograd,
    'noisy_sgd': optax.noisy_sgd, 'dpsgd': optax.dpsgd, 'radam': optax.radam, 'rmsprop': optax.rmsprop,
    'sgd': optax.sgd, 'sm3': optax.sm3, 'yogi': optax.yogi, 'optimistic_gradient_descent': optax.optimistic_gradient_descent
}

# Constants
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
    radial_reds: list of lists of baseline tuples (or RadialRedundancy)
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
    Sort baselines into groups with the same radial heading. These groups of baselines are
    radially redundant in a similar way to redcal.get_reds does. Returns a list of list of
    radially redundant baselines.

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
        uors = [[] for _ in range(np.max(clusters))]

        for cluster, bl in zip(clusters, ubl_pairs):
            uors[cluster - 1].append(bl)

        for group in uors:
            _uors[group[0]] = group

    # Filter out groups with less than min_ubl_per_orient baselines
    uors = [_uors[key] for key in _uors if len(_uors[key]) >= min_ubl_per_orient]
    return sorted(uors, key=len, reverse=True)


class RadialRedundancy:
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
            self.reds = redcal.get_reds(antpos, pols=pols, bl_error_tol=bl_error_tol)
        else:
            self.reds = reds

        # Get unique orientations
        self._radial_groups = get_unique_orientations(antpos, reds=self.reds, blvec_error_tol=blvec_error_tol)

        # Map baseline key to baseline length
        self.baseline_lengths = {}
        for group in self._radial_groups:
            for bl in group:
                ant1, ant2, _ = bl
                blmag = np.linalg.norm(self.antpos[ant2] - self.antpos[ant1])
                self.baseline_lengths[bl] = blmag

        # Map baselines to spatially redundant groups
        self._mapped_reds = {red[0]: red for red in self.reds}
        self._bl_to_red_key = {}
        for red in self.reds:
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
        """Get number of radially redundant groups"""
        return len(self._radial_groups)

    def __getitem__(self, index):
        """
        Get a list of baselines that are radially redundant at a given index from the list
        of unique orientations.
        """
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


def compute_spectral_filters(freqs, spectral_filter_half_width, eigenval_cutoff=1e-12):
    """
    Compute discrete prolate spheroidal sequences (DPSS) filters used to model the frequency-axis
    at a fixed u. Light wrapper over the top of hera_filters.dspec.dpss_operator

    Parameters:
    ----------
    freqs : np.ndarray
        Array of frequencies in Hz
    spectral_filter_half_width : float
        Fourier half width of the spectral filter in units of seconds
    eigenval_cutoff : float, optional, default=1e-12
        Cutoff for the eigenvalues of the DPSS filter

    Returns:
    -------
    spectral_filters : np.ndarray
        Array of spectral filters with shape (Nfreqs, Nfilters)
    """
    return dspec.dpss_operator(freqs, [0], [spectral_filter_half_width], eigenval_cutoff=[eigenval_cutoff])[0].real


def compute_spatial_filters_single_group(group, freqs, bls_lengths, spatial_filter_half_width=1, eigenval_cutoff=1e-12, umin=None, umax=None):
    """
    Compute prolate spheroidal wave function (PSWF) filters for a single radially redundant group.

    Parameters:
    ----------
    group : list
        List of baseline tuples that are radially redundant
    freqs : np.ndarray
        Array of frequencies in Hz
    bls_lengths : dict
        Dictionary of baseline lengths in meters where keys are baseline tuples
    spatial_filter_half_width : float, optional, default=1
        Fourier half width of the spatial filter. Value corresponds to maximum value from
        zenith where foregrounds are found in terms of the sky-coordinates (l, m). Here the
        default, 1, is equivalent to a modeling foregrounds out to the horizon, and allowing
        the uv-plane to be modeled at half-wavelength scales.
    eigenval_cutoff : float
        Cutoff for the eigenvalues of the PSWF filter
    umin : float, optional, default=None
        Minimum u-mode at which the filters are computed. If None, filter bounds will be computed from the minimum frequency value and shortest
        baseline length. Restricting the minimum u-mode can decrease the degrees of freedom in a nucal model if one is uininterested in u-modes below
        umin.
    umax : float, optional, default=None
        Maximum u-mode at which the filters are computed. If None, filter bounds will be computed from the maximum frequency value and longest
        baseline length. Restricting the maximum u-mode can significantly decrease the degrees of freedom in a nucal model particularly if the
        baseline group has a few long baselines an one is uininterested in u-modes above umax.

    Returns:
    -------
    spatial_filters : dict
        Dictionary of spatial filters for each baseline in the group
    """

    # Compute the minimum and maximum u values for the spatial filter
    group_bls_lengths = [bls_lengths[bl] for bl in group]
    if umin is None:
        umin = np.min(group_bls_lengths) / SPEED_OF_LIGHT * np.min(freqs)
    if umax is None:
        umax = np.max(group_bls_lengths) / SPEED_OF_LIGHT * np.max(freqs)

    # Create dictionary for storing spatial filters
    spatial_filters = {}

    # Compute spatial filters for each baseline in the group
    for bl in group:
        umodes = bls_lengths[bl] / SPEED_OF_LIGHT * freqs
        pswf, _ = dspec.pswf_operator(
            umodes, filter_centers=[0], filter_half_widths=[spatial_filter_half_width],
            eigenval_cutoff=[eigenval_cutoff], xmin=umin, xmax=umax
        )

        # Filters should be strictly real-valued
        spatial_filters[bl] = np.real(pswf)

    return spatial_filters


def compute_spatial_filters(radial_reds, freqs, spatial_filter_half_width=1, eigenval_cutoff=1e-12, umin=None, umax=None):
    """
    Compute prolate spheroidal wave function (PSWF) filters for each radially redundant group in radial_reds.
    Note that if you are using a large array with a large range of short and long baselines in an individual radially
    redundant group, it is advised to filter radial_reds using radial_reds.filter_reds before running this function
    to reduce the size of filters generated

    Parameters:
    ----------
    radial_reds : RadialRedundancy object
        RadialRedundancy object containing lists of radially redundant baselines.
    freqs : np.ndarray
        Frequencies found in the data in units of Hz
    spatial_half_width : float, default=1
        Filter half width used to generate PSWF filters. Default value of 1 cooresponds to
        modeling foregrounds out to the horizon.
    eigenval_cutoff : float, default=1e-12
        Sinc matrix eigenvalue cutoffs to use for included PSWF modes.
    umin : float, optional, default=None
        Minimum u-mode at which the filters are computed. If None, filter bounds will be computed from the minimum frequency value and shortest
        baseline length. Restricting the minimum u-mode can decrease the degrees of freedom in a nucal model if one is uininterested in u-modes below
        umin. If umin is not None, umin will be applied to all baseline groups in radial reds.
    umax : float, optional, default=None
        Maximum u-mode at which the filters are computed. If None, filter bounds will be computed from the maximum frequency value and longest
        baseline length. Restricting the maximum u-mode can significantly decrease the degrees of freedom in a nucal model particularly if the
        baseline group has a few long baselines an one is uininterested in u-modes above umax. If umax is not None, umax will be applied to all
        baseline groups in radial reds.

    Returns:
    -------
    spatial_filters : dictionary
        Dictionary containing baseline tuple / PSWF eigenvectors key-value pairs used for modeling
        foregrounds
    """
    # Create dictionary for all uv pswf eigenvectors
    spatial_filters = {}

    # Loop through each baseline in each radial group
    for group in radial_reds:
        # Compute spatial filters for each baseline in the group
        spatial_filters.update(
            compute_spatial_filters_single_group(
                group, freqs, radial_reds.baseline_lengths, spatial_filter_half_width,
                eigenval_cutoff, umin=umin, umax=umax
            )
        )

    return spatial_filters


def build_nucal_wgts(data_flags, data_nsamples, autocorrs, auto_flags, radial_reds, freqs, times_by_bl=None,
                     df=None, data_is_redsol=False, gain_flags=None, tol=1.0, antpos=None, min_u_cut=None,
                     max_u_cut=None, min_freq_cut=None, max_freq_cut=None, spw_range_flags=None):
    """
    Build linear weights for data in nucal (or calculating loss) defined as
    wgts = (noise variance * nsamples)^-1 * (0 if data or model is flagged). Light wrapper
    over abscal.build_data_wgts with additional flagging based cuts in uv and frequency.

    Parameters:
    ----------
        data_flags : DataContainer
            Containing flags on data to be calibrated
        data_nsamples : DataContainer
            Contains the number of samples in each data point
        autocorrs : DataContainer
             DataContainer with autocorrelation visibilities
        auto_flags : DataContainer
            DataContainer containing flags for autocorrelation visibilities
        radial_reds : RadialRedundancy object
            RadialRedundancy object containing a list of list baseline tuples of radially redundant
            groups
        freqs : np.ndarray
            Frequency values present in the data in units of Hz
        times_by_bl : dictionary
            Maps antenna pairs like (0,1) to float Julian Date. Optional if
            inferable from data_flags and all times have length > 1.
        df : float, default=None
            If None, inferred from data_flags.freqs
        data_is_redsol : bool, default=False
            If True, data_file only contains unique visibilities for each baseline group.
            In this case, gain_flags and tol are required and antpos is required if not derivable
            from data_flags. In this case, the noise variance is inferred from autocorrelations from
            all baselines in the represented unique baseline group.
        gain_flags : dictionary, default=None
            Used to exclude ants from the noise variance calculation from the autocorrelations
            Ignored if data_is_redsol is False.
        tol : float,
            Distance for baseline match tolerance in units of baseline vectors (e.g. meters).
            Ignored if data_is_redsol is False.
        antpos : dictionary
            Maps antenna number to ENU position in meters for antennas in the data.
            Ignored if data_is_redsol is False. If left as None, can be inferred from data_flags.data_antpos.
        min_u_cut : float
            Minimum u-magnitude value to include in calbration. All u-modes with magnitudes less than
            min_u_cut will have their weights set to 0.
        max_u_cut : float
            Maximum u-magnitude value to include in calbration. All u-modes with magnitudes greater than
            max_u_cut will have their weights set to 0.
        min_freq_cut : float
            Minimum frequency value to include in calibration in units of Hz. All frequency channels less than
            this value will be set to 0.
        max_freq_cut : float
            Maximum frequency value to include in calibration in units of Hz. All frequency channels greater than
            this value will be set to 0.
        spw_range_flags : list of tuples
            List of tuples containing the start and stop frequency of each spectral window to flag in units of Hz.

    Returns:
    -------
        wgts: Datacontainer
            Maps data_flags baseline to weights
    """
    # Build model flags from
    model_flags = {}
    for group in radial_reds:
        for key in group:
            # Get u-magnitudes of all samples for this baseline
            umag = radial_reds.baseline_lengths[key] * freqs / SPEED_OF_LIGHT
            flags = np.zeros_like(data_flags[key], dtype=bool)

            # Apply u-magnitude and frequency cuts
            if min_u_cut is not None:
                flags[:, umag < min_u_cut] = True
            if max_u_cut is not None:
                flags[:, umag > max_u_cut] = True
            if min_freq_cut is not None:
                flags[:, freqs < min_freq_cut] = True
            if max_freq_cut is not None:
                flags[:, freqs > max_freq_cut] = True
            if spw_range_flags is not None:
                for spw in spw_range_flags:
                    flags[:, (freqs > spw[0]) & (freqs < spw[1])] = True

            # Set model flags for all baselines in the group
            for bl in radial_reds.get_redundant_group(key):
                model_flags[bl] = flags

    # Add flags to DataContainer
    model_flags = DataContainer(model_flags)

    # Use abscal.build_data_wgts to build wgts for nucal
    wgts = abscal.build_data_wgts(
        data_flags, data_nsamples, model_flags, autocorrs, auto_flags, times_by_bl=times_by_bl, df=df,
        data_is_redsol=data_is_redsol, gain_flags=gain_flags, tol=tol, antpos=antpos
    )

    return wgts


def _linear_fit(XTX, Xy, solver='lu_solve', alpha=1e-15, cached_input={}):
    """
    Solves a linear system of equations using a variety of methods. This is a light wrapper
    around np.linalg.solve, np.linalg.lstsq, and scipy.linalg.lu_solve which helps fit nucal
    foreground models to the data.

    Parameters:
    ----------
        XTX : np.ndarray
            Matrix of shape (N, N) where N is the number of parameters in the model
        Xy : np.ndarray
            Matrix of shape (N) where N is the number of parameters in the model
        method : str, default='lu_solve'
            Method to use to solve the linear system of equations. Options are 'lu_solve', 'solve', 'pinv', 'lstsq'.
            'lu_solve' uses scipy.linalg.lu_solve to solve the linear system of equations, 'solve' uses np.linalg.solve.
            'pinv' uses np.linalg.pinv to solve the linear system of equations, and 'lstsq' uses np.linalg.lstsq. 'lu_solve'
            and 'solve' tend to be the faster methods, but 'lstsq' and 'pinv' are more robust.
        alpha : float, default=1e-15
            Parameter used for regularization. If method is 'lu_solve' or 'solve', this is added to the diagonal of XTX.
            If method is 'pinv' or 'lstsq', this is used as the rcond parameter for np.linalg.pinv and np.linalg.lstsq respectively.
        cached_input : dictionary, default={}
            Dictionary used to speed-up computation of linear fits for the 'lu_solve' and 'pinv' solvers.
            WARNING: Solvers will use cached_input if one is provided

    Returns:
    -------
        beta : np.ndarray
            Array of shape (N) where N is the number of parameters in the model.
        cached_output: dictionary
            Dictionary for storing computed results from the linear fit solvers which can
            speed-up computation if reused with the same XTX input. Dictionary is empty if
            solver method is 'lstsq' or 'solve'. Contains the matrix inverse (key 'XTXinv')
            for 'pinv' method and lu-decomposition for (key 'LU').
    """
    # Assert that the method is valid
    assert solver in [
        "lu_solve",
        "solve",
        "pinv",
        "lstsq",
    ], "method must be one of {}".format(["lu_solve", "solve", "pinv", "lstsq"])

    # Assert that the regularization tolerance is non-negative
    assert alpha >= 0.0, "alpha must be non-negative."

    # Add regularization tolerance to the diagonal of XTX
    # If XTX is a jax array, use the jax array indexing syntax
    if (solver == "lu_solve" or solver == "solve") and isinstance(XTX, jnp.ndarray):
        XTX = XTX.at[np.diag_indices_from(XTX)].add(alpha)
    elif solver == "lu_solve" or solver == "solve":
        XTX[np.diag_indices_from(XTX)] += alpha

    if solver == "lu_solve":
        # Factor XTX using scipy.linalg.lu_factor
        if "LU" in cached_input:
            L = cached_input.get('LU')
        else:
            L = linalg.lu_factor(XTX)

        # Solve the linear system of equations using scipy.linalg.lu_solve
        beta = linalg.lu_solve(L, Xy)

        # Save info
        cached_output = {'LU': L}

    elif solver == "solve":
        # Solve the linear system of equations using np.linalg.solve
        beta = np.linalg.solve(XTX, Xy)

        # Save info
        cached_output = {}

    elif solver == "pinv":
        # Compute the pseudo-inverse of XTX using np.linalg.pinv
        if "XTXinv" in cached_input:
            XTXinv = cached_input.get('XTXinv')
        else:
            XTXinv = np.linalg.pinv(XTX, rcond=alpha)

        # Compute the model parameters using the pseudo-inverse
        beta = np.dot(XTXinv, Xy)

        cached_output = {'XTXinv': XTXinv}

    elif solver == "lstsq":
        # Compute the model parameters using np.linalg.lstsq
        beta, res, rank, s = np.linalg.lstsq(XTX, Xy, rcond=alpha)

        # Save info
        cached_output = {}

    return beta, cached_output


def evaluate_foreground_model(radial_reds, fg_model_comps, spatial_filters, spectral_filters=None):
    """
    Evaluates a foreground model using the model components, spatial filters, and spectral filters.
    If the model components are time-dependent, the model is evaluated for each time sample in the data.
    If the model components are not time-dependent, the model is evaluated once and then broadcast to
    all time samples in the data.

    Parameters:
    ----------
        radial_reds : RadialRedundancy object
            List of lists of radially-redundant baseline tuples.
        fg_model_comps : dict
            Dictionary mapping baseline tuples to model components. Model components are fitted to the data
            using the DPSS filters to model variations in the spatial and spectral axes. If spectral_filters
            is None, the model components are assumed to be restricted to the spatial axis.
        spatial_filters : dict
            Dictionary mapping baseline tuples to PSWF spatial filters.
        spectral_filters : np.ndarray, default=None
            Array of shape (Nfreqs, Nspec) containing spectral filters for modeling the frequency axis.
            If None, the model components are assumed to be restricted to the spatial axis.

    Returns:
    -------
        model : dict
            Dictionary mapping baseline tuples to model visibilities.

    """
    # Determine whether to use spectral filters when evaluating the model
    use_spectral_filters = spectral_filters is not None

    # Assert that the model components match the baselines in the radial_reds
    for group in radial_reds:
        assert group[0] in fg_model_comps, "fg_model_comps must contain a model component for each baseline in radial_reds."
        if use_spectral_filters:
            assert fg_model_comps[group[0]].shape[1:] == (spectral_filters.shape[-1], spatial_filters[group[0]].shape[-1]), (
                f"The number of model components must match filter shapes."
            )
        else:
            assert fg_model_comps[group[0]].shape[-1] == spatial_filters[group[0]].shape[-1], (
                f"The number of model components in fg_model_comps must match the number of \
                  eigenvectors in spatial_filters for baseline {group[0]}."
            )

    # Initialize a dictionary to hold the model components
    model = {}

    # Determine the einsum path to use for evaluating the model
    einsum_path = "fm,fn,tmn->tf" if use_spectral_filters else "fm,tm->tf"

    # Loop over the groups in radial_reds
    for group in radial_reds:
        for bl in group:
            # Compute the model components for this baseline
            if use_spectral_filters:
                model[bl] = np.array(jnp.einsum(
                    einsum_path, spectral_filters, spatial_filters[bl], fg_model_comps[group[0]]
                ))
            else:
                model[bl] = np.array(jnp.einsum(
                    einsum_path, spatial_filters[bl], fg_model_comps[group[0]]
                ))

    # Return the model as a RedDataContainer
    return RedDataContainer(model, radial_reds.reds)


def fit_nucal_foreground_model(data, data_wgts, radial_reds, spatial_filters, spectral_filters=None, alpha=1e-12,
                               share_fg_model=False, return_model_comps=False, solver="lu_solve"):
    """
    Compute a foreground model for a set of radially redundant baselines. The model is computed by performing a linear
    least-squares fit using a set of DPSS filters to model visibilities within a radially redundant group. If only spatial
    filters are provided, the model is restricted to the spatial axis. If spectral filters are also provided, the model is fit
    allowing spectral variation at a fixed spatial scale. The model components are returned if return_model_comps is True.
    Otherwise, the model is evaluated using the model components and the spatial filters (and spectral filters if provided),
    and the model visibilities are returned.

    Parameters:
    -----------
    data : dictionary, DataContainer
        Data to be fit by u-model. This function assumes that visibilities are redundantly-averaged
        and will only use data from the first baseline in a spatially-redundant group.
    data_wgts : dictionary, DataContainer
        Weights associated with data
    radial_reds : RadialRedundancy, list of lists of tuples
        List of lists of radially redundant baseline tuples. Each list of redundant tuples represents a radial group of
        baselines.
    spatial_filters : dict
        Dictionary mapping baseline tuples to spatial filters. Dictionary is of the form {(ant1, ant2, pol):
        np.array([Nfreqs, Nspat])} where Nspat is the number of spatial filters.
    spectral_filters : np.ndarray, optional, Default is None.
        Array of DPSS filters with shape (Nfreqs, Nspec) where Nspec is the number of spectral DPSS filters. If None,
        the model is assumed to be restricted to the spatial axis with no spectral variation.
    solver : str, optional, Default is 'lu_solve'
        Solver to use for linear least-squares fit. Options are 'lu_solve', 'solve', 'pinv', and 'lstsq'.
    alpha : float, optional, Default is 1e-15.
        Regularization for linear least-squares fit.
    share_fg_model : bool, optional, Default is False.
        If True, the foreground model for each radially-redundant group is shared across the time axis.
        Otherwise, a nucal foreground will be independently computed for each time integration individually.
    return_model_comps: bool, optional, Default is False.
        If True, the model components for the foreground model are returned. Otherwise, only the model visibilities
        are returned.

    Returns:
    --------
    if return_model_comps:
        model_comps : list of np.ndarray
            List of model components projected on to the spectral axis. Each component is a 2D array with shape
            (Ntimes, Nspec, Nspat) where Ntimes is the number of times in the data, Nspec is the number of DPSS
            eigenvectors along the spectral axis, and Nspat is the number of spatial filters.
    else:
        model : dictionary
            Dictionary mapping baseline tuples to model visibilities. Dictionary is of the form
            {(ant1, ant2, pol): np.array([Ntimes, Nfreqs])}.
    """
    # Create empty dictionary for model components
    model_comps = {}

    # Loop over radial groups
    for group in radial_reds:
        # Get the data and design matrix for this group
        design_matrix = np.array([spatial_filters[bl] for bl in group])
        wgts_here = np.array([data_wgts[bl] for bl in group])
        data_here = np.array([data[bl] for bl in group])

        # Get number of fit parameters - this is the number of spectral filters times the number of spatial filters
        if spectral_filters is not None:
            ndim = spectral_filters.shape[1] * design_matrix.shape[-1]

        # Compute the XTX
        if share_fg_model:
            # Compute the XTX and XTWy
            # Below the indices "a" corresponds to the baseline axis, "f" corresponds to the frequency axis,
            # and "t" corresponds to the time axis. The indices "m" and "n" correspond to the model components.
            if spectral_filters is None:
                XTX = jnp.einsum("afm,atf,afn->mn", design_matrix, wgts_here, design_matrix)
                Xy = jnp.einsum("afm,atf->m", design_matrix, data_here * wgts_here)

                # Solve for model components
                beta, _ = _linear_fit(XTX, Xy, solver=solver, alpha=alpha)
            else:
                XTX = jnp.einsum(
                    "fm,afn,atf,fk,afj->mnkj",
                    spectral_filters, design_matrix, wgts_here, spectral_filters, design_matrix
                ).reshape(ndim, ndim)
                Xy = jnp.einsum("fm,afn,atf->mn", spectral_filters, design_matrix, data_here * wgts_here).reshape(ndim)

                # Solve for the foreground model components
                beta, _ = _linear_fit(XTX, Xy, alpha=alpha, solver=solver)
                beta = beta.reshape(spectral_filters.shape[-1], design_matrix.shape[-1])

            # Expand the model components to have time index
            beta = np.expand_dims(beta, axis=0)

        else:
            # Create empty list to hold the model components for each time
            beta = []

            # Loop over time
            for i in range(data_here.shape[1]):
                # Compute the XTX and XTWy
                if spectral_filters is None:
                    XTX = jnp.einsum("afm,af,afn->mn", design_matrix, wgts_here[:, i], design_matrix)
                    Xy = jnp.einsum("afm,af->m", design_matrix, data_here[:, i] * wgts_here[:, i])

                    # Solve for model components
                    beta.append(_linear_fit(XTX, Xy, solver=solver, alpha=alpha)[0])

                else:
                    XTX = jnp.einsum(
                        "fm,afn,af,fk,afj->mnkj",
                        spectral_filters, design_matrix, wgts_here[:, i], spectral_filters, design_matrix
                    ).reshape(ndim, ndim)
                    Xy = jnp.einsum("fm,afn,af->mn", spectral_filters, design_matrix, data_here[:, i] * wgts_here[:, i]).reshape(ndim)

                    # Solve for the foreground model components
                    _beta, _ = _linear_fit(XTX, Xy, alpha=alpha, solver=solver)
                    beta.append(_beta.reshape(spectral_filters.shape[-1], design_matrix.shape[-1]))

            # Pack solution into an array
            beta = np.array(beta)

        # Store the model components
        model_comps[group[0]] = beta

    if return_model_comps:
        return model_comps
    else:
        return evaluate_foreground_model(radial_reds, model_comps, spatial_filters=spatial_filters, spectral_filters=spectral_filters)


def project_u_model_comps_on_spec_axis(u_model_comps, spectral_filters):
    """
    Project u-model components on to the spectral axis using a pre-computed set of DPSS-filters for the spectral
    axis. Can be used with the output of fit_nucal_foreground_model when no spectral filters are provided.

    Parameters:
    -----------
    u_model_comps : dictionary of np.ndarray
        Dictionary of u-model PSWF fit components to project on to the spectral axis. Each set of u-model components is a
        a 2D array with shape (Ntimes, Nspat) where Ntimes is the number of times in the data and Nspat is the number of
        spatial filters.
    spectral_filters : np.ndarray
        Array of DPSS filters with shape (Nfreqs, Nspec) where Nspec is the number of DPSS eigenvectors modeling the
        spectral axis.

    Returns:
    --------
    model_comps : dictionary of np.ndarray
        Dictionary of model components projected on to the spectral axis. Each component is a 3D array with shape
        (Ntimes, Nfilters) where Ntimes is the number of times in the data and Nfilters is the number of DPSS
        eigenvectors.
    """
    # Compute the sum of each eigenvector in spectral filters
    const_eigen_vals = np.sum(spectral_filters, axis=0, keepdims=True)
    const_eigen_vals = np.expand_dims(const_eigen_vals, axis=-1)

    # Project u-model components on to the spectral axis
    model_comps = {}
    for key in u_model_comps:
        model_comps[key] = np.expand_dims(u_model_comps[key], axis=1) * const_eigen_vals

    return model_comps


@jax.jit
def _foreground_model(model_parameters, spectral_filters, spatial_filters):
    """
    Function for computing the foreground model from the foreground parameters and filters
    in the gradient descent loop.

    Parameters:
    ----------
    model_parameters : dictionary
        Foreground model components for computing the foreground model
    spectral_filters : jnp.ndarray
        Array of spectral filters with shape (Nfreqs, Nfilters)
    spatial_filters : List of jnp.ndarray
        List of jax arrays containing spatial DPSS filters for each baseline in the
        spectrally redundant group

    Returns:
    -------
    model_r : np.ndarray
        Array of real component of foreground model with shape (Ntimes, Nbls)
    model_i : np.ndarray
        Array of imaginary component of foreground model with shape (Ntimes, Nbls)
    """
    model_r, model_i = [], []

    # Loop over each radially redundant group
    # Below the indicies correspond to f -> frequency, b -> baseline, m -> spectral modes
    # n -> spatial modes, and t -> time
    for sf, fgr, fgi in zip(spatial_filters, model_parameters['fg_r'], model_parameters['fg_i']):
        model_r.append(jnp.einsum('fm,bfn,tmn->btf', spectral_filters, sf, fgr))
        model_i.append(jnp.einsum('fm,bfn,tmn->btf', spectral_filters, sf, fgi))

    # Stack models
    return jnp.vstack(model_r), jnp.vstack(model_i)


@jax.jit
def _mean_squared_error(model_parameters, data_r, data_i, wgts, fg_model_r, fg_model_i, idealized_blvecs):
    """
    Computes the mean squared error between the data and foreground model multiplied by the
    redundant calibration degenerate parameters. Used as the loss function in the gradient descent
    in SpectralRedundantCalibrator.post_redcal to solve for the redundant calibration degrees of freedom

    Parameters:
    ----------
    model_parameters : dictionary
        Parameters used to fit the DPSS-based foreground model and redundant calibration degeneracies.
        Keys are "fg_r", "fg_i", "amplitude", and "tip_tilt". Parameters "fg_r" and "fg_i" are the
        real and imaginary components of the DPSS foreground model but are not used in this function.
        "amplitude" is redundant calibration amplitude degeneracy and "tip_tilt" are the redundant
        calibration phase gradient degeneracies.
    data_r : np.ndarray
        Array of real component of data with shape (Ntimes, Nbls)
    data_i : np.ndarray
        Array of imaginary component of data with shape (Ntimes, Nbls)
    wgts : np.ndarray
        Array of weights with shape (Ntimes, Nbls)
    fg_model_r : np.ndarray
        Array of real component of foreground model with shape (Ntimes, Nbls)
    fg_model_i : np.ndarray
        Array of imaginary component of foreground model with shape (Ntimes, Nbls)
    idealized_blvecs : np.ndarray
        Array of baseline vectors with shape (Nbls, Ndims)

    Returns:
    -------
    loss : float
        Mean squared error between data and foreground model
    """
    # Dot baseline vector into tip-tilt parameters
    # Below the indicies correspond to b -> baseline, n -> tip-tilt parameters, t -> time, and f -> frequency
    phase = jnp.einsum('bn,ntf->btf', idealized_blvecs, model_parameters["tip_tilt"])

    # Compute model from foreground estimates and amplitude
    model_r = (model_parameters["amplitude"]) * (fg_model_r * jnp.cos(phase) - fg_model_i * jnp.sin(phase))
    model_i = (model_parameters["amplitude"]) * (fg_model_i * jnp.cos(phase) + fg_model_r * jnp.sin(phase))

    # Compute loss using weights and foreground model
    return jnp.sum((jnp.square(model_r - data_r) + jnp.square(model_i - data_i)) * wgts)


@jax.jit
def _calibration_loss_function(model_parameters, data_r, data_i, wgts, spectral_filters, spatial_filters, idealized_blvecs, alpha=0):
    """
    Function which computes the value of the loss from the degenerate parameters, DPSS foreground components, and the data

    Parameters:
    ----------
    model_parameters : dictionary
        Parameters for fitting
    data_r : np.ndarray
        Array of real component of data with shape (Ntimes, Nbls)
    data_i : np.ndarray
        Array of imaginary component of data with shape (Ntimes, Nbls)
    wgts : np.ndarray
        Array of weights with shape (Ntimes, Nbls)
    spectral_filters : np.ndarray
        Array of spectral filters with shape (Nfreqs, Nfilters)
    spatial_filters : List
        List of spatial filters for each baseline in the group
    idealized_blvecs : np.ndarray
        Array of idealized baseline vectors with shape (Nbls, Ndims)
    alpha : float, optional, default=0
        Regularization parameter to use for the loss function. If alpha is non-zero, the loss function will be regularized
        by the sum of the squares of the foreground model parameters.

    Returns:
    -------
    loss : float
        Mean squared error between data and foreground model
    """
    # Compute foreground model from the model_parameters and DPSS filters
    fg_model_r, fg_model_i = _foreground_model(model_parameters, spectral_filters, spatial_filters)

    # Regularize the loss
    param_loss = 0
    for fgr, fgi in zip(model_parameters['fg_r'], model_parameters['fg_i']):
        param_loss += (jnp.square(fgr).sum() + jnp.square(fgi).sum()) * alpha

    # Compute loss
    return _mean_squared_error(model_parameters, data_r, data_i, wgts, fg_model_r, fg_model_i, idealized_blvecs) + param_loss


def _nucal_post_redcal(
    data_r, data_i, wgts, model_parameters, optimizer, spectral_filters, spatial_filters, idealized_blvecs,
    major_cycle_maxiter=100, convergence_criteria=1e-10, minor_cycle_maxiter=10, alpha=1e-12
):
    """
    Function to perform frequency redundant calibration using gradient descent. Calibrates the
    data by fitting a foreground model to the data using DPSS filters and estimating the redundant
    calibration degeneracies that lead to the smoothest calibrated visibilities. Intended to only
    be only after performing redundant calibration with redcal.RedundantCalibrator.

    Parameters:
    ----------
    data_r : jnp.ndarray (or np.ndarray)
        Array of real component of data with shape (Ntimes, Nbls)
    data_i : jnp.ndarray (or np.ndarray)
        Array of imaginary component of data with shape (Ntimes, Nbls)
    wgts : np.ndarray
        Array of weights with shape (Ntimes, Nbls)
    model_parameters : dictionary
        Parameters used to fit the DPSS-based foreground model and redundant calibration degeneracies.
        Keys are "fg_r", "fg_i", "amplitude", and "tip_tilt". Parameters "fg_r" and "fg_i" are the
        real and imaginary components of the DPSS foreground model but are not used in this function.
        "amplitude" is redundant calibration amplitude degeneracy and "tip_tilt" are the redundant
        calibration phase gradient degeneracies.
    optimizer : optax optimizer
        Optimizer to use for gradient descent.
    spectral_filters : np.ndarray
        Array of spectral filters with shape (Nfreqs, Nfilters)
    spatial_filters : List
        List of spatial filters for each baseline in the group
    idealized_blvecs : np.ndarray
        Array of idealized baseline vectors with shape (Nbls, Ndims)
    major_cycle_maxiter : int, optional, default=100
        Maximum number of iterations to perform in the major portion of the gradient descent loop. A major cycle is defined as a gradient descent step
        in which the foreground model parameters and redundant calibration degeneracies are fit to the data.
        If convergence_criteria is not met after major_cycle_maxiter iterations, the optimization will stop.
    tol : float, optional, default=1e-10
        Tolerance for stopping criterion. If the difference of the loss between two iterations is less than tol,
        the optimization will stop.
    minor_cycle_maxiter : int, optional, default=0
        Maximum number of iterations of the minor cycle to perform after each major cycle. Minor cycles are performed by fixing the foreground
        model and solving for the calibration parameters. When minor_cycle_maxiter is 0, no minor cycles are performed. If subsequent losses
        are within convergence_criterea, the minor cycle will stop.
    alpha : float, optional, default=0
        Regularization parameter to use for the loss function. If alpha is non-zero, the loss function will be regularized
        by the sum of the squares of the foreground model parameters.

    Returns:
    -------
    model_parameters : dictionary
        Optimized parameters
    metadata : dictionary
        Dictionary containing metadata from the optimization. Contains the number of iterations ("niter") and the loss history
        ("loss_history"). The loss history is an array that stores the value of the loss at each major and minor cycle iteration.
    """
    # Initialize optimizer state using parameter guess
    opt_state = optimizer.init(model_parameters)

    # Initialize variables used in calibration loop
    losses = []
    previous_loss = np.inf

    # Start gradient descent
    for step in range(major_cycle_maxiter):
        # Compute loss and gradient
        loss, gradient = jax.value_and_grad(_calibration_loss_function)(
            model_parameters, data_r, data_i, wgts, spectral_filters=spectral_filters, spatial_filters=spatial_filters,
            idealized_blvecs=idealized_blvecs, alpha=alpha
        )
        # Update optimizer state and parameters
        updates, opt_state = optimizer.update(gradient, opt_state, model_parameters)
        model_parameters = optax.apply_updates(model_parameters, updates)

        if minor_cycle_maxiter > 0:
            minor_cycle_losses = []

            # Compute foreground model from the model_parameters and DPSS filters
            fg_model_r, fg_model_i = _foreground_model(model_parameters, spectral_filters, spatial_filters)
            for minor_step in range(minor_cycle_maxiter):
                # Since the foreground model is fixed, we can just use the _mean_square_error
                # function as our loss function
                minor_cycle_loss, gradient = jax.value_and_grad(_mean_squared_error)(
                    model_parameters, data_r, data_i, wgts, fg_model_r, fg_model_i, idealized_blvecs=idealized_blvecs,
                )
                # Update optimizer state and parameters
                updates, opt_state = optimizer.update(gradient, opt_state, model_parameters)
                model_parameters = optax.apply_updates(model_parameters, updates)

                # Store minor cycle loss values
                minor_cycle_losses.append(minor_cycle_loss)

                # Stop if subsequent losses are within tolerance
                if minor_step >= 1 and np.abs(minor_cycle_losses[-1] - minor_cycle_losses[-2]) < convergence_criteria:
                    break

            losses += minor_cycle_losses

        # Store loss values
        losses.append(loss)

        # Stop if subsequent losses are within tolerance
        if step >= 1 and np.abs(losses[-1] - previous_loss) < convergence_criteria:
            break

        previous_loss = loss

    # Save the metadata in dictionary
    metadata = {"niter": step + 1, "loss_history": np.array(losses)}

    return model_parameters, metadata


class SpectrallyRedundantCalibrator:
    """
    Class for performing spectral redundant calibration using the nucal implementation. This class is designed to
    be an easy-to-use, high-level interface for performing spectral redundant calibration, similarly to the redcal.RedundantCalibrator
    class. The main driver function, SpectrallyRedundantCalibrator.calibrate, estimates the DPSS-based foreground model components and
    redundant calibration degeneracies using gradient descent. The class also provides a number of helper functions for computing
    DPSS filters and estimating the degeneracies from the data.
    """
    def __init__(self, radial_reds):
        """
        Initialize the SpectrallyRedundantCalibrator class. Takes a RadialRedundancy object as input for
        computing the DPSS filters and determining which baselines are in each radially redundant group.

        Parameters:
        ----------
        radial_reds : RadialRedundancy object
            RadialRedundancy object containing a list of list baseline tuples of radially redundant
            groups. Can be generated using nucal.RadialRedundancy.
        """
        # Store the radial redundancy object and antpos
        self.radial_reds = radial_reds
        self.antpos = radial_reds.antpos

        # Initialize variables for tracking whether filters have been computed
        self._filters_computed = False
        self._most_recent_filter_params = {}

    def _compute_filters(self, freqs, spectral_filter_half_width, spatial_filter_half_width=1, eigenval_cutoff=1e-12, umin=None, umax=None):
        """
        """
        # Get all parameter names and local variables
        local_vars = locals()
        local_vars.pop("self")

        if self._filters_computed:
            # Variable for tracking if the filters need to be recomputed - by default, assume they do not
            recompute_filters = False

            # Loop over all parameters and check if they have changed
            for key in local_vars:
                if not np.array_equal(local_vars[key], self._most_recent_filter_params[key]):
                    recompute_filters = True
                    break

            if recompute_filters:
                self.spectral_filters = compute_spectral_filters(freqs, spectral_filter_half_width, eigenval_cutoff=eigenval_cutoff)
                self.spatial_filters = compute_spatial_filters(
                    self.radial_reds, freqs, spatial_filter_half_width, eigenval_cutoff=eigenval_cutoff, umin=umin, umax=umax
                )

                # Set most recent set of filter parameters
                for key in local_vars:
                    self._most_recent_filter_params[key] = local_vars[key]

                # Set filters computed to True
                self._filters_computed = True

        else:
            # Compute the spectral and spatial filters
            self.spectral_filters = compute_spectral_filters(freqs, spectral_filter_half_width, eigenval_cutoff=eigenval_cutoff)
            self.spatial_filters = compute_spatial_filters(
                self.radial_reds, freqs, spatial_filter_half_width, eigenval_cutoff=eigenval_cutoff, umin=umin, umax=umax
            )

            # Set most recent parameters
            for key in local_vars:
                self._most_recent_filter_params[key] = local_vars[key]

            # Set filters computed to True
            self._filters_computed = True

    def _estimate_degeneracies(self, data, model, wgts):
        """
        Estimate the redundant degeneracies from the data using traditional abscal techniques.

        Parameters:
        ----------
        data : DataContainer
            Data to be calibrated. Data is assumed to be redundantly averaged. DataContainer is of the form {(ant1, ant2, pol): np.array([Ntimes, Nfreqs])}
        model : DataContainer
            Model visibilities to use for estimating the bandpass. DataContainer is of the form {(ant1, ant2, pol): np.array([Ntimes, Nfreqs])}
        wgts : DataContainer
            Weights associated with data. DataContainer is of the form {(ant1, ant2, pol): np.array([Ntimes, Nfreqs])}

        Returns:
        -------
        amplitude : np.ndarray
            Array of shape (Nbls, Ntimes) containing the amplitude of the model visibilities
        tip_tilt : np.ndarray
            Array of shape (Nbls, Ntimes, Ndims) containing the tip-tilt parameters for each baseline
        """
        # Get the baselines in the model
        data_bls = [blkeys for blkeys in data]

        # Estimate the amplitude degeneracies from the model
        # abs_amp_logcal returns the amplitude degeneracies and works on both pols simulataneously
        amp_sol = abscal.abs_amp_logcal(
            data=data, model=model, wgts=wgts, verbose=False, return_gains=False
        )

        # Unpack solution into dictionary
        # Degeneracy as written in gradient descent is exp(2 * eta) because the amplitude degeneracy
        # in nucal is written as the square of the amplitude degeneracy in the abscal solution
        amplitude = {
            pol: np.exp(2 * amp_sol[f"eta_J{pol}"]) for pol in data.pols()
        }

        # Estimate the tip-tilt degeneracies from the model
        tip_tilt = {}
        for pol in data.pols():
            # Get the baselines in the model
            data_bls = [blkeys for blkeys in model if blkeys[2] == pol and blkeys[0] != blkeys[1]]

            # complex_phase_abscal returns the tip-tilt degeneracies
            meta, _ = abscal.complex_phase_abscal(
                data=data, model=model, reds=self.radial_reds.reds, data_bls=data_bls, model_bls=data_bls
            )

            # Tranpose the tip-tilt parameters to have shape (ndims, ntimes, nfreq)
            tip_tilt[pol] = np.transpose(meta["Lambda_sol"], (2, 0, 1))

        return amplitude, tip_tilt

    def post_redcal_nucal(
        self, data, data_wgts, cal_flags={}, spatial_estimate_only=False, linear_solver="lu_solve", alpha=0, share_fg_model=False,
        spectral_filter_half_width=30e-9, spatial_filter_half_width=1, eigenval_cutoff=1e-12, umin=None, umax=None, estimate_degeneracies=False,
        optimizer_name='novograd', learning_rate=1e-3, major_cycle_maxiter=100, minor_cycle_maxiter=0, convergence_criteria=1e-10, return_model=False
    ):
        """
        Estimates redundant calibration degeneracies by building a DPSS-based, sky-model and solving for the parameters which lead to the smoothest
        calibrated visibilities. Function starts by estimating a sky-model by using DPSS filters (which can start with spatial dependence or spectral
        and spatial dependence) which are fit to the data using linear-least squares. The sky-model is then used to estimate the degeneracies using
        traditional abscal techniques, or can be set such that the antenna gains are equal to 1+0j if one assumes that the data are well-calibrated.
        The degeneracies and sky-model are then refined using gradient descent. Once the gradient descent has converged, or reached the maximum number
        of iterations, the degeneracies and sky-model are returned. This function assumes that the input data are redundantly calibrated and redundantly averaged.

        Parameters:
        ----------
        data : DataContainer (or RedDataContainer)
            Data to be calibrated. Data is assumed to be redundantly averaged. DataContainer is of the form {(ant1, ant2, pol): np.array([Ntimes, Nfreqs])}
        data_wgts : DataContainer (or RedDataContainer)
            Weights associated with data. DataContainer is of the form {(ant1, ant2, pol): np.array([Ntimes, Nfreqs])}
        cal_flags : dictionary, default={}
            Dictionary mapping keys like (1, 'Jnn') to flag waterfalls. This dictionary is primarily used for computing the
            idealized antenna positions.
        spatial_estimate_only : bool, default=False
            If True, the initial estimate of the foreground model will be computed from the data assuming that the evolution foreground model
            is entirely restricted to the spatial axis. This estimate will then be projected onto the eigenmodes of the spectral DPSS modes
            for refinement in the gradient descent step. If False, the initial estimate of the foreground model will be computed from the
            data giving the model the flexibility to model the spatial and spectral axes. This option fits for fewer foreground parameters
            in the least-squares step, but can lead to slower convergence in the gradient descent step.
        linear_solver : str, default="lu_solve"
            Method to use for solving the linear system of equations when fitting the foreground models. Options are
            "lu_solve", "solve", "pinv", and "lstsq". "lu_solve" uses scipy.linalg.lu_solve to solve the linear system of
            equations, "solve" uses np.linalg.solve. "pinv" uses np.linalg.pinv to solve the linear system of equations, and
            "lstsq" uses np.linalg.lstsq. "lu_solve" and "solve" tend to be the faster methods, but "lstsq" and "pinv" are more
            robust.
        alpha : float, default=1e-12
            Regularization parameter for linear least-squares fit when computing the initial estimate of the foreground model.
            Regularization parameter is also used as a regularizer in the gradient descent step.
        share_fg_model : bool, default=False
            If True, the foreground model for each radially-redundant group is shared across the time axis for both the least-squares and
            gradient descent steps. One useful application of this option is when performing calibration of data across multiple nights at
            the same LST where the data have shape (N_nights, Nfreqs). In this case, the foreground model is expected to be the same across nights, so sharing the foreground model
            across nights can greatly reduce the number of parameters to fit. This parameter could also be used for subsequent times to share a
            sky model assuming the sky doesn't evolve much in the subsequent integrations. If False, a nucal foreground will be solved for independently
            for each time integration.
        spectral_filter_half_width : float, default=20e-9
            Fourier half-width of the spectral axis DPSS filters in units of seconds.
        spatial_filter_half_width : float, default=1
            Half-width of the spatial axis DPSS filters in units of wavelengths.
        eigenval_cutoff : float, default=1e-12
            Cutoff for the eigenvalues of the DPSS filters. Only DPSS eigenvectors with eigenvalues greater than eigenval_cutoff will be used
            for least-squares and gradient descent steps.
        umin : float, default=None
            Minimum u-magnitude value to include in calbration. All u-modes with magnitudes less than
            min_u_cut will have their weights set to 0. Can also be useful for decreasing the number
            for foreground eigenmodes as the number of eigenmodes is roughly proportional to (umax - umin).
        umax : float, default=None
            Maximum u-magnitude value to include in calbration. All u-modes with magnitudes greater than
            max_u_cut will have their weights set to 0. Can also be useful for decreasing the number
            for foreground eigenmodes as the number of eigenmodes is roughly proportional to (umax - umin).
        estimate_degeneracies : bool, default=False
            If True, the initial estimates of the redcal degeneracies will be computed from the data using traditional
            abscal techniques and the initial nucal model as the sky model. If False, the amplitude degeneracies will be
            initialized to 1 and tip-tilt degeneracies will be initialized to 0. If the data are well-calibrated,
            setting this option to False can improve the runtime of the calibration.
        optimizer_name : str, default="novograd"
            Name of the optimizer to use for gradient descent. Options are keys in nucal.OPTIMIZERS.
        learning_rate : float, default=1e-3
            Learning rate for the gradient descent optimizer
        major_cycle_maxiter : int, default=100
            Maximum number of iterations to run when performing gradient descent. Major cycles are defined as a gradient descent step
            in which the foreground model parameters and redundant calibration degeneracies are fit to the data. If the difference of the loss
            between two major cycle iterations is less than convergence_criteria, the optimization will stop.
        minor_cycle_maxiter : int, default=0
            Number of minor cycles to perform after each major cycle. Minor cycles are performed by fixing the foreground
            model and solving for the calibration parameters. Can be useful for improving convergence. If subsequent minor losses
            are within convergence_criteria, the minor cycle will stop for a given major cycle.
        convergence_criteria : float, default=1e-10
            Convergence criteria for stopping the optimization. If the difference in loss between two iterations is less than
            convergence_criteria, the optimization will stop.
        return_model : bool, default=False
            If True, the model visibilities will be returned.

        Returns:
        -------
        gains : dictionary
            Dictionary mapping antenna numbers to complex gains. Keys are antenna numbers and values are complex gains.
        model_parameters : dictionary
            Dictionary containing the model parameters for each polarization. Keys are polarization strings and values are
        metadata : dictionary
            Dictionary containing metadata from the optimization. Contains dictionaries for each polarization with the number of
            iterations ("niter") and the loss history of the gradient descent ("loss_history").

        If return_model:
            model : DataContainer
                DataContainer containing the model visibilities. DataContainer is of the form {(ant1, ant2, pol): np.array([Ntimes, Nfreqs])}
        """
        # Compute spectral and spatial filters
        self._compute_filters(
            freqs=data.freqs, spectral_filter_half_width=spectral_filter_half_width,
            spatial_filter_half_width=spatial_filter_half_width, eigenval_cutoff=eigenval_cutoff,
            umin=umin, umax=umax
        )

        # Assert that the optimizer is valid
        assert optimizer_name in OPTIMIZERS, f"Optimizer must be one of {OPTIMIZERS}. Got {optimizer_name}."
        optimizer = OPTIMIZERS[optimizer_name](learning_rate=learning_rate)

        # Compute the estimates of the model components from the data
        if spatial_estimate_only:
            init_model_comps = fit_nucal_foreground_model(
                data, data_wgts, self.radial_reds, self.spatial_filters, solver=linear_solver, share_fg_model=share_fg_model,
                return_model_comps=True, alpha=alpha
            )
            init_model_comps = project_u_model_comps_on_spec_axis(init_model_comps, self.spectral_filters)
        else:
            init_model_comps = fit_nucal_foreground_model(
                data, data_wgts, self.radial_reds, self.spatial_filters, solver=linear_solver, share_fg_model=share_fg_model,
                spectral_filters=self.spectral_filters, return_model_comps=True, alpha=alpha
            )

        # Compute idealized baseline vectors from antenna positions and calibration flags
        idealized_antpos = abscal._get_idealized_antpos(cal_flags, self.antpos, data.pols())

        if estimate_degeneracies:
            model = evaluate_foreground_model(
                self.radial_reds, init_model_comps, spatial_filters=self.spatial_filters, spectral_filters=self.spectral_filters
            )
            amplitude, tip_tilt = self._estimate_degeneracies(data, model, data_wgts)
        else:
            amplitude = {pol: np.ones((data.shape)) for pol in data.pols()}
            tip_tilt = {
                pol: np.zeros((idealized_antpos[list(idealized_antpos.keys())[0]].shape[0], data.shape[0], data.shape[1]))
                for pol in data.pols()
            }

        # Initialize model parameters and metadata dictionaries for storing results
        model_parameters = {}
        metadata = {}

        for pol in data.pols():
            # Initialize model parameters
            init_model_parameters = {}

            # Separate data into real and imaginary components
            data_real = jnp.array([
                data[blkey].real for rdgrp in self.radial_reds.get_pol(pol) for blkey in rdgrp
            ])
            data_imag = jnp.array([
                data[blkey].imag for rdgrp in self.radial_reds.get_pol(pol) for blkey in rdgrp
            ])

            # Initialize model parameters
            init_model_parameters = {
                "fg_r": [init_model_comps[rdgrp[0]].real for rdgrp in self.radial_reds.get_pol(pol)],
                "fg_i": [init_model_comps[rdgrp[0]].imag for rdgrp in self.radial_reds.get_pol(pol)],
                "amplitude": amplitude[pol],
                "tip_tilt": tip_tilt[pol]
            }
            # unpack wgts/filters
            wgts = jnp.array([
                data_wgts[blkey] for rdgrp in self.radial_reds.get_pol(pol) for blkey in rdgrp
            ])

            # Set spectral filters
            spatial_filters = [
                jnp.array([self.spatial_filters[blkey] for blkey in rdgrp])
                for rdgrp in self.radial_reds.get_pol(pol)
            ]

            # Compute idealized baseline vectors
            idealized_blvecs = jnp.array([
                idealized_antpos[blkey[1]] - idealized_antpos[blkey[0]]
                for rdgrp in self.radial_reds.get_pol(pol) for blkey in rdgrp
            ])

            # Run optimization
            model_parameters[pol], metadata[pol] = _nucal_post_redcal(
                data_real, data_imag, wgts, init_model_parameters, optimizer, spectral_filters=self.spectral_filters,
                spatial_filters=spatial_filters, idealized_blvecs=idealized_blvecs, major_cycle_maxiter=major_cycle_maxiter,
                convergence_criteria=convergence_criteria, minor_cycle_maxiter=minor_cycle_maxiter, alpha=alpha
            )

        if return_model:
            # Compute the foreground model from the model parameters
            fg_model_comps = {
                rdgrp[0]: model_parameters[pol][f"fg_r"][ri] + 1j * model_parameters[pol][f"fg_i"][ri]
                for pol in data.pols() for ri, rdgrp in enumerate(self.radial_reds.get_pol(pol))
            }
            model = evaluate_foreground_model(self.radial_reds, fg_model_comps, self.spatial_filters, self.spectral_filters)

        # Compute the gains from the model parameters
        gains = {}
        for pol in data.pols():
            for ant in idealized_antpos:
                gains[(ant, f"J{pol}")] = np.sqrt(model_parameters[pol][f"amplitude"]) * np.exp(
                    1j * np.tensordot(idealized_antpos[ant], model_parameters[pol][f"tip_tilt"], axes=(0, 0))
                )

        if return_model:
            return gains, model_parameters, metadata, model
        else:
            return gains, model_parameters, metadata
