from . import utils
from . import redcal
from . import abscal
from .datacontainer import DataContainer

import warnings
import numpy as np
from copy import deepcopy
from functools import partial
from hera_filters import dspec
import astropy.constants as const
from scipy.cluster.hierarchy import fclusterdata

# Optional import of Optax and Jax libraries
try:
    import optax

    # Approved Optax Optimizers
    OPTIMIZERS = {
        'adabelief': optax.adabelief, 'adafactor': optax.adafactor, 'adagrad': optax.adagrad, 'adam': optax.adam,
        'adamw': optax.adamw, 'fromage': optax.fromage, 'lamb': optax.lamb, 'lars': optax.lars,
        'noisy_sgd': optax.noisy_sgd, 'dpsgd': optax.dpsgd, 'radam': optax.radam, 'rmsprop': optax.rmsprop,
        'sgd': optax.sgd, 'sm3': optax.sm3, 'yogi': optax.yogi
    }
except:
    warnings.warn('Optax is not installed. Some functionality may not be available')

try:
    import jax
    from jax import numpy as jnp
    jax.config.update("jax_enable_x64", True)

except:
    warnings.warn('Jax is not installed. Some functionality may not be available')
    
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


def compute_spatial_filters(radial_reds, freqs, ell_half_width=1, eigenval_cutoff=1e-12):
    """
    Compute prolate spheroidal wave function (PSWF) filters for each radially redundant group in radial_reds. 
    Note that if you are using a large array with a large range short and long baselines in an individual radially
    redundant group, it is advised to filter radial_reds using radial_reds.filter_reds before running this function 
    to reduce the size of filters generated

    Parameters:
    ----------
    radial_reds : FrequencyRedundant object
        FrequencyRedundant object containing lists of radially redundant baselines. 
    freqs : np.ndarray
        Frequencies found in the data in units of Hz
    ell_half_width : float, default=1
        Filter half width used to generate PSWF filters. Default value of 1 cooresponds to
        modeling foregrounds out to the horizon.
    eigenval_cutoff : float, default=1e-12
        Sinc matrix eigenvalue cutoffs to use for included PSWF modes.

    Returns:
    -------
    spatial_modes : dictionary
        Dictionary containing baseline tuple / PSWF eigenvectors key-value pairs used for modeling 
        foregrounds
    """
    # Create dictionary for all uv pswf eigenvectors
    spatial_filters = {}

    # Get the minimum and maximum u-bounds used
    u_bounds = get_u_bounds(radial_reds, radial_reds.antpos, freqs)

    # Loop through each baseline in each radial group
    for gi, group in enumerate(radial_reds):
        umin, umax = u_bounds[gi]
        for bl in group:
            umodes = radial_reds.baseline_lengths[bl] / SPEED_OF_LIGHT * freqs
            pswf, _ = dspec.pswf_operator(umodes, [0], [ell_half_width], eigenval_cutoff=[eigenval_cutoff], xmin=umin, xmax=umax)
            
            # Filters should be strictly real-valued
            spatial_filters[bl] = np.real(pswf)

    return spatial_filters

def build_nucal_wgts(radial_reds, freqs, data_flags, data_nsamples, autocorrs, auto_flags, times_by_bl=None,
                     df=None, data_is_redsol=False, gain_flags=None, tol=1.0, antpos=None, 
                     min_u_cut=None, max_u_cut=None, min_freq_cut=None, max_freq_cut=None):
    """
    Build linear weights for data in nucal (or calculating loss) defined as
    wgts = (noise variance * nsamples)^-1 * (0 if data or model is flagged). Light wrapper
    over abscal.build_data_wgts with more stricts requirements. Adds additional flagging based 
    on u-cuts and frequency cuts.

    Parameters:
    ----------
        radial_reds : FrequencyRedundant object
            FrequencyRedundant object containing a list of list baseline tuples of radially redundant
            groups
        freqs : np.ndarray
            Frequency values present in the data in units of Hz     
        data_flags : DataContainer
            Contains flags on data
        data_nsamples : DataContainer
            Contains the number of samples in each data point
        autocorrs : DataContainer
             DataContainer with autocorrelation visibilities
        auto_flags : DataContainer
            DataContainer containing flags for autocorrelation visibilities
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
        max_u_cut : float
            Maximum frequency value to include in calibration in units of Hz. All frequency channels greater than
            this value will be set to 0.
                 
    Returns:
    -------
        wgts: Datacontainer
            Maps data_flags baseline to weights
    """
    # Build model flags from 
    model_flags = {}
    for group in radial_reds:
        for key in group:
            umag = radial_reds.baseline_lengths[key] * freqs / SPEED_OF_LIGHT
            wgts = np.zeros_like(data_flags[key], dtype=bool)

            if min_u_cut is not None:
                wgts[:, umag < min_u_cut] = True

            if max_u_cut is not None:
                wgts[:, umag > max_u_cut] = True

            if min_freq_cut is not None:
                wgts[:, freqs < min_freq_cut] = True

            if max_freq_cut is not None:
                wgts[:, freqs > max_freq_cut] = True
                
            model_flags[key] = wgts
    
    weights = abscal.build_data_wgts(
        data_flags, data_nsamples, model_flags, autocorrs, auto_flags, 
        times_by_bl=times_by_bl, df=df, data_is_redsol=data_is_redsol, 
        gain_flags=gain_flags, tol=tol, antpos=antpos
    )

    return weights

def estimate_degenerate_parameters(data, data_wgts, radial_reds, rc_flags, spatial_filters,
                                   niter=1, phs_max_iter=10, return_gains=False):
    """
    Relatively quick estimate of the redcal degenerate parameters using
    a common u-model
    
    Parameters:
    ----------
    data : dictionary, DataContainer
        DataContainer of redundantly calibrated visibilities
    data_wgts : dictionary, DataContainer
        Weights associated with redudantly calibrated visibilities
    radial_reds : FrequencyRedundancy
        FrequencyRedundancy object which contains of baseline tuple which are considered
        to be radially redundant
    rc_flags : dictionary, DataContainer
        pass
    spatial_filters : dictionary
        pass
    niter : int, default=1
        Number of iterations to attempt to 
    phs_max_iter : int, default=10
        pass
    return_gains : bool, default=False
        pass
        
    Returns:
    -------
    degen_params : dictionary
        pass
    model : dictionary
        pass
    model_comps : dictionary
        pass
    gains : dictionary
        pass
    """
    # 
    keys = [bl for group in radial_reds for bl in group]
    antpols = list(set([antpol for bl in keys for antpol in utils.split_bl(bl)]))
    
    # Keep only data found in 
    data_here = DataContainer(deepcopy({bl: data[bl] for bl in keys}))
    data_wgts_here = DataContainer(deepcopy({bl: data_wgts[bl] for bl in keys}))
    
    # Reset metadata
    data_here.antpos = data.antpos
    data_wgts_here.antpos = data.antpos
    data_here.freqs = data.freqs
    
    
    # Storage dictionaries
    cache = {}
    model_comps = {}
    
    # Estimate initial model of the uv-plane along radial spokes
    for i in range(niter):
        model = {}
        for group in radial_reds:
            design_matrix = jnp.array([spatial_filters[bl] for bl in group])
            wgts_arr = jnp.array([data_wgts_here[bl] for bl in group])
            data_arr = jnp.array([data_here[bl] for bl in group])

            # Compute XTX and inverse
            if i == 0:
                XTX = jnp.einsum('afm,atf,afn->tmn', design_matrix, wgts_arr, design_matrix)
                XTXinv = jnp.linalg.pinv(XTX)
                cache[group[0]] = XTXinv
            else:
                XTXinv = cache[group[0]]
            
            # Compute solution vector
            Xy = jnp.einsum('afm,atf->tm', design_matrix, wgts_arr * data_arr)
            beta = jnp.einsum('tmn,tm->tn', XTXinv, Xy)
            model_comps[group[0]] = beta

            # Evaluate model with solution vector
            for bl in group:
                model[bl] = np.einsum('fm,tm->tf', spatial_filters[bl], beta)
        
        # Add model to DataContainer
        model = DataContainer(model)
    
        # Estimate Degenerate Gain Parameters using abscal
        gains_here = abscal.post_redcal_abscal(model, data_here, data_wgts_here, rc_flags, verbose=False, phs_max_iter=phs_max_iter)

    
    # Get final set of degenerate parameters
    ants = sorted(list(rc_flags.keys()))
    idealized_antpos = abscal._get_idealized_antpos(rc_flags, data_here.antpos, data_here.pols(),
                                             data_wgts=data_wgts_here, keep_flagged_ants=True)

    # Repack data into new datacontainers for estimate of degenerate parameters
    data_here = DataContainer({bl: data[bl] for bl in keys})
    data_here.antpos = data.antpos
    data_here.freqs = data.freqs
    data_wgts_here = DataContainer({bl: data_wgts[bl] for bl in keys})
    
    # Use best-fit u-model to get amplitude and tip-tilt parameters
    amp_params = abscal.abs_amp_lincal(model, data_here, wgts=data_wgts_here, verbose=False, return_gains=False, gain_ants=ants)
    angle_wgts = DataContainer({bl: 2 * np.abs(model[bl])**2 * data_wgts_here[bl] for bl in model})
    phase_params = abscal.TT_phs_logcal(model, data_here, idealized_antpos, wgts=angle_wgts, verbose=False, assume_2D=False, return_gains=False, gain_ants=ants)
    degen_params = {**amp_params, **phase_params}
    
    # Return degenerate parameters and model components
    if return_gains:
        gains = abscal.post_redcal_abscal(model, data_here, data_wgts_here, rc_flags, verbose=False, phs_max_iter=phs_max_iter)
        return degen_params, model, model_comps, gains
    else:
        return degen_params, model, model_comps

@jax.jit
def foreground_model(fg_model_comps_r, fg_model_comps_i, spectral_filters, spatial_filters):
    """
    Function for computing foreground models from 
    
    Parameters:
    ----------
    fg_model_comps_r : jnp.ndarray
        Modeling components used to compute the sky model using DPSS vectors for the
        real component of the data.
    fg_model_comps_i : jnp.ndarray
        Modeling components used to compute the sky model using DPSS vectors for the
        imaginary component of the data.
    spec : jnp.ndarray
        Array of spectrally DPSS eigenvectors used for modeling foregrounds
    spat : list of jnp.ndarrays
        Array of spatial PSWF eigenvectors used for modeling foregrounds
    """
    model_r = jnp.einsum('fm,afn,mn->af', spectral_filters, spatial_filters, fg_model_comps_r, optimize=True)
    model_i = jnp.einsum('fm,afn,mn->af', spectral_filters, spatial_filters, fg_model_comps_i, optimize=True)
    return model_r, model_i

@jax.jit
def _mean_squared_error(params, fg_model_comps_r, fg_model_comps_i, data_r, data_i, wgts, spectral_filters, spatial_filters, blvecs):
    """
    Loss function used for solving for redcal degenerate parameters and a model of the sky
    
    Parameters:
    ----------
    params : dictionary
        pass
    fg_model_comps_r : jnp.ndarray
        pass
    fg_model_comps_i : jnp.ndarray
        pass
    data_r : jnp.ndarray
        pass
    data_r : jnp.ndarray
        pass
    wgts : jnp.ndarray
        pass
    spectral_filters : jnp.ndarray
        Array of spectrally DPSS eigenvectors used for modeling foregrounds
    spatial_filters : list of jnp.ndarrays
        Array of spatial PSWF eigenvectors used for modeling foregrounds
    
    Returns:
    -------
    loss : float
        pass
    """
    # Dot baseline vector into tip-tilt parameters
    phase = jnp.einsum('bn,nf->bf', blvecs, params["Phi"])
            
    # Compute foreground model from beta estimates
    fg_model_r, fg_model_i = foreground_model(fg_model_comps_r, fg_model_comps_i, spectral_filters, spatial_filters)

    # Compute model from foreground estimates and amplitude
    model_r = params["A"] * (fg_model_r * jnp.cos(phase) - fg_model_i * jnp.sin(phase))
    model_i = params["A"] * (fg_model_i * jnp.cos(phase) + fg_model_r * jnp.sin(phase))
    
    # Compute loss using weights and foreground model
    return jnp.sum((jnp.square(model_r - data_r) + jnp.square(model_i - data_i)) * wgts)
    
@partial(jax.value_and_grad, argnums=(0,))
def _iterate_through_groups(params, data, wgts, spectral_filters, spatial_filters, idealized_blvecs):
    """
    Function for iterating through groups of radially redundant baselines
    to compute the loss value

    Parameters: 
    ----------
    params : dictionary
        Parameter dictionary containing current iterations estimate of the parameter values
    data : list of jnp.ndarrays
        List of jnp.ndarrays of data for each radially redundant group of baselines
    wgts : list of jnp.ndarrays
        List of jnp.ndarrays of the data weights for each radially redundant group of baselines
    spec : jnp.ndarray
        Array of spectrally DPSS eigenvectors used for modeling foregrounds
    spat : list of jnp.ndarrays
        Array of spatial PSWF eigenvectors used for modeling foregrounds
    """
    loss = 0
    for _d, _w, fg_model_comps_r, fg_model_comps_i, _spat, blvecs in zip(data, wgts, params["fg_r"], params["fg_i"], spatial_filters, idealized_blvecs):
        loss += _mean_squared_error(
            params, fg_model_comps_r, fg_model_comps_i, _d.real, _d.imag, _w, spectral_filters, _spat, blvecs
        )
        
    return loss

def calibrate_single_integration(data, wgts, initial_params, optimizer, spectral_filters, spatial_filters, idealized_blvecs,
                                    maxiter=100, tol=1e-10):
    """
    Function for calibrating a single polarization/time integration

    Parameters:
    ----------
    data : list of jnp.ndarrays
        List of jnp.ndarrays of data for each radially redundant group of baselines
    wgts : list of jnp.ndarrays
        List of jnp.ndarrays of the data weights for each radially redundant group of baselines
    initial_params : pass
        pass
    optimizer : pass
        pass
    spectral_filters : jnp.ndarray
        Array of spectrally DPSS eigenvectors used for modeling foregrounds
    spatial_filters : list of jnp.ndarrays
        Array of spatial PSWF eigenvectors used for modeling foregrounds
    maxiter : int, default=100
        Maximum number of gradient descent steps to use when finding the optimal parameter values
    tol : float, default=1e-10
        Parameter used to . If the absolute difference between the value of the current step's loss
        and the previous step's loss is less than tol, 

    Return:
    ------
    solution : dict
        pass
    info : dict
        pass
    """
    # Initialize optimizer state using parameter guess
    opt_state = optimizer.init(initial_params)
    params = deepcopy(initial_params)

    # Initialize variables used in calibration loop
    losses = []

    # Start gradient descent
    for step in range(maxiter):
        # Compute loss and gradient
        loss, gradient = _iterate_through_groups(params, data, wgts, spectral_filters=spectral_filters, spatial_filters=spatial_filters, idealized_blvecs=idealized_blvecs)
        updates, opt_state = optimizer.update(gradient, opt_state, params)
        #params = optax.apply_updates(params, updates)
        
        # Store loss values
        losses.append(loss)

        # Stop if losses converge
        if step >= 1 and np.abs(losses[-1] - losses[-2]) < tol:
            break
            
    return params, {"loss": losses[-1], "niter": step + 1}

class NuCalibrator:
    """
    Class for performing frequency redundant calibration of previously redundantly
    calibrated data. This class handles the calibration and degeneracy correction of
    frequency redundantly calibrated gains.
    """
    def __init__(self, reds, radial_reds, freqs, eta_half_width=20e-9, ell_half_width=1, 
                  eigenval_cutoff=1e-12):
        """
        Parameters:
        ----------
        reds : list of lists
            List of groups of baselines that are considered to be spatially redundant
        radial_reds : FrequencyRedundancy
            FrequencyRedundancy object containing of lists of baselines that are considered to
            be radially redundant.
        freqs : np.ndarray
            pass
        """
        # Store radial reds
        self.radial_reds = radial_reds
        
        # Get idealized antenna positions and baseline vectors from reds
        self.idealized_antpos = redcal.reds_to_antpos(reds)
        self.idealized_blvecs = {red[0]: self.idealized_antpos[red[0][1]] - self.idealized_antpos[red[0][0]] for red in reds}
        
        # Get number of tip-tilt dimensions from 
        self.ndims = self.idealized_antpos[list(self.idealized_antpos.keys())[0]].shape[0]
            
        # Compute spatial filters
        self.spatial_filters = compute_spatial_filters(self.radial_reds, freqs, ell_half_width=ell_half_width, eigenval_cutoff=eigenval_cutoff)

        # Compute spectral dpss filters
        self.spectral_filters = dspec.dpss_operator(freqs, [0], [eta_half_width], eigenval_cutoff=[eigenval_cutoff])[0].real
    
    def fit_redcal_degens(self, data, rc_flags, freqs=None, pols=None, learning_rate=1e-3, maxiter=100, 
                  optimizer='adabelief', initial_est_niter=1, tol=1e-10, **opt_kwargs):
        """
        Parameters:
        ----------
        data : DataContainer, RedSol
            pass
        wgts : DataContainer
            pass
        eta_half_width : float, default=20e-9
            Filter half width along the spectral axis
        ell_half_width : float, default=1
            Filter half width along the spatial axis
        eigenval_cutoff : float, default=1e-12
            Cutoff value for eigenvalues of a given size
        learning_rate : float, default=1e-3
            Effective learning rate of chosen optimizer
        maxiter : int, default=100
            Maximum number of iterations to perform
        amp_estimate : dictionary, default=None
            pass
        tiptilt_estimate : dictionary, default=None
            pass
        optimizer : str, default='adabelief'
            Optimizer used when performing gradient descent
        opt_kwargs :
            Additional keyword arguments to be passed to the optimizer chosen. See optax documentation
            for additional details.

        Returns:
        -------
        gains : 
        """
        # Get frequency array data if not given
        if freqs is None:
            if hasattr(data, "freqs"):
                freqs = data.freqs
            else:
                raise ValueError("Frequency array not provided and not found in the data.")

        # Check to make sure all of the baselines in the radially redundant group also exist in the data provided
        for group in self.radial_reds:
            for bl in group:
                assert bl in data

        # Get number of times in the data or infer it
        if hasattr(data, 'times'):
            ntimes = data.times.shape[0]
        else:
            key = list(data.keys())[0]
            ntimes = data[key].shape[0]

        # If pols is None, get pols from the data
        if pols is None:
            pols = np.unique(list(map(lambda k: k[2], data.keys())))

        # Choose optimizer and initialize
        assert optimizer in OPTIMIZERS, "Invalid optimizer type chosen. Please refer to Optax documentation for available optimizers"
        optimizer = OPTIMIZERS[optimizer](learning_rate, **opt_kwargs)
                            
        # Populate with variables
        #data_wgts = build_nucal_wgts()
        data_wgts = {bl: np.ones_like(data[bl], dtype="float64") for bl in data}

        # TODO: compute estimate of the degenerate parameters from spatial filters if not given.
        degen_params, _, model_comps = estimate_degenerate_parameters(
            data, data_wgts, self.radial_reds, rc_flags, self.spatial_filters, niter=initial_est_niter
        )
        # Pack tip-tilts into single entry in dictionary
        for pol in pols:
            degen_params[f"Phi_J{pol}"] = np.transpose([degen_params[f"Phi_{ni}_J{pol}"] for ni in range(self.ndims)], axes=(1, 0, 2))

        degens = {**degen_params, **model_comps}

        
        # Sky Model Parameters should eventually be NPOLS, Ntimes, Ncomps
        # Tip-tilts should be NPOLS, NTIMES, NFREQS, NDIMS
        # Amplitude should be NPOLS, NTIMES, NFREQS
        # Solution keys should be stored as A_J{nn or ee} and Phi_{0, 1, ..., n}_J_{nn or ee}


        # For each time and each polarization in the data calibrate
        solution = {parameter + pol: [] for pol in pols for parameter in ["A_J", "Phi_J"]}
        for pol in pols:
            # Sum over spectral filters to project onto u-model
            const_eigvals = np.sum(self.spectral_filters, axis=0).reshape(-1, 1)

            idealized_blvecs = []
            for group in self.radial_reds.get_pol(pol):
                group_vecs = []
                for bl in group:
                    if bl in self.idealized_blvecs:
                        group_vecs.append(self.idealized_blvecs[bl])
                    elif utils.reverse_bl(bl) in self.idealized_blvecs:
                        group_vecs.append(self.idealized_blvecs[utils.reverse_bl(bl)])

                idealized_blvecs.append(jnp.array(group_vecs))

            spatial_filters = [jnp.array([self.spatial_filters[bl] for bl in group]) for group in self.radial_reds.get_pol(pol)]
            for tind in range(ntimes):
                # Group degenerate parameter estimates into dictionary for the current integration
                initial_params = {
                    f"A": degens[f"A_J{pol}"][tind],
                    f"Phi": degens[f"Phi_J{pol}"][tind],
                    f"fg_r": [np.real(model_comps[group[0]][tind]).reshape(1, -1) * const_eigvals for group in self.radial_reds.get_pol(pol)],
                    f"fg_i": [np.imag(model_comps[group[0]][tind]).reshape(1, -1) * const_eigvals for group in self.radial_reds.get_pol(pol)]
                }

                # Extract data/wgts into arrays for the current integration
                _data = [jnp.array([data[bl][tind] for bl in group]) for group in self.radial_reds.get_pol(pol)]
                _wgts = [jnp.array([data_wgts[bl][tind] for bl in group]) for group in self.radial_reds.get_pol(pol)]
                
                # Main optimization loop function
                fit_params, info = calibrate_single_integration(
                            _data, _wgts, initial_params, optimizer, spectral_filters=self.spectral_filters, 
                            spatial_filters=spatial_filters, maxiter=maxiter, idealized_blvecs=idealized_blvecs, tol=tol
                )

                # TODO: unpack solution and organize it in a sensible way
                # Consider nucal_sol object that gets added to. Nucal sol object could work like
                # dictionary similarly to Redsol
                solution[f"A_J{pol}"].append(fit_params['A'])
                solution[f"Phi_J{pol}"].append(fit_params['Phi'])

        return solution, info