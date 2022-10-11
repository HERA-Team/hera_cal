from multiprocessing.sharedctypes import Value
from . import utils
from . import redcal
from . import abscal

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
    Compute prolate spheroidal wave function (PSWF) filters for each radially redundant group in radial_reds. Note
    filtering radial_reds before running this function is advised to reduce the
    size of filters generated

    Parameters:
    ----------
    radial_reds : FrequencyRedundant object
        pass
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
        pass
    """
    spatial_modes = {}

    # Get the minimum and maximum u-bounds used
    u_bounds = get_u_bounds(radial_reds, freqs)

    for gi, group in radial_reds:
        umin, umax = u_bounds[gi]
        for bl in group:
            umodes = radial_reds.baseline_lengths[bl] / SPEED_OF_LIGHT * freqs
            pswf, _ = dspec.pswf_operator(umodes, [0], [ell_half_width], eigenval_cutoff=[eigenval_cutoff], xmin=umin, xmax=umax)
            spatial_modes[bl] = pswf

    return spatial_modes

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
            _wgts = np.zeros_like(data_flags[key], dtype=bool)

            if min_u_cut is not None:
                _wgts[:, umag < min_u_cut] = True

            if max_u_cut is not None:
                _wgts[:, umag > max_u_cut] = True

            if min_freq_cut is not None:
                _wgts[:, freqs < min_freq_cut] = True

            if max_freq_cut is not None:
                _wgts[:, freqs > max_freq_cut] = True
                
            model_flags[key] = _wgts
    
    weights = abscal.build_data_wgts(
        data_flags, data_nsamples, model_flags, autocorrs, auto_flags, 
        times_by_bl=times_by_bl, df=df, data_is_redsol=data_is_redsol, 
        gain_flags=gain_flags, tol=tol, antpos=antpos
    )

    # TODO: unpack into np.ndarrays for each group
    return weights

class NuCalibrator:
    """
    Class for performing frequency redundant calibration of previously redundantly
    calibrated data. This class handles the calibration and degeneracy correction of
    frequency redundantly calibrated gains.
    """
    def __init__(self, reds, radial_reds):
        """
        Parameters:
        ----------
        reds : list of lists
            pass
        radial_reds : FrequencyRedundancy
            pass
        """
        # Store radial reds
        self.radial_reds = radial_reds
        
        # Get idealized antenna positions
        self.idealized_antpos = redcal.reds_to_antpos(reds)
        
        # Get number of tip-tilt dimensions from 
        self.ndims = self.idealized_antpos[list(self.idealized_antpos.keys())[0]].shape[0]
            
        # Build function that will be used to calculate loss
        # TODO: Isolate confirm correct parameter is being used to 
        self.loss = jax.value_and_grad(self._iterate_through_groups)

        # Placeholder variable
        self.X = np.random.uniform(0, 1, size=(2, 1536))
    
    @partial(jax.jit, static_argnums=(0,))
    def test_function(self, X):
        """
        Placeholder for foreground model function. This might be a different function
        
        Parameters:
        ----------
        spec : jnp.ndarray
            pass
        spat : jnp.ndarray
            pass
        beta : jnp.ndarray
            pass
        """
        return jnp.dot(X.T, X).sum()
    
    @partial(jax.jit, static_argnums=(0,))
    def foreground_model(self, skymodes_r, skymodes_i, spec, spat):
        """
        Function for computing foreground models from 
        
        Parameters:
        ----------
        spec : jnp.ndarray
            pass
        spat : jnp.ndarray
            pass
        beta : jnp.ndarray
            pass
        """
        model_r = jnp.einsum('fm,afn,mn->af', spec, spat, skymodes_r, optimize=True)
        model_i = jnp.einsum('fm,afn,mn->af', spec, spat, skymodes_i, optimize=True)
        return model_r, model_i
    
    @partial(jax.jit, static_argnums=(0,))
    def loss_function(self, params, data, wgts, spec, spat):
        """
        Loss function used for solving for redcal degenerate parameters and a model of the sky
        
        Parameters:
        ----------
        data : jnp.ndarray
            pass
        wgts : jnp.ndarray
            pass
        spec : jnp.ndarray
            pass
        spat : jnp.ndarray
            pass
        
        Returns:
        -------
        loss : float
            pass
        """
        # Dot tip-tilt parameters into baseline vector
        phase = ...
        
        # Multiply in amplitude estimate
        gain = ...
        
        # Compute foreground model from beta estimates
        model_r, model_i = self.foreground_model(params['fg_r'], params['fg_i'], spec, spat)
        
        # Compute loss using weights and foreground model
        pass
    
    def _iterate_through_groups(self, params, data, wgts, spec, spat):
        """
        Function for iterating through groups of radially redundant baselines
        to compute the loss function

        Parameters: 
        ----------
        params : dictionary
            pass
        data : list of jnp.ndarrays
            pass
        wgts : list of jnp.ndarrays
            pass
        spec : jnp.ndarray
            pass
        spat : list of jnp.ndarrays
            pass
        """
        loss = 0
        for _d, _w, _spat in zip(data, wgts, spat):
            loss += self.loss_function(params, _d, _w, spec, _spat)
            
        return loss

    def _calibrate_single_integration(self, data, wgts, spec, spat, maxiter=100, return_min_loss=False):
        """
        Function for calibrating a single polarization/time integration
        """
        solution = []
        min_loss = 0
        # Start gradient descent
        for i in range(maxiter):
            # Loss function
            loss, gradient = self.loss(self.X)
            if loss < min_loss:
                min_loss = loss
                if return_min_loss:
                    pass
        
        return min_loss
    
    def calibrate(self, data, wgts, freqs=None, pols=["nn"], eta_half_width=20e-9, ell_half_width=1, 
                  eigenval_cutoff=1e-12, learning_rate=1e-3, maxiter=100, optimizer='adabelief', 
                  return_min_loss=False, **opt_kwargs):
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
        degen_guess: dictionary, default=None
            pass
        optimizer : str, default='adabelief'
            Optimizer used when performing gradient descent
        return_min_loss : bool, default=False
            Return solution with the minimum loss found. If False, solution found on the last iteration 
            will be returned.
        opt_kwargs :
            Additional keyword arguments to be passed to the optimizer chosen. See optax documentation
            for additional details.
        """
        # Think about number of polarizations. I don't think I can necessarily do them at the same time
        # If there's a baseline type missing because if there is a baseline that happens to not be in
        # the data, then arrays will different sizes. I could just hope it mostly works out but I don't
        # know if that's necessarily the best thing to do
        if freqs is None:
            if hasattr(data, "freqs"):
                freqs = data.freqs
            else:
                raise ValueError("Frequency array not provided and not found in the data.")

        if hasattr(data, 'times'):
            ntimes = data.times.shape[0]
        else:
            ntimes = 1

        # Choose optimizer
        assert optimizer in OPTIMIZERS, "Invalid optimizer type chosen. Please refer to Optax documentation for available optimizers"
        opt = OPTIMIZERS[optimizer](learning_rate, **opt_kwargs)
        
        if degen_guess is None:
            # Get estimate of gains from amplitude guess
            degen_guess = ...
        
        # Compute spatial filters used for calibration
        spat = compute_spatial_filters(self.radial_reds, ell_half_width=ell_half_width, eigenval_cutoff=eigenval_cutoff)
        spec = dspec.dpss_operator(freqs)

        # Set initial loss
        min_loss = np.inf
        losses = []
        
        # Sky Model Parameters should eventually be NPOLS, Ntimes, Ncomps
        # Tip-tilts should be NPOLS, NTIMES, NFREQS, NDIMS
        # Amplitude should be NPOLS, NTIMES, NFREQS
        
        solution, info = {}, {}
        for pol in pols:
            for tind in range(ntimes):
                pass
            
        return solution, info