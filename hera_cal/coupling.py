import numpy as np

# Jax libraries
import jax
import jaxopt
from jax import numpy as jnp
jax.config.update("jax_enable_x64", True)

from hera_cal.datacontainer import DataContainer

def project_baselines_to_grid(antpairs, antpos, ew_pair=(0, 1), ns_pair=(0, 11), ratio: int = 3):
    """
    Projects baseline vectors between antenna pairs onto a 2D coordinate system 
    defined by approximate east-west and north-south directions.

    Parameters
    ----------
    antpairs : list of tuple of int
        List of antenna index pairs [(i1, j1), (i2, j2), ...] to project.
    antpos : array_like
        Array of antenna positions with shape (N_antennas, 3).
    ew_pair : tuple of int, optional
        Antenna indices (i, j) that define the reference east-west direction.
        Default is (0, 1).
    ns_pair : tuple of int, optional
        Antenna indices (i, j) that define the reference north-south direction.
        Default is (0, 11).
    ratio : int, optional
        Scaling factor to normalize unit vectors (default is 3).

    Returns
    -------
    np.ndarray
        An array of shape (len(antpairs), 2) with [east, north] projections 
        for each baseline.
    """
    # Define scaled reference vectors
    unit_ew = (antpos[ew_pair[1]] - antpos[ew_pair[0]]) / ratio
    unit_ns = (antpos[ns_pair[1]] - antpos[ns_pair[0]]) / ratio

    # Orthogonalize NS with respect to EW
    unit_vec_ns = unit_ns - np.dot(unit_ns, unit_ew) / np.linalg.norm(unit_ew) ** 2 * unit_ew

    projections = []
    for ap1, ap2 in antpairs:
        vec = antpos[ap2] - antpos[ap1]
        north = np.dot(vec, unit_vec_ns) / np.linalg.norm(unit_vec_ns) ** 2
        east = np.dot(vec - north * unit_ns, unit_ew) / np.linalg.norm(unit_ew) ** 2
        projections.append([east, north])

    return np.array(projections)

def build_coupling_grid(antpos, uvw_grid, ratio: int = 1):
    """
    Builds the coupling grid for the antenna positions.

    Parameters
    ----------
    antpos : array_like
        The antenna positions to be projected.
    uvw_grid : array_like
        The UVW coordinates of the grid.
    ratio : int
        The ratio of the antenna positions to be projected.
    Returns
    -------
    -------
    tuple
        The coupling grid for the antenna positions.
    """
    # Get the antenna pairs
    antpair = np.array([antpos[i] for i in range(len(antpos))])
    # Project the antenna positions onto a grid defined by the ratio
    uvw_grid = project_baselines_to_grid(antpair, antpos, ratio)
    # Build the coupling grid
    coupling_grid = np.zeros((len(antpair), len(uvw_grid)))
    for i in range(len(antpair)):
        coupling_grid[i] = uvw_grid[i]
    return coupling_grid


def loss_function_redundantly_averaged(
    data: jnp.ndarray,
    alpha: float,
    coupling_coefficients: jnp.ndarray,
):
    """
    """
    pass

"""
Group of loss functions for coupling coefficients
"""
@jax.jit
def _scaled_log_1p(data, alpha):
    """
    Computes the scaled log(1 + x) function.

    Parameters
    ----------
        data : jnp.array
            The input data.
        alpha : float
            The scaling factor.
    
    Returns
    -------
        jnp.array
            The scaled log(1 + x) values.
    """
    return jnp.log1p(data * alpha)

@jax.jit
def _scaled_log_1p_normalized(data):
    """
    Computes the scaled log(max(x - 1, 0)) function.

    Parameters
    ----------
        data : jnp.array
            The input data.
        alpha : float
            The scaling factor.
    
    Returns
    -------
        jnp.array
            The scaled log(1 + x) values.
    """
    return jnp.log1p(jnp.maximum(data - 1, 0))

"""
Coupling Model for Non-Redundant Baselines
"""
@jax.jit
def masked_std(
    x: jnp.array, 
    mask: jnp.array
) -> jnp.array:
    """
    Computes the standard deviation of a masked jax array.

    Parameters
    ----------
        x : jnp.array, shape (time, max_len)
            The input array.
        mask : jnp.array, shape (max_len,)
            The mask array.
    Returns
    -------
        jnp.array, shape (time,)
            The standard deviation of the masked array.
    """
    # x: (time, max_len) and mask: (max_len,)
    n = jnp.sum(mask, axis=-1, keepdims=True)
    mean = jnp.sum(x * mask, axis=-1, keepdims=True) / n
    mean_abs_sq = jnp.sum((jnp.abs(x) ** 2) * mask, axis=-1, keepdims=True) / n
    var = mean_abs_sq - jnp.abs(mean) ** 2
    std = jnp.sqrt(var)
    return jnp.squeeze(std, axis=-1)  # (time,)

def pad_reds(
    reds: list[list[tuple[int]]], 
    pad_value: int=0
) -> tuple:
    """
    Pads the antenna indices and mask for each red to the maximum length.

    Parameters
    ----------
        reds : list of list of tuples
            List of redundant baselines, where each baseline is a tuple of antenna indices.
        pad_value : int, optional
            Value to use for padding. Default is 0.
    Returns
    -------
        tuple
            Padded antenna indices and mask.
    """
    # Find the maximum length of the largest redundant baseline group
    max_len = max(len(red[0]) for red in reds)
    padded_ai_list = []
    padded_aj_list = []
    mask_list = []
    
    # Loop over each red and pad the antenna indices and mask
    for ai, aj in reds:
        length = len(ai)
        pad_length = max_len - length
        ai_arr = jnp.concatenate([
            jnp.array(ai, dtype=int),
            jnp.full((pad_length,), pad_value, dtype=int)
        ])
        aj_arr = jnp.concatenate([
            jnp.array(aj, dtype=int),
            jnp.full((pad_length,), pad_value, dtype=int)
        ])
        mask_arr = jnp.concatenate([
            jnp.ones((length,), dtype=float),
            jnp.zeros((pad_length,), dtype=float)
        ])
        padded_ai_list.append(ai_arr)
        padded_aj_list.append(aj_arr)
        mask_list.append(mask_arr)
    
    # Stack the padded arrays and masks into a single array
    padded_ai = jnp.stack(padded_ai_list, axis=0)
    padded_aj = jnp.stack(padded_aj_list, axis=0)
    mask = jnp.stack(mask_list, axis=0)
    return padded_ai, padded_aj, mask

def redundancy_metric(
    vis: jnp.array, 
    padded_ai: jnp.ndarray, 
    padded_aj: jnp.ndarray, 
    mask: jnp.ndarray
) -> jnp.ndarray:
    """
    Computes the sum of the standard deviation of visibilities within a redundant group
    for all redundant groups provided. 
    
    Instead of a for loop, use vmap to compute the masked standard deviation for each red,
    then sum the result.
    
    Parameters
    ----------
        vis : jnp.array, shape (time, dim1, dim2)
        padded_ai : jnp.array, shape (num_reds, max_len)
        padded_aj : jnp.array, shape (num_reds, max_len)
        mask : jnp.array, shape (num_reds, max_len)
    
    Returns
    -------
        jnp.array, shape (time,)
    """
    # Define a per-red function
    def compute_std(ai, aj, red_mask):
        # Advanced indexing: vis[:, ai, aj] has shape (time, max_len)
        red_data = vis[:, ai, aj]
        return masked_std(red_data, red_mask)  # shape: (time,)
    
    # vmap over the first axis (reds)
    all_reds_std = jax.vmap(compute_std, in_axes=(0, 0, 0))(padded_ai, padded_aj, mask)
    # Now sum over the reds axis: result shape (time,)
    return jnp.sum(all_reds_std, axis=0)


@jax.jit
def non_redundantly_averaged_vis_coupling(vis, coupling):
    """
    Couples a visibility matrix of (Ntimes, N_antennas, N_antennas) with a coupling matrix of 
    size (Nfreqs, N_antennas, N_antennas). Here, the coupled visibilities are computed as:
    
        V^{1} = M^H V^{0} M

    where V^{0} is the original visibility matrix, M is the coupling matrix, and 
    V^{1} is the coupled visibility matrix. This model includes second order terms
    in the coupling matrix.
    
    Parameters
    ----------
    vis : array_like
        The visibility data. The shape should be (N_times, Nfreqs, Nants, Nants).
    coupling : array_like
        Matrix of coupling coefficients for the antennas.
        The shape should be (N_antennas, N_antennas).
    
    Returns
    -------
    array_like
        The de-coupled visibility data.
    """
    vis_coupled = jnp.einsum(
        "ab,...bc,cd->...ad", 
        coupling, 
        vis,
        jnp.conjugate(coupling).T
    )
    return vis_coupled

def loss_function_non_redundantly_averaged(
    coupling_coefficients: jnp.ndarray,
    vis: jnp.ndarray,
    padded_ai: jnp.ndarray,
    padded_aj: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """
    Computes the loss function for the coupling coefficients.

    Parameters
    ----------
        coupling_coefficients : jnp.ndarray
            The coupling coefficients.
        vis : jnp.ndarray
            The visibility data.
        padded_ai : jnp.ndarray
            Padded antenna indices.
        padded_aj : jnp.ndarray
            Padded antenna indices.
        mask : jnp.ndarray
            Mask for the visibility data.
        reds : np.ndarray
            The redundant baselines.
    
    Returns
    -------
        jnp.ndarray
            The loss value.
    """
    # Compute the coupled visibilities
    coupled_vis = non_redundantly_averaged_vis_coupling(vis, coupling_coefficients)
    
    # Compute the redundancy metric
    redundancy_metric_value = redundancy_metric(coupled_vis, padded_ai, padded_aj, mask)
    
    # Compute the loss as the sum of squared differences
    loss = jnp.mean(jnp.abs(redundancy_metric_value))
    
    return loss

def solve_coupling_coefficients_non_redundantly_averaged(
    data: DataContainer,
    flags: DataContainer,
    nsamples: DataContainer,
    antpos: dict,
    reds: np.ndarray=None,
    coupling_coefficients: np.ndarray=None,
):
    """
    Solve coupling coefficients for non-redundant baselines using a LBFGS solver.
    This function uses the JAX library for automatic differentiation and GPU acceleration.

    Parameters
    ----------
    data : DataContainer
        The visibility data.
    antpos : dict
        Dictionary of antenna positions.
    reds : np.ndarray, optional
        The redundant baselines.
    coupling_coefficients : np.ndarray, optional
        The coupling coefficients to be optimized.
    
    Returns
    -------
    coupling_coefficients : np.ndarray
        The optimized coupling coefficients.
    """
    # Get the visibility data


    return coupling_coefficients