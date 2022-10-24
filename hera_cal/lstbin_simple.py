"""
An attempt at a simpler LST binner that makes more assumptions but runs faster.

In particular, we assume that all baselines have the same time array and frequency array,
and that each is present throughout the data array. This allows a vectorization.
"""
import numpy as np
from . import utils

def simple_lst_bin(
    data: np.ndarray,
    data_lsts: np.ndarray,
    baselines: list[tuple[int, int]],
    pols: list[str],
    lst_bin_edges: np.ndarray,
    freq_array: np.ndarray,
    flags: np.ndarray | None = None,
    nsamples: np.ndarray | None = None,
    rephase: bool = True,
    antpos: np.ndarray | None = None,
    lat: float = -30.72152,
):
    required_shape = (len(baselines), len(pols), len(data_lsts), len(freq_array))
    if data.shape != required_shape:
        raise ValueError(
            f"data should have shape {required_shape} but got {data.shape}"
        )

    if flags is None:
        flags = np.zeros(data.shape, dtype=bool)

    if flags.shape != required_shape:
        raise ValueError(
            f"flags should have shape {required_shape} but got {flags.shape}"
        )

    if nsamples is None:
        nsamples = np.ones(data.shape, dtype=float)
    
    if nsamples.shape != required_shape:
        raise ValueError(
            f"nsampels should have shape {required_shape} but got {nsamples.shape}"
        )

    if len(lst_bin_edges) < 2:
        raise ValueError("lst_bin_edges must have at least 2 elements")

    # Ensure the lst bin edges start within (0, 2pi)
    while lst_bin_edges[0] < 0:
        lst_bin_edges += 2*np.pi
    while lst_bin_edges[0] >= 2*np.pi:
        lst_bin_edges -= 2*np.pi

    if not np.all(np.diff(lst_bin_edges) > 0):
        raise ValueError(
            "lst_bin_edges must be monotonically increasing."
        )

    # Now ensure that all the observed LSTs are wrapped so they start above the first bin edges
    data_lsts %= 2*np.pi
    data_lsts[data_lsts < lst_bin_edges[0]] += 2* np.pi

    lst_bin_centres = (lst_bin_edges[1:] + lst_bin_edges[:-1])/2

    grid_indices = np.digitize(data_lsts, lst_bin_edges, right=True) - 1

    # Now, any grid index that is less than zero, or len(edges) - 1 is not included in this grid.
    lst_mask = (grid_indices >= 0) & (grid_indices < len(lst_bin_centres))

    # TODO: check whether this creates a data copy. Don't want the extra RAM...
    data = data[:, :, lst_mask]
    flags = flags[:, :, lst_mask]
    nsamples = nsamples[:, :, lst_mask]
    data_lsts = data_lsts[lst_mask]

    # Now, rephase the data to the lst bin centres.
    if rephase:
        if freq_array is None or antpos is None:
            raise ValueError("freq_array and antpos is needed for rephase")

        # form baseline dictionary
        bls = odict([(k, antpos[k[0]] - antpos[k[1]]) for k in d.keys()])

        # get appropriate lst_shift for each integration, then rephase
        lst_shift = lst_bin_centres[grid_indices] - data_lsts

        # this makes a copy of the data in d
        d = utils.lst_rephase(d, bls, freq_array, lst_shift, lat=lat, inplace=False)

    # TODO: check for baseline conjugation stuff.

    davg = np.zeros((len(baselines), len(pols), len(lst_bin_centres), len(freq_array)), dtype=complex)
    flag_min = np.zeros(davg.shape, dtype=bool)
    std = np.ones(davg.shape, dtype=float)
    counts = np.zeros(davg.shape, dtype=float)
    for lstbin in range(len(lst_bin_centres)):
        # TODO: check that this doesn't make yet another copy...
        # This is just the data in this particular lst-bin.
        d = data[:, :, grid_indices==lstbin]

        (
            davg[:, :, lstbin], 
            flags[:, :, lstbin], 
            std[:, :, lstbin], 
            counts[:, :, lstbin]
        ) = lst_average(d)

    # TODO: should we put these back into datacontainers before returning?
    return lst_bin_centres, davg, flag_min, std, counts