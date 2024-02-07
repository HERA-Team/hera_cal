"""
Module for calibrating visibilities by comparing data which lie within the same LST-bin
"""
from __future__ import annotations
import linsolve
import numpy as np
from . import utils


def compute_offsets(
    data: np.ndarray,
    flags: np.ndarray,
    antpairs: list[tuple[int, int]],
    pols: list[str],
    cal_function: callable,
    day_flags: np.ndarray | None = None,
    bls_flags: np.ndarray | None = None,
    ref_operation: str = "multiply",
    ref_value: float = 1.0,
    **kwargs,
) -> tuple[dict, dict]:
    """
    Compute the offsets between days in an LST-bin using a given calibration function.

    Parameters:
    -----------
    data : np.ndarray
        Shape (Ntimes, Nbls, Nfreqs, Npols) of complex data.
    flags : np.ndarray
        Shape (Ntimes, Nbls, Nfreqs, Npols) of boolean flags.
    antpairs : list of tuples
        List of antenna pairs. Must contain the same number of baselines as the data and order must match
        the data.
    pols : list of strings
        List of polarizations. Must contain the same number of polarizations as the data and order must match 
        the data.
    cal_function : callable
        Function to use to solve for the offsets between days in an LST-bin.
    day_flags : np.ndarray, default=None
        Boolean array of shape (Ndays,) indicating which days are flagged across all baselines.
    bls_flags : np.ndarray, default=None
        Boolean array of shape (Nbls,) indicating which baselines are flagged across all days.
    ref_operation : str
        Operation to perform on the reference data. Default is 'multiply'.
    ref_value : float
        Value to use for the reference data. Default is 1.0.
    kwargs : dict
        Additional keyword arguments to pass to the cal_function.

    Returns:
    --------
    offsets : dict
        Dictionary of offsets between days in the lstbin.
    index_dict : dict
        Dictionary of indices for each baseline in the lstbin.
    """
    # Get shape of data
    ndays, nbls, _, npols = data.shape

    # If no day_flags is provided, assume all days are usable
    if day_flags is None:
        day_flags = np.zeros(ndays, dtype=bool)
    # If no bls_flags is provided, assume all baselines are usable
    if bls_flags is None:
        bls_flags = np.zeros(nbls, dtype=bool)

    # Dictionary for storing the baseline and polarization indices of each baseline
    index_dict = {}

    # Dictionary for storing the offsets between days for each baseline
    offsets = {}

    # Loop through all polarizations
    for pi in range(npols):
        # Loop through all baselines
        for bi in range(nbls):
            if bls_flags[bi]:
                continue

            _data, _flags = {}, {}

            # Loop through all days for a given baseline
            for di in range(ndays):
                # Skip if day is flagged
                if day_flags[di]:
                    continue

                # If the data is not flagged, add it to the dictionary
                if not np.all(flags[di, bi, :, pi]):
                    _data[(di, pols[pi])] = data[di, bi, :, pi][None]
                    _flags[(di, pols[pi])] = flags[di, bi, :, pi][None]

            # If more than one day for this baseline is unflagged, solve for the offset between days
            if len(_data) > 1:
                offsets[antpairs[bi] + (pols[pi],)] = hierachical_pairing(
                    cal_function,
                    data=_data,
                    flags=_flags,
                    ref_value=ref_value,
                    ref_operation=ref_operation,
                    **kwargs,
                )
                index_dict[antpairs[bi] + (pols[pi],)] = (bi, pi)

    return offsets, index_dict


def hierachical_pairing(
    pairing_function: callable,
    data: dict[tuple[int, str], np.ndarray],
    flags: dict[tuple[int, str], np.ndarray] | None = None,
    ref_value: float = 1.0,
    ref_operation: str = "multiply",
    **kwargs,
) -> dict[tuple[int, int, str], float]:
    """
    Hierachically pair data to solve for the offsets between days in an LST-bin. Days are paired
    together in a binary tree, with the offsets between days being solved for at each level. The
    final offset is the product of all the offsets between days.

    Parameters:
    -----------
    pairing_function : callable
        Function to use to solve for the offsets between days in an LST-bin.
    data : np.ndarray
        Shape (Ntimes, Nbls, Nfreqs, Npols) of complex data.
    flags : np.ndarray
        Shape (Ntimes, Nbls, Nfreqs, Npols) of boolean flags.
    ref_value : float
        Value to use for the reference data. Default is 1.0.
    ref_operation : str
        Operation to perform when comparing all the value of the offsets between days in the lstbin.
        Default is 'multiply'.
    kwargs : dict
        Additional keyword arguments to pass to the pairing_function.

    Returns:
    --------
    offsets : dict
        Dictionary of offsets between days in the lstbin.
    """
    # Get keys within the data
    bls = [bl for bl in data]
    keys = [(key,) for key in data]
    _data = {key: data[key[0]] for key in keys}

    # If flags are provided, use them
    if flags:
        _flags = {key: flags[key[0]] for key in keys}
    else:
        _flags = {key: np.zeros(data[key[0]].shape, dtype="bool") for key in keys}

    values = {}

    while len(keys) > 1:
        new_keys = []
        for key1, key2 in zip(keys[::2], keys[1::2]):
            values[key2] = pairing_function(
                data=_data, flags=_flags, key1=key1, key2=key2, **kwargs
            )
            new_keys.append(key1 + key2)
        # deal with stragglers
        if len(keys) % 2 == 1:
            values[keys[-1]] = pairing_function(
                data=_data, flags=_flags, key1=new_keys[-1], key2=keys[-1], **kwargs
            )
            new_keys = new_keys[:-1] + [new_keys[-1] + keys[-1]]

        keys = new_keys

    offsets = {}

    # Get the reference key
    bl0 = bls[0]

    for group, offset in values.items():
        for bl1 in group:
            offset0 = offsets.get((bl0, bl1), ref_value)
            if ref_operation == "multiply":
                offsets[(bl0, bl1)] = offset * offset0
            elif ref_operation == "add":
                offsets[(bl0, bl1)] = offset + offset0
            else:
                raise ValueError(
                    "Invalid ref_operation. Must be in ['multiply', 'add']"
                )

    return offsets


def _phase_slope_align_bls(data, flags, key1, key2, norm=True):
    """
    Function for comparing the phase slope between two days of data.
    """
    # Compute the product between the two groups
    d12 = data[key1] * np.conj(data[key2])

    # Normalize product
    if norm:
        ad12 = np.abs(d12)
        np.divide(d12, ad12, out=d12, where=(ad12 != 0))

    phase_offset = np.nanmedian(np.angle(d12), axis=1, keepdims=True)

    # Construct a phasor to phase-align the two groups
    phasor = np.exp(1j * phase_offset)
    rephased = data[key2] * phasor

    # Compute the combined data
    new_val = (
        flags[key1] * data[key2]
        + flags[key2] * data[key1]
        + (1 - flags[key1] - flags[key2]) * (data[key1] + rephased) / 2
    )

    # Identify regions where both groups are flagged and replace with 1 + 0j
    flags[key1 + key2] = np.logical_and(flags[key1], flags[key2])
    data[key1 + key2] = np.where(flags[key1 + key2], 1 + 0j, new_val)
    return phase_offset


def _delay_align_bls(data, flags, key1, key2, freqs, norm=True):
    """ 
    Function for comparing the delay between two days of data.
    """
    # Compute the product between the two groups
    d12 = data[key1] * np.conj(data[key2])

    # Normalize product
    if norm:
        ad12 = np.abs(d12)
        np.divide(d12, ad12, out=d12, where=(ad12 != 0))

    # Find the delay peak in the data product
    delay, _ = utils.fft_dly(
        d12,
        np.diff(freqs)[0],
        wgts=np.logical_not(flags[key1] | flags[key2]).astype(float),
    )

    # If fft_dly returned a nan, set the delay to zero
    delay[np.isnan(delay)] = 0

    # Construct a phasor to phase-align the two groups
    phasor = np.exp(2j * np.pi * delay * freqs)
    rephased = data[key2] * phasor

    # Compute the combined data
    new_val = (
        flags[key1] * data[key2]
        + flags[key2] * data[key1]
        + (1 - flags[key1] - flags[key2]) * (data[key1] + rephased) / 2
    )

    # Identify regions where both groups are flagged and replace with 1 + 0j
    flags[key1 + key2] = np.logical_and(flags[key1], flags[key2])
    data[key1 + key2] = np.where(flags[key1 + key2], 1 + 0j, new_val)
    return delay


def _tip_tilt_align(data, flags, key1, key2, norm=True):
    """Phase-align two groups, recording dly/off in dly_off_gps for gp2
    and the phase-aligned sum in _data. Returns gp1 + gp2, which
    keys the _data dict and represents group for next iteration."""
    d12 = data[key1] * np.conj(data[key2])
    if norm:
        ad12 = np.abs(d12)
        np.divide(d12, ad12, out=d12, where=(ad12 != 0))

    # Now that we know the slope, estimate the remaining phase offset
    angle = np.angle(d12)
    rephased = data[key2] * np.exp(1j * angle)
    new_val = (
        flags[key1] * data[key2]
        + flags[key2] * data[key1]
        + (1 - flags[key1] - flags[key2]) * (data[key1] + rephased) / 2
    )

    # Identify regions where both groups are flagged and replace with 1 + 0j
    flags[key1 + key2] = np.logical_and(flags[key1], flags[key2])
    data[key1 + key2] = np.where(flags[key1 + key2], 1 + 0j, new_val)
    return angle


def _amplitude_align(data, flags, key1, key2):
    """ 
    Function for comparing the amplitude between two days of data.
    """
    d12 = np.abs(data[key1]) / np.abs(data[key2])

    # Average the two groups together, weighted by their amplitudes
    rescaled = data[key2] * d12
    new_val = (
        flags[key1] * data[key2]
        + flags[key2] * data[key1]
        + (1 - flags[key1] - flags[key2]) * (data[key1] + rescaled) / 2
    )

    # Identify regions where both groups are flagged and replace with 1 + 0j
    flags[key1 + key2] = np.logical_and(flags[key1], flags[key2])
    data[key1 + key2] = np.where(flags[key1 + key2], 1 + 0j, new_val)
    return d12


def delay_slope_calibration(
    data: np.ndarray,
    flags: np.ndarray,
    nsamples: np.ndarray,
    freqs: np.ndarray,
    antpairs: list[tuple[int, int]],
    antpos: list[dict[int, np.ndarray]] | dict[int, np.ndarray],
    pols: list[str],
    sparse: bool = True,
    solver_method: str = "default",
    day_flags: np.ndarray | None = None,
    bls_flags: np.ndarray | None = None,
) -> tuple[dict, dict]:
    """
    Solve for the delay slope degeneracy of each day in an LST-bin.

    Parameters:
    -----------
    data : np.ndarray
        Shape (Ndays, Nbls, Nfreqs, Npols) of complex data.
    flags : np.ndarray
        Shape (Ndays, Nbls, Nfreqs, Npols) of boolean flags.
    nsamples : np.ndarray
        Number of samples in each time-frequency bin. Shape (Ndays, Nbls, Nfreqs, Npols).
    freqs : np.ndarray
        Frequency array in Hz.
    antpairs : list of tuples
        List of antenna pairs. Must contain the same number of baselines as the data and order must match
        the data.
    antpos : list of dictionarys, or dictionary
        List of dictionaries of idealized antenna positions. If only one dictionary is provided, it will be used for all days.
        If a list of dictionaries is provided, the length of the list must match the number of days.
        Each dictionary must contain the same number of antennas and the same keys.
    pols : list of strings
        List of polarizations. Must contain the same number of polarizations as the data and order must match 
        the data.
    sparse : bool, default=True
        If True, linsolve will use sparse matrices to solve for the delay slopes.
    solver_method : str, default='default'
        Method for linsolve to use when solving for the calibration terms. 
        Options are 'default', 'pinv', 'solve', and 'lstsq'.
    day_flags : np.ndarray, default=None
        Boolean array of shape (Ndays,) indicating which days are flagged across all baselines.
    bls_flags : np.ndarray, default=None
        Boolean array of shape (Nbls,) indicating which baselines are flagged across all days.

    Returns:
    --------
    delay_slope : np.ndarray
        Shape (Ntimes, Nfreqs, Npols) of delay slopes in nanoseconds.
    """
    # Loop through all baselines
    delays, index_dict = compute_offsets(
        data,
        flags,
        antpairs,
        pols,
        cal_function=_delay_align_bls,
        day_flags=day_flags,
        bls_flags=bls_flags,
        freqs=freqs,
        ref_value=0.0,
        ref_operation="add",
    )

    # Shape of the data
    ndays, _, _, _ = data.shape

    # Get the antennas from the antpairs
    ants = list(set(sum(map(list, antpairs), [])))

    # Check if antpos is a dictionary or a list of dictionaries
    use_same_antpos = True if isinstance(antpos, dict) else False

    # Solutions
    solutions = {}
    gains = {}

    # Calibration polarizations indepedently
    for pol in pols:
        # Setup equations
        ls_data, const, wgts = {}, {}, {}

        # Get the baselines for this polarization
        baselines = [bl for bl in delays if pol == bl[-1]]

        for bl in baselines:
            bi, pi = index_dict[bl]

            # If only one antpos is provided share it across all days, otherwise use day dependent antpos
            if use_same_antpos:
                blvec = antpos[bl[1]] - antpos[bl[0]]
                const.update(
                    {
                        f"b_{bl[0]}_{bl[1]}_{ni}": blvec[ni]
                        for ni in range(blvec.shape[0])
                    }
                )
            else:
                blvec_shape = []
                # Loop through all of the days for this baseline
                for di in range(ndays):
                    blvec = antpos[di][bl[1]] - antpos[di][bl[0]]
                    const.update(
                        {
                            f"b_{bl[0]}_{bl[1]}_{ni}_{di}": blvec[ni]
                            for ni in range(blvec.shape[0])
                        }
                    )
                    blvec_shape.append(blvec.shape[0])

            # Loop through all of the tip-tilt offsets
            for day1, day2 in delays[bl]:
                # Form the data key
                if use_same_antpos:
                    data_key_1 = " + ".join(
                        [
                            f"b_{bl[0]}_{bl[1]}_{ni} * T{ni}_{day2[0]}"
                            for ni in range(blvec.shape[0])
                        ]
                    )
                    data_key_2 = " - ".join(
                        [
                            f"b_{bl[0]}_{bl[1]}_{ni} * T{ni}_{day1[0]}"
                            for ni in range(blvec.shape[0])
                        ]
                    )
                else:
                    data_key_1 = " + ".join(
                        [
                            f"b_{bl[0]}_{bl[1]}_{ni}_{day2[0]} * T{ni}_{day2[0]}"
                            for ni in range(blvec_shape[day2[0]])
                        ]
                    )
                    data_key_2 = " - ".join(
                        [
                            f"b_{bl[0]}_{bl[1]}_{ni}_{day1[0]} * T{ni}_{day1[0]}"
                            for ni in range(blvec_shape[day1[0]])
                        ]
                    )

                # Load data from the blgrp into the linear system
                ls_data[data_key_1 + " - " + data_key_2] = delays[bl][(day1, day2)][0]

                # Weight by flags and nsamples
                wgt = np.logical_not(flags[day1[0], bi, :, pi]).astype(
                    float
                ) * np.logical_not(flags[day2[0], bi, :, pi]).astype(float)
                wgt *= np.sqrt(
                    nsamples[day1[0], bi, :, pi] * nsamples[day2[0], bi, :, pi]
                )
                wgts[data_key_1 + " - " + data_key_2] = np.median(wgt)

        # Solve for the delay slope
        ls = linsolve.LinearSolver(ls_data, wgts=wgts, sparse=sparse, **const)
        fit = ls.solve(mode=solver_method)

        # Pack the solution into a dictionary
        solutions[pol] = fit
        _gains = {}
        for ant in ants:
            phase = []
            for ti in range(ndays):
                if use_same_antpos:
                    delay = np.sum(
                        [
                            antpos[ant][n] * fit[f"T{n}_{ti}"]
                            for n in range(antpos[ant].shape[0])
                        ],
                        axis=0,
                    )
                else:
                    delay = np.sum(
                        [
                            antpos[ti][ant][n] * fit[f"T{n}_{ti}"]
                            for n in range(antpos[ti][ant].shape[0])
                        ],
                        axis=0,
                    )
                phase.append(delay * freqs)

            _gains[(ant, "J" + "nn")] = np.exp(2j * np.pi * np.array(phase))
        gains.update(_gains)

    return gains, solutions


def global_phase_slope_calibration(
    data: np.ndarray,
    flags: np.ndarray,
    nsamples: np.ndarray,
    antpairs: list[tuple[int, int]],
    antpos: list[dict[int, np.ndarray]] | dict[int, np.ndarray],
    pols: list[str],
    day_flags: np.ndarray | None = None,
    bls_flags: np.ndarray | None = None,
    sparse: bool = True,
    solver_method: str = "default",
) -> tuple[dict, dict]:
    """
    Solve for the global phase slope degeneracy of each day in an LST-bin.

    Parameters:
    -----------
    data : np.ndarray
        Shape (Ndays, Nbls, Nfreqs, Npols) of complex data.
    flags : np.ndarray
        Shape (Ndays, Nbls, Nfreqs, Npols) of boolean flags.
    nsamples : np.ndarray
        Number of samples in each time-frequency bin. Shape (Ndays, Nbls, Nfreqs, Npols).
    antpairs : list of tuples
        List of antenna pairs. Must contain the same number of baselines as the data and order must match
        the data.
    antpos : list of dicts or dict
        List of dictionaries of idealized antenna positions. If only one dictionary is provided, it will be used for all days.
        If a list of dictionaries is provided, the length of the list must match the number of days.
        Each dictionary must contain the same number of antennas and the same keys.
    pols : list of strings
        List of polarizations. Must contain the same number of polarizations as the data and order must match 
        the data.
    sparse : bool, default=True
        If True, linsolve will use sparse matrices to solve for the delay slopes.
    solver_method : str, default='default'
        Method for linsolve to use when solving for the calibration terms. 
        Options are 'default', 'pinv', 'solve', and 'lstsq'.
    day_flags : np.ndarray, default=None
        Boolean array of shape (Ndays,) indicating which days are flagged across all baselines.
    bls_flags : np.ndarray, default=None
        Boolean array of shape (Nbls,) indicating which baselines are flagged across all days.
    
    Returns:
    --------
    gains : dict
        Dictionary of gains with keys (ant, pol) and values of shape (Ndays, Nfreqs).
    phase_slope : np.ndarray
        Shape (Ntimes, Nfreqs, Npols) of phase slopes in radians per meter.
    """
    # Loop through all baselines
    phase_slopes, index_dict = compute_offsets(
        data,
        flags,
        antpairs,
        pols,
        cal_function=_phase_slope_align_bls,
        day_flags=day_flags,
        bls_flags=bls_flags,
        ref_value=0.0,
        ref_operation="add",
    )

    # Get the antennas from the antpairs
    ants = list(set(sum(map(list, antpairs), [])))

    # Get shape of data
    ndays, _, _, _ = data.shape

    # Check if antpos is a dictionary or a list of dictionaries
    use_same_antpos = True if isinstance(antpos, dict) else False
    gains = {}
    solutions = {}

    # Calibration polarizations indepedently
    for pol in pols:
        # Setup equations
        ls_data, const, wgts = {}, {}, {}

        # Get the baselines for this polarization
        baselines = [bl for bl in phase_slopes if pol == bl[-1]]

        for bl in baselines:
            # Get the baseline and polarzation index for this baseline
            bi, pi = index_dict[bl]

            # Loop through all of the tip-tilt solutions
            if use_same_antpos:
                blvec = antpos[bl[1]] - antpos[bl[0]]
                const.update(
                    {
                        f"b_{bl[0]}_{bl[1]}_{ni}": blvec[ni]
                        for ni in range(blvec.shape[0])
                    }
                )
            else:
                # Loop through all of the days for this baseline
                blvec_shape = []
                for di in range(ndays):
                    blvec = antpos[di][bl[1]] - antpos[di][bl[0]]
                    const.update(
                        {
                            f"b_{bl[0]}_{bl[1]}_{ni}_{di}": blvec[ni]
                            for ni in range(blvec.shape[0])
                        }
                    )
                    blvec_shape.append(blvec.shape[0])

            # Loop through all of the tip-tilt offsets
            for day1, day2 in phase_slopes[bl]:
                if use_same_antpos:
                    data_key_1 = " + ".join(
                        [
                            f"b_{bl[0]}_{bl[1]}_{ni} * Phi{ni}_{day2[0]}"
                            for ni in range(blvec.shape[0])
                        ]
                    )
                    data_key_2 = " - ".join(
                        [
                            f"b_{bl[0]}_{bl[1]}_{ni} * Phi{ni}_{day1[0]}"
                            for ni in range(blvec.shape[0])
                        ]
                    )
                else:
                    data_key_1 = " + ".join(
                        [
                            f"b_{bl[0]}_{bl[1]}_{ni}_{day2[0]} * Phi{ni}_{day2[0]}"
                            for ni in range(blvec_shape[day2[0]])
                        ]
                    )
                    data_key_2 = " - ".join(
                        [
                            f"b_{bl[0]}_{bl[1]}_{ni}_{day1[0]} * Phi{ni}_{day1[0]}"
                            for ni in range(blvec_shape[day1[0]])
                        ]
                    )
                ls_data[data_key_1 + " - " + data_key_2] = phase_slopes[bl][
                    (day1, day2)
                ][0]

                # Weight by flags and nsamples
                wgt = np.logical_not(flags[day1[0], bi, :, pi]).astype(
                    float
                ) * np.logical_not(flags[day2[0], bi, :, pi]).astype(float)
                wgt *= np.sqrt(
                    nsamples[day1[0], bi, :, pi] * nsamples[day2[0], bi, :, pi]
                )
                wgts[data_key_1 + " - " + data_key_2] = np.nanmedian(wgt)

        # Solve system of equations
        solver = linsolve.LinearSolver(ls_data, wgts=wgts, sparse=sparse, **const)
        fit = solver.solve(mode=solver_method)

        # Pack the solution into a dictionary
        solutions[pol] = fit

        _gains = {}
        for ant in ants:
            phase = []
            for ti in range(ndays):
                if use_same_antpos:
                    _phase = np.sum(
                        [
                            antpos[ant][ni] * fit[f"Phi{ni}_{ti}"]
                            for ni in range(antpos[ant].shape[0])
                        ],
                        axis=0,
                    )
                else:
                    _phase = np.sum(
                        [
                            antpos[ti][ant][ni] * fit[f"Phi{ni}_{ti}"]
                            for ni in range(antpos[ti][ant].shape[0])
                        ],
                        axis=0,
                    )
                phase.append(_phase)

            _gains[(ant, "J" + pol)] = np.exp(1j * np.array(phase))

        gains.update(_gains)

    return gains, solutions


def tip_tilt_calibration(
    data: np.ndarray,
    flags: np.ndarray,
    nsamples: np.ndarray,
    antpairs: list[tuple[int, int]],
    antpos: list[dict[int, np.ndarray]] | dict[int, np.ndarray],
    pols: list[str],
    day_flags: np.ndarray | None = None,
    bls_flags: np.ndarray | None = None,
    sparse: bool = True,
    solver_method: str = "default",
) -> tuple[dict, dict]:
    """
    Solve for the per-frequency phase slope degeneracy of each day in an LST-bin.

    Parameters:
    -----------
    data : np.ndarray
        Shape (Ndays, Nbls, Nfreqs, Npols) of complex data.
    flags : np.ndarray
        Shape (Ndays, Nbls, Nfreqs, Npols) of boolean flags.
    nsamples : np.ndarray
        Number of samples in each time-frequency bin. Shape (Ndays, Nbls, Nfreqs, Npols).
    antpairs : list of tuples
        List of antenna pairs. Must contain the same number of baselines as the data and order must match
        the data.
    antpos : list of dicts or dict
        List of dictionaries of idealized antenna positions. If only one dictionary is provided, it will be used for all days.
        If a list of dictionaries is provided, the length of the list must match the number of days.
        Each dictionary must contain the same number of antennas and the same keys.
    pols : list of strings
        List of polarizations. Must contain the same number of polarizations as the data and order must match 
        the data.
    sparse : bool, default=True
        If True, linsolve will use sparse matrices to solve for the delay slopes.
    solver_method : str, default='default'
        Method for linsolve to use when solving for the calibration terms. 
        Options are 'default', 'pinv', 'solve', and 'lstsq'.
    day_flags : np.ndarray, default=None
        Boolean array of shape (Ndays,) indicating which days are flagged across all baselines.
    bls_flags : np.ndarray, default=None
        Boolean array of shape (Nbls,) indicating which baselines are flagged across all days.

    Returns:
    --------
    phase_slope : np.ndarray
        Shape (Ntimes, Nfreqs, Npols) of phase slopes in radians per meter.
    """
    # Loop through all baselines
    tip_tilts, index_dict = compute_offsets(
        data,
        flags,
        antpairs,
        pols,
        cal_function=_tip_tilt_align,
        day_flags=day_flags,
        bls_flags=bls_flags,
        ref_value=0.0,
        ref_operation="add",
    )

    # Get the antennas from the antpairs
    ants = list(set(sum(map(list, antpairs), [])))

    # Get shape of data
    ndays, _, _, _ = data.shape

    # Check if antpos is a dictionary or a list of dictionaries
    use_same_antpos = True if isinstance(antpos, dict) else False
    gains = {}
    solutions = {}

    # Calibration polarizations indepedently
    for pol in pols:
        # Setup equations
        ls_data, const, wgts = {}, {}, {}

        # Get the baselines for this polarization
        baselines = [bl for bl in tip_tilts if pol == bl[-1]]

        for bl in baselines:
            # Get the baseline and polarzation index for this baseline
            bi, pi = index_dict[bl]

            # Loop through all of the tip-tilt solutions
            if use_same_antpos:
                blvec = antpos[bl[1]] - antpos[bl[0]]
                const.update(
                    {
                        f"b_{bl[0]}_{bl[1]}_{ni}": blvec[ni]
                        for ni in range(blvec.shape[0])
                    }
                )
            else:
                # Loop through all of the days for this baseline
                blvec_shape = []
                for di in range(ndays):
                    blvec = antpos[di][bl[1]] - antpos[di][bl[0]]
                    const.update(
                        {
                            f"b_{bl[0]}_{bl[1]}_{ni}_{di}": blvec[ni]
                            for ni in range(blvec.shape[0])
                        }
                    )
                    blvec_shape.append(blvec.shape[0])

            # Loop through all of the tip-tilt offsets
            for day1, day2 in tip_tilts[bl]:
                if use_same_antpos:
                    data_key_1 = " + ".join(
                        [
                            f"b_{bl[0]}_{bl[1]}_{ni} * TT{ni}_{day2[0]}"
                            for ni in range(blvec.shape[0])
                        ]
                    )
                    data_key_2 = " - ".join(
                        [
                            f"b_{bl[0]}_{bl[1]}_{ni} * TT{ni}_{day1[0]}"
                            for ni in range(blvec.shape[0])
                        ]
                    )
                else:
                    data_key_1 = " + ".join(
                        [
                            f"b_{bl[0]}_{bl[1]}_{ni}_{day2[0]} * TT{ni}_{day2[0]}"
                            for ni in range(blvec_shape[day2[0]])
                        ]
                    )
                    data_key_2 = " - ".join(
                        [
                            f"b_{bl[0]}_{bl[1]}_{ni}_{day1[0]} * TT{ni}_{day1[0]}"
                            for ni in range(blvec_shape[day1[0]])
                        ]
                    )
                ls_data[data_key_1 + " - " + data_key_2] = tip_tilts[bl][(day1, day2)][
                    0
                ]

                # Weight by flags and nsamples
                wgt = np.logical_not(flags[day1[0], bi, :, pi]).astype(
                    float
                ) * np.logical_not(flags[day2[0], bi, :, pi]).astype(float)
                wgt *= np.sqrt(
                    nsamples[day1[0], bi, :, pi] * nsamples[day2[0], bi, :, pi]
                )
                wgts[data_key_1 + " - " + data_key_2] = wgt

        # Solve system of equations
        solver = linsolve.LinearSolver(ls_data, wgts=wgts, sparse=sparse, **const)
        fit = solver.solve(mode=solver_method)

        # Pack the solution into a dictionary
        solutions[pol] = fit

        _gains = {}
        for ant in ants:
            phase = []
            for ti in range(ndays):
                if use_same_antpos:
                    _phase = np.sum(
                        [
                            antpos[ant][ni] * fit[f"TT{ni}_{ti}"]
                            for ni in range(antpos[ant].shape[0])
                        ],
                        axis=0,
                    )
                else:
                    _phase = np.sum(
                        [
                            antpos[ti][ant][ni] * fit[f"TT{ni}_{ti}"]
                            for ni in range(antpos[ti][ant].shape[0])
                        ],
                        axis=0,
                    )
                phase.append(_phase)

            _gains[(ant, "J" + pol)] = np.exp(1j * np.array(phase))

        gains.update(_gains)

    return gains, solutions


def amplitude_calibration(
    data: np.ndarray,
    flags: np.ndarray,
    nsamples: np.ndarray,
    antpairs: list[tuple[int, int]],
    pols: list[str],
    day_flags: np.ndarray | None = None,
    bls_flags: np.ndarray | None = None,
    sparse: bool = True,
    solver_method: str = "default",
) -> tuple[dict, dict]:
    """
    Solves for the frequency-dependent amplitude degeneracy of each day in an LST-bin.

    Parameters:
    -----------
    data : np.ndarray
        Shape (Ndays, Nbls, Nfreqs, Npols) of complex data.
    flags : np.ndarray
        Shape (Ndays, Nbls, Nfreqs, Npols) of boolean flags.
    nsamples : np.ndarray
        Number of samples in each time-frequency bin. Shape (Ndays, Nbls, Nfreqs, Npols).
    antpairs : list of tuples
        List of antenna pairs. Must contain the same number of baselines as the data and order must match
        the data.
    pols : list of strings
        List of polarizations. Must contain the same number of polarizations as the data and order must match 
        the data.
    day_flags : np.ndarray, default=None
        Boolean array of shape (Ndays,) indicating which days are flagged across all baselines.
    bls_flags : np.ndarray, default=None
        Boolean array of shape (Nbls,) indicating which baselines are flagged across all days.
    sparse : bool, default=True
        If True, linsolve will use sparse matrices to solve for the delay slopes.
    solver_method : str, default='default'
        Method for linsolve to use when solving for the amplitude and phase calibration terms. 
        Options are 'default', 'pinv', 'solve', and 'lstsq'.

    Returns:
    --------
    amplitude : np.ndarray
        Shape (Ntimes, Nfreqs, Npols) of amplitudes.
    """
    # Get shape of data
    ndays, _, nfreqs, _ = data.shape

    # Get the antennas from the antpairs
    ants = list(set(sum(map(list, antpairs), [])))

    # Loop through all baselines and compute the amplitude offsets
    amps, index_dict = compute_offsets(
        data,
        flags,
        antpairs,
        pols,
        cal_function=_amplitude_align,
        day_flags=day_flags,
        bls_flags=bls_flags,
        ref_value=1.0,
        ref_operation="multiply",
    )

    # Store solutions in a dictionary keyed by polarization
    solutions = {}
    gains = {}

    # Solve for the amplitude offsets for each polarization independently
    for pol in pols:
        # Setup equations
        ls_data = {}
        const = {}
        wgts = {}

        baselines = [bl for bl in amps if pol == bl[-1]]
        for bl in baselines:
            bi, pi = index_dict[bl]
            for day1, day2 in amps[bl]:
                # Construct the data key
                data_key_1 = f"a_{day2[0]}_{bl[0]}_{bl[1]} * eta_{day2[0]} - a_{day1[0]}_{bl[0]}_{bl[1]} * eta_{day1[0]}"
                const[f"a_{day2[0]}_{bl[0]}_{bl[1]}"] = 2.0
                const[f"a_{day1[0]}_{bl[0]}_{bl[1]}"] = 2.0

                # Load data from the blgrp into the linear system
                ls_data[data_key_1] = np.log(amps[bl][(day1, day2)][0])

                # Weight by flags and nsamples
                wgt = np.logical_not(flags[day1[0], bi, :, pi]).astype(
                    float
                ) * np.logical_not(flags[day2[0], bi, :, pi]).astype(float)
                wgt *= np.sqrt(
                    nsamples[day1[0], bi, :, pi] * nsamples[day2[0], bi, :, pi]
                )
                wgts[data_key_1] = wgt

        # Solve for the amplitude offsets
        solver = linsolve.LinearSolver(ls_data, wgts=wgts, sparse=sparse, **const)
        fit = solver.solve(mode=solver_method)
        solutions[pol] = fit

        # Compute gain amplitudes
        gain_amp = np.exp(
            [
                -fit.get(f"eta_{day_index}", np.zeros(nfreqs))
                for day_index in range(ndays)
            ]
        ).astype(np.complex128)

        # Evaluate gains - gain dictionary values have shape (Ndays, Nfreqs)
        gains.update({(ant, "J" + pol): gain_amp for ant in ants})

    return gains, solutions


def apply_lstcal_inplace(
    data: np.ndarray,
    gains: dict,
    antpairs: list[tuples],
    pols: list[str],
    gain_convention: str = "divide",
):
    """
    Apply the gains to the data inplace.

    Parameters:
    -----------
    data : np.ndarray
        Shape (Ndays, Nbls, Nfreqs, Npols) of complex data.
    gains : dict
        Dictionary of gains with keys (ant, pol) and values of shape (Ndays, Nfreqs).
    antpairs : list of tuples
        List of antenna pairs. Must contain the same number of baselines as the data and order must match
        the data.
    pols : list of strings
        List of polarizations. Must contain the same number of polarizations as the data and order must match
        the data.
    gain_convention : str, default='divide'
        Convention for applying the gains. Options are 'divide' and 'multiply'.
    """
    exponent = {"divide": 1, "multiply": -1}[gain_convention]

    # Check the shape of the data
    assert data.shape[1] == len(
        antpairs
    ), "Data shape does not match the number of antenna pairs."
    assert data.shape[3] == len(
        pols
    ), "Data shape does not match the number of polarizations."

    for pi, pol in enumerate(pols):
        for ai, ap in enumerate(antpairs):
            bl = ap + (pol,)
            antpol1, antpol2 = utils.split_bl(bl)

            # Apply the gain calibration
            if antpol1 == antpol2:
                data[:, ai, :, pi] /= (np.abs(gains[antpol1]) ** 2) ** exponent
            else:
                data[:, ai, :, pi] /= gains[antpol1] ** exponent
                data[:, ai, :, pi] /= np.conj(gains[antpol2]) ** exponent


def calibrate_data(
    data: np.ndarray,
    flags: np.ndarray,
    nsamples: np.ndarray,
    freqs: np.ndarray,
    idealized_antpos: list[dict] | dict,
    antpairs: list[tuple],
    pols: list[str],
    phs_max_iter: int = 10,
    phs_conv_crit: float = 1e-6,
    day_flags: np.ndarray | None = None,
    bls_flags: np.ndarray | None = None,
    sparse: bool = True,
    solver_method: str = "default",
) -> dict:
    """
    Calibrate the data using the LST-binned calibration scheme. This function performs delay slope calibration,
    global phase slope calibration, tip-tilt calibration, and amplitude calibration using the abscal degrees of freedom
    by comparing days within an LST bin. Calibration here in performed in place.

    Parameters:
    -----------
    data : np.ndarray
        Shape (Ndays, Nbls, Nfreqs, Npols) of complex data.
    flags : np.ndarray
        Shape (Ndays, Nbls, Nfreqs, Npols) of boolean flags.
    nsamples : np.ndarray
        Number of samples in each time-frequency bin. Shape (Ndays, Nbls, Nfreqs, Npols).
    freqs : np.ndarray
        Frequency array in Hz.
    idealized_antpos : list of dicts or dict
        List of dictionaries of idealized antenna positions. If only one dictionary is provided, it will be used for all days.
        If a list of dictionaries is provided, the length of the list must match the number of days.
        Each dictionary must contain the same number of antennas and the same keys.
    antpairs : list of tuples
        List of antenna pairs. Must contain the same number of baselines as the data and order must match
        the data.
    pols : list of strings
        List of polarizations. Must contain the same number of polarizations as the data and order must match 
        the data.
    phs_max_iter : int, default=10
        Maximum number of iterations to perform for the phase calibration.
    conv_crit : float, default=1e-6
        Convergence criterion for the phase calibration.
    day_flags : np.ndarray, default=None
        Boolean array of shape (Ndays,) indicating which days are flagged across all baselines.
    bls_flags : np.ndarray, default=None
        Boolean array of shape (Nbls,) indicating which baselines are flagged across all days.
    sparse : bool, default=True
        If True, linsolve will use sparse matrices to solve for the amplitude and phase calibration terms in linsolve step.
    solver_method : str, default='default'
        Method for linsolve to use when solving for the amplitude and phase calibration terms. 
        Options are 'default', 'pinv', 'solve', and 'lstsq'.
    
    Returns:
    --------
    gains : dict
        Dictionary of gains with keys (ant, pol) and values of shape (Ndays, Nfreqs).

    """
    # Initialize gains
    gains = {}

    # Perform global delay slope calibration
    delta_gains, _ = delay_slope_calibration(
        data=data,
        flags=flags,
        nsamples=nsamples,
        freqs=freqs,
        antpairs=antpairs,
        antpos=idealized_antpos,
        pols=pols,
        day_flags=day_flags,
        bls_flags=bls_flags,
        sparse=sparse,
        solver_method=solver_method,
    )
    apply_lstcal_inplace(
        data=data,
        gains=delta_gains,
        antpairs=antpairs,
        pols=pols,
        gain_convention="divide",
    )
    # Update gains
    gains = {k: gains.get(k, 1 + 0j) * delta_gains[k] for k in delta_gains}

    # Perform global phase-slope calibration
    delta_gains, _ = global_phase_slope_calibration(
        data=data,
        flags=flags,
        nsamples=nsamples,
        antpairs=antpairs,
        antpos=idealized_antpos,
        pols=pols,
        day_flags=day_flags,
        bls_flags=bls_flags,
        sparse=sparse,
        solver_method=solver_method,
    )
    apply_lstcal_inplace(
        data=data,
        gains=delta_gains,
        antpairs=antpairs,
        pols=pols,
        gain_convention="divide",
    )
    # Update gains
    gains = {k: gains[k] * delta_gains[k] for k in gains}

    # Perform per-frequency tip-tilt phase calibration
    for _ in range(phs_max_iter):
        delta_gains, _ = tip_tilt_calibration(
            data=data,
            flags=flags,
            nsamples=nsamples,
            antpairs=antpairs,
            antpos=idealized_antpos,
            pols=pols,
            day_flags=day_flags,
            bls_flags=bls_flags,
            sparse=sparse,
            solver_method=solver_method,
        )
        apply_lstcal_inplace(
            data=data,
            gains=delta_gains,
            antpairs=antpairs,
            pols=pols,
        )
        gains = {k: gains.get(k, 1 + 0j) * delta_gains[k] for k in delta_gains}
        crit = np.median(
            np.linalg.norm([delta_gains[k] - 1.0 for k in delta_gains], axis=(0, 1))
        )
        if crit < phs_conv_crit:
            break

    # Perform per-frequency logarithmic absolute amplitude calibration
    delta_gains, _ = amplitude_calibration(
        data=data,
        flags=flags,
        nsamples=nsamples,
        antpairs=antpairs,
        pols=pols,
        day_flags=day_flags,
        bls_flags=bls_flags,
        sparse=sparse,
        solver_method=solver_method,
    )

    # Calibrate the data inplace
    apply_lstcal_inplace(
        data=data,
        gains=delta_gains,
        antpairs=antpairs,
        pols=pols,
        gain_convention="divide",
    )

    # Update gains
    gains = {k: gains.get(k, 1 + 0j) * delta_gains[k] for k in delta_gains}

    return gains