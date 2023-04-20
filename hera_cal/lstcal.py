"""
Module for calibrating visibilities by comparing data which lie within the same LST-bin
"""
import linsolve
import numpy as np

from . import redcal
from . import lstbin_simple


def delay_slope_calibration(data, flags, nsamples, freqs, antpairs, antpos, solve_for_phase_slope=False):
    """
    Solve for the delay slope of each day in an LST-bin.

    Parameters:
    -----------
    data : np.ndarray
        Shape (Ntimes, Nbls, Nfreqs, Npols) of complex data.
    flags : np.ndarray
        Shape (Ntimes, Nbls, Nfreqs, Npols) of boolean flags.
    nsamples : np.ndarray
        Number of samples in each time-frequency bin. Shape (Ntimes, Nbls, Nfreqs, Npols).
    freqs : np.ndarray
        Frequency array in Hz.
    antpairs : list of tuples
        List of tuples of antenna pairs.
    antpos : dict
        Dictionary of antenna positions in ENU coordinates.
    solve_for_phase_slope : bool
        If True, solve for the phase slope as well as the delay slope.

    Returns:
    --------
    delay_slope : np.ndarray
        Shape (Ntimes, Nfreqs, Npols) of delay slopes in nanoseconds.
    """
    dlys = []
    good_pairs = []
    nsamples_dict = {}
    
    for j in range(data.shape[1]):
        data_dict = {}
        for i in range(data.shape[0]):
            all_nans = np.all(np.isnan(data[i, j]))
            all_zeros = np.all(data[i, j] == 0)
            if not (all_nans or all_zeros) and np.mean(flags[i, j]) < 0.5:
                data_dict[(i,)] = data[i, j].T
                nsamples_dict[(antpairs[j], i,)] = nsamples[i, j] * np.logical_not(flags[i, j]).astype(float)

        if len(data_dict) > 1:
            dlys_offs = redcal._firstcal_align_bls(list(data_dict.keys()), freqs, data_dict)
            dlys.append(dlys_offs)
            good_pairs.append(antpairs[j])
                                      
    # Setup equations
    ls_data = {}
    const = {}
    wgts = {}
            
    # Loop through all of the delay solutions
    for (blgrp, pairs) in zip(dlys, good_pairs):
        x, y, z = (pos[pairs[1]] - pos[pairs[0]])
        const[f'b_{pairs[0]}_{pairs[1]}_x'] = x
        const[f'b_{pairs[0]}_{pairs[1]}_y'] = y
        const[f'b_{pairs[0]}_{pairs[1]}_z'] = z
        for k in blgrp:                            
            data_key_1 = f'b_{pairs[0]}_{pairs[1]}_x * Tx_{k[1][0]} + b_{pairs[0]}_{pairs[1]}_y * Ty_{k[1][0]} + b_{pairs[0]}_{pairs[1]}_z * Tz_{k[1][0]}'
            data_key_2 = f'- b_{pairs[0]}_{pairs[1]}_x * Tx_{k[0][0]} - b_{pairs[0]}_{pairs[1]}_y * Ty_{k[0][0]} - b_{pairs[0]}_{pairs[1]}_z * Tz_{k[0][0]}'
            ls_data[data_key_1 + data_key_2] = blgrp[k][0][0]

            
            wgt = np.sqrt(nsamples_dict[(pairs, k[0][0],)] * nsamples_dict[(pairs, k[1][0],)])
            wgts[data_key_1 + data_key_2] = np.nanmean(wgt)
                               
            if solve_for_phase_slope:
                data_key_3 = f'b_{pairs[0]}_{pairs[1]}_x * Px_{k[1][0]} + b_{pairs[0]}_{pairs[1]}_y * Py_{k[1][0]} + b_{pairs[0]}_{pairs[1]}_z * Pz_{k[1][0]}'
                data_key_4 = f'- b_{pairs[0]}_{pairs[1]}_x * Px_{k[0][0]} - b_{pairs[0]}_{pairs[1]}_y * Py_{k[0][0]} - b_{pairs[0]}_{pairs[1]}_z * Pz_{k[0][0]}'
                ls_data[data_key_3 + data_key_4] = blgrp[k][1][0]
                wgts[data_key_3 + data_key_4] = np.nanmean(wgt)
                                
    ls = linsolve.LinearSolver(ls_data, wgts=wgts, **const)
    sol = ls.solve()   
    
    if not solve_for_phase_slope:
        _sol = {}
        for k in sol:
            _sol[k.replace('T', 'P')] = 0

        sol.update(_sol)
    return sol


def _tip_tilt_align(bls, data, norm=True, wrap_pnt=(np.pi / 2)):
    '''
    Given a redundant group of bls, find per-baseline dly/off params that
    bring them into phase alignment using hierarchical pairing.

    Parameters:
    -----------
    bls : list of tuples
        List of antenna-pair tuples, e.g. [(0, 1), (0, 2), (1, 2)].
    data : dict
        Dictionary of complex visibility data, shaped (Ntimes, Nfreqs).
    norm : bool
        If True, normalize data by its absolute value before alignment.
    wrap_pnt : float
        Point at which to wrap phase. Default is pi/2.

    Returns:
    --------
    dly_off_gps : dict
        Dictionary of delay and offset parameters for each baseline, keyed
    '''
    grps = [(bl,) for bl in bls]  # start with each bl in its own group
    _data = {bl: data[bl[0]] for bl in grps}
    Ntimes, Nfreqs = data[bls[0]].shape
    times = np.arange(Ntimes)
    TT_gps = {}

    def process_pair(gp1, gp2):
        '''Phase-align two groups, recording dly/off in dly_off_gps for gp2
        and the phase-aligned sum in _data. Returns gp1 + gp2, which
        keys the _data dict and represents group for next iteration.'''
        d12 = _data[gp1] * np.conj(_data[gp2])
        if norm:
            ad12 = np.abs(d12)
            np.divide(d12, ad12, out=d12, where=(ad12 != 0))
        
        # Now that we know the slope, estimate the remaining phase offset
        TT_gps[gp2] = np.angle(d12)
        _data[gp1 + gp2] = _data[gp1] + _data[gp2] * np.exp(1j * TT_gps[gp2])
        return gp1 + gp2

    # Main N log N loop
    while len(grps) > 1:
        new_grps = []
        for gp1, gp2 in zip(grps[::2], grps[1::2]):
            new_grps.append(process_pair(gp1, gp2))
        # deal with stragglers
        if len(grps) % 2 == 1:
            new_grps = new_grps[:-1] + [process_pair(new_grps[-1], grps[-1])]
        grps = new_grps
        
    bl0 = bls[0]  # everything is effectively phase referenced off first bl
    tip_tilts = {}
    for gp, TT in TT_gps.items():
        for bl in gp:
            if (bl0, bl) in tip_tilts:
                TT0 = tip_tilts.get((bl0, bl))
            else:
                TT0 = np.zeros((Ntimes, Nfreqs))
            tip_tilts[(bl0, bl)] = TT0 + TT

    tip_tilts = {k: v for k, v in tip_tilts.items()}
    return tip_tilts

def phase_slope_calibration(data, flags, nsamples, antpairs, antpos):
    """
    Solve for the per-frequency phase slope of each day in an LST-bin.

    Parameters:
    -----------
    data : np.ndarray
        Shape (Ntimes, Nbls, Nfreqs, Npols) of complex data.
    flags : np.ndarray
        Shape (Ntimes, Nbls, Nfreqs, Npols) of boolean flags.
    nsamples : np.ndarray
        Number of samples in each time-frequency bin. Shape (Ntimes, Nbls, Nfreqs, Npols).
    freqs : np.ndarray
        Frequency array in Hz.
    antpairs : list of tuples
        List of antenna pairs.
    antpos : dict
        Dictionary of antenna positions in ENU frame in meters.

    Returns:
    --------
    phase_slope : np.ndarray
        Shape (Ntimes, Nfreqs, Npols) of phase slopes in radians per meter.
    """
    flag_dict = {}
    nsamples_dict = {}
    tip_tilts = []
    good_pairs = []
    for j in range(data.shape[1]):
        data_dict = {}
        for i in range(data.shape[0]):
            all_nans = np.all(np.isnan(data[i, j]))
            all_zeros = np.all(np.isclose(data[i, j], 0))
            if not (all_nans or all_zeros) and np.mean(flags[i, j]) < 0.5:
                data_dict[(i, 'ee')] = data[i, j].T
                flag_dict[(antpairs[j], i, 'ee')] = flags[i, j, :, 0]
                nsamples_dict[(antpairs[j], i, 'ee')] = nsamples[i, j, :, 0]

        if len(data_dict) > 0:
            _tip_tilts = _tip_tilt_align(list(data_dict.keys()), data_dict)
            tip_tilts.append(_tip_tilts)
            good_pairs.append(antpairs[j])
            
        
    # Setup equations
    ls_data = {}
    const = {}
    wgts = {}
    for (blgrp, pairs) in zip(tip_tilts, good_pairs):
        x, y, z = (pos[pairs[1]] - pos[pairs[0]])
        const[f'b_{pairs[0]}_{pairs[1]}_x'] = x
        const[f'b_{pairs[0]}_{pairs[1]}_y'] = y
        const[f'b_{pairs[0]}_{pairs[1]}_z'] = z            
        
        for k in blgrp:
            data_key_1 = f'b_{pairs[0]}_{pairs[1]}_x * TTx_{k[1][0]} + b_{pairs[0]}_{pairs[1]}_y * TTy_{k[1][0]} + b_{pairs[0]}_{pairs[1]}_z * TTz_{k[1][0]}'
            data_key_2 = f'- b_{pairs[0]}_{pairs[1]}_x * TTx_{k[0][0]} - b_{pairs[0]}_{pairs[1]}_y * TTy_{k[0][0]} - b_{pairs[0]}_{pairs[1]}_z * TTz_{k[0][0]}'
            ls_data[data_key_1 + data_key_2] = blgrp[k][0]
            
            wgt = np.logical_not(flag_dict[(pairs, k[0][0], 'ee')]).astype(float) * np.logical_not(flag_dict[(pairs, k[1][0], 'ee')]).astype(float)
            wgt *= np.sqrt(nsamples_dict[(pairs, k[0][0], 'ee')] * nsamples_dict[(pairs, k[1][0], 'ee')])
            wgts[data_key_1 + data_key_2] = wgt
    
    
    ls = linsolve.LinearSolver(ls_data, wgts=wgts, **const)
    sol = ls.solve()   

    return sol, wgts

def _amplitude_align(bls, data):
    """
    Given a redundant group of bls, find per-baseline dly/off params that
    bring them into phase alignment using hierarchical pairing.

    Parameters:
    -----------
    bls : list of tuples
        List of antenna pairs.
    data : dict
        Dictionary of complex visibility data keyed by (i,j) antenna pair.
    """
    grps = [(bl,) for bl in bls]  # start with each bl in its own group
    _data = {bl: np.abs(data[bl[0]]) for bl in grps}
    Ntimes, Nfreqs = data[bls[0]].shape
    amp_grps = {}

    def process_pair(gp1, gp2):
        '''Phase-align two groups, recording dly/off in dly_off_gps for gp2
        and the phase-aligned sum in _data. Returns gp1 + gp2, which
        keys the _data dict and represents group for next iteration.'''
        d12 = np.abs(_data[gp1] / _data[gp2])
        
        # Now that we know the slope, estimate the remaining phase offset
        amp_grps[gp2] = d12
        _data[gp1 + gp2] = (_data[gp1] + _data[gp2] * d12) / 2
        return gp1 + gp2

    # Main N log N loop
    while len(grps) > 1:
        new_grps = []
        for gp1, gp2 in zip(grps[::2], grps[1::2]):
            new_grps.append(process_pair(gp1, gp2))
        # deal with stragglers
        if len(grps) % 2 == 1:
            new_grps = new_grps[:-1] + [process_pair(new_grps[-1], grps[-1])]
        grps = new_grps
    
    bl0 = bls[0]  # everything is effectively scale referenced off first bl
    amplitudes = {}
    for gp, TT in amp_grps.items():
        for bl in gp:
            if (bl0, bl) in amplitudes:
                TT0 = amplitudes[(bl0, bl)]
            else:
                TT0 = np.ones((Ntimes, Nfreqs))
            amplitudes[(bl0, bl)] = TT * TT0

    return amplitudes

def amplitude_calibration(data, flags, nsamples, antpairs):
    """
    Solve for the frequency-amplitude of each day in an LST-bin.

    Parameters:
    -----------
    data : np.ndarray
        Shape (Ntimes, Nbls, Nfreqs, Npols) of complex data.
    flags : np.ndarray
        Shape (Ntimes, Nbls, Nfreqs, Npols) of boolean flags.
    nsamples : np.ndarray
        Number of samples in each time-frequency bin. Shape (Ntimes, Nbls, Nfreqs, Npols).
    freqs : np.ndarray
        Frequency array in Hz.
    antpairs : list of tuples
        List of antenna pairs.

    Returns:
    --------
    amplitude : np.ndarray
        Shape (Ntimes, Nfreqs, Npols) of amplitudes.
    """
    flag_dict = {}
    nsamples_dict = {}
    amps = []
    good_pairs = []
    for j in range(data.shape[1]):
        data_dict = {}
        for i in range(data.shape[0]):
            all_nans = np.all(np.isnan(data[i, j]))
            all_zeros = np.all(np.isclose(data[i, j], 0))
            if not (all_nans or all_zeros):
                data_dict[(i, 'ee')] = data[i, j].T
                flag_dict[(antpairs[j], i, 'ee')] = flags[i, j, :, 0]
                nsamples_dict[(antpairs[j], i, 'ee')] = nsamples[i, j, :, 0]

        if len(data_dict) > 0:
            _amps = _amplitude_align(list(data_dict.keys()), data_dict)
            amps.append(_amps)
            good_pairs.append(antpairs[j])
                        
    # Setup equations
    ls_data = {}
    const = {}
    wgts = {}
    for (blgrp, pairs) in zip(amps, good_pairs):        
        for k in blgrp:
            data_key_1 = f'a_{k[1][0]}_{pairs[0]}_{pairs[1]} * amp_{k[1][0]} - a_{k[0][0]}_{pairs[0]}_{pairs[1]} * amp_{k[0][0]}'
            const[f'a_{k[1][0]}_{pairs[0]}_{pairs[1]}'] = 1.0
            const[f'a_{k[0][0]}_{pairs[0]}_{pairs[1]}'] = 1.0
            ls_data[data_key_1] = np.log(blgrp[k][0])
            wgt = np.logical_not(flag_dict[(pairs, k[0][0], 'ee')]).astype(float) * np.logical_not(flag_dict[(pairs, k[1][0], 'ee')]).astype(float)
            wgt *= np.sqrt(nsamples_dict[(pairs, k[0][0], 'ee')] * nsamples_dict[(pairs, k[1][0], 'ee')])
            wgts[data_key_1] = wgt
    
    
    ls = linsolve.LinearSolver(ls_data, wgts=wgts, **const)
    sol = ls.solve()   

    return sol, wgts

def robust_divide(num, den):
    """Prevent division by zero.

    This function will compute division between two array-like objects by setting
    values to infinity when the denominator is small for the given data type. This
    avoids floating point exception warnings that may hide genuine problems
    in the data.

    Parameters
    ----------
    num : array
        The numerator.
    den : array
        The denominator.

    Returns
    -------
    out : array
        The result of dividing num / den. Elements where b is small (or zero) are set
        to infinity.

    """
    thresh = np.finfo(den.dtype).eps
    out = np.true_divide(num, den, where=(np.abs(den) > thresh))
    out = np.where(np.abs(den) > thresh, out, np.inf)
    return out

def modified_zscore(data, flags, nsamples, sigma=5.0, axis=-1):
    """
    Identify outliers in an LST-bin by computing the modified z-score.

    Parameters:
    -----------
    data : np.ndarray
        Shape (Ntimes, Nbls, Nfreqs, Npols) of complex data.
    flags : np.ndarray
        Shape (Ntimes, Nbls, Nfreqs, Npols) of boolean flags.
    nsamples : np.ndarray
        Number of samples in each time-frequency bin. Shape (Ntimes, Nbls, Nfreqs, Npols).
    sigma : float, default=5.0
        Number of standard deviations to use for outlier rejection.
    axis : int, default=-1
        Axis along which to compute the modified z-score.

    Returns:
    --------
    outlier_flags : np.ndarray
        Shape (Ntimes, Nbls, Nfreqs, Npols) of boolean flags.
    """
    # Make a copy of the data and set flags to NaN
    _data = np.copy(data)
    _data[flags] = np.nan

    # Compute the modified z-score
    med_data = np.nanmedian(data, axis=axis, keepdims=True)
    d_rs = data - med_data
    d_sq = np.abs(d_rs) ** 2
    sig = np.sqrt(np.nanmedian(d_sq, axis=axis, keepdims=True) / 0.456)
    zscore = robust_divide(d_rs, sig)
    return np.abs(zscore) > sigma

def flag_lst_data_products(data, flags, nsamples, inplace=True):
    """
    Flag data products in an LST-bin.

    Parameters:
    -----------
    data : np.ndarray
        Shape (Ntimes, Nbls, Nfreqs, Npols) of complex data.
    flags : np.ndarray
        Shape (Ntimes, Nbls, Nfreqs, Npols) of boolean flags.
    nsamples : np.ndarray
        Number of samples in each time-frequency bin. Shape (Ntimes, Nbls, Nfreqs, Npols).

    Returns:
    --------
    if inplace:
        None
    else:
        flagged_data : np.ndarray
            Shape (Ntimes, Nbls, Nfreqs, Npols) of complex data.
        flags : np.ndarray
            Shape (Ntimes, Nbls, Nfreqs, Npols) of boolean flags.
    """
    pass