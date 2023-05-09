"""
Module for calibrating visibilities by comparing data which lie within the same LST-bin
"""
import yaml
import linsolve
import numpy as np
from . import utils, redcal, red_groups, lstbin_simple

def _build_data_dict(data, flags, nsamples, antpairs, freqs, cal_function, day_flags=None, pack_data=False, pack_flags=False, pack_nsamples=False):
    """
    Build a dictionary of data for each baseline in the lstbin.

    Parameters:
    -----------
    data : np.ndarray
        Shape (Ntimes, Nbls, Nfreqs, Npols) of complex data.
    flags : np.ndarray
        Shape (Ntimes, Nbls, Nfreqs, Npols) of boolean flags.
    nsamples : np.ndarray
        Number of samples in each time-frequency bin. Shape (Ntimes, Nbls, Nfreqs, Npols).
    antpairs : list of tuples
        List of tuples of antenna pairs.
    freqs : np.ndarray
        Frequency array in Hz.
    cal_function : callable
        Function to use to solve for the offsets between days in an LST-bin.
    day_flags : np.ndarray
        Boolean array of shape (Ndays,) indicating which days to use in the lstbin.
    pack_data : bool
        If True, pack data into a single array for each baseline.
    pack_flags : bool
        If True, pack flags into a single array for each baseline.
    pack_nsamples : bool
        If True, pack nsamples into a single array for each baseline.

    Returns:
    --------
    data_dict : dict
        Dictionary of data for each baseline in the lstbin.
    """
    # If no day_flags is provided, assume all days are usable
    if day_flags is None:
        day_flags = np.zeros(data.shape[0], dtype=bool)

    # Get shape of data
    Ndays, Nbls, Nfreqs, Npols = data.shape

    # Dictionary for storing data, flags, and nsamples for each baseline
    data_dict = {}
    nsamples_dict = {}
    flag_dict = {}

    # Dictionary for storing the offsets between days
    offsets = {}

    # Loop through all polarizations
    for pi in range(Npols):

        # Loop through all baselines
        for bi in range(Nbls):
            _data_dict = {}
            _flag_dict = {}
            _nsamples_dict = {}

            # Loop through all days for a given baseline
            for di in range(Ndays):
                # Skip if day is flagged
                if day_flags[di]:
                    continue
                
                # Skip if all data is zero or nan
                # all_nans = np.all(np.isnan(data[di, bi, :, pi]))
                # all_zeros = np.all(data[di, bi, :, pi] == 0)
                if not np.all(flags[di, bi, :, pi]):
                    _data_dict[(antpairs[bi], pols[pi], di,)] = data[di, bi, :, pi][None]
                    _flag_dict[(antpairs[bi], pols[pi], di,)] = flags[di, bi, :, pi][None]
                    _nsamples_dict[(antpairs[bi], pols[pi], di,)] = nsamples[di, bi, :, pi][None]

            # If there is data for this baseline, solve for the offset between days
            if len(_data_dict) > 1:
                offsets[antpairs[bi] + (pols[pi], )] = cal_function(
                    list(_data_dict.keys()), freqs, _data_dict, _flag_dict
                )

            # Update data_dict with data from this loop
            if not pack_data:
                data_dict.update(_data_dict)
            if not pack_flags:
                flag_dict.update(_flag_dict)
            if not pack_nsamples:
                nsamples_dict.update(_nsamples_dict)

    return offsets, data_dict, flag_dict, nsamples_dict

def _delay_align_bls(bls, freqs, data, flags=None, norm=True, wrap_pnt=(np.pi / 2)):
    """
    Given a redundant group of bls, find per-baseline dly/off params that
    bring them into phase alignment using hierarchical pairing.
    """
    fftfreqs = np.fft.fftfreq(freqs.shape[-1], np.median(np.diff(freqs)))
    dtau = fftfreqs[1] - fftfreqs[0]
    grps = [(bl,) for bl in bls]  # start with each bl in its own group
    _data = {bl: data[bl[0]] for bl in grps}

    if flags:
        # If flags are provided, use them
        _flags = {bl: flags[bl[0]] for bl in grps}
    else:
        # If flags are not provided, assume there are no flags
        _flags = {bl: np.zeros(data[bl[0]].shape, dtype='bool') for bl in grps}

    Ntimes, Nfreqs = data[bls[0]].shape
    dly_off_gps = {}

    def process_pair(gp1, gp2):
        """
        Phase-align two groups, recording dly/off in dly_off_gps for gp2
        and the phase-aligned sum in _data. Returns gp1 + gp2, which
        keys the _data dict and represents group for next iteration.
        """
        d12 = _data[gp1] * np.conj(_data[gp2])
        if norm:
            ad12 = np.abs(d12)
            np.divide(d12, ad12, out=d12, where=(ad12 != 0))
        
        # Find the frequency of the peak in the cross-correlation
        dly, off = utils.fft_dly(
            d12, np.diff(freqs)[0], wgts=np.logical_not(_flags[gp1] | _flags[gp2]).astype(float)
        )

        # If fft_dly returned a nan, set the delay and offset to zero 
        dly[np.isnan(dly)] = 0
        off[np.isnan(off)] = 0

        # Construct a phasor to phase-align the two groups
        phasor = np.exp(np.complex64(2j * np.pi) * dly * freqs)
        
        # Record the delay and offset between the two groups
        dly_off_gps[gp2] = dly, off

        # Now that we know the slope, estimate the remaining phase offset
        _data[gp1 + gp2] = _data[gp1] + _data[gp2] * phasor * np.exp(np.complex64(1j) * off)
        _flags[gp1 + gp2] = _flags[gp1] | _flags[gp2]
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
    dly_offs = {}
    for gp, (dly, off) in dly_off_gps.items():
        for bl in gp:
            dly0, off0 = dly_offs.get((bl0, bl), (0, 0))
            dly_offs[(bl0, bl)] = (dly0 + dly, off0 + off)
    dly_offs = {k: (v[0], _wrap_phs(v[1], wrap_pnt=wrap_pnt)) for k, v in dly_offs.items()}
    return dly_offs

def delay_slope_calibration(data, flags, nsamples, freqs, antpairs, antpos, sparse=True, day_flags=None, solve_for_phase_slope=False):
    """
    Solve for the delay slope of each day in an LST-bin.

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
        List of tuples of antenna pairs.
    antpos : list of dictionarys, or dictionary
        Dictionary of antenna positions in ENU coordinates.
    solve_for_phase_slope : bool
        If True, solve for the phase slope as well as the delay slope.

    Returns:
    --------
    delay_slope : np.ndarray
        Shape (Ntimes, Nfreqs, Npols) of delay slopes in nanoseconds.
    """
    # Loop through all baselines
    dlys, antpairs_used, _, flag_dict, _ = _build_data_dict(
        data, flags, nsamples, antpairs, freqs, _delay_align_bls, day_flags=day_flags
        pack_flags=True
    )

    # Check if antpos is a dictionary or a list of dictionaries
    use_same_pos = True if isinstance(antpos, dict) else False
                                      
    # Setup equations
    ls_data = {}
    const = {}
    wgts = {}
            
    # Loop through all of the delay solutions
    for keys in dlys:
        blvec = (antpos[pairs[1]] - antpos[pairs[0]])
        const.update({f'b_{pairs[0]}_{pairs[1]}_{i}': blvec[i] for i in range(blvec.shape)})
        
        for k in blgrp:                            
            # Form the data key
            data_key_1 = " + ".join([f'b_{pairs[0]}_{pairs[1]}_{i} * T{i}_{k[1][0]}' for i in range(blvec.shape)])
            data_key_2 = " - ".join([f'b_{pairs[0]}_{pairs[1]}_{i} * T{i}_{k[0][0]}' for i in range(blvec.shape)])
            data_key = data_key_1 + " - " + data_key_2    

            # Load data from the blgrp into the linear system
            ls_data[data_key] = blgrp[k][0][0]

            # Weight the data by the fraction of unflagged data
            wgt = np.logical_or(flag_dict[blgrp[k][0][0]], flag_dict[blgrp[k][0][0]])
            wgts[data_key] = np.nanmean(wgt)
                               
            if solve_for_phase_slope:
                phase_key_1 = " + ".join([f'b_{pairs[0]}_{pairs[1]}_{i} * P{i}_{k[1][0]}' for i in range(blvec.shape)])
                phase_key_2 = " - ".join([f'b_{pairs[0]}_{pairs[1]}_{i} * P{i}_{k[0][0]}' for i in range(blvec.shape)])
                phase_key = phase_key_1 + " - " + phase_key_2    
                ls_data[phase_key] = blgrp[k][1][0]
                wgts[phase_key] = np.nanmean(wgt)
                                
    # Solve for the delay slope
    ls = linsolve.LinearSolver(ls_data, wgts=wgts, sparse=sparse, **const)
    sol = ls.solve()   
    
    if not solve_for_phase_slope:
        _sol = {}
        for k in sol:
            _sol[k.replace('T', 'P')] = 0

        sol.update(_sol)


    return sol

def _tip_tilt_align(bls, data, flags, norm=True):
    '''
    Given a redundant group of bls, find per-frequency tip-tilt parameters that
    bring them into phase alignment using hierarchical pairing.

    Parameters:
    -----------
    bls : list of tuples
        List of antenna-pair tuples, e.g. [(0, 1), (0, 2), (1, 2)].
    data : dict
        Dictionary of complex visibility data, shaped (Ntimes, Nfreqs).
    norm : bool
        If True, normalize data by its absolute value before alignment.

    Returns:
    --------
    dly_off_gps : dict
        Dictionary of delay and offset parameters for each baseline, keyed
    '''
    grps = [(bl,) for bl in bls]  # start with each bl in its own group
    _data = {bl: data[bl[0]] for bl in grps}
    _flags = {bl: flags[bl[0]] for bl in grps}
    Ntimes, Nfreqs = data[bls[0]].shape
    angle = {}

    def process_pair(gp1, gp2):
        '''Phase-align two groups, recording dly/off in dly_off_gps for gp2
        and the phase-aligned sum in _data. Returns gp1 + gp2, which
        keys the _data dict and represents group for next iteration.'''
        d12 = _data[gp1] * np.conj(_data[gp2])
        if norm:
            ad12 = np.abs(d12)
            np.divide(d12, ad12, out=d12, where=(ad12 != 0))
        
        # Now that we know the slope, estimate the remaining phase offset
        angle[gp2] = np.angle(d12)
        rephased = _data[gp2] * np.exp(1j * TT_gps[gp2])
        new_val = (
            _flags[gp1] * _data[gp2] + _flags[gp2] * _data[gp1] + (1 - _flags[gp1] - _flags[gp2]) * (_data[gp1] + rephased) / 2
        )
        
        # Identify regions where both groups are flagged and replace with 1 + 0j
        _flags[gp1 + gp2] = np.logical_and(_flags[gp1], _flags[gp2])
        _data[gp1 + gp2] = np.where(_flags[gp1 + gp2], 1 + 0j, new_val)
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
    angle = {}
    for gp, TT in TT_gps.items():
        for bl in gp:
            if (bl0, bl) in angle:
                TT0 = angle.get((bl0, bl))
            else:
                TT0 = np.zeros((Ntimes, Nfreqs))
            angle[(bl0, bl)] = TT0 + TT

    angles = {k: v for k, v in angle.items()}
    return angles

def phase_slope_calibration(data, flags, nsamples, antpairs, antpos, freqs, day_flags=None, sparse=True):
    """
    Solve for the per-frequency phase slope of each day in an LST-bin.

    Parameters:
    -----------
    data : np.ndarray
        Shape (Ndays, Nbls, Nfreqs, Npols) of complex data.
    flags : np.ndarray
        Shape (Ndays, Nbls, Nfreqs, Npols) of boolean flags.
    nsamples : np.ndarray
        Number of samples in each time-frequency bin. Shape (Ndays, Nbls, Nfreqs, Npols).
    antpairs : list of tuples
        List of antenna pairs.
    antpos : list of dictionaries, or dictionary
        Dictionary of antenna positions in ENU frame in meters.
    freqs : np.ndarray
        Frequency array in Hz.

    Returns:
    --------
    phase_slope : np.ndarray
        Shape (Ntimes, Nfreqs, Npols) of phase slopes in radians per meter.
    """
     # Loop through all baselines
    tip_tilts, antpairs_used, flag_dict, nsamples_dict = _build_data_dict(
        data, flags, nsamples, antpairs, freqs, _tip_tilt_align, day_flags=day_flags,
        return_flags=True, return_nsamples=True
    )

    # Check if antpos is a dictionary or a list of dictionaries
    use_same_pos = True if isinstance(antpos, dict) else False
    
    # Calibration polarizations indepedently
    for pol in pols:

        # Setup equations
        ls_data = {}
        const = {}
        wgts = {}
        for (blgrp, pairs) in zip(tip_tilts, antpairs_used):
            blvec = antpos[pairs[1]] - antpos[pairs[0]]
            const.update({f'b_{pairs[0]}_{pairs[1]}_{i}': blvec[i] for i in range(blvec.shape[0])})
            
            for k in blgrp:
                data_key_1 = " + ".join([f'b_{pairs[0]}_{pairs[1]}_{i} * TT{i}_{k[1][0]}' for i in range(blvec.shape[0])])
                data_key_2 = " - ".join([f'b_{pairs[0]}_{pairs[1]}_{i} * TT{i}_{k[0][0]}' for i in range(blvec.shape[0])])
                ls_data[data_key_1 + " - " + data_key_2] = blgrp[k][0]
                wgt = np.logical_not(flag_dict[(pairs, k[0][0], 'ee')]).astype(float) * np.logical_not(flag_dict[(pairs, k[1][0], 'ee')]).astype(float)
                wgt *= np.sqrt(nsamples_dict[(pairs, k[0][0], 'ee')] * nsamples_dict[(pairs, k[1][0], 'ee')])
                wgts[data_key_1 + " - " + data_key_2] = wgt
        
        ls = linsolve.LinearSolver(ls_data, wgts=wgts, sparse=sparse, **const)
        sol = ls.solve()   
    return sol

def _amplitude_align(bls, data, flags):
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
    _data = {bl: data[bl[0]] for bl in grps}
    _flags = {bl: flags[bl[0]] for bl in grps}
    Ntimes, Nfreqs = data[bls[0]].shape
    amp_grps = {}

    def process_pair(gp1, gp2):
        '''Phase-align two groups, recording dly/off in dly_off_gps for gp2
        and the phase-aligned sum in _data. Returns gp1 + gp2, which
        keys the _data dict and represents group for next iteration.'''
        d12 = np.abs(_data[gp1] / _data[gp2])

        # Record the amplitude scalar for gp2
        amp_grps[gp2] = d12
        
        # Average the two groups together, weighted by their amplitudes
        rephased =  _data[gp2] * d12
        new_val = (
            _flags[gp1] * _data[gp2] + _flags[gp2] * _data[gp1] +
            (1 - _flags[gp1] - _flags[gp2]) * (_data[gp1] + rephased) / 2
        )
        
        # Identify regions where both groups are flagged and replace with 1 + 0j
        _flags[gp1 + gp2] = np.logical_and(_flags[gp1], _flags[gp2])
        _data[gp1 + gp2] = np.where(_flags[gp1 + gp2], 1 + 0j, new_val)
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
    for gp, amp in amp_grps.items():
        for bl in gp:
            if (bl0, bl) in amplitudes:
                amp0 = amplitudes[(bl0, bl)]
            else:
                amp0 = np.ones((Ntimes, Nfreqs))
            amplitudes[(bl0, bl)] = amp * amp0

    return amplitudes

def amplitude_calibration(data, flags, nsamples, freqs, antpairs, pols, sparse=True):
    """
    Solve for the frequency-amplitude of each day in an LST-bin.

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
        List of antenna pairs.

    Returns:
    --------
    amplitude : np.ndarray
        Shape (Ntimes, Nfreqs, Npols) of amplitudes.
    """
     # Loop through all baselines
    amps, antpairs_used, flags_dict, nsamples_dict = _build_data_dict(
        data, flags, nsamples, antpairs, freqs, _amplitude_align,
        return_flags=True, return_nsamples=True
    )
                        
    # Store solutions in a dictionary keyed by polarization
    solutions = {}

    # Solve for the amplitude offsets for each polarization independently
    for pol in pols:
        # Setup equations
        ls_data = {}
        const = {}
        wgts = {}
        for (blgrp, pairs) in zip(amps[pol], antpairs_used):
            for k in blgrp:
                data_key_1 = f'a_{k[1][0]}_{pairs[0]}_{pairs[1]} * amp_{k[1][0]} - a_{k[0][0]}_{pairs[0]}_{pairs[1]} * amp_{k[0][0]}'
                const[f'a_{k[1][0]}_{pairs[0]}_{pairs[1]}'] = 1.0
                const[f'a_{k[0][0]}_{pairs[0]}_{pairs[1]}'] = 1.0
                ls_data[data_key_1] = np.log(blgrp[k][0])
                wgt = np.logical_not(flags_dict[(pairs, k[0][0], pol)]).astype(float) * np.logical_not(flags_dict[(pairs, k[1][0], pol)]).astype(float)
                wgt *= np.sqrt(nsamples_dict[(pairs, k[0][0], pol)] * nsamples_dict[(pairs, k[1][0], pol)])
                wgts[data_key_1] = wgt
        
        # Solve for the amplitude offsets
        ls = linsolve.LinearSolver(ls_data, wgts=wgts, sparse=sparse, **const)
        sol = ls.solve()   
        solutions[pol] = sol

    return solutions

def complex_phase_calibration(data, flags, nsamples, freqs, antpairs, pols=['nn', 'ee']):
    """
    Performs complex phase calibration on the data in an LST-bin.
    """
    raise NotImplementedError("abscal.complex_phase_abscal not currently implemented.")

def apply_lstcal_in_place(data, gains, antpairs, times, pols, gain_convention='divide'):
    """
    Apply the gains to the data in-place.

    Parameters:
    -----------
    data : np.ndarray
        Shape (Ndays, Nbls, Nfreqs, Npols) of complex data.
    gains : dict
        Dictionary of gains for each time and antenna-polarization pair.
    antpairs : list of tuples
        List of antenna pairs.
    times : list of floats
        List of times in JD.
    pols : list of strings
        List of polarizations.
    gain_convention : str, default='divide'
        Convention for applying the gains. Options are 'divide' and 'multiply'.
    """
    exponent = {'divide': 1, 'multiply': -1}[gain_convention]

    # Check the shape of the data
    assert data.shape[0] == len(times), "Data shape does not match the number of times."
    assert data.shape[1] == len(antpairs), "Data shape does not match the number of antenna pairs."
    assert data.shape[3] == len(pols), "Data shape does not match the number of polarizations."

    for pi, pol in enumerate(pols):
        for ti, time in enumerate(times):
            for ai, ap in enumerate(antpairs):
                bl = ap + (pol,)
                antpol1, antpol2 = utils.split_bl(bl)

                # Apply the gain calibration
                if antpol1 == antpol2:
                    data[ti, ai, :, pi] /= (np.abs(gains[time][antpol1]) ** 2) ** exponent
                else:
                    data[ti, ai, :, pi] /= gains[time][antpol1] ** exponent
                    data[ti, ai, :, pi] /= np.conj(gains[time][antpol2]) ** exponent

                

def calibrate_data(data, flags, nsamples, freqs, antpairs, pols, phs_max_iter=100, phs_conv_crit=1e-10, 
                   phase_method="logcal", day_flags=None, sparse=True):
    """
    Calibrate the data in an LST-bin.

    Parameters:
    -----------
    data : np.ndarray
        Shape (Ndays, Nbls, Nfreqs, Npols) of complex data.
    flags : np.ndarray
        Shape (Ndays, Nbls, Nfreqs, Npols) of boolean flags.
    nsamples : np.ndarray
        Number of samples in each time-frequency bin. Shape (Ndays, Nbls, Nfreqs, Npols).
    antpairs : list of tuples
        List of antenna pairs.
    times : list of floats
        List of times in JD.
    pols : list of strings
        List of polarizations.
    phs_max_iter : int, default=100
        Maximum number of iterations to perform for the phase calibration.
    conv_crit : float, default=1e-10
        Convergence criterion for the phase calibration.
    phase_method : str, default="logcal"
        Method to use for phase calibration. Options are "complex_phase" and "logcal".
    flag_bad_days : bool, default=True
        If True, flag days with bad data before performing calibration.

    Returns:
    --------
    gains : dict
        Dictionary of gains for each time and antenna-polarization pair.
    
    """
    if day_flags is None:
        day_flags = np.zeros(data.shape[0], dtype=bool)
    # Perform per-frequency logarithmic absolute amplitude calibration
    gains = amplitude_calibration(data, flags, nsamples, freqs, antpairs, pols, sparse=sparse)

    # Calibrate the data inplace
    apply_lstcal_in_place(data, gains, antpairs, pols, gain_convention="divide")

    if phase_method == "complex_phase":
        # Perform per-frequency complex phase calibration
        delta_gains = complex_phase_calibration(data, flags, nsamples, antpairs, pols, day_flags=day_flags)

        # Calibrate the data inplace
        apply_lstcal_in_place(data, gains, antpairs, pols, gain_convention="divide")

        # Update gains
        gains = {k: gains[k] * delta_gains[k] for k in gains}

    elif phase_method == "logcal":
        # Perform global delay slope calibration
        delta_gains = delay_slope_calibration(day_flags=day_flags)
        apply_lstcal_in_place(data, delta_gains, antpairs, pols, gain_convention="divide")
        gains = {k: gains[k] * delta_gains[k] for k in gains}

        # Perform global phase-slope calibration
        delta_gains = delay_slope_calibration(day_flags=day_flags)
        apply_lstcal_in_place(data, delta_gains, antpairs, pols, gain_convention="divide")
        gains = {k: gains[k] * delta_gains[k] for k in gains}

        # Perform per-frequency tip-tilt phase calibration
        for _ in range(phs_max_iter):
            delta_gains = phase_slope_calibration(day_flags=day_flags)
            apply_lstcal_in_place(data, delta_gains, antpairs, pols, gain_convention="divide")
            gains = {k: gains[k] * delta_gains[k] for k in gains}
            crit = np.median(np.linalg.norm([gains[k] - 1.0 for k in gains.keys()], axis=(0, 1)))
            if crit < phs_conv_crit:
                break
    else:
        raise ValueError(f"Unrecognized phase_method: {phase_method}")
    
    return gains

def config_lst_bin_calibration(config_file):
    """
    """
    

def run_lst_calibration(
        config_file, outfile_index=0, calibrate_bad_days=True, 
        sigma_clip_min_N=4, sigma_clip_thresh=4.0
    ):
    """
    Run LST-binning calibration on a set of LST-binned data files.

    Parameters:
    -----------
    config_file : str
        Path to the configuration file.
    pols : list of strings, default=["nn", "ee"]
        List of polarizations to calibrate.
    outfile_index : int, default=0
        Index of the output file to calibrate.
    inplace : bool, default=True
        If True, calibrate the data in-place.
    calibrate_bad_days : bool, default=True
        If True, calibrate days with bad data.
    sigma_clip_min_N : int, default=4
        Minimum number of samples to use for sigma clipping.
    sigma_clip_thresh : float, default=4.0
        Sigma clipping threshold.
    
    Returns:
    --------
    data : np.ndarray
        Calibrated data of shape (Ndays, Nbls, Nfreqs, Npols) of complex data.
    flags : np.ndarray
        Shape (Ndays, Nbls, Nfreqs, Npols) of boolean flags.
    nsamples : np.ndarray
        Number of samples in each time-frequency bin. Shape (Ndays, Nbls, Nfreqs, Npols).
    """
    # Load the configuration file
    with open(config_file, "r") as fl:
        configuration = yaml.safe_load(fl)

    # Configuration parameters
    config_opts = configuration['config_params']
    lst_grid = configuration['lst_grid']
    matched_files = configuration['matched_files']
    metadata = configuration['metadata']

    # Load the metadata from first file
    meta = lstbin_simple.FastUVH5Meta(
        matched_files[outfile_index][0][0], 
        blts_are_rectangular=metadata['blts_are_rectangular'],
        time_axis_faster_than_bls=metadata['time_axis_faster_than_bls'],
    )

    # Get the redundant groups
    antpos = dict(zip(meta.antenna_numbers, meta.antpos_enu))
    reds = red_groups.RedundantGroups.from_antpos(
        antpos=antpos,
        include_autos=True
    )

    # Get the LST-bins and data files
    lst_bins = lst_grid[outfile_index]
    data_files = matched_files[outfile_index]
    data_files = [df for df in data_files if df]

    # Get the metadata for each file
    data_metas = [[
        lstbin_simple.FastUVH5Meta(
            df, 
            blts_are_rectangular=metadata['blts_are_rectangular'], 
            time_axis_faster_than_bls=metadata['time_axis_faster_than_bls']
        ) for df in dflist
        ] 
        for dflist in data_files
    ]

    # Get additional metadata from the metadata file
    x_orientation = metadata['x_orientation']
    start_jd = metadata['start_jd']
    integration_time = metadata['integration_time']
    dlst = config_opts['dlst']
    freq_array = np.squeeze(meta.freq_array)

    # Find all baselines in the LST-bin
    all_baselines, all_pols = lstbin_simple.get_all_unflagged_baselines(
        data_metas, 
        include_autos=True, 
        redundantly_averaged=True,
        reds=reds
    )

    # Get the LST bin edges
    lst_bin_edges = np.array(
        [x - dlst/2 for x in lst_bins] + [lst_bins[-1] + dlst/2]
    )
    tinds, time_arrays, all_lsts, file_list, cals = lstbin_simple.filter_required_files_by_times(
        (lst_bin_edges[0], lst_bin_edges[-1]),
        data_metas
    )
    all_lsts = np.concatenate(all_lsts)

    # Get the LST data for each file
    bin_lst, data, flags, nsamples, binned_times = lstbin_simple.lst_bin_files_for_baselines(
        data_files=file_list,
        lst_bin_edges=lst_bin_edges,
        antpairs=all_baselines,
        pols=all_pols,
        freqs=freq_array,
        cal_files=cals,
        time_arrays=time_arrays,
        time_idx=tinds,
        lsts=all_lsts,
        redundantly_averaged=True,
        rephase=True,
        reds=reds
    )

    # Loop through all of the LST-bins
    for (_data, _flags, _nsamples) in zip(data, flags, nsamples):
        # Get the antenna pairs
        day_flags = flag_lst_data_products(_data, _flags, _nsamples)

        # Calibrate the data
        gains = calibrate_data(_data, _flags, _nsamples, day_flags=day_flags)

        # Average the data
        if calibrate_bad_days:
            model_arr, model_flags, _ = lstbin_simple.lst_average(
                data=_data, flags=_flags, nsamples=_nsamples, sigma_clip_thresh=sigma_clip_thresh,
                sigma_clip_min_N=sigma_clip_min_N
            )

            # Calibrate the "bad" days
            delta_gains = single_file_calibrate_data(model_arr, _data, _flags, _nsamples, day_mask)
            apply_lstcal_in_place(_data, delta_gains, antpairs, times, pols, gain_convention="divide")
            gains = {k: gains[k] * delta_gains[k] for k in gains}

            # Check to see if the data are still bad
            day_flags = flag_lst_data_products(_data, _flags, _nsamples)

        # Flag bad days in the original flag array
        flags[day_flags] = True

        # Save the gains ...
        save_meta_data(gains)

    return data, flags, nsamples

def robust_divide(num, den):
    """
    Prevent division by zero.

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
        Shape (Ndays, Nbls, Nfreqs, Npols) of boolean flags.
    """
    # Make a copy of the data and set flags to NaN
    _data = np.copy(data)
    _data[flags] = np.nan

    # Compute the modified z-score
    med_data = np.nanmedian(_data, axis=axis, keepdims=True)
    d_rs = _data - med_data
    d_sq = np.abs(d_rs) ** 2
    sig = np.sqrt(np.nanmedian(d_sq, axis=axis, keepdims=True) / 0.456)
    zscore = robust_divide(d_rs, sig)
    return np.abs(zscore) > sigma

def flag_lst_data_products(data, flags, nsamples):
    """
    Flag data products in an LST-bin.

    Parameters:
    -----------
    data : np.ndarray
        Shape (Ntimes, Nbls, Nfreqs, Npols) of complex data.
    flags : np.ndarray
        Shape (Ndays, Nbls, Nfreqs, Npols) of boolean flags.
    nsamples : np.ndarray
        Number of samples in each time-frequency bin. Shape (Ntimes, Nbls, Nfreqs, Npols).

    Returns:
    --------
    day_flags : np.ndarray
        Shape (Ndays) of boolean flags.
    """
    pass

def save_meta_data(filepath, metadata):
    """
    """
    pass