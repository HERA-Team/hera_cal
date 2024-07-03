from __future__ import annotations

import numpy as np
from .. import abscal, utils
from hera_filters import dspec
from .binning import LSTStack


def _expand_degeneracies_to_ant_gains(
    stack: LSTStack,
    amplitude_parameters: dict,
    phase_gains: dict,
    inpaint_bands: tuple,
    auto_stack: LSTStack = None,
    smooth_gains: bool = True,
    smoothing_scale: float = 10e6,
    eigenval_cutoff: float = 1e-12,
):
    """
    This function expands the degenerate calibration parameters to per-antenna gains. The function
    also smooths the gains using DPSS basis functions if smooth_gains is set to True.

    Parameters:
    ----------
        stack : LSTStack
            The LSTStack object to calibrate
        amplitude_parameters : dict
            A dictionary containing the amplitude calibration parameters.
        phase_gains : dict
            A dictionary containing the gains produced from tip-tilt calibration.
        inpaint_bands : tuple of slices
            Defines the frequency bands to use for smoothing. Each slice object in the tuple
            defines a separate frequency band to use for smoothing.
        auto_stack : LSTStack, default=None
            An LSTStack object containing the stack of auto-correlations. If provided, the
            auto-correlations will be calibrated using the same gains calculated from the
            cross-correlations.
        smooth_gains : bool, default=True
            Boolean flag to smooth the gains.
        smoothing_scale : float, default=10e6
            The scale of the smoothing function used to smooth the gains. This is the width of the
            smoothing function in Hz.
        eigenval_cutoff : float, default=1e-12
            The cutoff for the eigenvalues of the DPSS eigenvectors.

    Returns:
    -------
        gains : dict
            A dictionary containing the calibrated gains for each baseline. The keys are tuples
            containing the antenna numbers and polarization of the baseline.

    """
    # Get antpairs for the stack and auto_stack
    antpairs = stack.antpairs[:]
    if auto_stack:
        antpairs.extend(auto_stack.antpairs)

    # Get the unique antennas
    gain_ants = set()
    for ap in antpairs:
        gain_ants.update(ap)
    gain_ants = list(gain_ants)

    # Get the unique antenna-polarizations
    unique_pols = list(
        set(sum(map(list, [utils.split_pol(pol) for pol in stack.pols]), []))
    )

    # Pre-compute matrices for smoothing fits
    if smooth_gains:
        smoothing_functions = []
        for band in inpaint_bands:
            smoothing_functions.append(
                dspec.dpss_operator(
                    x=stack.freq_array[band],
                    filter_centers=[0],
                    filter_half_widths=[1 / smoothing_scale],
                    eigenval_cutoff=[eigenval_cutoff],
                )[0]
            )

        # Get consensus flagging pattern
        flags_here = np.all(stack.flags, axis=1)

        # Compute matrices for linear least-squares fits
        fmats = {pol: [] for pol in unique_pols}
        for polidx, pol in enumerate(stack.pols):
            split_pol1, split_pol2 = utils.split_pol(pol)
            if split_pol1 != split_pol2:
                continue
            for bandidx, band in enumerate(inpaint_bands):
                # Get weights and basis functions for the fit
                wgts = np.logical_not(flags_here[:, band, polidx]).astype(float)
                basis = smoothing_functions[bandidx]

                # Compute matrices for linear least-squares fits
                xtxinv = np.linalg.pinv([np.dot(basis.T * wi, basis) for wi in wgts])
                fmat = np.array(
                    [np.dot(_xtxinv, basis.T) * _w for _xtxinv, _w in zip(xtxinv, wgts)]
                )
                fmats[split_pol1].append(fmat)

    gains = {}
    for polidx, pol in enumerate(stack.pols):
        split_pol1, split_pol2 = utils.split_pol(pol)
        if split_pol1 != split_pol2:
            continue
        for ant in gain_ants:
            for bandidx, band in enumerate(inpaint_bands):
                raw_ant_gain = amplitude_parameters[f"A_{split_pol1}"] * (
                    phase_gains.get(
                        (ant, split_pol1),
                        np.ones_like(amplitude_parameters[f"A_{split_pol1}"]),
                    )
                )

                if smooth_gains:
                    # Rephase antenna gains
                    tau, _ = utils.fft_dly(
                        raw_ant_gain,
                        np.diff(stack.freq_array[band])[0],
                        wgts=np.logical_not(stack.flags[:, 0, band, polidx]),
                    )
                    rephasor = np.exp(-2.0j * np.pi * tau * stack.freq_array[band])
                    raw_ant_gain *= rephasor

                    basis = smoothing_functions[bandidx]
                    smooth_ant_gain = np.array(
                        [
                            np.dot(basis, _fmat.dot(_raw_gain))
                            for _fmat, _raw_gain in zip(
                                fmats[split_pol1][bandidx], raw_ant_gain
                            )
                        ]
                    )

                    # Rephase antenna gains
                    gains[(ant, split_pol1)] = smooth_ant_gain * rephasor.conj()
                else:
                    gains[(ant, split_pol1)] = raw_ant_gain

    return gains


def _lstbin_amplitude_calibration(
    stack: LSTStack,
    model: np.ndarray,
    auto_stack: LSTStack = None,
    auto_model: np.ndarray = None,
    use_autos_for_abscal: bool = True,
):
    """
    This function performs amplitude calibration on LSTStack object by comparing each day to an input
    reference model. Each day is calibrated independently using only the amplitude absolute calibration
    degrees of freedom (per-antenna calibration is not currently supported).
    """
    # Dictionaries for storing data used in amplitude calibration function
    data_here = {}
    wgts_here = {}
    abscal_model = {}

    # Loop through baselines and polarizations
    for polidx, pol in enumerate(stack.pols):
        for apidx, (ant1, ant2) in enumerate(stack.antpairs):

            blpol = (ant1, ant2, pol)

            # Move to the next blpol if there is not a model for the data or the entire baseline is flagged
            if np.all(stack.flags[:, apidx, :, polidx]):
                continue

            # Get model, weights, and data for each baseline
            abscal_model[blpol] = model[apidx, :, polidx] * np.ones(
                (len(stack.nights), 1)
            )
            data_here[blpol] = stack.data[:, apidx, :, polidx]
            wgts_here[blpol] = stack.nsamples[:, apidx, :, polidx] * (
                ~stack.flags[:, apidx, :, polidx]
            ).astype(float)

    # If autos are provided and use_autos_for_abscal is True, use them for amplitude calibration
    if use_autos_for_abscal and auto_stack is not None:
        for polidx, pol in enumerate(stack.pols):
            for apidx, (ant1, ant2) in enumerate(auto_stack.antpairs):

                blpol = (ant1, ant2, pol)

                # Move to the next blpol if there is not a model for the data or the entire baseline is flagged
                if np.all(auto_stack.flags[:, apidx, :, polidx]):
                    continue

                # Get model, weights, and data for each baseline
                abscal_model[blpol] = auto_model[apidx, :, polidx] * np.ones(
                    (len(auto_stack.nights), 1)
                )
                data_here[blpol] = auto_stack.data[:, apidx, :, polidx]
                wgts_here[blpol] = auto_stack.nsamples[:, apidx, :, polidx] * (
                    ~auto_stack.flags[:, apidx, :, polidx]
                ).astype(float)

    # Perform amplitude calibration
    solution = abscal.abs_amp_lincal(
        model=abscal_model, data=data_here, wgts=wgts_here, verbose=False
    )

    calibration_parameters = {}

    # Get the unique antenna-polarizations
    unique_pols = list(
        set(sum(map(list, [utils.split_pol(pol) for pol in stack.pols]), []))
    )

    for pol in unique_pols:
        # Calibration parameters store in an N_nights by N_freqs array
        polidx = stack.pols.index(utils.join_pol(pol, pol))
        amplitude_gain = np.where(
            np.all(stack.flags[..., polidx], axis=1), 1.0 + 0.0j, solution[f"A_{pol}"]
        )

        calibration_parameters[f"A_{pol}"] = amplitude_gain

    return calibration_parameters


def _lstbin_phase_calibration(
    stack: LSTStack,
    model: np.ndarray,
    all_reds: list[list[tuple]],
):
    """
    This function performs phase calibration on LSTStack object by comparing each day to an input
    reference model. Each day is calibrated independently using only the tip-tilt degrees of freedom
    (per-antenna calibration is not currently supported).
    """

    # Get antennas
    gain_ants = set()
    for ap in stack.antpairs:
        gain_ants.update(ap)

    gain_ants = list(gain_ants)

    transformed_antpos = None

    # Gains for the phase component
    phase_gains = {
        (ant, utils.split_pol(pol)[0]): [] for ant in gain_ants for pol in stack.pols
    }

    # Add tip-tilt gain parameters to calibration_parameters
    unique_pols = list(
        set(sum(map(list, [utils.split_pol(pol) for pol in stack.pols]), []))
    )
    calibration_parameters = {f"T_{pol}": [] for pol in unique_pols}

    for nightidx, _ in enumerate(stack.nights):
        for polidx, pol in enumerate(stack.pols):
            cal_bls = []

            data_here = {}
            model_here = {}

            split_pol1, split_pol2 = utils.split_pol(pol)

            if split_pol1 != split_pol2:
                continue

            if np.all(stack.flags[nightidx, :, :, polidx]):
                # Store phase gains
                for ant in gain_ants:
                    phase_gains[(ant, f"{split_pol1}")].append(
                        np.ones(stack.freq_array.shape, dtype=complex)
                    )

                # Store placeholder for degenerate parameters
                calibration_parameters[f"T_{split_pol1}"].append(None)
                continue

            for apidx, (ant1, ant2) in enumerate(stack.antpairs):
                blpol = (ant1, ant2, pol)

                if np.all(stack.flags[nightidx, apidx, :, polidx]):
                    continue

                cal_bls.append(blpol)
                data_here[blpol] = stack.data[nightidx, apidx, :, polidx][np.newaxis]
                model_here[blpol] = model[apidx, :, polidx][np.newaxis]

            # Compute phase calibration
            metadata, delta_gains = abscal.complex_phase_abscal(
                data=data_here,
                model=model_here,
                reds=all_reds,
                model_bls=cal_bls,
                data_bls=cal_bls,
                transformed_antpos=transformed_antpos,
            )
            transformed_antpos = metadata["transformed_antpos"]

            # Store degenerate parameters
            calibration_parameters[f"T_{split_pol1}"].append(metadata["Lambda_sol"])

            # Store phase gains
            for ant in gain_ants:
                gain_here = delta_gains.get(
                    (ant, split_pol1),
                    np.ones((1, stack.freq_array.shape[0]), dtype=complex),
                )[0]
                phase_gains[(ant, split_pol1)].append(
                    np.where(
                        np.all(stack.flags[nightidx, :, :, polidx], axis=0),
                        1.0 + 0.0j,
                        gain_here,
                    )
                )

    for key in phase_gains:
        phase_gains[key] = np.array(phase_gains[key])

    for pol in unique_pols:
        for nightly_cal_params in calibration_parameters[f"T_{pol}"]:
            if isinstance(nightly_cal_params, np.ndarray):
                cal_params_shape = nightly_cal_params.shape
                break

        # Fill in tip-tilt gains with zeros if night was flagged
        calibration_parameters[f"T_{pol}"] = np.array(
            [
                (
                    nightly_cal_params
                    if nightly_cal_params is not None
                    else np.zeros(cal_params_shape)
                )
                for nightly_cal_params in calibration_parameters[f"T_{pol}"]
            ]
        )

    return calibration_parameters, phase_gains


def lstbin_absolute_calibration(
    stack: LSTStack,
    model: np.ndarray,
    all_reds: list[list[tuple]],
    inpaint_bands: tuple = (slice(0, None, None),),
    auto_stack: np.ndarray = None,
    auto_model: np.ndarray = None,
    run_amplitude_cal: bool = True,
    run_phase_cal: bool = True,
    smoothing_scale: float = 10e6,
    eigenval_cutoff: float = 1e-12,
    calibrate_inplace: bool = True,
    smooth_gains: bool = True,
    use_autos_for_abscal: bool = True,
):
    """
    This function performs calibration on LSTStack object by comparing each day to an input
    reference model. Each day is calibrated independently using only the absolute calibration
    degrees of freedom (per-antenna calibration is not currently supported). The calibration
    is performed in two steps:

        1. Amplitude calibration: The amplitude of the data is scaled to match the model using
           hera_cal.abscal.abs_amp_lincal to perform gain fits.
        2. Phase calibration: The phase of the data is adjusted to match the model by fitting
           for the tip-tilt abscal degeneracy using hera_cal.abscal.complex_phase_abscal. Note
           that while the tip-tilt degeneracy may be solved for in greater than 2 dimensions
           in the general case, this function only solves for the 2D tip-tilt degeneracy.


    Parameters:
    ----------
        stack : LSTStack
            The LSTStack object to calibrate
        model : np.ndarray
            The reference model to calibrate the data to. The model should have the same
            number of baselines, frequencies, and polarizations as the data in stack. The model
            can simply be the mean of the data in the stack across nights (zeroth axis in LSTStack)
            if data are already abscal'd.
        all_reds : list of list of tuples
            A list of lists of redundant baseline groups. Each element of the list is a list of tuples
            containing the redundant baseline groups. Each tuple contains the antenna numbers
            of the antennas in the redundant baseline group.
        inpaint_bands : tuple of slices, default=(slice(0, None, None),)
            Defines the frequency bands to use for smoothing. Each slice object in the tuple
            defines a separate frequency band to use for smoothing. The default is to smooth over
            all frequencies at once.
        auto_stack : LSTStack, default=None
            An LSTStack object containing the stack of auto-correlations. If provided, the
            auto-correlations will be calibrated using the same gains calculated from the
            cross-correlations. Note that the auto-correlations are used to calculate the gain
            amplitude if use_autos_for_abscal is set to True.
        auto_model : np.ndarray, default=None
            The reference model to calibrate the auto-correlations to. The model should have the
            same number of baselines, frequencies, and polarizations as the auto-correlations in
            auto_stack.
        run_amplitude_cal : bool, default=True
            Boolean flag to run amplitude calibration.
        run_phase_cal : bool, default=True
            Boolean flag to run tip-tilt calibration.
        smoothing_scale : float, default=10e6
            The scale of the smoothing function used to smooth the gains. This is the width of the
            smoothing function in Hz.
        smooth_gains : bool, default=True
            Boolean flag to smooth the gains.
        eigenval_cutoff : float, default=1e-12
            The cutoff for the eigenvalues of the DPSS eigenvectors.
        calibrate_inplace : bool, default=True
            Boolean flag to calibrate the data in place.
        use_autos_for_abscal : bool, default=True
            Boolean flag to use the auto-correlations for absolute calibration. If set to True,
            the auto-correlations will be used to calculate the gain amplitude if they are provided.
            Note that the auto_model must also be provided if this flag is set to True and autos are
            provided. If set to False, the auto-correlations will not be used for calibration.

    Returns:
    -------
        calibration_parameters : dict
            A dictionary containing the calibration parameters. The keys are as follows:
                - 'A_J{pol}' : The amplitude calibration parameters for each polarization which has
                               shape (N_nights, Nfreqs)
                - 'T_J{pol}' : The tip-tilt calibration parameters for each polarization which has
                               shape (N_nights, Nfreqs, Ndims) where Ndims in the number of tip-tilt
                               degeneracies. For now, this is always 2.
        gains : dict
            A dictionary containing the calibrated gains for each baseline. The keys are tuples
            containing the antenna numbers and polarization of the baseline. If return_gains is
            set to False, this dictionary will be empty.
    """
    # Check to see if calibration modes are set
    if not (run_amplitude_cal or run_phase_cal):
        raise ValueError("At least one calibration mode must be used")

    # Check to see if the model has the same shape as the stack
    if not (stack.data.shape[1:] == model.shape):
        raise ValueError(
            "Model must have the same number of antpairs/freqs/pols as stack.data"
        )

    # Check to see if auto_model is provided if use_autos_for_abscal is True
    if auto_stack is not None and use_autos_for_abscal and auto_model is None:
        raise ValueError(
            "auto_model must be provided if auto_stack is provided and use_autos_for_abscal is True"
        )

    # Dictionary for storing calibration parameters
    calibration_parameters = {}

    # Check to see if all nights are flagged
    all_nights_flagged = np.all(stack.flags)

    # Get the unique antenna-polarizations
    unique_pols = list(
        set(sum(map(list, [utils.split_pol(pol) for pol in stack.pols]), []))
    )

    # Perform amplitude calibration
    if run_amplitude_cal and not all_nights_flagged:
        # Store gain solutions in paramter dictionary
        amp_cal_params = _lstbin_amplitude_calibration(
            stack=stack,
            model=model,
            auto_stack=auto_stack,
            auto_model=auto_model,
            use_autos_for_abscal=use_autos_for_abscal,
        )
        for key in amp_cal_params:
            calibration_parameters[key] = amp_cal_params[key]

    else:
        # Fill in amplitude w/ ones if not running amplitude calibration
        for pol in unique_pols:
            calibration_parameters[f"A_{pol}"] = np.ones(
                (len(stack.nights), stack.freq_array.size), dtype=complex
            )

    if run_phase_cal and not all_nights_flagged:
        # Perform phase calibration
        phs_cal_params, phase_gains = _lstbin_phase_calibration(
            stack=stack,
            model=model,
            all_reds=all_reds,
        )
        for key in phs_cal_params:
            calibration_parameters[key] = phs_cal_params[key]

    else:
        # Fill in phase w/ ones if not running phase calibration
        # Tip-tilt gains are zeros with dimensions (N_nights by N_freqs by Ndims)
        phase_gains = {}

        for pol in unique_pols:
            calibration_parameters[f"T_{pol}"] = np.zeros(
                (len(stack.nights), stack.freq_array.size, 2), dtype=complex
            )

    # Compute antenna gains from calibration parameters and smooth
    gains = _expand_degeneracies_to_ant_gains(
        stack=stack,
        amplitude_parameters=calibration_parameters,
        phase_gains=phase_gains,
        inpaint_bands=inpaint_bands,
        auto_stack=auto_stack,
        smooth_gains=smooth_gains,
        smoothing_scale=smoothing_scale,
        eigenval_cutoff=eigenval_cutoff,
    )

    # Iterate for each baseline in the stack and calibrate out gains
    if calibrate_inplace:
        for polidx, pol in enumerate(stack.pols):
            for apidx, (ant1, ant2) in enumerate(stack.antpairs):
                antpol1, antpol2 = utils.split_bl((ant1, ant2, pol))

                # Compute gain and calibrate out
                bl_gain = gains[antpol1] * gains[antpol2].conj()
                stack.data[:, apidx, :, polidx] /= bl_gain

            if auto_stack:
                for apidx, (ant1, ant2) in enumerate(auto_stack.antpairs):
                    antpol1, antpol2 = utils.split_bl((ant1, ant2, pol))

                    # Compute gain and calibrate out
                    auto_gain = gains[antpol1] * gains[antpol2].conj()
                    auto_stack.data[:, apidx, :, polidx] /= auto_gain

    return calibration_parameters, gains
