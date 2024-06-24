from __future__ import annotations

import numpy as np
from .. import abscal
from hera_filters import dspec
from .binning import LSTStack


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
    return_gains: bool = True,
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
            can simply be the mean of the data in stack.
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
        eigenval_cutoff : float, default=1e-12
            The cutoff for the eigenvalues of the DPSS eigenvectors.
        calibrate_inplace : bool, default=True
            Boolean flag to calibrate the data in place.
        return_gains : bool, default=True
            Boolean flag to return the gains. If set to False, the function will return an empty
            dictionary for the gains.
        use_autos_for_abscal : bool, default=True
            Boolean flag to use the auto-correlations for absolute calibration. If set to True,
            the auto-correlations will be used to calculate the gain amplitude if they are provided.
            Note that the auto_model must also be provided if this flag is set to True and autos are
            provided. If set to False, the auto-correlations will not be used for calibration.

    Returns:
    -------
        calibration_parameters : dict
            A dictionary containing the calibration parameters. The keys are as follows:
                - 'A_J{pol}' : The amplitude calibration parameters for each polarization.
                - 'T_J{pol}' : The tip-tilt calibration parameters for each polarization
        gains : dict
            A dictionary containing the calibrated gains for each baseline. The keys are tuples
            containing the antenna numbers and polarization of the baseline. If return_gains is
            set to False, this dictionary will be empty.
    """
    # Assert some calibration done
    if not (run_amplitude_cal or run_phase_cal):
        raise ValueError("At least one calibration mode must be used")

    if not (stack.data.shape[1:] == model.shape):
        raise ValueError(
            "Model must have the same number of antpairs/freqs/pols as stack.data"
        )

    if auto_stack is not None and use_autos_for_abscal and auto_model is None:
        raise ValueError(
            "auto_model must be provided if auto_stack is provided and use_autos_for_abscal is True"
        )

    # Get variables used for both functions
    antpairs = stack.antpairs[:]
    pols = stack.pols

    # Function for storing calibration parameters
    calibration_parameters = {}

    # Get DPSS for each band
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

    # Dictionaries for storing data used in abscal functions
    data_here = {}
    wgts_here = {}
    abscal_model = {}

    # Loop through baselines and polarizations
    for polidx, pol in enumerate(pols):
        for apidx, (ant1, ant2) in enumerate(antpairs):

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
        for polidx, pol in enumerate(pols):
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

    # Check to see if all nights are flagged
    all_nights_flagged = len(data_here) == 0

    # Perform amplitude calibration
    if run_amplitude_cal and not all_nights_flagged:
        # Store gain solutions in paramter dictionary
        solution = abscal.abs_amp_lincal(
            model=abscal_model, data=data_here, wgts=wgts_here, verbose=False
        )

        for pol in pols:
            # Calibration parameters store in an N_nights by N_freqs array
            amplitude_gain = np.where(
                np.isfinite(solution[f"A_J{pol}"]), solution[f"A_J{pol}"], 1.0 + 0.0j
            )

            calibration_parameters[f"A_J{pol}"] = amplitude_gain
    else:
        # Fill in amplitude w/ ones if no running amplitude calibration
        for pol in pols:
            calibration_parameters[f"A_J{pol}"] = np.ones(
                (len(stack.nights), stack.freq_array.size), dtype=complex
            )

    phase_gains = {}

    if run_phase_cal and not all_nights_flagged:
        gain_ants = set()
        for ap in antpairs:
            gain_ants.update(ap)
        gain_ants = list(gain_ants)

        transformed_antpos = None

        # Gains for the phase component
        phase_gains = {(ant, "J" + pol): [] for ant in gain_ants for pol in pols}

        # Add tip-tilt gain parameters to calibration_parameters
        calibration_parameters.update({f"T_J{pol}": [] for pol in pols})

        for nightidx, night in enumerate(stack.nights):
            for polidx, pol in enumerate(pols):
                cal_bls = []

                _data_here = {}
                _abscal_model = {}

                if np.all(stack.flags[nightidx, :, :, polidx]):
                    # Store phase gains
                    for ant in gain_ants:
                        phase_gains[(ant, f"J{pol}")].append(
                            np.ones(stack.freq_array.shape, dtype=complex)
                        )

                    # Store placeholder for degenerate parameters
                    calibration_parameters[f"T_J{pol}"].append(None)
                    continue

                for apidx, (ant1, ant2) in enumerate(antpairs):
                    blpol = (ant1, ant2, pol)

                    if np.all(stack.flags[nightidx, apidx, :, polidx]):
                        continue

                    cal_bls.append((ant1, ant2, pol))
                    _data_here[blpol] = stack.data[nightidx, apidx, :, polidx][
                        np.newaxis
                    ]
                    _abscal_model[blpol] = model[apidx, :, polidx][np.newaxis]

                # Compute phase calibration
                metadata, delta_gains = abscal.complex_phase_abscal(
                    data=_data_here,
                    model=_abscal_model,
                    reds=all_reds,
                    model_bls=cal_bls,
                    data_bls=cal_bls,
                    transformed_antpos=transformed_antpos,
                )
                transformed_antpos = metadata["transformed_antpos"]

                # Store degenerate parameters
                calibration_parameters[f"T_J{pol}"].append(metadata["Lambda_sol"])

                # Store phase gains
                for ant in gain_ants:
                    _gain_here = delta_gains.get(
                        (ant, f"J{pol}"),
                        np.ones((1, stack.freq_array.shape[0]), dtype=complex),
                    )[0]
                    phase_gains[(ant, f"J{pol}")].append(
                        np.where(np.isfinite(_gain_here), _gain_here, 1.0 + 0.0j)
                    )

        for key in phase_gains:
            phase_gains[key] = np.array(phase_gains[key])

        for pol in pols:
            for nightly_cal_params in calibration_parameters[f"T_J{pol}"]:
                if isinstance(nightly_cal_params, np.ndarray):
                    cal_params_shape = nightly_cal_params.shape
                    break

            # Fill in tip-tilt gains with zeros if night was flagged
            calibration_parameters[f"T_J{pol}"] = np.array(
                [
                    (
                        nightly_cal_params
                        if nightly_cal_params is not None
                        else np.zeros(cal_params_shape)
                    )
                    for nightly_cal_params in calibration_parameters[f"T_J{pol}"]
                ]
            )

    else:
        # Fill in phase w/ ones if no running phase calibration
        # Tip-tilt gains are zeros with dimensions (N_nights by N_freqs by Ndims)
        for pol in pols:
            calibration_parameters[f"T_J{pol}"] = np.zeros(
                (len(stack.nights), stack.freq_array.size, 2), dtype=complex
            )

    # Pre-compute matrices for smoothing fits
    if smooth_gains:
        fmats = {pol: [] for pol in pols}
        for polidx, pol in enumerate(pols):
            for bandidx, band in enumerate(inpaint_bands):
                # Get weights and basis functions for the fit
                wgts = np.logical_not(stack.flags[:, 0, band, polidx]).astype(float)
                basis = smoothing_functions[bandidx]

                # Compute matrices for linear least-squares fits
                xtxinv = np.linalg.pinv([np.dot(basis.T * wi, basis) for wi in wgts])
                fmat = np.array(
                    [np.dot(_xtxinv, basis.T) * _w for _xtxinv, _w in zip(xtxinv, wgts)]
                )
                fmats[pol].append(fmat)

    # Dictionary for gain parameters
    gains = {}

    # Iterate for each baseline the array
    for polidx, pol in enumerate(pols):
        for apidx, (ant1, ant2) in enumerate(antpairs):
            bl_gain = np.ones((stack.nights.size, stack.freq_array.size), dtype=complex)
            for bandidx, band in enumerate(inpaint_bands):
                # Construct the raw gain and remove nans and infs
                raw_gain = calibration_parameters[f"A_J{pol}"] ** 2 * (
                    phase_gains.get(
                        (ant1, "J" + pol),
                        np.ones_like(calibration_parameters[f"A_J{pol}"]),
                    )
                    * phase_gains.get(
                        (ant2, "J" + pol),
                        np.ones_like(calibration_parameters[f"A_J{pol}"]),
                    ).conj()
                )
                raw_gain = np.where(
                    stack.flags[:, 0, band, polidx], 1.0 + 0.0j, raw_gain[:, band]
                )

                # Compute smooth gain for each parameter and remove zeros/nans/infs
                if smooth_gains:
                    basis = smoothing_functions[bandidx]
                    bl_gain_here = np.array(
                        [
                            np.dot(basis, _fmat.dot(_raw_gain))
                            for _fmat, _raw_gain in zip(fmats[pol][bandidx], raw_gain)
                        ]
                    )
                else:
                    bl_gain_here = raw_gain

                bl_gain_here = np.where(
                    np.isfinite(bl_gain_here), bl_gain_here, 1.0 + 0.0j
                )
                bl_gain_here = np.where(
                    np.isclose(bl_gain_here, 0), 1.0 + 0.0j, bl_gain_here
                )
                bl_gain[:, band] = bl_gain_here

            # Calibrate out smoothed gains
            if calibrate_inplace:
                stack.data[:, apidx, :, polidx] /= bl_gain

            # Store gains to return
            if return_gains:
                gains[(ant1, ant2, pol)] = bl_gain

    if auto_stack:
        for polidx, pol in enumerate(pols):
            for apidx, (ant1, ant2) in enumerate(auto_stack.antpairs):
                bl_gain = np.ones(
                    (stack.nights.size, stack.freq_array.size), dtype=complex
                )
                for bandidx, band in enumerate(inpaint_bands):
                    # Construct the raw gain and remove nans and infs
                    raw_gain = calibration_parameters[f"A_J{pol}"] ** 2
                    raw_gain = np.where(
                        stack.flags[:, 0, band, polidx], 1.0 + 0.0j, raw_gain[:, band]
                    )

                    # Compute smooth gain for each parameter and remove zeros/nans/infs

                    if smooth_gains:
                        basis = smoothing_functions[bandidx]
                        bl_gain_here = np.array(
                            [
                                np.dot(basis, _fmat.dot(_raw_gain))
                                for _fmat, _raw_gain in zip(
                                    fmats[pol][bandidx], raw_gain
                                )
                            ]
                        )
                    else:
                        bl_gain_here = raw_gain

                    bl_gain_here = np.where(
                        np.isfinite(bl_gain_here), bl_gain_here, 1.0 + 0.0j
                    )
                    bl_gain_here = np.where(
                        np.isclose(bl_gain_here, 0), 1.0 + 0.0j, bl_gain_here
                    )
                    bl_gain[:, band] = bl_gain_here

                # Calibrate out smoothed gains
                if calibrate_inplace:
                    auto_stack.data[:, apidx, :, polidx] /= bl_gain

                # Store gains to return
                if return_gains:
                    gains[(ant1, ant2, pol)] = bl_gain

    return calibration_parameters, gains
