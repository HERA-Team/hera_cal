from __future__ import annotations

import numpy as np
import logging
import warnings
from .. import abscal
from hera_filters import dspec
from .binning import LSTStack


def lstbin_calibration(
    stack: LSTStack,
    model: np.ndarray,
    all_reds: list[list[tuple]],
    inpaint_bands: tuple = (slice(0, None, None),),
    auto_stack: np.ndarray = None,
    run_amplitude_cal: bool = True,
    run_phase_cal: bool = True,
    smoothing_scale: float = 10e6,
    eigenval_cutoff: float = 1e-12,
    calibrate_inplace: bool = True,
    return_gains: bool = True,
):
    """
    Perform calibration on LSTStack object by comparing each day to a reference model.
    Each day is calibrated independently using only the absolute calibration degrees of freedom
    (per-antenna calibration is not currently supported), and the calibration parameters are
    smoothed in frequency.


    Parameters:
    ----------
        stack : LSTStack
        model : dictionary
        inpaint_bands : tuple
        run_amplitude_cal : bool
        run_phase_cal : bool
        smoothing_scale : float

    """
    # Assert some calibration done
    assert (
        run_amplitude_cal or run_phase_cal
    ), "At least one calibration mode must be used"
    assert (
        stack.data.shape[1:] == model.shape
    ), "Model must have the same number of antpairs/freqs/pols as stack.data"

    # Get variables used for both functions
    antpairs = stack.antpairs[:]
    pols = stack.pols

    # Function for storing calibration parameters
    calibration_parameters = {}

    # Get DPSS for each band
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
            amplitude_gain = np.where(
                np.isclose(amplitude_gain, 0.0), 1.0 + 0.0j, amplitude_gain
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

                    calibration_parameters[f"T_J{pol}"].append(
                        np.zeros((1, stack.data.shape[2], 2))
                    )
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
                calibration_parameters[f"T_J{pol}"].append(
                    np.squeeze(metadata["Lambda_sol"])
                )

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
            calibration_parameters[f"T_J{pol}"] = np.array(
                calibration_parameters[f"T_J{pol}"]
            )

    else:
        # Fill in phase w/ ones if no running phase calibration
        # Tip-tilt gains are zeros with dimensions (N_nights by N_freqs by Ndims)
        for pol in pols:
            calibration_parameters[f"T_J{pol}"] = np.zeros(
                (len(stack.nights), stack.freq_array.size, 2), dtype=complex
            )

    # Pre-compute matrices for smoothing fits
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
                basis = smoothing_functions[bandidx]

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
                bl_gain_here = np.array(
                    [
                        np.dot(basis, _fmat.dot(_raw_gain))
                        for _fmat, _raw_gain in zip(fmats[pol][bandidx], raw_gain)
                    ]
                )
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
                    basis = smoothing_functions[bandidx]

                    # Construct the raw gain and remove nans and infs
                    raw_gain = calibration_parameters[f"A_J{pol}"] ** 2
                    raw_gain = np.where(
                        stack.flags[:, 0, band, polidx], 1.0 + 0.0j, raw_gain[:, band]
                    )

                    # Compute smooth gain for each parameter and remove zeros/nans/infs
                    bl_gain_here = np.array(
                        [
                            np.dot(basis, _fmat.dot(_raw_gain))
                            for _fmat, _raw_gain in zip(fmats[pol][bandidx], raw_gain)
                        ]
                    )
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
