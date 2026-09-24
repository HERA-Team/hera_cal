"""
Fit the sky position and rotation measure of polarized point sources.

Used to refine catalog RA, Dec, and rotation measure (RM) of polarized sources in
per-night data before building and subtracting a Faraday-rotating model of each
source. Also gives the delay at which a Faraday-rotating source appears as a
function of frequency.
"""
from __future__ import annotations

import warnings

import numpy as np
from astropy import constants
from astropy.coordinates import EarthLocation
from scipy.optimize import minimize_scalar
from hera_cal import datacontainer, io
from hera_cal.utils import radec_to_lmn, unpack_data_containers

SPEED_OF_LIGHT = constants.c.value  # m/s


def _fit_polarized_source_position(
    vis: np.ndarray,
    weights: np.ndarray,
    uvw: np.ndarray,
    ra: float,
    dec: float,
    rotation_measure: float,
    times: np.ndarray,
    freqs: np.ndarray,
    location,
) -> tuple[float, float]:
    """
    Fit the sky position of a polarized point source near an initial guess.

    Phases the visibilities to the current best-guess position, removes the
    Faraday rotation, and solves a weighted linear system for the small
    positional offsets (delta_l, delta_m) in direction-cosine space. The
    offsets are projected back to RA/Dec.

    Parameters
    ----------
    vis : np.ndarray, shape (n_bls, n_times, n_freqs)
        Complex visibilities.
    weights : np.ndarray, shape (n_bls, n_times, n_freqs)
        Non-negative real weights.
    uvw : np.ndarray, shape (n_bls, 3, n_freqs)
        UVW coordinates in wavelengths.
    ra : float
        Current best-guess right ascension in degrees.
    dec : float
        Current best-guess declination in degrees.
    rotation_measure : float
        Current best-guess rotation measure in rad/m².
    times : np.ndarray, shape (n_times,)
        Julian dates.
    freqs : np.ndarray, shape (n_freqs,)
        Frequencies in Hz.
    location : astropy.coordinates.EarthLocation
        Observatory location used for coordinate transforms.

    Returns
    -------
    ra_fit : float
        Refined right ascension in degrees.
    dec_fit : float
        Refined declination in degrees.
    """
    # Faraday de-rotation phasor, shape (n_freqs,).
    lambda_sq = (constants.c.value / freqs) ** 2
    rm_phasor = np.exp(2j * lambda_sq * rotation_measure)

    # Design matrix M is time-independent: columns are [1, 2πiu, 2πiv].
    # Shape: (n_bls * n_freqs, 3).
    u_flat = uvw[:, 0, :].ravel()
    v_flat = uvw[:, 1, :].ravel()
    M = np.stack(
        [
            np.ones(len(u_flat)),
            2j * np.pi * u_flat,
            2j * np.pi * v_flat,
        ],
        axis=1,
    )

    # Accumulate weighted normal equations over time.
    XTX = np.zeros((3, 3), dtype=complex)
    XTy = np.zeros(3, dtype=complex)

    # Direction cosines of the current sky position at all times, shape (3, n_times).
    lmn0 = radec_to_lmn(ra, dec, times, location)

    for ti in range(vis.shape[1]):
        # Phase-shift visibilities to the current sky position.
        phase0 = np.einsum("bcf,ct->btf", uvw, lmn0[:, ti : ti + 1])  # (n_bls, 1, n_freqs)

        vis_t = (
            vis[:, ti, :]
            * np.exp(-2j * np.pi * phase0[:, 0, :])
            * rm_phasor[None, :]
        )  # (n_bls, n_freqs)
        w_t = weights[:, ti, :]  # (n_bls, n_freqs)

        vis_flat_t = vis_t.ravel()
        w_flat_t = w_t.ravel()

        WM = w_flat_t[:, None] * M  # (n_vis, 3)
        XTX += WM.conj().T @ M
        XTy += WM.conj().T @ vis_flat_t

    # Solve normal equations for [amplitude, delta_l, delta_m].
    x = np.linalg.solve(XTX, XTy)

    delta_l = (x[1] / x[0]).real
    delta_m = (x[2] / x[0]).real
    n = np.sqrt(1 - delta_l**2 - delta_m**2)

    dec_rad = np.radians(dec)
    ra_fit = ra + np.degrees(
        np.arctan2(
            delta_l,
            n * np.cos(dec_rad) - delta_m * np.sin(dec_rad),
        )
    )
    dec_fit = np.degrees(
        np.arcsin(delta_m * np.cos(dec_rad) + n * np.sin(dec_rad))
    )

    return ra_fit, dec_fit


def _fit_rotation_measure(
    vis: np.ndarray,
    weights: np.ndarray,
    uvw: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    ra: float,
    dec: float,
    start_rm: float,
    location,
    drm: float = 5.0,
    dtest: int = 500,
) -> float:
    """
    Fit the Faraday rotation measure (RM) via a coherent grid search refined
    with scalar minimization.

    Phases the visibilities to the supplied sky position, collapses over
    baselines and times into a Stokes-Q/U spectrum, then evaluates the
    coherent sum over a grid of trial RM values. The grid maximum is used as
    the starting point for a bounded scalar minimization
    (``scipy.optimize.minimize_scalar``) that returns a sub-grid-spacing
    result.

    Parameters
    ----------
    vis : np.ndarray, shape (n_bls, n_times, n_freqs)
        Complex visibilities.
    weights : np.ndarray, shape (n_bls, n_times, n_freqs)
        Non-negative real weights.
    uvw : np.ndarray, shape (n_bls, 3, n_freqs)
        UVW coordinates in wavelengths.
    times : np.ndarray, shape (n_times,)
        Julian dates.
    freqs : np.ndarray, shape (n_freqs,)
        Frequencies in Hz.
    ra : float
        Right ascension of the source in degrees.
    dec : float
        Declination of the source in degrees.
    start_rm : float
        Central value of the RM search grid in rad/m².
    location : astropy.coordinates.EarthLocation
        Observatory location used for coordinate transforms.
    drm : float, optional
        Half-width of the RM search window in rad/m². Default is 5.
    dtest : int, optional
        Number of RM trial values on the coarse grid. Default is 500.

    Returns
    -------
    float
        Best-fit rotation measure in rad/m².
    """
    # Phase-rotate to source position and compute a weighted-average spectrum.
    lmn = radec_to_lmn(ra, dec, times, location)  # shape (3, n_times)
    phasor = np.exp(-2j * np.pi * np.einsum("bcf,ct->btf", uvw, lmn))

    vis_phased = vis * phasor  # (n_bls, n_times, n_freqs)
    weight_sum = np.sum(weights, axis=(0, 1))  # (n_freqs,)
    # Avoid division by zero for fully-flagged channels.
    safe_weight_sum = np.where(weight_sum > 0, weight_sum, 1.0)
    spectrum = (
        np.sum(vis_phased * weights, axis=(0, 1)) / safe_weight_sum
    )  # (n_freqs,)

    # Coarse grid search to identify the basin of the maximum.
    lambda_sq = (constants.c.value / freqs) ** 2
    test_rm = np.linspace(start_rm - drm, start_rm + drm, dtest)
    faraday_response = np.array(
        [
            np.abs(np.nanmean(spectrum * np.exp(2j * lambda_sq * rm)))
            for rm in test_rm
        ]
    )
    best_idx = np.argmax(faraday_response)
    grid_spacing = test_rm[1] - test_rm[0]

    # Refine with bounded scalar minimization within ±2 grid spacings of peak.
    def neg_faraday_response(rm):
        return -np.abs(np.nanmean(spectrum * np.exp(2j * lambda_sq * rm)))

    result = minimize_scalar(
        neg_faraday_response,
        bounds=(
            test_rm[best_idx] - 2 * grid_spacing,
            test_rm[best_idx] + 2 * grid_spacing,
        ),
        method="bounded",
    )
    return result.x


def iteratively_fit_polarized_source_params(
    data: datacontainer.DataContainer,
    flags: datacontainer.DataContainer,
    nsamples: datacontainer.DataContainer,
    right_ascension: float,
    declination: float,
    rotation_measure: float,
    location: EarthLocation,
    maxiter: int = 10,
    drm: float = 5.0,
    dtest: int = 5000,
    pol: str = "pQ",
    verbose: bool = False,
) -> tuple[float, float, float]:
    """
    Iteratively fit the RA, Dec, and rotation measure of a polarized source.

    Alternates between refining the sky position (via
    :func:`_fit_polarized_source_position`) and the rotation measure (via
    :func:`_fit_rotation_measure`) until convergence or ``maxiter``
    iterations are reached. Convergence means RA and Dec change by less than
    1e-4 degrees and RM by less than 1e-3 rad/m² in one iteration. If
    ``maxiter`` is reached first, a ``RuntimeWarning`` is raised and the last
    estimate is returned.

    Parameters
    ----------
    data : datacontainer.DataContainer
        Visibility data for the source.
    flags : datacontainer.DataContainer
        Boolean flags corresponding to ``data``. Flagged samples are given
        zero weight.
    nsamples : datacontainer.DataContainer
        Number of samples contributing to each visibility measurement, used
        as weights.
    right_ascension : float
        Initial right ascension in degrees.
    declination : float
        Initial declination in degrees.
    rotation_measure : float
        Initial rotation measure in rad/m².
    location : astropy.coordinates.EarthLocation
        Observatory location used for coordinate transforms.
    maxiter : int, optional
        Maximum number of RA/Dec ↔ RM alternation iterations. Default is 10.
    drm : float, optional
        Half-width of the RM search window passed to
        :func:`_fit_rotation_measure` in rad/m². Default is 5.
    dtest : int, optional
        Number of coarse-grid trial RM values passed to
        :func:`_fit_rotation_measure`. Default is 5000.
    pol : str, optional
        Polarization string to extract from the data containers. Default is
        ``"pQ"``.
    verbose : bool, optional
        If True, print per-iteration diagnostics and the convergence message.
        Default is False.

    Returns
    -------
    fit_ra : float
        Best-fit right ascension in degrees.
    fit_dec : float
        Best-fit declination in degrees.
    fit_rm : float
        Best-fit rotation measure in rad/m².
    """
    # Unpack datacontainers into numpy arrays.
    vis, weights, uvw, times, freqs = unpack_data_containers(
        data,
        flags,
        nsamples,
        pol=pol,
    )
    weights = np.where(np.isfinite(vis), weights, 0.0)
    vis = np.where(np.isfinite(vis), vis, 0.0)

    # If all data are flagged, return the original source parameters unchanged.
    if np.sum(weights) == 0.0:
        return right_ascension, declination, rotation_measure

    fit_ra, fit_dec, fit_rm = right_ascension, declination, rotation_measure
    converged = False
    for fi in range(maxiter):
        fit_ra, fit_dec = _fit_polarized_source_position(
            vis,
            weights,
            uvw,
            right_ascension,
            declination,
            rotation_measure,
            times,
            freqs,
            location,
        )

        fit_rm = _fit_rotation_measure(
            vis,
            weights,
            uvw,
            times,
            freqs,
            fit_ra,
            fit_dec,
            rotation_measure,
            location,
            drm=drm,
            dtest=dtest,
        )

        ra_tol = abs(fit_ra - right_ascension)
        dec_tol = abs(fit_dec - declination)
        rm_tol = abs(fit_rm - rotation_measure)

        if verbose:
            print("RA:", fit_ra, "DEC:", fit_dec, "RM:", fit_rm)

        if ra_tol < 1e-4 and dec_tol < 1e-4 and rm_tol < 1e-3:
            converged = True
            if verbose:
                print(f"Converged at iteration {fi}.")
            break

        # Update running estimates for the next iteration.
        right_ascension = fit_ra
        declination = fit_dec
        rotation_measure = fit_rm

    if maxiter > 0 and not converged:
        warnings.warn(
            f"Polarized source fit did not converge in {maxiter} iterations; the last "
            f"step changed RA by {ra_tol:.2e} deg, Dec by {dec_tol:.2e} deg, and RM by "
            f"{rm_tol:.2e} rad/m^2. Returning the last estimate.",
            RuntimeWarning,
        )

    return fit_ra, fit_dec, fit_rm


def estimate_polarized_source_delay(
    freqs: float | np.ndarray,
    rotation_measure: float | np.ndarray,
) -> np.ndarray:
    """
    Compute the Faraday-rotation peak delay as a function of frequency.

    The delay is derived from the dispersion relation
    ``phi = RM * lambda^2``, differentiated with respect to frequency:

    .. math::

        \\tau(\\nu) = \\frac{2 c^2 \\, \\mathrm{RM}}{\\pi \\, \\nu^3}

    Parameters
    ----------
    freqs : float or np.ndarray
        Observed frequencies in Hz.
    rotation_measure : float or np.ndarray
        Rotation measure in rad m⁻².

    Returns
    -------
    delay : np.ndarray
        Peak delay in seconds at each frequency.
    """
    return 2.0 * (SPEED_OF_LIGHT ** 2 / freqs ** 3) * rotation_measure / np.pi


def estimate_freq_from_polarized_source_delay(
    delay: float | np.ndarray,
    rotation_measure: float | np.ndarray,
) -> np.ndarray:
    """
    Invert :func:`estimate_polarized_source_delay` to recover frequency.

    Parameters
    ----------
    delay : float or np.ndarray
        Peak delay in seconds.
    rotation_measure : float or np.ndarray
        Rotation measure in rad m⁻².

    Returns
    -------
    freqs : np.ndarray
        Frequency in Hz at which the given delay occurs.
    """
    return (delay * np.pi / (2.0 * rotation_measure * SPEED_OF_LIGHT ** 2)) ** (-1.0 / 3.0)


def subtract_polarized_source_model(
    data: datacontainer.DataContainer,
    flags: datacontainer.DataContainer,
    model_file: str,
    baseline: tuple = None,
    extra_flags: np.ndarray = None,
    freq_range: tuple = None,
):
    """
    Subtract a phased polarized-source model from one baseline of data, in place.

    The model file is the per-night polarized source filtering's product for one
    source and one component (its Faraday-rotating, scintillation, or smooth
    foreground part): the baseline-averaged visibilities phased to the source,
    stored on a single placeholder baseline, with the source's ICRS position in
    the ``SOURCE_RA`` and ``SOURCE_DEC`` extra keywords (degrees). Each of its
    polarizations is moved back to the source's true position on the data
    baseline by the geometric phasor ``exp(+2πi b·s(t) ν/c)``, with
    ``b = antpos[ant2] - antpos[ant1]`` (the same baseline convention as
    ``unpack_data_containers``) and ``s(t)`` the source's direction cosines from
    ``radec_to_lmn``, which inverts the phasing used to build the model, and is
    then subtracted from the data.

    Parameters
    ----------
    data : datacontainer.DataContainer
        Visibilities with ``antpos``, ``freqs``, and ``times`` attributes.
        Modified in place.
    flags : datacontainer.DataContainer
        Boolean flags keyed like ``data``. Where the data or the model are
        flagged nothing is subtracted.
    model_file : str
        Path to the model's uvh5 file, described above. Its times and
        frequencies must match the data's.
    baseline : tuple of int, optional
        The ``(ant1, ant2)`` to subtract from, in the orientation stored in
        ``data``. Required when ``data`` holds more than one antpair.
    extra_flags : np.ndarray of bool, optional
        Additional flags of shape ``(n_times, n_freqs)``, treated like ``flags``.
    freq_range : tuple of float, optional
        ``(f_min, f_max)`` in Hz: subtract only in channels with ``f_min <= freq <= f_max``.
        Default: all channels.

    Raises
    ------
    ValueError
        If ``baseline`` is needed but not given or is not stored in ``data``,
        or if the model's times or frequencies do not match the data's.
    KeyError
        If the model file lacks the ``SOURCE_RA`` / ``SOURCE_DEC`` keywords, or
        holds a polarization the data do not.
    """
    if baseline is None:
        antpairs = list(data.antpairs())
        if len(antpairs) != 1:
            raise ValueError("baseline must be given when data holds more than one antpair.")
        baseline = antpairs[0]
    baseline = tuple(baseline)
    if baseline not in data.antpairs():
        raise ValueError(f"{baseline} is not stored in data (in that orientation).")
    ant1, ant2 = baseline
    blvec = data.antpos[ant2] - data.antpos[ant1]

    hd_model = io.HERAData(model_file)
    model, model_flags, _ = hd_model.read()
    if (
        len(model.times) != len(data.times)
        or len(model.freqs) != len(data.freqs)
        or not np.allclose(model.times, data.times, rtol=0, atol=1e-8)
        or not np.allclose(model.freqs, data.freqs, rtol=0, atol=1.0)
    ):
        raise ValueError(f"{model_file} is not on the same time and frequency grid as the data.")
    ra, dec = hd_model.extra_keywords["SOURCE_RA"], hd_model.extra_keywords["SOURCE_DEC"]
    lmn = radec_to_lmn(ra, dec, data.times, hd_model.telescope.location)
    phasor = np.exp(2j * np.pi * np.outer(blvec @ lmn, data.freqs) / SPEED_OF_LIGHT)

    model_antpair = list(model.antpairs())[0]
    for pol in model.pols():
        bl = baseline + (pol,)
        flagged = model_flags[model_antpair + (pol,)] | flags[bl]
        if extra_flags is not None:
            flagged = flagged | extra_flags
        model_here = np.where(flagged, 0, model[model_antpair + (pol,)] * phasor)
        if freq_range is not None:
            in_range = (data.freqs >= freq_range[0]) & (data.freqs <= freq_range[1])
            model_here = np.where(in_range[np.newaxis, :], model_here, 0)
        data[bl] -= model_here
