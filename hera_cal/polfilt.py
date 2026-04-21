"""
Radio astronomy signal processing utilities.

Provides coordinate transforms, polarized-source delay estimation, and
visibility model computation for calibration pipelines.
"""
import numpy as np
import astropy.units as u
from astropy import constants
from astropy.time import Time
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from scipy.optimize import minimize_scalar

from hera_cal import datacontainer, utils

# Constants
SIDEREAL_DAY_SECONDS = 86164.0905  # Sidereal day in seconds
SPEED_OF_LIGHT = constants.c.value  # m/s


def unpack_data_containers(
    data: datacontainer.DataContainer,
    flags: datacontainer.DataContainer,
    nsamples: datacontainer.DataContainer,
    pol: str = "ee",
    antpos: dict = None,
    freqs: np.ndarray = None,
    time_slice: slice = slice(0, None),
    freq_slice: slice = slice(0, None),
    antpairs: list = None,
    weight_by_nsamples: bool = True,
):
    """
    Unpack HERA data containers into arrays suitable for vectorized operations.

    Extracts visibility data, flags, and metadata from HERA DataContainer
    objects and formats them for use with the imaging and fitting algorithms.
    Both each baseline and its conjugate are included to enforce Hermitian
    symmetry in the visibility data.

    Parameters
    ----------
    data : datacontainer.DataContainer
        Visibility data to be unpacked.
    flags : datacontainer.DataContainer
        Boolean flags corresponding to the visibility data. Flagged samples
        are zeroed out in the returned weights array.
    nsamples : datacontainer.DataContainer
        Number of samples contributing to each visibility measurement. Used
        as weights when ``weight_by_nsamples`` is True.
    pol : str, optional
        Polarization string to extract (e.g. ``"ee"``, ``"nn"``).
        Default is ``"ee"``.
    antpos : dict, optional
        Antenna positions in metres, keyed by antenna number. If None, falls
        back to ``data.antpos``.
    freqs : np.ndarray, optional
        Frequency array in Hz. If None, falls back to ``data.freqs``.
    time_slice : slice, optional
        Slice applied along the time axis. Default selects all times.
    freq_slice : slice, optional
        Slice applied along the frequency axis. Default selects all channels.
    antpairs : list of tuple, optional
        Antenna pairs ``(ant1, ant2)`` to include. If None, all pairs in
        ``data`` are used.
    weight_by_nsamples : bool, optional
        If True, weights are ``nsamples * ~flags``; otherwise weights are
        ``~flags`` (i.e. binary unflagged mask). Default is True.

    Returns
    -------
    vis : np.ndarray, shape (2 * n_antpairs, n_times, n_freqs)
        Complex visibility data. The factor of two arises from including each
        baseline and its conjugate.
    weights : np.ndarray, shape (2 * n_antpairs, n_times, n_freqs)
        Non-negative real weights for each visibility sample.
    uvw : np.ndarray, shape (2 * n_antpairs, 3, n_freqs)
        UVW coordinates in units of wavelengths, computed per frequency
        channel.
    times : np.ndarray, shape (n_times,)
        Julian dates for each time sample after applying ``time_slice``.
    freqs : np.ndarray, shape (n_freqs,)
        Frequencies in Hz after applying ``freq_slice``.
    """
    if antpairs is None:
        antpairs = data.antpairs()

    if freqs is None:
        freqs = data.freqs

    if antpos is None:
        antpos = data.antpos

    vis_list = []
    weights_list = []
    uvw_list = []

    for ap in antpairs:
        blpol = ap + (pol,)
        blvec = antpos[ap[1]] - antpos[ap[0]]

        # Weights: optionally scale by nsamples, then zero flagged samples.
        if weight_by_nsamples:
            weight = nsamples[blpol][time_slice, freq_slice] * (
                ~flags[blpol][time_slice, freq_slice]
            ).astype(float)
        else:
            weight = (~flags[blpol][time_slice, freq_slice]).astype(float)

        vis_list.extend(
            [
                data[blpol][time_slice, freq_slice],
                data[utils.reverse_bl(blpol)][time_slice, freq_slice],
            ]
        )
        weights_list.extend([weight, weight])

        # UVW in wavelengths: shape (3, n_freqs).
        uvw_baseline = (
            blvec[:, None] * freqs[freq_slice][None] / constants.c.value
        )
        uvw_list.extend([uvw_baseline, -uvw_baseline])

    # Final shapes:
    #   vis, weights : (n_bls, n_times, n_freqs)
    #   uvw          : (n_bls, 3, n_freqs)
    vis = np.array(vis_list)
    weights = np.array(weights_list)
    uvw = np.array(uvw_list)
    times = data.times[time_slice]
    freqs_out = freqs[freq_slice]

    return vis, weights, uvw, times, freqs_out


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

    for ti in range(vis.shape[1]):
        # Phase-shift visibilities to the current sky position.
        lmn0 = radec_to_lmn(ra, dec, times[ti : ti + 1], location)  # shape (3, 1)
        phase0 = np.einsum("bcf,ct->btf", uvw, lmn0)  # (n_bls, 1, n_freqs)

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
    with scalar minimisation.

    Phases the visibilities to the supplied sky position, collapses over
    baselines and times into a Stokes-Q/U spectrum, then evaluates the
    coherent sum over a grid of trial RM values. The grid maximum is used as
    the starting point for a bounded scalar minimisation
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

    # Refine with bounded scalar minimisation within ±2 grid spacings of peak.
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
    iterations are reached.

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
            if verbose:
                print(f"Converged at iteration {fi}.")
            break

        # Update running estimates for the next iteration.
        right_ascension = fit_ra
        declination = fit_dec
        rotation_measure = fit_rm

    return fit_ra, fit_dec, fit_rm


def radec_to_lmn(
    ra: float,
    dec: float,
    times,
    location: EarthLocation,
) -> np.ndarray:
    """
    Convert a fixed ICRS position to direction cosines in the local
    topocentric frame.

    The direction cosines ``(l, m, n)`` are derived by first transforming
    the sky coordinate to the local Az/Alt frame at the observer's location,
    then projecting onto the East, North, and Up axes respectively.

    Parameters
    ----------
    ra : float
        Right ascension in degrees.
    dec : float
        Declination in degrees.
    times : array-like of float or `~astropy.time.Time`
        Observation times as Julian-date floats or an existing ``Time``
        object.
    location : `~astropy.coordinates.EarthLocation`
        Observer location on Earth.

    Returns
    -------
    lmn : np.ndarray, shape (3, n_times)
        Direction cosines ``[l, m, n]`` at each time, where ``l`` is East,
        ``m`` is North, and ``n`` is Up.
    """
    if not isinstance(times, Time):
        times = Time(times, format="jd")

    source = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    altaz = source.transform_to(AltAz(obstime=times, location=location))

    az = altaz.az.rad
    alt = altaz.alt.rad

    l = np.cos(alt) * np.sin(az)   # East
    m = np.cos(alt) * np.cos(az)   # North
    n = np.sin(alt)                 # Up

    return np.array([l, m, n])


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
