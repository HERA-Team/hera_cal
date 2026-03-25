"""
Radio astronomy signal processing utilities.

Provides coordinate transforms, polarized-source delay estimation, and
visibility model computation for calibration pipelines.
"""
from copy import deepcopy

from astropy import constants
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
import astropy.units as u
import numpy as np
from tqdm import tqdm
from hera_filters import dspec
from hera_cal.smooth_cal import _linear_fit
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
    freqs_out = freqs[freq_slice]  # Bug fix: was returning the un-sliced `freqs`

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
        l0, m0, n0 = radec_to_azalt(ra, dec, times[ti : ti + 1], location)
        lmn0 = np.stack([l0, m0, n0])  # shape (3, 1)
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
    method='scipy'
) -> float:
    """
    Fit the Faraday rotation measure (RM) via a coherent grid search.

    Phases the visibilities to the supplied sky position, collapses over
    baselines and times into a Stokes-Q/U spectrum, then evaluates the
    coherent sum over a grid of trial RM values. The RM that maximises the
    amplitude of the de-rotated spectrum is returned.

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
        Number of RM trial values. Default is 500.

    Returns
    -------
    float
        Best-fit rotation measure in rad/m².
    """
    # Phase-rotate to source position and compute a weighted-average spectrum.
    l, m, n = radec_to_azalt(ra, dec, times, location)
    lmn = np.array([l, m, n])  # shape (3, n_times)
    phasor = np.exp(-2j * np.pi * np.einsum("bcf,ct->btf", uvw, lmn))

    vis_phased = vis * phasor  # (n_bls, n_times, n_freqs)
    weight_sum = np.sum(weights, axis=(0, 1))  # (n_freqs,)
    # Avoid division by zero for fully-flagged channels.
    safe_weight_sum = np.where(weight_sum > 0, weight_sum, 1.0)
    spectrum = (
        np.sum(vis_phased * weights, axis=(0, 1)) / safe_weight_sum
    )  # (n_freqs,)

    # Grid search: find the RM that maximises the coherent de-rotated sum.
    lambda_sq = (constants.c.value / freqs) ** 2
    test_rm = np.linspace(start_rm - drm, start_rm + drm, dtest)
    faraday_response = np.array(
        [
            np.abs(np.nanmean(spectrum * np.exp(2j * lambda_sq * rm)))
            for rm in test_rm
        ]
    )
    best_idx = np.argmax(faraday_response)

    if method == 'grid_search':

        # Fall back to grid value if peak is at an edge (can't interpolate)
        if best_idx == 0 or best_idx == len(faraday_response) - 1:
            return test_rm[best_idx]

        # Parabolic interpolation using the three points around the peak
        y_left = faraday_response[best_idx - 1]
        y_peak = faraday_response[best_idx]
        y_right = faraday_response[best_idx + 1]

        # Analytic vertex of the parabola through these three points
        # offset is in units of grid spacing, between -0.5 and +0.5
        denom = 2 * (2 * y_peak - y_left - y_right)
        if denom == 0:
            return test_rm[best_idx]
        offset = (y_right - y_left) / denom

        grid_spacing = test_rm[1] - test_rm[0]
        return test_rm[best_idx] + offset * grid_spacing

    elif method == "max":
        return test_rm[best_idx]

    elif method == "scipy":
        from scipy.optimize import minimize_scalar

        best_idx = np.argmax(faraday_response)
        grid_spacing = test_rm[1] - test_rm[0]

        def neg_faraday_response(rm):
            return -np.abs(np.nanmean(spectrum * np.exp(2j * lambda_sq * rm)))

        result = minimize_scalar(
            neg_faraday_response,
            bounds=(test_rm[best_idx] - 2 * grid_spacing,
                    test_rm[best_idx] + 2 * grid_spacing),
            method='bounded'
        )
        return result.x

    else:
        raise ValueError("Blah")


def iteratively_fit_polarized_source_params(
    data: datacontainer.DataContainer,
    flags: datacontainer.DataContainer,
    nsamples: datacontainer.DataContainer,
    right_ascension,
    declination,
    rotation_measure,
    location,
    maxiter: int = 10,
    drm=5.0,
    dtest=5000,
    method='grid_search',
    verbose=False,
):
    """
    Iteratively fit RA, Dec, and rotation measure for a set of polarized sources.

    For each source, alternates between refining the sky position (via
    ``_fit_polarized_source_position``) and the rotation measure (via
    ``_fit_rotation_measure``) until convergence or ``maxiter`` iterations.

    Parameters
    ----------
    data_list : list of datacontainer.DataContainer
        One visibility DataContainer per source.
    flags_list : list of datacontainer.DataContainer
        One flags DataContainer per source, matching ``data_list``.
    nsamples_list : list of datacontainer.DataContainer
        One nsamples DataContainer per source, matching ``data_list``.
    sources : dict
        Dictionary keyed by source name. Each value must be a dict with keys:

        - ``"ra"`` : float — initial right ascension in degrees
        - ``"dec"`` : float — initial declination in degrees
        - ``"rotation_measure"`` : float — initial RM in rad/m²

    location : astropy.coordinates.EarthLocation
        Observatory location used for coordinate transforms.
    maxiter : int, optional
        Maximum number of RA/Dec ↔ RM alternation iterations. Default is 10.

    Returns
    -------
    dict
        Keyed by source name. Each value is a dict with keys ``"ra"``,
        ``"dec"``, and ``"rotation_measure"`` holding the best-fit values.
    """
    # Unpack datacontainers in numpy arrays
    vis, weights, uvw, times, freqs = unpack_data_containers(
        data,
        flags,
        nsamples,
        pol='pQ'
    )
    weights = np.where(np.isfinite(vis), weights, 0.0)
    vis = np.where(np.isfinite(vis), vis, 0.0)

    # If all of the data are flagged, return original source values
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
            method=method,
            dtest=dtest,
        )

        ra_tol = abs(fit_ra - right_ascension)
        dec_tol = abs(fit_dec - declination)
        rm_tol = abs(fit_rm - rotation_measure)

        if verbose:
            print("RA:", fit_ra, "DEC:", fit_dec, "RM:", fit_rm)

        if ra_tol < 1e-4 and dec_tol < 1e-4 and rm_tol < 1e-3:
            print(f"Converged at {fi} iterations")
            break

        # Update running estimates for the next iteration
        right_ascension = fit_ra
        declination = fit_dec
        rotation_measure = fit_rm

    return fit_ra, fit_dec, fit_rm


def radec_to_azalt(
    ra: float,
    dec: float,
    times,
    location: EarthLocation,
):
    """
    Convert a fixed ICRS position to direction cosines and Az/Alt.

    Parameters
    ----------
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.
    times : list of str or `~astropy.time.Time`
        Observation times as Julian-date floats or an existing ``Time`` object.
    location : `~astropy.coordinates.EarthLocation`
        Observer location on Earth.

    Returns
    -------
    l, m, n : np.ndarray
        Direction cosines (East, North, Up) at each time.
    """
    # Convert times to `Time` object if needed, and set up coordinate transformations
    obstimes = Time(times, format="jd")
    source = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    altaz = source.transform_to(AltAz(obstime=obstimes, location=location))

    # Extract azimuth and altitude in radians
    az = altaz.az.rad
    alt = altaz.alt.rad

    # Compute the direction cosines in the local topocentric frame
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


def _compute_source_weights(
    freqs: np.ndarray,
    baseline_length_m: float,
    rotation_measure: float,
    band_slices: list[slice],
    downweight_value: float = 0.1,
    foreground_delay_buffer_sec: float = 0.0,
) -> np.ndarray:
    """
    Build per-frequency weights that suppress bands where the polarized delay
    is either below the foreground horizon plus some buffer or beyond the Nyquist limit.

    Bands where the RM delay is usable receive weight 1; others receive
    ``downweight_value`` (not zero, to retain mild regularisation).

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array in Hz.
    baseline_length_m : float
        Physical baseline length in metres, used to set the foreground horizon.
    rotation_measure : float
        Rotation measure of the source in rad m⁻².
    band_slices : list of slice
        Frequency-axis slices defining each sub-band of the array.
    downweight_value : float, optional
        Weight assigned to unusable bands (default 0.1).

    Returns
    -------
    weights : np.ndarray
        Shape ``(freqs.size,)`` array of per-frequency weights.
    """
    rm_delay = estimate_polarized_source_delay(freqs, rotation_measure)
    min_delay = baseline_length_m / SPEED_OF_LIGHT + foreground_delay_buffer_sec
    nyquist_limit = 1.0 / (2.0 * np.median(np.diff(freqs)))

    usable = (rm_delay > min_delay) & (rm_delay < nyquist_limit)

    weights = np.zeros_like(freqs)
    for band_slice in band_slices:
        band_usable = np.any(usable[band_slice])
        weights[band_slice] = 1.0 if band_usable else downweight_value

    return weights


def fit_polarized_source_model_single_bl(
    data,
    flags,
    nsamples,
    sources: dict,
    pol: str,
    key: tuple,
    freqs: np.ndarray,
    times: np.ndarray,
    location: EarthLocation,
    bands: list[slice],
    band_slices: list[slice],
    disable: bool = False,
    use_nsample_wgts: bool = False,
    elevation_threshold: float = 0.75,
    foreground_delay_buffer_sec: float = 500e-9,
    polarized_temporal_hw_mHz: float = 0.1,
    polarized_spectral_hw_sec: float = 50e-9,
    eigenval_cutoff: float = 1e-12,
    sign: float = 1.0
) -> tuple[dict, dict, list]:
    """
    Fit a polarized visibility model for each source in ``sources``.

    For every source the function:

    1. Identifies the time window around the source's meridian transit.
    2. High-pass filters the data to remove foreground contamination.
    3. De-phases the filtered visibilities by the geometric baseline delay.
    4. Fits a DPSS model in time and frequency, applying an RM phasor to
       isolate the left- and right-hand polarized contributions.
    5. Reconstructs the full model visibility and stores it.

    Parameters
    ----------
    data : datacontainer.DataContainer
        Visibility data container with keys like ``data[key + (pol,)]``
        (shape ``(ntimes_total, nfreqs)``) and ``data.lsts`` / ``data.times``.
    flags : array-like
        Flag array (currently unused; reserved for future masking).
    nsamples : array-like
        Sample-count array used to build per-integration weights.
    sources : dict
        Mapping of source name → parameter dict.  Each entry must contain:
        ``"ra"`` (degrees), ``"rotation_measure"`` (rad m⁻²).
    pol : str
        Polarisation product key (e.g. ``"ee"`` or ``"nn"``).
    key : tuple
        Baseline key used to index into ``data`` and ``nsamples``.
    freqs : np.ndarray
        Frequency array in Hz, shape ``(nfreqs,)``.
    times : np.ndarray
        Time array in Julian days, shape ``(ntimes_total,)``.
    location : `~astropy.coordinates.EarthLocation`
        Observer location on Earth.
    bands : list of slice
        List of frequency-axis slices defining the sub-bands to filter and fit.
    band_slices : list of slice
        Sub-band slices into the frequency axis.
    disable : bool, optional
        Suppress the tqdm progress bar when ``True`` (default ``False``).
    foreground_delay_buffer_sec : float, optional
        Buffer time to add to the foreground delay (default 0.0).
    polarized_temporal_hw_mHz : float, optional
        DPSS temporal half-bandwidth in hours (default 0.1).
    polarized_spectral_hw_sec : float, optional
        Spectral half-bandwidth for the polarized DPSS fitting (default 50e-9 seconds).
    eigenval_cutoff : float, optional
        Cutoff for eigenvalues in the DPSS fitting (default 1e-12).
    LSQR_ATOL : float, optional
        Absolute tolerance for LSQR convergence (default 1e-10).
    LSQR_BTOL : float, optional
        Relative tolerance for LSQR convergence (default 1e-10).

    Returns
    -------
    source_models : dict
        Fitted model visibilities for each source, shape ``(ntimes, nfreqs)``.
    all_filtered_data : dict
        Foreground-filtered, de-phased data for each source.
    time_slices : list of slice
        Time slices used for each source, in the same order as ``sources``.
    """
    # Baseline vector in metres
    blvec = data.antpos[key[1]] - data.antpos[key[0]]
    baseline_length_m = np.linalg.norm(blvec)
    foreground_hw = baseline_length_m / SPEED_OF_LIGHT + foreground_delay_buffer_sec

    # Time axis relative to first sample, in seconds
    times_sec = np.copy(times) * 24.0 * 3600.0 / 1e3
    times_sec -= times_sec[0]

    # Precompute per-source frequency weights
    source_weights = [
        _compute_source_weights(
            freqs,
            baseline_length_m,
            sources[src]["rotation_measure"],
            band_slices,
            foreground_delay_buffer_sec=foreground_delay_buffer_sec,
        )
        for src in sources
    ]

    lmns = [
        radec_to_azalt(
            sources[src]["ra"],
            sources[src]["dec"],
            times,
            location,
        )
        for src in sources
    ]

    source_models: dict = {}
    all_filtered_data: dict = {}
    time_slices: list = []

    for si, source in enumerate(tqdm(sources, disable=disable)):
        # convert the source RA from degrees to hours for comparison with LSTs
        elevation = lmns[si][2]  # sin(alt) at each time
        peak_index = int(np.argmax(elevation))
        in_window = elevation >= (elevation_threshold * elevation[peak_index])
        rng = np.arange(0, in_window.size)
        time_slice = slice(rng[in_window].min(), rng[in_window].max())
        time_slices.append(time_slice)

        vis = data[key + (pol,)]
        nsamp = nsamples[key + (pol,)]

        # Weights: zero where the visibility is identically zero or flagged
        if use_nsample_wgts:
            auto_weight = nsamp
        else:
            auto_weight = (nsamp >= 0.0).astype(float)

        filter_weight = (nsamp >= 0.0).astype(float)

        weights = np.where(np.isclose(vis[time_slice], 0.0), 0.0, 1.0) * auto_weight[time_slice]
        fweights = np.where(np.isclose(vis[time_slice], 0.0), 0.0, 1.0) * filter_weight[time_slice]
        # weights *= elevation[time_slice][:, None]

        # Geometric phase toward this source
        lmn = lmns[si][:, time_slice]
        phasor = np.exp(
            2j * np.pi * sign
            * np.dot(blvec, lmn)[:, None]
            * freqs[None]
            / SPEED_OF_LIGHT
        )

        # RM phasor for Faraday rotation, applied during fitting to isolate polarized components
        rm = sources[source]["rotation_measure"]
        rm_phasor = np.exp(-2j * (freqs / SPEED_OF_LIGHT) ** -2 * rm)

        # Initialize output arrays for this source
        source_models[source] = np.zeros_like(vis[time_slice])
        all_filtered_data[source] = np.zeros_like(vis[time_slice])

        # Build the DPSS basis in time for the foreground filtering; this is shared across all bands for this source since the temporal filter is the same.
        time_basis, _ = dspec.dpss_operator(
            times_sec[time_slice],
            [0],
            [polarized_temporal_hw_mHz],
            eigenval_cutoff=[eigenval_cutoff],
        )

        for band_slice in bands:
            # Build the raw data array for filtering, treating flagged or zeroed data as zero
            raw = np.where(
                np.isfinite(vis[time_slice][:, band_slice]),
                vis[time_slice][:, band_slice],
                0.0,
            )

            # Build the weights array for filtering, zeroing out flagged or zeroed data to exclude it from the fit
            fwgts = np.where(
                np.isfinite(vis[time_slice][:, band_slice]),
                fweights[:, band_slice],
                0.0,
            )

            wgts = np.where(
                np.isfinite(vis[time_slice][:, band_slice]),
                weights[:, band_slice],
                0.0,
            )

            # Apply the foreground filter to isolate the polarized signal
            _, filtered, _ = dspec.fourier_filter(
                freqs[band_slice],
                raw,
                fwgts,
                filter_centers=[0],
                filter_half_widths=[foreground_hw],
                mode="dpss_solve",
                eigenval_cutoff=[eigenval_cutoff],
                max_contiguous_edge_flags=freqs[band_slice].size,
            )

            # De-phase by the geometric delay before time/frequency fitting
            filtered *= phasor[:, band_slice].conj()

            if np.any(np.isnan(filtered)) or np.sum(source_weights[si][band_slice]) == 0:
                all_filtered_data[source][:, band_slice] = filtered
                continue

            # Build DPSS bases in frequency for the polarized model fitting
            freq_basis = dspec.dpss_operator(
                freqs[band_slice],
                [0],
                [polarized_spectral_hw_sec],
                eigenval_cutoff=[eigenval_cutoff],
            )[0].real

            # Apply the source weights to the fitting; this suppresses bands where the polarized signal is expected to be unmeasurable,
            # improving stability.
            band_wgts = wgts * source_weights[si][band_slice][None]
            rm_phasor_band = rm_phasor[band_slice]

            def _fit(data_2d):
                XTX = np.einsum(
                    "ti,fj,tf,tm,fn->ijmn", time_basis.conj(), freq_basis.conj(),
                    band_wgts, time_basis, freq_basis, optimize=True
                )
                ncomps = time_basis.shape[-1] * freq_basis.shape[-1]
                XTX = np.reshape(XTX, (ncomps, ncomps))

                # Calculate X^T W y using the property (A \otimes B) vec(y) = (A Y B)
                XTWy = np.ravel(np.dot(np.dot(np.transpose(time_basis.conj()), (data_2d * band_wgts)), freq_basis.conj()))

                # Compute beta and reshape into a 2D array
                beta, _ = _linear_fit(XTX, XTWy, solver="pinv")
                beta = np.reshape(beta, (time_basis.shape[-1], freq_basis.shape[-1]))
                # return time_basis.dot(
                #    dspec.sparse_linear_fit_2D(
                #        data_2d,
                #        band_wgts,
                #        time_basis,
                #        freq_basis,
                #        atol=LSQR_ATOL,
                #        btol=LSQR_BTOL,
                #        precondition_solver=True,
                #    )[0]
                # ).dot(freq_basis.T)
                return time_basis.dot(beta).dot(freq_basis.T)

            p_model_left = _fit(filtered * rm_phasor_band)
            p_model_right = _fit(filtered * rm_phasor_band.conj())

            all_filtered_data[source][:, band_slice] = filtered
            source_models[source][:, band_slice] = (
                p_model_left * rm_phasor_band.conj()
                + p_model_right * rm_phasor_band
            ) * phasor[:, band_slice]

    return source_models, all_filtered_data, time_slices, weights


def deproject_polarized_source(
    data,
    nsamples,
    data_filtered,
    sources,
    times,
    location: EarthLocation,
):
    """
    De-project the fitted polarized source model from the original data.
    This function takes the original data, the fitted polarized model (after foreground filtering and de-phasing),
    and the source direction cosines, and computes the contribution of the polarized source to the original data.
    It then subtracts this contribution from the original data to yield a "de-projected" dataset where the polarized source has been removed.
    """
    lmns = [
        radec_to_azalt(
            sources[src]["ra"],
            sources[src]["dec"],
            times,
            location,
        )
        for src in sources
    ]

    data_proj = deepcopy(data)

    for ti in tqdm(range(data.shape[0])):
        steering_vec = []
        data_stack = []
        nsamples_stack = []
        keys = []
        unfilt_data_stack = []
        for key in data:
            ai, aj, _ = key

            if ai == aj:
                continue
            if np.sum(nsamples[key][ti]) == 0:
                continue

            blvec = data.antpos[aj] - data.antpos[ai]
            steering_vec.append(np.exp(2j * np.pi * (blvec[0] * l[ti] + blvec[1] * m[ti] + blvec[2] * (1 - n[ti])) * data.freqs / constants.c.value))
            data_stack.append(data_filtered[key][ti])
            nsamples_stack.append(np.sqrt(nsamples[key][ti]))
            keys.append(key)
            unfilt_data_stack.append(data[key][ti])

        nsamples_stack = np.array(nsamples_stack)
        data_stack = np.array(data_stack) * np.median(nsamples_stack, axis=1, keepdims=True)
        unfilt_data_stack = np.array(unfilt_data_stack) * np.median(nsamples_stack, axis=1, keepdims=True)
        steering_vec = np.array(steering_vec)  # / nsamples_stack
        inv_nsamples_stack = 1 / np.median(nsamples_stack, axis=1, keepdims=True)

        # Inner product s^† v  → (Nfreq,)
        alpha = np.einsum("a...,a...->...", steering_vec.conj() * inv_nsamples_stack, data_stack)

        # Norm s^† s → (Nfreq,)
        norm = np.einsum("a...,a...->...", steering_vec.conj() * inv_nsamples_stack, steering_vec)

        # Projected data
        proj_data = (unfilt_data_stack - steering_vec * (alpha / norm))
        proj_data *= inv_nsamples_stack

        for ki, key in enumerate(keys):
            data_proj[key][ti] = proj_data[ki]
