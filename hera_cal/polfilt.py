"""
Radio astronomy signal processing utilities.

Provides coordinate transforms, polarized-source delay estimation, and
visibility model computation for calibration pipelines.
"""

from astropy import constants
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
import astropy.units as u
import numpy as np
from tqdm import tqdm
from hera_filters import dspec

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

SIDEREAL_DAY_SECONDS = 86164.0905  # Sidereal day in seconds
SPEED_OF_LIGHT = constants.c.value  # m/s

# ---------------------------------------------------------------------------
# Default solver / filter hyper-parameters
# (centralised here so callers can override without touching call sites)
# ---------------------------------------------------------------------------

# Foreground filter: extra buffer beyond the geometric baseline delay (seconds)
# FOREGROUND_DELAY_BUFFER_SEC = 500e-9

# Spectral DPSS half-bandwidth used when fitting the polarized model (seconds)
# POLARIZED_SPECTRAL_HW_SEC = 50e-9

# LSQR convergence tolerances for sparse_linear_fit_2D
# LSQR_ATOL = 1e-10
# LSQR_BTOL = 1e-10

# Eigenvalue cutoff for DPSS operators
# DPSS_EIGENVAL_CUTOFF = 1e-12


# ---------------------------------------------------------------------------
# Coordinate utilities
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Polarized-source delay utilities
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Model computation
# ---------------------------------------------------------------------------

def fit_polarized_source_models(
    data,
    flags,
    nsamples,
    sources: dict,
    pol: str,
    blvec: np.ndarray,
    key: tuple,
    freqs: np.ndarray,
    times: np.ndarray,
    band_slices: list[slice],
    ntimes: int = 1000,
    disable: bool = False,
    model: dict | None = None,
    foreground_delay_buffer_sec: float = 0.0,
    polarized_temporal_hw_mHz: float = 0.1,
    polarized_spectral_hw_sec: float = 50e-9,
    eigenval_cutoff: float = 1e-12,
    LSQR_ATOL: float = 1e-10,
    LSQR_BTOL: float = 1e-10,
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
    blvec : np.ndarray
        Baseline vector in metres, shape ``(3,)``.
    key : tuple
        Baseline key used to index into ``data`` and ``nsamples``.
    freqs : np.ndarray
        Frequency array in Hz, shape ``(nfreqs,)``.
    times : np.ndarray
        Time array in Julian days, shape ``(ntimes_total,)``.
    band_slices : list of slice
        Sub-band slices into the frequency axis.
    ntimes : int, optional
        Number of time samples to include in the fitting window (default 1000).
    disable : bool, optional
        Suppress the tqdm progress bar when ``True`` (default ``False``).
    model : dict or None, optional
        Pre-existing model to subtract before filtering, keyed by source name.
        Pass ``None`` (default) to skip subtraction.
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
    baseline_length_m = np.linalg.norm(blvec)
    foreground_hw = baseline_length_m / SPEED_OF_LIGHT + foreground_delay_buffer_sec

    # Convert LSTs from radians to hours for easy comparison with RA
    lsts_hours = np.copy(data.lsts) * 12.0 / np.pi

    # Unwrap LSTs so that any wrap-around near midnight is handled correctly
    lst_midpoint = (lsts_hours[-1] + lsts_hours[0]) / 2.0
    if lsts_hours[-1] <= lsts_hours[0]:
        lst_midpoint = lsts_hours[-1] + np.median(np.diff(lsts_hours))
    lsts_hours[lst_midpoint < lsts_hours] -= 24.0

    # Time axis relative to first sample, in seconds
    times_sec = np.copy(times) * 24.0 * 3600.0
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
        )
        for src in sources
    ]

    source_models: dict = {}
    all_filtered_data: dict = {}
    time_slices: list = []

    for si, source in enumerate(tqdm(sources, disable=disable)):
        ra_hours = sources[source]["ra"] / 15.0
        peak_index = int(np.argmin(np.abs(ra_hours - lsts_hours)))
        time_slice = slice(peak_index - ntimes // 2, peak_index + ntimes // 2)
        time_slices.append(time_slice)

        vis = data[key + (pol,)]
        nsamp = nsamples[key + (pol,)]

        # Weights: zero where the visibility is identically zero or flagged
        auto_weight = (nsamp >= 0.0).astype(float)
        wgts = np.where(np.isclose(vis[time_slice], 0.0), 0.0, 1.0) * auto_weight[time_slice]

        # Geometric phase toward this source
        phasor = np.exp(
            2j * np.pi
            * np.dot(blvec, lmns[si][:, time_slice])[:, None]
            * freqs[None]
            / SPEED_OF_LIGHT
        )

        rm = sources[source]["rotation_measure"]
        rm_phasor = np.exp(-2j * (freqs / SPEED_OF_LIGHT) ** -2 * rm)

        source_models[source] = np.zeros_like(vis[time_slice])
        all_filtered_data[source] = np.zeros_like(vis[time_slice])

        for band_slice in band_slices:
            prior_model = model[source][:, band_slice] if model else 0.0

            raw = np.where(
                np.isfinite(vis[time_slice][:, band_slice]),
                vis[time_slice][:, band_slice] - prior_model,
                0.0,
            )

            _, filtered, _ = dspec.fourier_filter(
                freqs[band_slice],
                raw,
                wgts[:, band_slice],
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

            time_basis, _ = dspec.dpss_operator(
                times_sec[time_slice],
                [0],
                [polarized_temporal_hw_mHz],
                eigenval_cutoff=[eigenval_cutoff],
            )
            freq_basis = dspec.dpss_operator(
                freqs[band_slice],
                [0],
                [polarized_spectral_hw_sec],
                eigenval_cutoff=[eigenval_cutoff],
            )[0].real

            band_wgts = wgts[:, band_slice] * source_weights[si][band_slice][None]
            rm_phasor_band = rm_phasor[band_slice]

            def _fit(data_2d):
                return time_basis.dot(
                    dspec.sparse_linear_fit_2D(
                        data_2d,
                        band_wgts,
                        time_basis,
                        freq_basis,
                        atol=LSQR_ATOL,
                        btol=LSQR_BTOL,
                        precondition_solver=True,
                    )[0]
                ).dot(freq_basis.T)

            p_model_left  = _fit(filtered * rm_phasor_band)
            p_model_right = _fit(filtered * rm_phasor_band.conj())

            all_filtered_data[source][:, band_slice] = filtered
            source_models[source][:, band_slice] = (
                p_model_left * rm_phasor_band.conj()
                + p_model_right * rm_phasor_band
            ) * phasor[:, band_slice]

    return source_models, all_filtered_data, time_slices