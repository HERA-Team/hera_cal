from __future__ import annotations

import numpy as np
import logging
import warnings
from .binning import LSTStack
from .. import vis_clean
from hera_filters import dspec
from scipy import constants
from typing import Sequence
from .. import types as tp
from scipy import signal, linalg
from hera_qm.time_series_metrics import true_stretches

logger = logging.getLogger(__name__)


def get_masked_data(
    data: np.ndarray,
    flags: np.ndarray,
    nsamples: np.ndarray,
    inpainted_mode: bool = False,
) -> tuple[np.ma.MaskedArray, np.ndarray, np.ma.MaskedArray]:
    """
    The assumptions on the input data here are that

    data that is flagged IS flagged and not inpainted and should not be used at all
    if nsamples < 0 but unflagged, the data is flagged but inpainted.
    if nsamples > 0 and unflagged, the data is "normal" and should be used in any mode.

    Assumptions on the output data are that

    the data.mask represents what should be used when averaging the data:
    - if inpainted_mode is False, then the mask is [flags & nsamples == -1] (i.e.anything that was originally flagged, whether inpainted or not)
    - if inpainted_mode is True, then the mask is [flags] (i.e. anything that was originally flagged and not inpainted)

    flags represent what was originally flagged (regardless of inpainting)
    for nsamples, the elements where nsamples < 0 are set to zero in non-inpainting mode
      but in inpaint mode, the nsamples is forced to be the mean over frequency,
      and the mask is set the same way as the data.
    """
    flags = flags | ~np.isfinite(data)  # un-recoverable
    orig_flags = flags | (nsamples < 0)

    if not inpainted_mode:
        data = np.ma.masked_array(data, mask=orig_flags)
        nsamples = np.ma.masked_array(nsamples, mask=orig_flags)
    else:
        data = np.ma.masked_array(data, mask=flags)

        # First set the mask to all original flags, so we don't count nsamples < 0
        nsamples = np.ma.masked_array(nsamples, mask=orig_flags)
        # Take a mean over the freq axis, which forces nsamples to be uniform over frequency
        nsamples = np.ma.mean(nsamples, axis=2)[:, :, None, :] * np.ones((1, 1, data.shape[2], 1))

        # Set the mask the same as data.mask (i.e. mask out only non-inpainted data)
        nsamples.mask = data.mask

    return data, orig_flags, nsamples


def get_lst_median_and_mad(
    data: np.ndarray | np.ma.MaskedArray,
):
    """Compute the median absolute deviation of a set of data over its zeroth axis.

    Flagged data will be ignored in flagged mode, but included in inpainted mode.
    Nsamples is not taken into account at all, unless Nsamples=0.
    """
    fnc = np.ma.median if isinstance(data, np.ma.MaskedArray) else np.median
    med = fnc(data, axis=0)
    madrl = fnc(np.abs(data.real - med.real), axis=0) * 1.482579
    madim = fnc(np.abs(data.imag - med.imag), axis=0) * 1.482579
    return med, madrl + madim * 1j


def compute_std(
    data: np.ma.MaskedArray,
    nsamples: np.ma.MaskedArray,
    mean: np.ndarray | None = None,
):
    r"""
    Compute the standard deviation of a set of data over its zeroth axis.

    The input data is expected to be complex visibilities. The standard deviation is
    computed component-wise for real and imaginary parts, as

    .. math:: \sigma  = \sqrt{\frac{1}{N} \sum_i (1-f_i) n_i (d_i - \bar{d})^2},

    where :math:`n_i` is the number of samples for the i-th integration, :math:`d_i` is
    the data for the i-th integration (either real or imaginary component, separately),
    :math:`f_i` is a binary flag which is 1 if the data is flagged, :math:`\bar{d}` is
    the mean of the data, and :math:`N` is the total number of un-flagged samples over
    integrations.

    If not given directly, the mean of the data is computed as

    .. math:: \bar{d} = \frac{\sum_i (1 - f_i) n_i d_i}{N}.

    Parameters
    ----------
    data : np.ma.MaskedArray
        The data to compute the statistics over. The shape should be such that the
        first axis is the one averaged over (e.g. nights), and other axes are arbitrary.
        The mask of the array should be set to True where the data is flagged. The
        correct form of the MaskedArray can be obtained using :func:`get_masked_data`
        *with inpainted_mode=False*.
    nsamples : np.ma.MaskedArray
        The number of samples for each measurement. Same shape as ``data``, and having
        the same mask. The masked array can be computed with :func:`get_masked_data`.
    mean : np.ndarray, optional
        The mean of the data over the first axis. If not given, it will be computed from
        the data and nsamples. Providing it directly can serve two purposes: firstly, it
        allows more rapid computation of the standard deviation without re-computing the
        mean, and secondly, it allows using a different mean than the one computed from
        the data (e.g. if an inpainted mean is desired, rather than a flagged-mode mean).
        Note that if the mean is not provided, it will be computed under whatever
        assumptions led to the masking of the data and nsamples, which *should* be
        that flagged data (even if inpainted) is not included.

    Returns
    -------
    std : np.ndarray
        The standard deviation of the data. The shape is the same as the data, but with
        the first axis removed. It will be a complex array, regardless of the dtype
        of data.
    norm : np.ndarray
        The total number of un-flagged samples over the first axis. This is the same
        as the total number of samples in the mean, but it is returned here for
        convenience.
    """
    logger.info("Calculating std")

    norm = np.sum(nsamples, axis=0)
    normalizable = norm > 0

    if mean is None:
        mean = np.sum(data * nsamples, axis=0)
        mean[normalizable] /= norm[normalizable]
        mean[~normalizable] = np.nan

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice.")
        std = np.square(data.real - mean.real) + 1j * np.square(
            data.imag - mean.imag
        )
        std = np.sum(std * nsamples, axis=0)
        std[normalizable] /= norm[normalizable]
        std = np.sqrt(std.real) + 1j * np.sqrt(std.imag)

        std[~normalizable] = np.inf

    return std.data, norm


def lst_average(
    data: np.ma.MaskedArray,
    nsamples: np.ma.MaskedArray,
    flags: np.ndarray,
    get_std: bool = True,
    fill_value: float = np.nan,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute statistics of a set of data over its zeroth axis.

    The idea here is that the data's zeroth axis is "nights", and that each night is
    at the same LST. However, this function is agnostic to the meaning of the first
    axis. It just computes the mean, std, and nsamples over it.

    However, it *is* important that the second-last axis is the frequency axis, because
    this axis is checked for whether the nsamples is uniform over frequency.

    A distinction is made between the averaging of the *data* compared to *nsamples*.
    The average of *nsamples* is to be used in calculating expected variance, and
    therefore represents the number of actual data samples in the average (as opposed
    to inpainted values). The determination of which nsamples to include in this
    average is determined by the *flags* array (which thus represents all the flags
    on the data, irrespective of whether they were inpainted or not).

    The average of the *data* may include flagged samples that were subsequently
    inpainted, in order to maintain spectral smoothness. The determination of which
    samples to include in this average is determined by the mask of the data and
    nsamples arrays, which should be MaskedArrays. In principle, this array can be
    the *same* as the flags array, in which case flagged data are not used at all.

    Parameters
    ----------
    data : np.ma.MaskedArray
        The data to compute the statistics over. Shape ``(nnights, nbl, nfreq, npol)``.
        See notes above for interpretation of the mask. The masked array can be
        computed with :func:`get_masked_data`.
    nsamples : np.ma.MaskedArray
        The number of samples for each measurement. Same shape as ``data``, and having
        the same mask. The masked array can be computed with :func:`get_masked_data`.
    flags
        The flags for each measurement. Same shape as ``data``. These represent the flags
        considered in the averaging of nsamples -- i.e. the determination of the
        expected variance.
    get_std
        Whether to compute the standard deviation of the data.
    fill_value
        The value to use for the mean when there are no samples. This must be either
        nan, zero or inf.

    Returns
    -------
    out_data, out_flags, out_std, out_nsamples
        The reduced data, flags, standard deviation (across nights) and nsamples.
        Shape ``(nbl, nfreq, npol)``.
    """
    # all data is assumed to be in the same LST bin.
    assert (all(isinstance(x, np.ma.MaskedArray) for x in (data, nsamples)))

    if fill_value not in (np.nan, 0, np.inf):
        raise ValueError("fill_value must be nan, 0, or inf")

    # Here we do a check to make sure Nsamples is uniform across frequency
    ndiff = np.diff(nsamples, axis=2)
    if np.any(ndiff != 0):
        warnings.warn(
            "Nsamples is not uniform across frequency. This will result in spectral structure."
        )

    # Norm is the total number of samples over the nights. In the in-painted case,
    # it *counts* in-painted data as samples. In the non-inpainted case, it does not.
    norm = np.sum(nsamples, axis=0)

    # Ndays binned is the number of days that count towards the mean. This is the same
    # in in-painted and flagged mode.
    ndays_binned = np.sum((~flags).astype(int), axis=0)

    logger.info("Calculating mean")

    # np.sum works the same for both masked and non-masked arrays.
    meandata = np.sum(data * nsamples, axis=0)

    lstbin_flagged = np.all(data.mask, axis=0)

    normalizable = norm > 0

    meandata[normalizable] /= norm[normalizable]

    # Multiply by the fill value, which works for complex numbers when fill_value is
    # inf, 0 or nan.
    meandata[~normalizable] *= fill_value

    # While the previous nsamples is different for in-painted and flagged mode, which is
    # what we want for the mean, for the std and nsamples we want to treat flags as really
    # flagged.
    nsamples.mask = flags

    if get_std:
        std, norm = compute_std(data, nsamples, meandata)
    else:
        std = None

    return meandata.data, lstbin_flagged, std, norm.data, ndays_binned


def reduce_lst_bins(
    lststack: LSTStack | None = None,
    data: np.ndarray | None = None,
    flags: np.ndarray | None = None,
    nsamples: np.ndarray | None = None,
    inpainted_mode: bool = True,
    get_mad: bool = False,
    get_std: bool = True,
    mean_fill_value: float = np.nan,
) -> dict[str, np.ndarray]:
    """
    Reduce LST-stacked data over the time axis.

    Use this function to reduce an array of data, whose first axis contains all the
    integrations within a particular LST-bin (generally from different nights), across
    the night (or integration) axis. This will produce different statistics (e.g.
    mean, std, median).

    Parameters
    ----------
    data
        The data over which to perform the reduction. The shape should be
        ``(nintegrations_in_lst, nbl, nfreq, npol)`` -- i.e. the same as a UVData
        object's data_array if the blt axis is pushed out to two dimensions (time, bl).
        Note that this is the same format as ``LSTStack.data``.
    flags
        The flags, in the same shape as ``data``.
    nsamples
        The nsamples, same shape as ``data``. Importantly, in-painted samples must be
        represented with negative nsamples.
    inpainted_mode
        Whether to use the inpainted samples when calculating the statistics. If False,
        the inpainted samples are ignored. If True, the inpainted samples are used, and
        the statistics are calculated over the un-inpainted samples.
    get_mad
        Whether to compute the median and median absolute deviation of the data in each
        LST bin, in addition to the mean and standard deviation.
    get_std
        Whether to compute the standard deviation of the data in each LST bin.
    fill_value
        The value to use for the mean when there are no samples. This must be either
        nan, zero or inf.

    Returns
    -------
    dict
        The reduced data in a dictionary. Keys are 'data' (the lst-binned mean),
        'nsamples', 'flags', 'days_binned' (the number of days that went into each bin),
        'std' (standard deviation), 'median' and 'mad'. All values are arrays of the
        same shape: ``(nbl, nfreq, npol)`` unless None.
    """
    if lststack is not None:
        data = lststack.data
        flags = lststack.flags
        nsamples = lststack.nsamples

    if any(x is None for x in (data, flags, nsamples)):
        raise ValueError(
            "data, flags, and nsamples must all be provided, "
            "or lststack must be provided."
        )

    assert data.shape == flags.shape == nsamples.shape

    if data.shape[0] == 0:
        # We have no data to reduce (no data in this LST bin)
        return {
            'data': np.nan * np.ones(data.shape[1:], dtype=complex),
            'flags': np.ones(data.shape[1:], dtype=bool),
            'std': np.inf * np.ones(data.shape[1:], dtype=complex),
            'nsamples': np.zeros(data.shape[1:]),
            'days_binned': np.zeros(data.shape[1:]),
            'median': np.nan * np.ones(data.shape[1:], dtype=complex) if get_mad else None,
            'mad': np.inf * np.ones(data.shape[1:], dtype=complex) if get_mad else None,
        }

    data, flags, nsamples = get_masked_data(
        data, flags, nsamples, inpainted_mode=inpainted_mode
    )

    o = {'mad': None, 'median': None}
    o['data'], o['flags'], o['std'], o['nsamples'], o['days_binned'] = lst_average(
        data, flags=flags, nsamples=nsamples, get_std=get_std, fill_value=mean_fill_value
    )
    if get_mad:
        o['median'], o['mad'] = get_lst_median_and_mad(data)

    return o


def average_and_inpaint_simultaneously_single_bl(
    freqs: np.ndarray,
    stackd: np.ndarray,
    stackf: np.ndarray,
    stackn: np.ndarray,
    base_noise_var: np.ndarray,
    df: un.Quantity['frequency'] | float,
    filter_half_widths: Sequence[float],
    filter_centers: Sequence[float] = (0.0,),
    avg_flgs: np.ndarray | None = None,
    inpaint_bands: tuple[slice] = (slice(0, None, None),),
    max_gap_factor: float = 2.0,
    max_convolved_flag_frac: float = 0.667,
    use_unbiased_estimator: bool = False,
    sample_cov_fraction: float = 0.0,
    eigenval_cutoff: float = (0.01,),
    cache: dict | None = None,
):
    """
    Average and inpaint simultaneously for a single baseline.

    Parameters
    ----------
    freqs : np.ndarray
        The frequencies of the data.
    stackd : np.ndarray
        The stacked data, shape (n_nights, nfreq).
    stackf : np.ndarray
        The stacked flags, shape (n_nights, nfreq).
    stackn : np.ndarray
        The stacked nsamples, shape (n_nights, nfreq).
    base_noise_var : np.ndarray
        The expected noise variance for each night, shape (n_nights, nfreq).
    df : un.Quantity['frequency'] | float
        The frequency resolution. If a float, assumed to be in Hz.
    filter_centers : Sequence[float]
        The centers of the DPSS filters.
    avg_flgs : np.ndarray | None
        If passed, should be True when ALL data is flagged over nights, False otherwise.
        If not passed, will be calculated from ``stackf``. If passed, this array is
        **modified in-place**.
    inpaint_bands : tuple[slice]
        The bands to inpaint individually, as slices over frequency channels.
    max_gap_factor : float
        The maximum allowed gap factor.
    max_convolved_flag_frac : float
        The maximum allowed fraction of convolved flags.
    use_unbiased_estimator : bool
        Whether to use an unbiased estimator.
    sample_cov_fraction : float
        A factor to use to down-weight off-diagonal elements of the sample covariance.
        ``sample_cov_fraction==0`` means use variance only, while
        ``sample_cov_fraction==1`` means use the full sample covariance.
    filter_half_widths : Sequence[float]
        The half-widths of the DPSS filters in nanoseconds.
    eigenval_cutoff : float
        The eigenvalue cutoff for determining DPSS filters.
    cache : dict
        A cache for storing DPSS filter matrices.

    Returns
    -------
    mean
        The inpainted mean as a numpy array of shape (nfreqs,)
    avg_flgs
        The flags as determined after simultaneous inpainting on the mean. If
        ``avg_flgs`` was given as input, it will be modified in-place (and also returned).
    models
        The inpainting models as a numpy array of shape (n_nights, nfreq).
    """
    # DPSS inpainting model
    model = np.zeros_like(stackd)
    mask = (~stackf).astype(float)
    avg_flgs = avg_flgs if avg_flgs is not None else np.all(stackf, axis=0)

    if hasattr(df, 'unit'):
        df = df.to_value("Hz")

    n_nights = stackd.shape[0]

    # Get median nsamples across the band
    nsamples_by_night = np.median(stackn, axis=1, keepdims=True)
    if np.any(nsamples_by_night != stackn):
        raise ValueError(
            'average_and_inpaint_simultaneously assumes that nsamples is constant over '
            'frequency for a given night and baseline.'
        )

    # Arrays for inpainted mean and total samples
    inpainted_mean = np.zeros(len(freqs), dtype=stackd.dtype)
    total_nsamples = np.zeros(len(freqs), dtype=float)

    cache = cache or {}

    for band in inpaint_bands:
        # if the band is already entirely flagged for all nights, continue
        if np.all(stackf[:, band]):
            continue

        # if there are too-large flag gaps even after a simple LST-stacking, continue
        max_allowed_gap_size = max_gap_factor * filter_half_widths[0]**-1 / df
        convolution_kernel = np.append(
            np.linspace(0, 1, int(max_allowed_gap_size) - 1, endpoint=False),
            np.linspace(1, 0, int(max_allowed_gap_size))
        )
        convolution_kernel /= np.sum(convolution_kernel)
        convolved_flags = signal.convolve(avg_flgs[band], convolution_kernel, mode='same') > max_convolved_flag_frac
        flagged_stretches = true_stretches(convolved_flags)
        longest_gap = np.max([ts.stop - ts.start for ts in flagged_stretches]) if len(flagged_stretches) > 0 else 0

        # Flag if combined gap is too large
        if longest_gap > max_allowed_gap_size:
            avg_flgs[band] = True
            continue

        # Get basis functions
        basis = dspec.dpss_operator(
            freqs[band],
            filter_centers=filter_centers,
            filter_half_widths=filter_half_widths,
            cache=cache,
            eigenval_cutoff=eigenval_cutoff
        )[0].real

        # Do the caching on a per night basis - find better way to do this
        CNinv_1sample_dpss = []
        CNinv_1sample_dpss_inv = []
        for night_index in range(n_nights):
            # compute fits for dpss basis functions
            hash_key = dspec._fourier_filter_hash(
                filter_centers=filter_centers,
                filter_half_widths=filter_half_widths,
                x=freqs[band],
                w=(base_noise_var[night_index, band] * mask[night_index, band])
            )

            # If key exists in cache, load in filter and inverse
            if hash_key in cache:
                _CNinv_1sample_dpss, _CNinv_1sample_dpss_inv = cache[hash_key]
                CNinv_1sample_dpss.append(_CNinv_1sample_dpss)
                CNinv_1sample_dpss_inv.append(_CNinv_1sample_dpss_inv)
            else:
                _CNinv_1sample_dpss = np.dot(basis.T * mask[night_index, band] / base_noise_var[night_index, band], basis)
                _CNinv_1sample_dpss_inv = np.linalg.pinv(_CNinv_1sample_dpss)
                CNinv_1sample_dpss.append(_CNinv_1sample_dpss)
                CNinv_1sample_dpss_inv.append(_CNinv_1sample_dpss_inv)
                cache[hash_key] = (_CNinv_1sample_dpss, _CNinv_1sample_dpss_inv)

        # Compute data covariance
        CNinv_dpss = CNinv_1sample_dpss * nsamples_by_night[:, np.newaxis]
        CNinv_dpss_inv = CNinv_1sample_dpss_inv / nsamples_by_night[:, np.newaxis]
        sum_CNinv_dpss = np.sum(CNinv_dpss, axis=0)

        # compute matrix product + get per-day DPSS fits
        noise_var = base_noise_var / nsamples_by_night
        CNinv_data_dpss = np.array([
            basis.T.dot(weighted_data) for weighted_data in mask[:, band] / noise_var[:, band] * stackd[:, band]
        ])
        print(CNinv_data_dpss.shape, CNinv_data_dpss.shape)
        dpss_fits = np.array([
            a.dot(b) if np.all(np.isfinite(b)) else a.dot(np.zeros_like(b))
            for a, b in zip(CNinv_dpss_inv, CNinv_data_dpss)
        ])

        # Find nights that are entirely flagged in this band
        is_unflagged_night = (~np.all(stackf[:, band], axis=1))

        # Compute weighted sample mean from per-day DPSS-fits and noise-weighted covariance matrix
        inv_sum_CNinv_dpss = np.linalg.pinv(sum_CNinv_dpss)
        sample_mean_dpss = inv_sum_CNinv_dpss @ np.einsum('nde,nd->e', CNinv_dpss[is_unflagged_night], dpss_fits[is_unflagged_night])

        # Compute weighted sample covariance of DPSS coefficients, then restrict it to the diagonal variance, then invert
        weighted_diff_dpss = np.array([
            np.dot(CNinv_dpss[i], (dpss_fits - sample_mean_dpss)[i])
            for i in range(n_nights) if is_unflagged_night[i]
        ])
        sample_cov_dpss = np.sum([
            np.outer(_weighted_diff_dpss, _weighted_diff_dpss.conj())
            for _weighted_diff_dpss in weighted_diff_dpss
        ], axis=0)

        # Calculate effective nights
        effective_nights = np.sum(np.mean(mask[:, band], axis=1))

        # TODO: Sample covariance is overestimated and has a fudge factor, this is probably close to correct
        # but not exactly right. We should revisit this.
        if effective_nights <= 1:
            sample_cov_dpss_inv = np.zeros_like(sample_cov_dpss)
        else:
            sample_cov_dpss = (
                inv_sum_CNinv_dpss @ (sample_cov_dpss - use_unbiased_estimator * sum_CNinv_dpss) @ inv_sum_CNinv_dpss
            ) * effective_nights ** 2 / (effective_nights - 1)
            sample_cov_dpss = (
                sample_cov_fraction * sample_cov_dpss
                + (1.0 - sample_cov_fraction) * np.diag(np.diag(sample_cov_dpss))
            )
            sample_cov_dpss_inv = np.linalg.pinv(sample_cov_dpss)

        CNinv_data_dpss = np.where(np.isfinite(CNinv_data_dpss), CNinv_data_dpss, 0)
        LU_decomp = [
            linalg.lu_factor(nightly_inv + sample_cov_dpss_inv)
            for nightly_inv in CNinv_dpss
        ]  # LU decomposition of Sigma_{N,i}^-1
        sample_cov_inv_sample_mean_dpss = sample_cov_dpss_inv.dot(sample_mean_dpss)
        post_mean = np.array([
            linalg.lu_solve(_LU_decomp, (nightly_weighted_data + sample_cov_inv_sample_mean_dpss))
            for _LU_decomp, nightly_weighted_data in zip(LU_decomp, CNinv_data_dpss)
        ])
        model[:, band] = basis.dot(post_mean.T).T

        # If we've made it this far, set averaged flags to False
        avg_flgs[band] = False

    # Shortcut here if everything is flagged.
    # Note that we can have avg_flgs be all flagged when not all of stackf is flagged
    # because we flag the average on "largest gaps". That's why we shortcut early here.
    if np.all(avg_flgs):
        return np.nan * inpainted_mean, avg_flgs, model

    # Inpainted mean is going to be sum(n_i * {model if flagged else data_i}) / sum(n_i)
    # where n_i is the nsamples for the i-th integration. The total_nsamples is
    # simply sum(n_i) for all i (originally flagged or not).
    inpainted_mean[:] = 0.0
    total_nsamples[:] = 0.0
    for nightidx, (d, f, n) in enumerate(zip(stackd, stackf, stackn)):
        # If an entire integration is flagged, don't use it at all
        # in the averaging -- it doesn't contibute any knowledge.
        if np.all(f):
            continue

        # Make model variable here
        inpainted_mean += n * np.where(f, model[nightidx], d)
        total_nsamples += n

    with np.errstate(divide='ignore', invalid='ignore'):
        inpainted_mean /= total_nsamples
        inpainted_mean[total_nsamples == 0] *= np.nan

    return inpainted_mean, avg_flgs, model


def average_and_inpaint_simultaneously(
    stack: LSTStack,
    auto_stack: LSTStack,
    inpaint_bands: tuple[slice] = (slice(0, None, None),),
    return_models: bool = True,
    cache: dict = {},
    filter_properties: dict | None = None,
    eigenval_cutoff: list[float] = [1e-12],
    round_filter_half_width: bool = True,
    max_gap_factor: float = 2.0,
    max_convolved_flag_frac: float = 0.667,
    use_unbiased_estimator: bool = False,
    sample_cov_fraction: float = 0.0
):
    """
    Average and inpaint simultaneously for all baselines in a stack.

    Parameters
    ----------
    stack : LSTStack
        The LSTStack object to average and inpaint.
    auto_stack : LSTStack
        The LSTStack object containing the auto-correlations.
    inpaint_bands : tuple[slice]
        The bands to inpaint individually, as slices over frequency channels.
    return_models : bool
        Whether to return the inpainting models.
    cache : dict
        A cache for storing DPSS filter matrices.
    filter_properties : dict
        The properties of the DPSS filters.
    eigenval_cutoff : list[float]
        The eigenvalue cutoff for determining DPSS filters.
    round_filter_half_width : bool
        Whether to round the filter half-width to the nearest nanosecond. This helps
        with caching.
    inpaint_max_gap_factor : float
        A factor determining the maximum allowed flagging gap.
    inpaint_max_convolved_flag_frac : float
        The maximum allowed fraction of convolved flags.
    use_unbiased_estimator : bool
        Whether to use an unbiased estimator.
    inpaint_sample_cov_fraction : float
        A factor to use to down-weight off-diagonal elements of the sample covariance.

    Returns
    -------
    lstavg : dict
        The LST-averaged data, flags, standard deviation, and nsamples.
    all_models : dict
        The inpainting models for each baseline (if return_models is True).
    """
    filter_properties = filter_properties or {}

    # Dictionary for model storage
    all_models = {}

    # Time axis is outer axis for all LSTStacks.
    antpos, ants = stack.get_ENU_antpos(pick_data_ants=False)
    antpos = dict(zip(ants, antpos))

    complete_flags = stack.flagged_or_inpainted()

    # First, perform a simple flagged-mode average over the nights.
    # lstavg is a dict of arrays with keys being 'mean', 'std', 'nsamples', 'flags'.
    lstavg = reduce_lst_bins(
        stack, get_std=False, get_mad=False, inpainted_mode=False, mean_fill_value=0.0
    )

    # Compute noise variance
    if auto_stack.data.shape[1] != 1:
        raise NotImplementedError('This code only works with redundantly averaged data, which has only one unique auto per polarization')

    for iap, antpair in enumerate(stack.antpairs):
        # Get the baseline vector and length
        bl_vec = (antpos[antpair[1]] - antpos[antpair[0]])[:]
        bl_len = np.linalg.norm(bl_vec) / constants.c
        filter_centers, filter_half_widths = vis_clean.gen_filter_properties(
            ax='freq',
            bl_len=max(bl_len, 7.0 / constants.c),
            **filter_properties,
        )

        # Round up filter half width to the nearest nanosecond
        # allows the cache to be hit more frequently
        if round_filter_half_width:
            filter_half_widths = [np.ceil(filter_half_widths[0] * 1e9) * 1e-9]

        for polidx, pol in enumerate(stack.pols):
            # Get easy access to the data, flags, and nsamples for this baseline-pol pair
            stackd = stack.data[:, iap, :, polidx]
            stackf = complete_flags[:, iap, :, polidx]
            stackn = np.abs(stack.nsamples[:, iap, :, polidx])
            flagged_mean = lstavg['data'][iap, :, polidx]
            avg_flgs = lstavg['flags'][iap, :, polidx]

            # Compute noise variance for all days in stack
            base_noise_var = np.abs(auto_stack.data[:, 0, :, polidx]) ** 2 / (stack.dt * stack.df).value

            # Shortcut early if there are no flags in the stack. In that case,
            # the LST-average is the same as the flagged-mode mean.
            if (not np.any(stackf)) or np.all(stackf):
                continue

            flagged_mean[:], _, model = average_and_inpaint_simultaneously_single_bl(
                freqs=stack.freq_array,
                stackd=stackd,
                stackf=stackf,
                stackn=stackn,
                avg_flgs=avg_flgs,
                base_noise_var=base_noise_var,
                df=stack.df,
                filter_centers=filter_centers,
                inpaint_bands=inpaint_bands,
                max_gap_factor=max_gap_factor,
                max_convolved_flag_frac=max_convolved_flag_frac,
                use_unbiased_estimator=use_unbiased_estimator,
                sample_cov_fraction=sample_cov_fraction,
                filter_half_widths=filter_half_widths,
                eigenval_cutoff=eigenval_cutoff,
                cache=cache
            )

            if return_models:
                all_models[(antpair[0], antpair[1], pol)] = model.copy()

    # Set data that is flagged to nan
    lstavg['data'][lstavg['flags']] = np.nan

    return lstavg, all_models
