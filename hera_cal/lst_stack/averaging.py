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
    data: np.ndarray | np.ma.MaskedArray,
    nsamples: np.ndarray | np.ma.MaskedArray,
    meandata: np.ndarray,
    flags: np.ndarray | None = None
):
    logger.info("Calculating std")

    if flags is not None:
        if isinstance(data, np.ma.MaskedArray):
            data.mask = flags
        else:
            data = np.ma.masked_array(data, mask=flags)

        if isinstance(nsamples, np.ma.MaskedArray):
            nsamples.mask = flags
        else:
            nsamples = np.ma.masked_array(nsamples, mask=flags)

    norm = np.sum(nsamples, axis=0)

    normalizable = norm > 0

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice.")
        std = np.square(data.real - meandata.real) + 1j * np.square(
            data.imag - meandata.imag
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

    Returns
    -------
    out_data, out_flags, out_std, out_nsamples
        The reduced data, flags, standard deviation (across nights) and nsamples.
        Shape ``(nbl, nfreq, npol)``.
    """
    # all data is assumed to be in the same LST bin.
    assert (all(isinstance(x, np.ma.MaskedArray) for x in (data, nsamples)))

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


def average_and_inpaint_simultaneously(
    stack: LSTStack,
    inpaint_bands: Sequence[tuple[int, int] | slice] = (slice(0, None),),
    return_models: bool = True,
    cache: dict | None = None,
    filter_properties: dict | None = None,
    **kwargs,
) -> dict[tp.Baseline, np.ndarray]:
    r"""
    Perform an average over nights while simultaneously inpainting.

    For any particular baseline-pol-channel, this function will first perform a
    flagged-mode average:

    .. math:: \bar{d} = \frac{\sum_i (1 - f_i) n_i d_i}{N}

    where :math:`n_i` is the number of samples for the i-th integration, :math:`d_i` is
    the data for the i-th integration, :math:`f_i` is the flag for the i-th integration,
    and :math:`N` is the total number of samples:

    .. math:: N = \sum_i (1 - f_i) n_i

    Then it will create a model of the data by inpainting the averaged data:

    .. math:: m = \text{FourierFilter}(\bar{d})

    Finally, it will do a weighted average of the data and the model:

    .. math:: \bar{d}_{\text{inp}} = \frac{\sum_i n_i (f_i m + (1 - f_i) d_i)}{\sum_i n_i}.

    The output dictionary will have :math:`\bar{d}_{\text{inp}}` as the data, :math:`N`
    as the nsamples (i.e. only the sum of un-flagged samples), and the binned flags will
    simply be where the inpaint model either fails or is not attempted.

    Parameters
    ----------
    stack
        An LSTStack object containing the data to average.
    inpaint_bands
        The frequency bands to inpaint independently for each baseline-night. This
        should be a sequence of tuples, where each tuple is a start and end frequency
        to inpaint, or slices.
    return_models
        Whether to return the inpainted models.
    cache
        A dict-like object to use as a cache for the Fourier filter.
    filter_properties
        A dictionary of params to use for the Fourier filter. This is passed to
        :func:`hera_cal.vis_clean.gen_filter_properties`.

    Other Parameters
    ----------------
    kwargs
        Passed to :func:`hera_filters.dspec.fourier_filter`.

    Returns
    -------
    dict
        A dictionary of the LST-averaged data. This is in the same format as returned
        by :func:`reduce_lst_bins`.
    dict
        A dictionary of the inpainted models, keyed by (ant1, ant2, pol). If
        ``return_models`` is False, the dict is empty.
    """
    model = np.zeros(stack.Nfreqs, dtype=stack.data_array.dtype)
    filter_properties = filter_properties or {}

    all_models = {}

    # Time axis is outer axis for all LSTStacks.
    uvws = stack.uvw_array.reshape((stack.Ntimes, stack.Nbls, 3))

    complete_flags = stack.flagged_or_inpainted()

    # First, perform a simple flagged-mode average over the nights.
    # lstavg is a dict of arrays with keys being 'mean', 'std', 'nsamples', 'flags'.
    lstavg = reduce_lst_bins(
        stack, get_std=False, get_mad=False, inpainted_mode=False, mean_fill_value=0.0
    )

    inpainted_mean = np.zeros(stack.Nfreqs, dtype=stack.data.dtype)
    total_nsamples = np.zeros(stack.Nfreqs, dtype=float)

    for iap, antpair in enumerate(stack.antpairs):
        for polidx, pol in enumerate(stack.pols):
            # Get the data, flags, and nsamples for this baseline-pol pair, for the
            # whole LST-stack. Note that the incoming data may already be in-painted
            # on a per-day basis, in which case the nsamples for those data will be
            # negative, and they will be unflagged. Here, we want to use the original
            # per-day flags and nsamples (i.e. we effectively use the pre-inpainted
            # data).
            stackd = stack.data[:, iap, :, polidx]
            stackf = complete_flags[:, iap, :, polidx]
            stackn = np.abs(stack.nsamples[:, iap, :, polidx])

            # Also get the lst-avg data, flags, and nsamples for this baseline-pol pair.
            flagged_mean = lstavg['data'][iap, :, polidx]
            wgts = lstavg['nsamples'][iap, :, polidx]
            flgs = lstavg['flags'][iap, :, polidx]

            # Shortcut early if there are no flags in the stack. In that case,
            # the LST-average is the same as the flagged-mode mean.
            if (not np.any(stackf)) or np.all(stackf):
                continue

            # Get the baseline vector and length
            bl_vec = uvws[0, iap, :2]
            bl_len = np.linalg.norm(bl_vec) / constants.c
            filter_centers, filter_half_widths = vis_clean.gen_filter_properties(
                ax='freq',
                bl_len=max(bl_len, 7.0 / constants.c),
                **filter_properties,
            )

            for band in inpaint_bands:
                model[band], _, info = dspec.fourier_filter(
                    stack.freq_array[band],
                    flagged_mean[band],
                    wgts=wgts[band],
                    mode='dpss_solve',
                    max_contiguous_edge_flags=stack.Nfreqs,
                    cache=cache,
                    filter_centers=filter_centers,
                    filter_half_widths=filter_half_widths,
                    **kwargs,  # noqa: E225
                )

                # Update the flags. All successfully-inpainted data is unflagged.
                # If in-painting is unsuccessful, we flag the data.
                flgs[band] = (info['status']['axis_1'][0] == 'skipped')

            if return_models:
                all_models[(*antpair, pol)] = model.copy()

            # Inpainted mean is going to be sum(n_i * {model if flagged else data_i}) / sum(n_i)
            # where n_i is the nsamples for the i-th integration The total_nsamples is
            # simply sum(n_i) for all i (originally flagged or not).
            inpainted_mean[:] = 0.0
            total_nsamples[:] = 0.0
            for d, f, n in zip(stackd, stackf, stackn):
                # If an entire integration is flagged, don't use it at all
                # in the averaging -- it doesn't contibute any knowledge.
                if np.all(f):
                    continue

                inpainted_mean += n * np.where(f, model, d)
                total_nsamples += n

            with np.errstate(divide='ignore', invalid='ignore'):
                inpainted_mean /= total_nsamples
                inpainted_mean[total_nsamples == 0] *= np.nan

            # Overwrite the original averaged data with the inpainted mean.
            # The nsamples remains the same for inpainted vs. flagged mean (we don't
            # count inpainted samples as samples, but we do count them as data).
            flagged_mean[:] = inpainted_mean

    # Set data that is flagged to nan
    lstavg['data'][lstavg['flags']] = np.nan

    return lstavg, all_models
