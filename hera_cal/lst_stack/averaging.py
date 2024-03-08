from __future__ import annotations

import numpy as np
import logging
from typing import Literal
import warnings
from .flagging import threshold_flags, sigma_clip

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
    flags = flags | (np.isnan(data) | np.isinf(data))  # un-recoverable
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


def lst_average(
    data: np.ndarray | np.ma.MaskedArray,
    nsamples: np.ndarray | np.ma.MaskedArray,
    flags: np.ndarray,
    inpainted_mode: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute statistics of a set of data over its zeroth axis.

    The idea here is that the data's zeroth axis is "nights", and that each night is
    at the same LST. However, this function is agnostic to the meaning of the first
    axis. It just computes the mean, std, and nsamples over the first axis.

    This function is meant to be used on a single element of a list returned by
    :func:`simple_lst_bin`.

    Parameters
    ----------
    data
        The data to compute the statistics over. Shape ``(nnights, nbl, nfreq, npol)``.
    nsamples
        The number of samples for each measurement. Same shape as ``data``.
    flags
        The flags for each measurement. Same shape as ``data``.
    inpainted_mode
        Whether to use the inpainted samples when calculating the statistics. If False,
        the inpainted samples are ignored. If True, the inpainted samples are used, and
        the statistics are calculated over the un-inpainted samples.

    Returns
    -------
    out_data, out_flags, out_std, out_nsamples
        The reduced data, flags, standard deviation (across nights) and nsamples.
        Shape ``(nbl, nfreq, npol)``.
    """
    # all data is assumed to be in the same LST bin.

    if not isinstance(data, np.ma.MaskedArray):
        # Generally, we want a MaskedArray for 'data', where the mask is *either*
        # the flags or the 'non-inpainted flags', as obtained by `threshold_flags`.
        # However, if this hasn't been called, and we just have an array, apply flags
        # appropriately here.
        data, flags, nsamples = get_masked_data(
            data, flags, nsamples, inpainted_mode=inpainted_mode
        )

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
    # Multiply by nan instead of just setting as nan, so both real and imag parts are nan
    meandata[~normalizable] *= np.nan

    # While the previous nsamples is different for in-painted and flagged mode, which is
    # what we want for the mean, for the std and nsamples we want to treat flags as really
    # flagged.
    nsamples.mask = flags
    norm = np.sum(nsamples, axis=0)
    normalizable = norm > 0

    # get other stats
    logger.info("Calculating std")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice.")
        std = np.square(data.real - meandata.real) + 1j * np.square(
            data.imag - meandata.imag
        )
        std = np.sum(std * nsamples, axis=0)
        std[normalizable] /= norm[normalizable]
        std = np.sqrt(std.real) + 1j * np.sqrt(std.imag)

    std[~normalizable] = np.inf

    logger.info(f"Mean of meandata: {np.mean(meandata)}. Mean of std: {np.mean(std)}. Total nsamples: {np.sum(norm)}")
    return meandata.data, lstbin_flagged, std.data, norm.data, ndays_binned


def reduce_lst_bins(
    data: list[np.ndarray],
    flags: list[np.ndarray],
    nsamples: list[np.ndarray],
    inpainted_mode: bool = True,
    get_mad: bool = False,
) -> dict[str, np.ndarray]:
    """
    From a list of LST-binned data, produce reduced statistics.

    Use this function to reduce a list of `nlst_bins` arrays, each with multiple time
    integrations in them (i.e. the output of :func:`lst_align`) to arrays of shape
    ``(nbl, nlst_bins, nfreq, npol)``, each representing different statistics of the
    data in each LST bin (eg. mean, std, etc.).

    Parameters
    ----------
    data
        The data to perform the reduction over. The length of the list is the number
        of LST bins. Each array in the list should have shape
        ``(nintegrations_in_lst, nbl, nfreq, npol)`` -- i.e. the same as a UVData
        object's data_array if the blt axis is pushed out to two dimensions (time, bl).
    flags
        A list, the same length/shape as ``data``, containing the flags.
    nsamples
        A list, the same length/shape as ``data``, containing the number of samples
        for each measurement.
    where_inpainted
        A list, the same length/shape as ``data``, containing a boolean mask indicating
        which samples have been inpainted.
    inpainted_mode
        Whether to use the inpainted samples when calculating the statistics. If False,
        the inpainted samples are ignored. If True, the inpainted samples are used, and
        the statistics are calculated over the un-inpainted samples.
    get_mad
        Whether to compute the median and median absolute deviation of the data in each
        LST bin, in addition to the mean and standard deviation.

    Returns
    -------
    dict
        The reduced data in a dictionary. Keys are 'data' (the lst-binned mean),
        'nsamples', 'flags', 'days_binned' (the number of days that went into each bin),
        'std' (standard deviation) and *optionally* 'median' and 'mad' (if `get_mad` is
        True). All values are arrays of the same shape: ``(nlst_bins, nbl, nfreq, npol)``.
    """
    nlst_bins = len(data)
    (_, nbl, nfreq, npol) = data[0].shape

    for d, f, n in zip(data, flags, nsamples):
        assert d.shape == f.shape == n.shape

    out_data = np.zeros((nlst_bins, nbl, nfreq, npol), dtype=complex) * np.nan
    out_flags = np.ones(out_data.shape, dtype=bool)
    out_std = np.ones(out_data.shape, dtype=complex) * np.inf
    out_nsamples = np.zeros(out_data.shape, dtype=float)
    days_binned = np.zeros(out_data.shape, dtype=int)

    if get_mad:
        mad = np.ones(out_data.shape, dtype=complex) * np.inf
        med = np.ones(out_data.shape, dtype=complex) * np.nan

    for lstbin, (d, n, f) in enumerate(
        zip(data, nsamples, flags)
    ):
        logger.info(f"Computing LST bin {lstbin + 1} / {nlst_bins}")

        if d.size:  # If not, keep the default values initialized above
            d, f, n = get_masked_data(d, f, n, inpainted_mode=inpainted_mode)

            (
                out_data[lstbin],
                out_flags[lstbin],
                out_std[lstbin],
                out_nsamples[lstbin],
                days_binned[lstbin],
            ) = lst_average(d, n, f, inpainted_mode=inpainted_mode)

            if get_mad:
                med[lstbin], mad[lstbin] = get_lst_median_and_mad(d)

    out = {
        "data": out_data,
        "flags": out_flags,
        "std": out_std,
        "nsamples": out_nsamples,
        "days_binned": days_binned,
    }
    if get_mad:
        out["mad"] = mad
        out["median"] = med

    return out
