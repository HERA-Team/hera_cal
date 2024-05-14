from __future__ import annotations

import numpy as np
import logging
import warnings
from .binning import LSTStack

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


def lst_average(
    data: np.ma.MaskedArray,
    nsamples: np.ma.MaskedArray,
    flags: np.ndarray,
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
    lststack: LSTStack | None = None,
    data: np.ndarray | None = None,
    flags: np.ndarray | None = None,
    nsamples: np.ndarray | None = None,
    inpainted_mode: bool = True,
    get_mad: bool = False,
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
        data, flags=flags, nsamples=nsamples
    )
    if get_mad:
        o['median'], o['mad'] = get_lst_median_and_mad(data)

    return o


def average_and_inpaint_simultaneously(
    stack: LSTStack,
    lstavg: dict[str, np.ndarray],
    inpaint_bands: Sequence[tuple[int, int] | slice] = (slice(0, None)),
    return_models: bool = True,
    cache: dict | None = None,
    filter_properties: dict | None = None,
    **kwargs,
) -> list[np.ndarray]:
    """
    Perform an average over nights while simultaneously inpainting.

    This function updates the ``data`` key in ``lstavg`` in-place. If you want to keep
    the original data, make a copy of it first.

    Parameters
    ----------
    stack
        An LSTStack object containing the data to average.
    lstavg
        A dictionary of LST-averaged data, with at _least_ keys 'data' and 'nsamples'.
        The averaging for this data must be in *flagged* mode. This can be the output
        of :func:`reduce_lst_bins`.
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
        A dictionary of the inpainted models, keyed by (ant1, ant2, pol). If
        ``return_models`` is False, the dict is empty.
    """
    model = np.zeros(stacks[0].Nfreqs, dtype=stacks[0].data_array.dtype)

    all_models = {}

    # Time axis is outer axis for all LSTStacks.
    uvws = stack.uvw_array.reshape((stack.Ntimes, stack.Nbls, 3))

    newmean = np.ones_like(avg['data']) * np.nan
    complete_flags = stack.flagged_or_inpainted()

    for iap, antpair in enumerate(stack.antpairs):
        for polidx, pol in enumerate(stack.pols):

            bl_vec = uvws[0, iap, :2]
            bl_len = np.linalg.norm(bl_vec) / constants.c
            filter_centers, filter_half_widths = vis_clean.gen_filter_properties(
                ax='freq',
                bl_len=bl_len,
                **filter_properties,
            )

            flagged_mean = lstavg['data'][iap, :, polidx]
            wgts = lstavg['nsamples'][iap, :, polidx]

            for band in inpaint_bands:
                model[band], _, info = dspec.fourier_filter(
                    stack.freq_array[band],
                    flagged_mean[band],
                    wgts=wgts[band],
                    mode='dpss_solve',
                    max_contiguous_edge_flags=stack.Nfreqs,
                    cache=cache
                    **kwargs,  # noqa: E225
                )

            if return_models:
                all_models[(*antpair, pol)] = model.copy()

            # fill in the data with the model
            data = stack.data[:, iap, :, polidx]
            flags = complete_flags[:, iap, :, polidx]
            nsamples = np.abs(stack.nsamples[:, iap, :, polidx])

            this = np.zeros(stack.Nfreqs, dtype=stack.data.dtype)
            nn = np.zeros(stack.Nfreqs, dtype=float)
            for d, f, n in zip(data, flags, nsamples):
                # If an entire integration is flagged, don't use it at all
                # in the averaging -- it doesn't contibute any knowledge.
                if np.all(f):
                    continue

                this += n * np.where(f, model, d)
                nn += n

            with np.errstate(divide='ignore', invalid='ignore'):
                this /= nn
                this[nn == 0] *= np.nan

            newmean[iap, :, polidx] = this

        lstavg['data'] = newmean

    return all_models
