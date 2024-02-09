import numpy as np
import logging
from typing import Literal
import warnings

logger = logging.getLogger(__name__)


def reduce_lst_bins(
    data: list[np.ndarray],
    flags: list[np.ndarray],
    nsamples: list[np.ndarray],
    where_inpainted: list[np.ndarray] | None = None,
    inpainted_mode: bool = False,
    mutable: bool = False,
    sigma_clip_thresh: float | None = None,
    sigma_clip_min_N: int = 4,
    sigma_clip_type: str = "direct",
    sigma_clip_subbands: list[int] | None = None,
    sigma_clip_scale: list[np.ndarray] | None = None,
    flag_below_min_N: bool = False,
    flag_thresh: float = 0.7,
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
        ``(nintegrations_in_lst, nbl, nfreq, npol)``.
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
    mutable
        Whether the input data (and flags and nsamples) can be modified in place within
        the algorithm. Setting to true saves memory, and is safe for a one-shot script.
    sigma_clip_thresh
        The number of standard deviations to use as a threshold for sigma clipping.
        If None (default), no sigma clipping is performed. Note that sigma-clipping is performed
        per baseline, frequency, and polarization.
    sigma_clip_min_N
        The minimum number of unflagged samples required to perform sigma clipping.
    flag_below_min_N
        Whether to flag data that has fewer than ``sigma_clip_min_N`` unflagged samples.
    flag_thresh
        The fraction of integrations for a particular (antpair, pol, channel) combination
        within an LST-bin that can be flagged before that combination is flagged
        in the LST-average.
    get_mad
        Whether to compute the median and median absolute deviation of the data in each
        LST bin, in addition to the mean and standard deviation.

    Returns
    -------
    dict
        The reduced data in a dictionary. Keys are 'data' (the lst-binned mean),
        'nsamples', 'flags', 'days_binned' (the number of days that went into each bin),
        'std' (standard deviation) and *otionally* 'median' and 'mad' (if `get_mad` is
        True). All values are arrays of the same shape: ``(nbl, nlst_bins, nfreq, npol)``.
    """
    nlst_bins = len(data)
    (_, nbl, nfreq, npol) = data[0].shape

    for d, f, n in zip(data, flags, nsamples):
        assert d.shape == f.shape == n.shape

    # Do this just so that we can save memory if the call to this function already
    # has allocated memory.

    out_data = np.zeros((nbl, nlst_bins, nfreq, npol), dtype=complex)
    out_flags = np.zeros(out_data.shape, dtype=bool)
    out_std = np.ones(out_data.shape, dtype=complex)
    out_nsamples = np.zeros(out_data.shape, dtype=float)
    days_binned = np.zeros(out_nsamples.shape, dtype=int)

    if get_mad:
        mad = np.ones(out_data.shape, dtype=complex)
        med = np.ones(out_data.shape, dtype=complex)

    if where_inpainted is None:
        where_inpainted = [None] * nlst_bins

    if sigma_clip_scale is None:
        sigma_clip_scale = [None] * nlst_bins

    for lstbin, (d, n, f, clip_scale, inpf) in enumerate(
        zip(data, nsamples, flags, sigma_clip_scale, where_inpainted)
    ):
        logger.info(f"Computing LST bin {lstbin+1} / {nlst_bins}")

        # TODO: check that this doesn't make yet another copy...
        # This is just the data in this particular lst-bin.

        if d.size:
            d, f = get_masked_data(
                d,
                n,
                f,
                inpainted=inpf,
                inpainted_mode=inpainted_mode,
                flag_thresh=flag_thresh,
            )

            (
                out_data[:, lstbin],
                out_flags[:, lstbin],
                out_std[:, lstbin],
                out_nsamples[:, lstbin],
                days_binned[:, lstbin],
            ) = lst_average(
                d,
                n,
                f,
                inpainted_mode=inpainted_mode,
                sigma_clip_thresh=sigma_clip_thresh,
                sigma_clip_min_N=sigma_clip_min_N,
                sigma_clip_subbands=sigma_clip_subbands,
                sigma_clip_type=sigma_clip_type,
                sigma_clip_scale=clip_scale,
                flag_below_min_N=flag_below_min_N,
            )

            if get_mad:
                med[:, lstbin], mad[:, lstbin] = get_lst_median_and_mad(d)
        else:
            out_data[:, lstbin] *= np.nan
            out_flags[:, lstbin] = True
            out_std[:, lstbin] *= np.inf
            out_nsamples[:, lstbin] = 0.0

            if get_mad:
                mad[:, lstbin] *= np.inf
                med[:, lstbin] *= np.nan

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


def get_masked_data(
    data: np.ndarray,
    nsamples: np.ndarray,
    flags: np.ndarray,
    inpainted: np.ndarray | None = None,
    inpainted_mode: bool = False,
    flag_thresh: float = 0.7,
) -> np.ma.MaskedArray:
    if not inpainted_mode:
        # Act like nothing is inpainted.
        inpainted = np.zeros(flags.shape, dtype=bool)
    elif inpainted is None:
        # Flag if a whole blt is flagged:
        allf = np.all(flags, axis=2)[:, :, None, :]

        # Assume everything else that's flagged is inpainted.
        inpainted = flags.copy() * (~allf)

    flags = flags | np.isnan(data) | np.isinf(data) | (nsamples == 0)

    # Threshold flags over time here, because we want the new flags to be treated on the
    # same footing as the inpainted flags.
    threshold_flags(flags, inplace=True, flag_thresh=flag_thresh)

    logger.info(
        f"In inpainted_mode: {inpainted_mode}. Got {np.sum(inpainted)} inpainted samples, {np.sum(flags)} total flags, {np.sum(flags & ~inpainted)} non-inpainted flags."
    )
    data = np.ma.masked_array(data, mask=(flags & ~inpainted))
    return data, flags


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


def threshold_flags(
    flags: np.ndarray,
    inplace: bool = False,
    flag_thresh: float = 0.7,
):
    """
    Thresholds the input flags array based on the flag fraction.

    Parameters
    ----------
    flags : numpy.ndarray
        A numpy array of shape (Nnights, ...) representing the flags.
    inplace : bool, optional
        If True, modifies the input flags array in place. If False, creates a copy of
        the flags array.
    flag_thresh : float, optional
        The threshold value for the flag fraction.

    Returns
    -------
    numpy.ndarray
        A numpy array of shape (N, ...) with the thresholded flags.

    Examples
    --------
    >>> flags = np.array([[True, False, True], [False, True, False]])
    >>> threshold_flags(flags, inplace=True, flag_thresh=0.5)
    array([[ True, False,  True],
           [False,  True, False]])
    """
    if not inplace:
        flags = flags.copy()

    # Flag entire LST bins if there are too many flags over time
    flag_frac = np.sum(flags, axis=0) / flags.shape[0]
    nflags = np.sum(flags)
    logger.info(f"Percent of data flagged before thresholding: {100*nflags/flags.size:.2f}%")
    flags |= flag_frac > flag_thresh
    logger.info(
        f"Flagged a further {100*(np.sum(flags) - nflags)/flags.size:.2f}% of visibilities due to flag_frac > {flag_thresh}"
    )

    return flags


def sigma_clip(
    array: np.ndarray | np.ma.MaskedArray,
    threshold: float = 4.0,
    min_N: int = 4,
    median_axis: int = 0,
    threshold_axis: int = 0,
    clip_type: Literal["direct", "mean", "median"] = "direct",
    flag_bands: list[tuple[int, int]] | None = None,
    scale: np.ndarray | None = None,
):
    """
    One-iteration robust sigma clipping algorithm.

    Parameters
    ----------
    array
        ndarray of *real* data, of any dimension. If a MaskedArray, the mask is
        respected. If not, a masked array is created that masks out NaN values in the
        array. While the input array can be of any dimension, it must have at least
        ``ndim > max(median_axis, threshold_axis)``. In the context of this module
        (lst binning), we expect the array to be of shape
        ``(Nnights, Nbls, Nfreqs, Npols)``.
    threshold
        Threshold to cut above, in units of the standard deviation.
    min_N
        minimum length of array to sigma clip, below which no sigma
        clipping is performed. Non-clipped values are *not* flagged.
    median_axis
        Axis along which to perform the median to determine the zscore of individual
        data.
    threshold_axis
        Axis along which to perform the thresholding, if multiple data are to be
        combined before thresholding. This is only applicable if ``clip_type`` is
        ``mean`` or ``median``. In this case, if for example a 2D array is passed in
        and ``threshold_axis=1`` (but no ``flag_bands`` is passed), then the mean of
        the absolute zscores is take along the final axis, and the output flags are
        applied homogeneously across this axis based on this mean.
    clip_type
        The type of sigma clipping to perform. If ``direct``, each datum is flagged
        individually. If ``mean`` or ``median``, an entire sub-band of the data is
        flagged if its mean (absolute) zscore is beyond the threshold.
    flag_bands
        A list of tuples specifying the start and end indices of the threshold axis
        over which to perform sigma clipping. They are used in a ``slice`` object,
        so that the end is exclusive but the start is inclusive. If None, the entire
        threshold axis is used at once.
    scale
        If given, interpreted as the expected standard deviation of the data
        (over nights). If not given, estimated from the data using the median
        absolute deviation. If given, must be an array with eitherthe same
        shape as ``array`` OR the same shape as ``array`` with the ``median_axis``
        removed. If the former, each variate over the ``median_axis`` is scaled
        independently. If the latter, the same scale is applied to all variates
        (generally nights).

    Output
    ------
    clip_flags
        A boolean array with same shape as input array,
        with clipped values set to True.
    """
    # ensure array is an array
    if not isinstance(array, np.ndarray):  # this covers np.ma.MaskedArray as well
        array = np.asarray(array)

    if np.iscomplexobj(array):
        raise ValueError("array must be real")

    # ensure array passes min_N criterion:
    if array.shape[median_axis] < min_N:
        return np.zeros_like(array, dtype=bool)

    if not isinstance(array, np.ma.MaskedArray):
        array = np.ma.MaskedArray(array, mask=np.isnan(array))

    location = np.expand_dims(np.ma.median(array, axis=median_axis), axis=median_axis)

    if scale is None:
        scale = np.expand_dims(
            np.ma.median(np.abs(array - location), axis=median_axis) * 1.482579,
            axis=median_axis,
        )
    elif scale.ndim == array.ndim - 1:
        scale = np.expand_dims(scale, axis=median_axis)

    if (scale.shape != array.shape and scale.shape[median_axis] != 1) or scale.ndim != array.ndim:
        raise ValueError(
            "scale must have same shape as array or array with median_axis removed."
            f"Got {scale.shape}, needed {array.shape}"
        )

    if flag_bands is None:
        # Use entire threshold axis together
        flag_bands = [(0, array.shape[threshold_axis])]

    zscore = np.abs(array - location) / scale

    clip_flags = np.zeros(array.shape, dtype=bool)

    for band in flag_bands:
        # build the slice index. Don't use np.take with axis= parameter because it
        # creates a new array, instead of a view.
        mask = [slice(None)] * array.ndim
        mask[threshold_axis] = slice(*band)
        mask = tuple(mask)

        subz = zscore[mask]
        subflags = clip_flags[mask]

        if clip_type == "direct":
            # In this mode, each datum is flagged individually.
            subflags[:] = subz > threshold
        elif clip_type in ["mean", "median"]:
            # In this mode, an entire sub-band of the data is flagged if its mean
            # (absolute) zscore is beyond the threshold.
            thisf = getattr(np.ma, clip_type)(subz, axis=threshold_axis) > threshold
            subflags[:] = np.expand_dims(thisf, axis=threshold_axis)
        else:
            raise ValueError(f"clip_type must be 'direct', 'mean' or 'median', got {clip_type}")

    return clip_flags


def lst_average(
    data: np.ndarray | np.ma.MaskedArray,
    nsamples: np.ndarray,
    flags: np.ndarray,
    inpainted_mode: bool = False,
    sigma_clip_thresh: float | None = None,
    sigma_clip_min_N: int = 4,
    flag_below_min_N: bool = False,
    sigma_clip_subbands: list[int] | None = None,
    sigma_clip_type: Literal["direct", "mean", "median"] = "direct",
    sigma_clip_scale: np.ndarray | None = None,
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
    flag_thresh
        The fraction of integrations for a particular (antpair, pol, channel) combination
        within an LST-bin that can be flagged before that combination is flagged
        in the LST-average.
    sigma_clip_thresh
        The number of standard deviations to use as a threshold for sigma clipping.
        If None (default), no sigma clipping is performed. Note that sigma-clipping is performed
        per baseline, frequency, and polarization.
    sigma_clip_min_N
        The minimum number of unflagged samples required to perform sigma clipping.
    flag_below_min_N
        Whether to flag data that has fewer than ``sigma_clip_min_N`` unflagged samples.
    sigma_clip_subbands
        A list of integers specifying the start and end indices of the frequency axis
        to perform sigma clipping over. If None, the entire frequency axis is used at
        once. Given a list of integers e.g. ``[0, 10, 20]``, the sub-bands will be
        defined as [(0, 10), (10, 20)], where each 2-tuple defines a standard Python
        slice object (i.e. end is exclusive, start is inclusive).
    sigma_clip_type
        The type of sigma clipping to perform. If ``direct``, each datum is flagged
        individually. If ``mean`` or ``median``, an entire sub-band of the data is
        flagged if its mean (absolute) zscore is beyond the threshold.

    Returns
    -------
    out_data, out_flags, out_std, out_nsamples
        The reduced data, flags, standard deviation (across nights) and nsamples.
        Shape ``(nbl, nfreq, npol)``.
    """
    # data has shape (nnights, nbl, npols, nfreqs)
    # all data is assumed to be in the same LST bin.

    if not isinstance(data, np.ma.MaskedArray):
        # Generally, we want a MaskedArray for 'data', where the mask is *either*
        # the flags or the 'non-inpainted flags', as obtained by `threshold_flags`.
        # However, if this hasn't been called, and we just have an array, apply flags
        # appropriately here.
        data, flags = get_masked_data(data, nsamples, flags, inpainted_mode=inpainted_mode)

    # Now do sigma-clipping.
    if sigma_clip_thresh is not None:
        if inpainted_mode and sigma_clip_type == "direct":
            warnings.warn(
                "Direct-mode sigma-clipping in in-painted mode is a bad idea, because it creates "
                "non-uniform flags over frequency, which can cause artificial spectral "
                "structure. In-painted mode specifically attempts to avoid this."
            )

        nflags = np.sum(flags)
        kw = {
            "threshold": sigma_clip_thresh,
            "min_N": sigma_clip_min_N,
            "clip_type": sigma_clip_type,
            "median_axis": 0,
            "threshold_axis": 0 if sigma_clip_type == "direct" else -2,
            "flag_bands": (
                list(zip(sigma_clip_subbands[:-1], sigma_clip_subbands[1:]))
                if sigma_clip_subbands
                else None
            ),
            "scale": sigma_clip_scale,
        }
        clip_flags = sigma_clip(data.real, **kw)
        clip_flags |= sigma_clip(data.imag, **kw)

        # Need to restore min_N condition properly here because it's not done properly in sigma_clip
        sc_min_N = np.sum(~flags, axis=0) < sigma_clip_min_N
        clip_flags[:, sc_min_N] = False

        flags |= clip_flags

        data.mask |= clip_flags

        logger.info(
            f"Flagged a further {100*(np.sum(flags) - nflags)/flags.size:.2f}% of visibilities due to sigma clipping"
        )

    # Here we do a check to make sure Nsamples is uniform across frequency.
    # Do this before setting non_inpainted to zero nsamples.
    ndiff = np.diff(nsamples, axis=2)
    if np.any(ndiff != 0):
        warnings.warn(
            "Nsamples is not uniform across frequency. This will result in spectral structure."
        )

    nsamples = np.ma.masked_array(nsamples, mask=data.mask)

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

    if flag_below_min_N:
        lstbin_flagged[ndays_binned < sigma_clip_min_N] = True

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
        std = np.square(data.real - meandata.real) + 1j * np.square(data.imag - meandata.imag)
        std = np.sum(std * nsamples, axis=0)
        std[normalizable] /= norm[normalizable]
        std = np.sqrt(std.real) + 1j * np.sqrt(std.imag)

    std[~normalizable] = np.inf

    logger.info(
        f"Mean of meandata: {np.mean(meandata)}. Mean of std: {np.mean(std)}. Total nsamples: {np.sum(norm)}"
    )
    return meandata.data, lstbin_flagged, std.data, norm.data, ndays_binned
