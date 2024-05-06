from __future__ import annotations

import numpy as np
from typing import Literal
import logging

logger = logging.getLogger(__name__)


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
    logger.info(
        f"Percent of data flagged before thresholding: {100 * nflags / flags.size:.2f}%"
    )
    flags |= flag_frac > flag_thresh
    logger.info(
        f"Flagged a further {100 * (np.sum(flags) - nflags) / flags.size:.2f}% of visibilities due to flag_frac > {flag_thresh}"
    )

    return flags


def sigma_clip(
    array: np.ndarray | np.ma.MaskedArray,
    threshold: float = 4.0,
    min_N: int = 4,
    median_axis: int = 0,
    threshold_axis: int = 0,
    clip_type: Literal['direct', 'mean', 'median'] = 'direct',
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
        scale = np.expand_dims(np.ma.median(np.abs(array - location), axis=median_axis) * 1.482579, axis=median_axis)
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

        if clip_type == 'direct':
            # In this mode, each datum is flagged individually.
            subflags[:] = subz > threshold
        elif clip_type in ['mean', 'median']:
            # In this mode, an entire sub-band of the data is flagged if its mean
            # (absolute) zscore is beyond the threshold.
            thisf = getattr(np.ma, clip_type)(subz, axis=threshold_axis) > threshold
            subflags[:] = np.expand_dims(thisf, axis=threshold_axis)
        else:
            raise ValueError(
                f"clip_type must be 'direct', 'mean' or 'median', got {clip_type}"
            )

    return clip_flags
