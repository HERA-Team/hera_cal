"""Statistics for LST-binned data."""
from __future__ import annotations

from scipy.stats import rv_continuous, gamma, chi2
from ..datacontainer import DataContainer, RedDataContainer
import attrs
import numpy as np
from functools import partial
from .. import utils
from pyuvdata import UVFlag
from ..red_groups import RedundantGroups
from .. import types as tp
from .binning import LSTStack


@attrs.define(slots=False)
class LSTBinStats:
    """Class that holds basic LST-binned data and statistics"""
    mean: DataContainer = attrs.field()
    std: DataContainer = attrs.field()
    nsamples: DataContainer = attrs.field()
    flags: DataContainer = attrs.field()
    days_binned: DataContainer = attrs.field()
    median: DataContainer | None = attrs.field(default=None)
    mad: DataContainer | None = attrs.field(default=None)

    @classmethod
    def from_reduced_data(
        cls,
        antpairs: list[tuple[int, int]],
        pols: list[str],
        rdc: dict[str, np.ndarray],
        reds: RedundantGroups | None = None,
    ):
        """Return a class given a dictionary computed by reduce_lst_bins."""
        outdict = {}
        kls = partial(RedDataContainer, reds=reds) if reds is not None else DataContainer

        for k, v in rdc.items():
            if v is None:
                continue
            if k == 'data':
                k = 'mean'
            dct = {antpair + (pol,): v[i, :, j] for i, antpair in enumerate(antpairs) for j, pol in enumerate(pols)}
            outdict[k] = kls(dct)

        return cls(**outdict)

    @property
    def bls(self):
        return self.mean.bls()


def get_nightly_predicted_variance(
    bl: tuple[int, int, str],
    stack: LSTStack,
    auto_stats: LSTBinStats,
) -> np.ndarray:
    """
    Get predicted thermal variance on a single bl for each night & freq.

    Output is an array of shape (Nnights, Nfreqs).
    """
    auto = auto_stats.mean[(bl[0], bl[0], bl[2])] * auto_stats.mean[(bl[1], bl[1], bl[2])]

    dtdf = (stack.dt * stack.df).to_value("")

    gf = stack.get_flags(bl)
    with np.errstate(divide='ignore'):
        # Some nsamples can be zero, so we ignore warnings from that.
        # Also note that in the stack, inpainted samples have -nsamples, rather than
        # zero. These will thus return a finite expected variance, even though they
        # technically have no unflagged samples.
        per_day_expected_var = np.abs(auto / dtdf / stack.get_nsamples(bl))
    per_day_expected_var[gf] = np.inf
    return per_day_expected_var


def get_nightly_predicted_variance_stack(
    stack: LSTStack,
    auto_stats: LSTBinStats,
    flag_if_inpainted: bool = True,
) -> np.ndarray:
    """
    Get predicted thermal variance for each bl, night & freq.

    Output is an array with same shape as stack.data, containing the variance.
    """
    auto = np.array([
        [
            auto_stats.mean[(ap[0], ap[0], pol)] * auto_stats.mean[(ap[1], ap[1], pol)]
            for pol in stack.pols
        ] for ap in stack.antpairs
    ]).transpose((0, 2, 1))  # nbls, nfreq, npol

    dtdf = (stack.dt * stack.df).to_value("")

    with np.errstate(divide='ignore', invalid='ignore'):
        # Some nsamples can be zero, so we ignore warnings from that.
        # Also note that in the stack, inpainted samples have -nsamples, rather than
        # zero. These will thus return a finite expected variance, even though they
        # technically have no unflagged samples.
        per_day_expected_var = np.abs(auto / dtdf / stack.nsamples)

    gf = stack.flagged_or_inpainted() if flag_if_inpainted else stack.flags
    per_day_expected_var[gf] = np.inf

    return per_day_expected_var


def get_squared_zscores(
    auto_stats: LSTBinStats,
    cross_stats: LSTBinStats,
    stack: LSTStack,
    central: str = 'mean',
    std: str = 'autos'
) -> LSTStack:
    """
    Obtain squared Z-scores as a UVFlag object in metrics mode.

    The Z-score is defined as:

    .. math:: Z_i \\equiv \\sqrt{\frac{2n_i}{\\sigma^2}}\frac{M}{M-n_i}(V_i - \bar{V}).

    Where :math:`n_i` is the number of samples, :math:`\\sigma^2` is the variance of the
    visibility, and M is the total number of samples. This definition and associated
    distributions can be found in [this memo](https://github.com/HERA-Team/H6C-analysis/blob/main/docs/statistics_of_visibilities.ipynb).

    Parameters
    ----------
    auto_stats : LSTBinStats
        LST-binned statistics for the autos. Used to predict the variance of the
        cross-correlations if std=='autos'.
    cross_stats : LSTBinStats
        LST-binned statistics for the cross-correlations. Used to compute the Z-scores,
        by obtaining the mean or median.
    stack : LSTStack
        The LST-stack of data over nights, as an LSTStack object (which is essentially
        a UVData object).
    central : {'mean', 'median'}
        The central value to use for the Z-score. If 'mean', the mean of the
        visibilities over nights is used. If 'median', the median is used.
    std : {'autos', 'std', 'mad'}
        The standard deviation to use for the Z-score. If 'autos', the autos are used
        to predict the variance of the cross-correlations. If 'std' or 'mad', the
        cross-correlations are used, and the statistic is estimated over nights from
        the sample.

    Returns
    -------
    zstack : LSTStack
        The squared Z-scores as a UVFlag object in metrics mode.

    See Also
    --------
    stats.zsquare
        The predicted distribution of the results of this function.
    """
    if central not in ("mean", "median"):
        raise ValueError("central must be 'mean' or 'median'")

    if std not in {'autos', 'std', 'mad'}:
        raise ValueError("std must be 'autos', 'std' or 'mad'")

    zsq = np.zeros(stack.data.shape, dtype=np.float32)

    for iap, ap in enumerate(stack.antpairs):
        for ipol, pol in enumerate(stack.pols):
            bl = (*ap, pol)
            data = stack.get_data(bl)
            nsamps = stack.get_nsamples(bl)  # per-night nsamples
            M = cross_stats.nsamples[bl]     # total nsamples over nights
            norm = nsamps * (M / (M - nsamps))

            zsq_view = zsq[:, iap, :, ipol]
            zsq_view[:] = norm * np.abs(data - getattr(cross_stats, central)[bl])**2

            # Divide variance by 2 to get the variance of the real/imaginary parts.
            # That is, z = (data - centre) / sqrt(variance / 2), so that each component
            # (real/imag) is a Gaussian with variance equal to 1. Then zsq = |z|^2 is just
            # the sum of two standard normal variables squared, which is chi2(2).

            if std == 'autos':
                # This is the variance such that each of the components (real/imag) is a
                # Gaussian with variance equal to this value / 2.
                zsq_view /= get_nightly_predicted_variance(bl, stack, auto_stats) / 2
            else:
                # This variance is also the variance of the magnitude of the complex number,
                # the same as using the autos method.
                zsq_view /= np.abs(np.where(
                    cross_stats.flags[bl], np.inf, getattr(cross_stats, std)[bl]
                ))**2 / 2

    # convert zstack to UVFlag object
    zstack = UVFlag(stack._uvd, mode='metric', use_future_array_shapes=True)
    zstack.metric_array = zsq.reshape(stack.data_array.shape)

    return LSTStack(zstack)


def get_squared_zscores_flagged(
    stack: LSTStack,
    variance: np.ndarray | None = None,
    auto_stats: LSTBinStats | None = None,
):
    """
    Obtain squared Z-scores as a UVFlag object in metrics mode.

    This function is similar to :func:`get_squared_zscores` -- it takes in the same
    information and returns the same output. However, it is more limited in its options:
    it only implements the 'autos' method for the standard deviation, and the 'mean'
    method for the central value. It is also faster, because it vectorizes the computation
    over all baselines, polarizations, and frequencies. Finally, it allows the nightly
    variance to be passed in, instead of computed within the function, which speeds up
    calculations if the nightly variance is already known but the Z^2 scores need to be
    re-evaluated many times (e.g. if iterative flagging is being performed).

    Parameters
    ----------
    stack : LSTStack
        The LST-stack of data over nights, as an LSTStack object (which is essentially
        a UVData object).
    variance : np.ndarray | None
        The nightly variance of the data, with the same shape as ``stack.data``.
        If None, it is computed from the autos. Either this or the auto_stats must be
        given.
    auto_stats : LSTBinStats | None
        LST-binned statistics for the autos. Used to predict the variance of the
        cross-correlations if ``variance`` is not given. Must be provided if ``variance``
        is not.

    Returns
    -------
    zstack : LSTStack
        The squared Z-scores as an LSTStack-ed UVFlag object in metrics mode (the
        zscores can be accessed via ``zstack.metrics``).
    """
    zsq = np.zeros(stack.data.shape, dtype=np.float32)

    # inpainted data should _not_ be counted in the nsamples here,
    # because its not counted in the mean (vbar) below.
    nsamples = np.abs(stack.nsamples) * ~stack.flagged_or_inpainted() * np.isfinite(stack.data)

    if variance is None:
        variance = get_nightly_predicted_variance_stack(stack, auto_stats, flag_if_inpainted=True) / 2

    # Get V mean
    vbar = np.nansum(stack.data * nsamples, axis=0)
    M = np.sum(nsamples, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        vbar /= M
        norm = M / (M - nsamples)
        zsq = norm * np.abs(stack.data - vbar)**2 / variance

    with np.errstate(invalid='ignore'):
        # Either zero or one visibility in the mean, or no samples at that point -- undefined z-score.
        mask = (M == 0) | (M == nsamples) | (nsamples == 0)
        zsq[mask] = np.nan

    # convert zstack to UVFlag object
    zstack = UVFlag(stack._uvd, mode='metric', use_future_array_shapes=True)
    zstack.metric_array = zsq.reshape(stack.data_array.shape)
    return LSTStack(zstack)


def get_selected_bls(
    bls: list[tp.Baseline],
    days_binned: DataContainer | None = None,
    selectors=None,
    min_days: int = 7
):
    """Get a subset of baselines that satisfy the given selectors.

    Parameters
    ----------
    bls : list[tuple[int, int, str]]
        List of baselines to select from.
    days_binned : DataContainer
        A DataContainer with the number of days binned for each baseline.
    selectors : list[Callable] | None
        A list of callables that take a baseline (int, int str) and return a boolean,
        for whether to include that baseline.
    min_days : int
        The minimum number of days a baseline must be binned to be included.
    """
    if selectors is None:
        selectors = []
    elif callable(selectors):
        selectors = [selectors]

    if min_days:
        selectors.append(lambda bl: bl in days_binned and (np.median(days_binned[bl]) >= min_days))

    def select(bl):
        return all(sel(bl) for sel in selectors)

    return [bl for bl in bls if select(bl)]


def downselect_zscores(
    zscores: LSTStack,
    antpairs: list[tp.Antpair] | None = None,
    band: tuple[int, int] | slice | None = None,
    nights: list[int] | None = None,
    pols: str | None = None,
    bl_selectors=None,
    flags: np.ndarray | None = None,
) -> list[np.ndarray]:
    """
    Downselect zscores to a subset of baselines, nights, and polarizations.

    This does not use the select() method on the UVFlag object, because we don't need
    to do _all_ the selections on the metadata and everything here: we just return
    a simple array, which is intended to be directly used in some averaging or plot
    after this.

    Parameters
    ----------
    zscores : LSTStack
        The LSTStack object to downselect. This should be an LSTStack wrapping
        a UVFlag object in metrics mode (i.e. the output of get_squared_zscores).
    bls : list[tuple[int, int]] | None
        List of baselines to get the data from. If None, all baselines are used.
    band : tuple[int, int] | slice | None
        The frequency band to use. If None, all frequencies are used.
    nights : int | list[int] | 'all'
        The nights to use. By default, all nights are used.
    pols : str | list[str]
        The polarizations to use. By default, all polarizations are used.
    bl_selectors : list[Callable] | None
        A list of callables that take a baseline (int, int str) and return a boolean,
        for whether to include that baseline. This can be used instead of passing
        bls directly.
    flags : np.ndarray | None
        Any flags for which to mask out particular zscore values. Must be the same
        shape as zscores.metrics

    Returns
    -------
    zscores : np.ma.MaskedArray
        The downselected zscores, as an array with the same form as zscores.metric_array,
        but with some axes smaller after subselection. The mask is from the flags
        in stack.
    """
    nbls = zscores.Nbls
    datapols = utils.polnum2str(
        zscores.polarization_array, x_orientation=zscores.x_orientation
    )
    datapairs = list(
        zip(zscores.ant_1_array[:nbls], zscores.ant_2_array[:nbls])
    )

    if bl_selectors is not None:
        allbls = [(a, b, p) for a, b in datapairs for p in datapols]
        bls = get_selected_bls(allbls, min_days=0, selectors=bl_selectors)
        selpols = {p for a, b, p in bls}
        antpairs = list({bl[:2] for bl in bls})  # only antpairs

        if 'ee' in selpols and 'nn' in selpols:
            pols = None
        elif 'ee' in selpols:
            pols = 'ee'
        elif 'nn' in selpols:
            pols = 'nn'

    # Get pol indices
    if pols is None:
        pols = slice(None)
    elif isinstance(pols, str):
        pols = [datapols.index(pols)]
    else:
        pols = [datapols.index(p) for p in pols]

    # Get bl indices
    if antpairs is None:
        antpairs = slice(None)
    elif isinstance(antpairs, tuple) and len(antpairs) == 2:
        antpairs = [datapairs.index(antpairs)]
    elif isinstance(antpairs, list):
        antpairs = [datapairs.index(ap) for ap in antpairs]

    if band is None:
        band = slice(None)
    elif isinstance(band, tuple):
        band = slice(band[0], band[1])

    if not isinstance(band, slice):
        raise TypeError("band must be a tuple of (low, high) or a slice")

    if flags is None:
        flags = ~np.isfinite(zscores.metrics)
    else:
        flags = flags | ~np.isfinite(zscores.metrics)

    zsq = np.ma.MaskedArray(
        zscores.metrics[:, :, band],
        mask=flags[:, :, band]
    )

    zsq = zsq[:, antpairs][..., pols]
    # Get time indices
    if nights is not None:
        if not hasattr(nights, '__len__'):
            nights = [nights]

        zsq = zsq[np.isin(zscores.nights, nights)]

    return zsq


def get_compressed_zscores(zscores: list[LSTStack], flags: list[np.ndarray] | None = None, **kwargs) -> np.ndarray:
    """
    Get a subset of data from a list of UVData objects.

    Returns a single flat array with all the datapoints that satisfy the subset criteria.

    Parameters
    ----------
    zscores : list[LSTStack]
        The LSTStack objects to downselect. This should be an LSTStack wrapping
        a UVFlag object in metrics mode (i.e. the output of get_squared_zscores).

    Other Parameters
    ----------------
    All other parameters passed on to :func:`downselect_zscores`.

    Returns
    -------
    zscores : np.ndarray
        The downselected zscores, with flagged values removed, compressed into a 1D
        array.
    """

    allz = []
    if flags is None:
        flags = [None] * len(zscores)

    for zsq, flg in zip(zscores, flags):
        subset = downselect_zscores(zsq, flags=flg, **kwargs)
        allz.append(subset.compressed())

    return np.concatenate(allz)


def reduce_stack_over_axis(
    func: callable,
    data: np.ndarray,
    axis: Literal['bls', 'nights', 'freqs', 'pols', 'antpairs'] | list[str],
) -> np.ndarray:
    """Reduce an LSTStack data-like array over a given axis, and return the result.

    Parameters
    ----------
    func : callable
        The function to use to reduce the data. This should take an array and return
        a single value along a given axis. E.g. `np.mean`, `np.median`, `np.std`.
    data : np.ndarray
        The data to reduce. This should be an array of shape (Nnights, Nantpairs, Nfreqs, Npols).
    axis : {'bls', 'nights', 'freqs', 'pols', 'antpairs'}
        The axis to reduce over. Here, "bls" will reduce over _both_ antpairs and pols.
        If a list, reduce over all axes in the list.

    Returns
    -------
    reduced : np.ndarray
        The reduced data, with original shape, sans the axis that was reduced over.
    """
    axes = set()
    if isinstance(axis, str):
        axis = [axis]

    for ax in axis:
        if ax == 'antpairs':
            axes.add(1)
        elif ax == 'bls':
            axes.add(1)
            axes.add(3)
        elif ax == 'nights':
            axes.add(0)
        elif ax == 'freqs':
            axes.add(2)
        elif ax == 'pols':
            axes.add(3)
        else:
            raise ValueError(
                f"got {ax} for axis. Must be one of 'bls', 'nights', 'freqs', 'pols'"
            )

    axes = tuple(sorted(axes))

    return func(data, axis=axes)
