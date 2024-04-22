"""Statistics for LST-binned data."""
from __future__ import annotations

from scipy.stats import rv_continuous, gamma, chi2
from ..datacontainer import DataContainer, RedDataContainer
import attrs
import numpy as np
from functools import partial
from .. import utils
from pyuvdata import UVData, UVFlag
from ..red_groups import RedundantGroups
from .. import types as tp
from .binning import LSTStack


class MixtureModel(rv_continuous):
    """A distribution model from mixing multiple models.

    Taken from https://stackoverflow.com/a/72315113/1467820
    """

    def __init__(self, submodels, *args, weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.submodels = submodels
        if weights is None:
            weights = [1 for _ in submodels]
        if len(weights) != len(submodels):
            raise (
                ValueError(
                    f"There are {len(submodels)} submodels and {len(weights)} weights, but they must be equal."
                )
            )
        self.weights = [w / sum(weights) for w in weights]

    def _pdf(self, x):
        pdf = self.submodels[0].pdf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            pdf += submodel.pdf(x) * weight
        return pdf

    def _sf(self, x):
        sf = self.submodels[0].sf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            sf += submodel.sf(x) * weight
        return sf

    def _cdf(self, x):
        cdf = self.submodels[0].cdf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            cdf += submodel.cdf(x) * weight
        return cdf

    def rvs(self, size):
        submodel_choices = np.random.choice(
            len(self.submodels), size=size, p=self.weights
        )
        submodel_samples = [submodel.rvs(size=size) for submodel in self.submodels]
        return np.choose(submodel_choices, submodel_samples)


def zsquare_predicted_dist(df: int = 2):
    if df in {1, 2}:
        return chi2(df=df)
    else:
        raise ValueError(
            "df should be either 1 (if using Z^2 of real/imag separately), "
            "or 2 (if using |Z^2|)."
        )


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
) -> list[np.ndarray]:
    """
    Compute the predicted variance from the autos over nights/freqs for a particular bl.

    Output is a list with the length of the number of lst bins, and each element is
    an array of shape (Nnights, Nfreqs).
    """

    auto = auto_stats.mean[(bl[0], bl[0], bl[2])] * auto_stats.mean[(bl[1], bl[1], bl[2])]

    dtdf = stack.dt * stack.df

    gf = stack.get_flags(bl)
    per_day_expected_var = np.abs(auto / dtdf / stack.get_nsamples(bl))
    per_day_expected_var[gf] = np.inf

    return per_day_expected_var


def get_squared_zscores(
    auto_stats: LSTBinStats,
    cross_stats: LSTBinStats,
    stack: LSTStack,
    central: str = 'mean',
    std: str = 'autos'
) -> list[UVFlag]:
    """
    Obtain squared Z-scores as a list of UVFlag objects in metrics mode.
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
    """
    zstack = UVFlag(stack, mode='metric', label='zsquare')  # [UVFlag(stack, mode='metric', label='zsquare') for stack in stacks]

    if central not in ("mean", "median"):
        raise ValueError("central must be 'mean' or 'median'")

    for bl in cross_stats.bls:
        if std == 'autos':
            variance = get_nightly_predicted_variance(
                bl, stack, auto_stats, stack.dt, stack.df
            )
        elif std in {'std', 'mad'}:
            variance = np.abs(np.where(cross_stats.flags[bl], np.inf, getattr(cross_stats, std)[bl]))**2
        else:
            raise ValueError("std must be 'autos', 'std' or 'mad'")

        data = zstack.get_data(bl)
        flg = zstack.get_flags(bl)

        centre = getattr(cross_stats, central)[bl]

        z = np.abs(data - centre)**2 / (variance / 2)  # TODO: check that we need /2 for std, mad

        z[flg] = np.nan

        zstack.set_data(z[:, :, None], bl)

    return zstack


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
    stack: LSTStack,
    bls: list[tp.Baseline] | None = None,
    band: tuple[int, int] | slice | None = None,
    nights='all',
    pols='all',
    bl_selectors=None
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
    stack : LSTStack
        The LSTStack object that the zscores were computed from. This is used to
        get the flags.
    bls : list[tuple[int, int, str]] | None
        List of baselines to get the data from. If None, all baselines are used.
    band : tuple[int, int] | slice | None
        The frequency band to use. If None, all frequencies are used.
    nights : int | list[int] | 'all'
        The nights to use. If 'all', all nights are used.
    pols : str | list[str]
        The polarizations to use. If 'all', all polarizations are used.
    bl_selectors : list[Callable] | None
        A list of callables that take a baseline (int, int str) and return a boolean,
        for whether to include that baseline. This can be used instead of passing
        bls directly.

    Returns
    -------
    zscores : np.ma.MaskedArray
        The downselected zscores, as an array with the same form as zscores.metric_array,
        but with some axes smaller after subselection. The mask is from the flags
        in stack.
    """

    allz = []

    nbls = zscores.Nbls
    datapols = utils.polnum2str(
        zscores.polarization_array, x_orientation=zscores.x_orientation
    )
    datapairs = list(
        zip(zscores.ant_1_array[:nbls], zscores.ant_2_array[:nbls])
    )

    if selectors is not None:
        allbls = [(a, b, p) for a, b in datapairs for p in datapols]
        bls = get_selected_bls(allbls, min_days=0, selectors=selectors)
        selpols = {p for a, b, p in bls}
        bls = list({bl[:2] for bl in bls})  # only antpairs

        if 'ee' in selpols and 'nn' in selpols:
            pols = 'all'
        elif 'ee' in selpols:
            pols = 'ee'
        elif 'nn' in selpols:
            pols = 'nn'

    # Get pol indices
    if pols == 'all':
        pols = slice(None)
    elif isinstance(pols, str):
        pols = [datapols.index(pols)]
    else:
        pols = [datapols.index(p) for p in pols]

    # Get bl indices
    if bls is None:
        bls = datapairs
    if isinstance(bls, tuple) and len(bls) == 2:
        bls = [datapairs.index(bls)]
    elif isinstance(bls, list):
        bls = [datapairs.index(bl) for bl in bls]

    if band is None:
        band = slice(None)
    elif isinstance(band, tuple):
        band = slice(band[0], band[1])

    if not isinstance(band, slice):
        raise TypeError("band must be a tuple of (low, high) or a slice")

    zsq = np.ma.MaskedArray(zscores.metrics[:, :, band], mask=stack.flags[:, :, band])

    # make sure flagged stuff is nan
    zsq = zsq[:, bls][..., pols]
    # Get time indices
    if nights != 'all':
        if isinstance(nights, int):
            nights = [nights]

        zsq = zsq[[zscores.nights.tolist().index(n) for n in nights]]

    return zsq


def get_compressed_zscores(
    zscores: list[LSTStack],
    stacks: list[LSTStack],
    bls: list[tp.Baseline] | None = None,
    band: tupe[int, int] | slice | None = None,
    nights: Literal['all'] | list[int] = 'all',
    pols: Literal['all'] | list[str] = 'all',
    bl_selectors: list[callable] | None = None
) -> np.ndarray:
    """
    Get a subset of data from a list of UVData objects.

    Returns a single flat array with all the datapoints that satisfy the subset criteria.

    Parameters
    ----------
    zscores : list[LSTStack]
        The LSTStack objects to downselect. This should be an LSTStack wrapping
        a UVFlag object in metrics mode (i.e. the output of get_squared_zscores).
    stack : list[LSTStack]
        The LSTStack objects that the zscores were computed from. This is used to
        get the flags.
    bls : list[tuple[int, int, str]] | None
        List of baselines to get the data from. If None, all baselines are used.
    band : tuple[int, int] | slice | None
        The frequency band to use. If None, all frequencies are used.
    nights : int | list[int] | 'all'
        The nights to use. If 'all', all nights are used.
    pols : str | list[str]
        The polarizations to use. If 'all', all polarizations are used.
    bl_selectors : list[Callable] | None
        A list of callables that take a baseline (int, int str) and return a boolean,
        for whether to include that baseline. This can be used instead of passing
        bls directly.

    Returns
    -------
    zscores : np.ndarray
        The downselected zscores, with flagged values removed, compressed into a 1D
        array.
    """

    allz = []
    for stack, zsq in zip(stacks, zscores):
        subset = downselect_zscores(zsq, stack, bls=bls, band=band, nights=nights, pols=pols, bl_selectors=bl_selectors)
        allz.append(subset.comressed())

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
    if isinstance(mean_over, str):
        mean_over = [mean_over]
    for mo in mean_over:
        if mo == 'bls':
            axes.add(1)
            axes.add(3)
        elif mo == 'nights':
            axes.add(0)
        elif mo == 'band':
            axes.add(2)
        elif mo == 'pols':
            axes.add(3)
        else:
            raise ValueError(f"got {mo} for mean_over")

    axes = tuple(sorted(axes))

    return func(data, axis=axes)


def night_to_night_excess_var_distribution(ndays_binned: int) -> rv_continuous:
    """Get a distribution for the excess variance of a single lst-averaged observation.

    See https://reionization.org/manual_uploads/HERA123_LST_Bin_Statistics-v3.pdf.

    Parameters
    ----------
    ndays_binned : int
        The number of nights binned in the LST bin.

    Returns
    -------
    dist : rv_continuous
        A scipy distribution object representing the excess variance distribution.
    """
    return gamma(a=(ndays_binned - 1) / 2, scale=2 / (ndays_binned - 1))


def n2n_excess_var_pred_dist(
    days_binned: DataContainer, bls, freq_inds=slice(None), min_n: int = 1
) -> rv_continuous:
    """Get a scipy distribution representing the theoretical distribution of excess variance.

    This will return a MixtureModel -- i.e. it will be the expected distribution of all frequencies
    and baselines asked for (not their average).

    See https://reionization.org/manual_uploads/HERA123_LST_Bin_Statistics-v3.pdf.

    Parameters
    ----------
    days_binned : DataContainer
        The number of days binned for each baseline and frequency.
    bls : list[tuple[int, int, str]]
        The baselines to include in the distribution.
    freq_inds : slice
        The frequency indices to include in the distribution.
    min_n : int
        The minimum number of days binned to include in the distribution.

    Returns
    -------
    dist : MixtureModel
        A mixture model representing the distribution of excess variance.
    """
    if not hasattr(bls[0], "__len__"):
        bls = [bls]

    all_ns = np.concatenate(tuple(days_binned[bl][freq_inds] for bl in bls))
    unique_days_binned, counts = np.unique(all_ns, return_counts=True)
    indx = np.argwhere(unique_days_binned >= min_n)[:, 0]
    unique_days_binned = unique_days_binned[indx]
    counts = counts[indx]

    return MixtureModel(
        [night_to_night_excess_var_distribution(nn) for nn in unique_days_binned],
        weights=counts,
    )


def night_to_night_excess_var_avg_pred_dist(
    days_binned: DataContainer,
    bls, freq_inds=slice(None), min_n: int = 1
):
    """Get a scipy distribution representing the theoretical distribution of averaged excess variance.

    This will return the expected distribution of the averaged excess variance for the
    requested baselines and frequencies. Note this is NOT the excess averaged variance (i.e.
    we're averaging the mean-one excess over the baselines/frequencies, rather than averaging
    the observed variance and dividing by the averaged expected variance).

    This is exact for non-redundantly averaged data, and an approximation for red-avg data.
    Gotten from https://stats.stackexchange.com/a/191912/81338

    Parameters
    ----------
    days_binned : DataContainer
        The number of days binned for each baseline and frequency.
    bls : list[tuple[int, int, str]]
        The baselines to include in the distribution.
    freq_inds : slice
        The frequency indices to include in the distribution.
    min_n : int
        The minimum number of days binned to include in the distribution.

    Returns
    -------
    dist : rv_continuous
        A gamma distribution representing the distribution of averaged excess variance.

    See Also
    --------
    n2n_excess_var_pred_dist
        The distribution of a collection of excess variances (rather than the
        distribution of their average).
    night_to_night_excess_var_distribution
        The distribution of a single excess variance.
    """
    if not hasattr(bls[0], "__len__"):
        bls = [bls]

    ndays_binned = np.concatenate(
        tuple(days_binned[bl][freq_inds] for bl in bls)
    )
    ndays_binned = ndays_binned[ndays_binned >= min_n]

    M = len(ndays_binned)
    ksum = np.sum(M**2 / 2 / np.sum(1 / (ndays_binned - 1)))
    thetasum = 1 / ksum

    return gamma(a=ksum, scale=thetasum)
