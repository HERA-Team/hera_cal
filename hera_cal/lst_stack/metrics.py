"""Statistics for LST-binned data."""
from __future__ import annotations

from scipy.stats import rv_continuous, gamma, chi2
from ..datacontainer import DataContainer, RedDataContainer
import attrs
import numpy as np
from functools import partial
from .. import utils
from pyuvdata import UVData
from ..red_groups import RedundantGroups


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
        rvs = np.choose(submodel_choices, submodel_samples)
        return rvs


def zsquare_predicted_dist(df: int = 2):
    if df not in (1, 2):
        raise ValueError(
            "df should be either 1 (if using Z^2 of real/imag separately), "
            "or 2 (if using |Z^2|)."
        )
    return chi2(df=df)


@attrs.define(slots=False)
class LSTBinStats:
    """Class that holds basic LST-binned data and statistics"""
    mean: DataContainer = attrs.field()
    std: DataContainer = attrs.field()
    nsamples: DataContainer = attrs.field()
    flags: DataContainer = attrs.field()
    median: DataContainer = attrs.field()
    mad: DataContainer = attrs.field()
    days_binned: DataContainer = attrs.field()

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
            dct = {antpair + (pol,): v[:, i, :, j] for i, antpair in enumerate(antpairs) for j, pol in enumerate(pols)}
            outdict[k] = kls(dct)

        return cls(**outdict)

    @property
    def bls(self):
        return self.mean.bls()


def get_nightly_predicted_variance(
    bl: tuple[int, int, str],
    stacks: list[UVData],
    auto_stats: LSTBinStats,
    dt: float,
    df: float,
) -> list[np.ndarray]:
    """
    Compute the predicted variance from the autos over nights/freqs for a particular bl.

    Output is a list with the length of the number of lst bins, and each element is
    an array of shape (Nnights, Nfreqs).
    """

    auto = auto_stats.mean[(bl[0], bl[0], bl[2])] * auto_stats.mean[(bl[1], bl[1], bl[2])]

    per_day_expected_var = [None] * len(stacks)
    dtdf = dt * df

    for i, stack in enumerate(stacks):
        gf = stack.get_flags(bl)
        per_day_expected_var[i] = np.abs(auto[i] / dtdf / stack.get_nsamples(bl))
        per_day_expected_var[i][gf] = np.inf

    return per_day_expected_var


def get_zscores(
    auto_stats: LSTBinStats,
    cross_stats: LSTBinStats,
    dt: float,
    df: float,
    stacks: list[UVData],
    central: str = 'mean',
    std: str = 'autos'
) -> list[UVData]:
    """
    Obtain Z-scores as a list of UVData objects.
    """
    zstacks = [stack.copy() for stack in stacks]  # UVData objects

    if central not in ("mean", "median"):
        raise ValueError("central must be 'mean' or 'median'")

    for bl in cross_stats.bls:
        if std == 'autos':
            variance = get_nightly_predicted_variance(bl, stacks, auto_stats, dt, df)
        elif std in {'std', 'mad'}:
            variance = np.abs(np.where(cross_stats.flags[bl], np.inf, getattr(cross_stats, std)[bl]))**2
        else:
            raise ValueError("std must be 'autos', 'std' or 'mad'")

        for i, (zstack, v) in enumerate(zip(zstacks, variance)):
            data = zstack.get_data(bl)
            flg = zstack.get_flags(bl)

            centre = getattr(cross_stats, central)[bl][i]

            z = np.abs(data - centre)**2 / (v / 2)  # TODO: check that we need /2 for std, mad

            z[flg] = np.nan

            zstack.set_data(z[:, :, None], bl)

    return zstacks


def get_selected_bls(
    bls: list[tuple[int, int, str]],
    days_binned: DataContainer | None = None,
    selectors=None,
    min_days: int = 7
):
    if selectors is None:
        selectors = []
    elif callable(selectors):
        selectors = [selectors]

    if min_days:
        selectors.append(lambda bl: bl in days_binned and (np.median(days_binned[bl]) >= min_days))

    def select(bl):
        return all(sel(bl) for sel in selectors)

    return [bl for bl in bls if select(bl)]


def _get_data_subset(
    zscores: list[UVData],
    bls: list[tuple[int, int, str]] | None = None,
    band: tuple[int, int] | None = None,
    nights='all',
    pols='all',
    selector=None
):

    allz = []

    nbls = zscores[0].Nbls
    datapols = utils.polnum2str(zscores[0].polarization_array, x_orientation=zscores[0].x_orientation).tolist()
    datapairs = [(a, b) for a, b in zip(zscores[0].ant_1_array[:nbls], zscores[0].ant_2_array[:nbls])]

    if selector is not None:
        allbls = [(a, b, p) for a, b in datapairs for p in datapols]
        bls = get_selected_bls(allbls, min_days=0, selectors=selector)
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

    for lstbin, zscore in enumerate(zscores):
        newshape = (zscore.Ntimes, zscore.Nbls, -1, zscore.Npols)
        data = zscore.data_array[:, band].real.reshape(newshape)
        flg = zscore.flag_array[:, band].reshape(newshape)

        # make sure flagged stuff is nan
        data[flg] *= np.nan
        data = data[:, bls][..., pols]
        # Get time indices
        if nights != 'all':
            data_nights = zscore.time_array[::zscore.Nbls].astype(int).tolist()

            if isinstance(nights, int):
                data = data[[data_nights.index(nights)]]
            else:
                data = data[[data_nights.index(n) for n in nights]]

        allz.append(data)
    return allz


def get_data_subset(
    data: list[UVData], bls=None, band=None, nights='all', pols='all', selector=None
):
    """
    Get a subset of data from a list of UVData objects.

    Returns a single flat array with all the datapoints that satisfy the subset criteria.
    """
    subsets = _get_data_subset(zscores=data, bls=bls, band=band, nights=nights, pols=pols, selector=selector)
    allz = []
    for subset in subsets:
        subset = subset.flatten()
        subset = subset[~np.isnan(subset)]
        allz.append(subset)
    return np.concatenate(allz)


def get_data_subset_mean(
    data: list[UVData], bls=None, band=None, nights='all', pols='all', mean_over=None, selector=None
):
    """
    Get a subset of data from a list of UVData objects, and return the mean over the requested axes.

    Returns a list of arrays, where the resulting arrays have shape (Nnights, Nbls, Nfreq, Npol),
    excluding the axes that were averaged over.
    """
    subsets = _get_data_subset(zscores=data, bls=bls, band=band, nights=nights, pols=pols, selector=selector)

    if not mean_over:
        return subsets

    # result is a list of arrays of shape (Nnights, Nbls, Nfreq, Npol), one for each LST bin
    out = []
    axes = ()
    if isinstance(mean_over, str):
        mean_over = [mean_over]
    for mo in mean_over:
        if mo == 'bls':
            axes += (1, 3)
        elif mo == 'nights':
            axes += (0,)
        elif mo == 'band':
            axes += (2,)
        else:
            raise ValueError(f"got {mo} for mean_over")
    axes = tuple(sorted(axes))

    for data in subsets:
        out.append(np.nanmean(data, axis=axes))
    return out


def night_to_night_excess_var_distribution(cls, ndays_binned: int):
    return gamma(a=(ndays_binned - 1) / 2, scale=2 / (ndays_binned - 1))


def n2n_excess_var_pred_dist(
    days_binned: DataContainer, bls, freq_inds=slice(None), min_n: int = 1
) -> rv_continuous:
    """Get a scipy distribution representing the theoretical distribution of excess variance.

    This will return a MixtureModel -- i.e. it will be the expected distribution of all frequencies
    and baselines asked for (not their average).

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


def getmean(
    self, rdc: str | RedDataContainer | DataContainer, bls=None, min_days: int = 7
):
    if isinstance(rdc, str):
        rdc = getattr(self, rdc)
    if bls is None:
        bls = self.bls()

    return np.nanmean(
        [np.where(self.days_binned[bl] >= min_days, rdc[bl], np.nan) for bl in bls],
        axis=0,
    )
