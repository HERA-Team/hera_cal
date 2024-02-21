"""Statistics for LST-binned data."""
from __future__ import annotations

from scipy.stats import rv_continuous
from scipy.special import gamma
from ..datacontainer import DataContainer, RedDataContainer
import attrs
import numpy as np
from functools import partial, cached_property
from .. import noise
from .. import utils
from astropy import units
from pyuvdata import UVData, UVFlag
from .config import LSTConfig
from .binning import lst_bin_files_from_config
from .. import io


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


@attrs.define(slots=False, kw_only=True, frozen=True)
class LSTStack:
    """Class containing methods to calculate statistics of the LST-binned data."""
    uvd: UVData = attrs.field()

    @uvd.validator
    def _validate_data(self, attribute, value):
        if not isinstance(value, UVData):
            raise ValueError("uvd must be a UVData object")

        if not value.blts_are_rectangular:
            raise ValueError("blts must be rectangular in UVData object")

        if not value.time_axis_faster_than_bls:
            raise ValueError("time axis must be faster than bls in UVData object")

        if value.integration_time is None:
            raise ValueError("integration_time must be defined in UVData object")

    def __getattr__(self, name):
        return getattr(self.uvd, name)

    @cached_property
    def df(self):
        """The frequency resolution of the data."""
        return np.median(np.diff(self.freq_array)) * units.Hz

    @cached_property
    def dt(self):
        """The time resolution of the data."""
        return (np.median(self.integration_time) * units.day).to(units.s)


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
            outdict[k] = kls(
                {antpair + (pol,): v[i] for i, antpair in enumerate(antpairs) for j, pol in enumerate(pols)}
            )
        return cls(**outdict)

    @property
    def bls(self):
        return self.mean.bls()


@attrs.define(slots=False)
class LSTBinMetricsCalculator:
    """Class with methods to compute metrics that reduce over nights."""
    auto_stats: LSTBinStats = attrs.field()
    stats: LSTBinStats = attrs.field()

    @cached_property
    def _cls(self):
        if isinstance(self.stats.mean, RedDataContainer):
            return partial(RedDataContainer, reds=self.stats.mean.reds)
        else:
            return DataContainer


@attrs.define(slots=False)
class LSTBinMetricsReduceNights(LSTBinMetricsCalculator):
    """Class with methods to compute metrics that reduce over nights."""
    def night_to_night_variance_predicted(self, stacks: list[LSTStack]) -> DataContainer:
        out = {}
        for bl in self.stats.mean.bls():
            auto = self.auto_stats.mean[(bl[0], bl[0], bl[2])] * self.auto_stats.mean[(bl[1], bl[1], bl[2])]
            out[bl] = np.zeros_like(self.stats.mean[bl])

            for i, stack in enumerate(stacks):
                gf = stack.get_flags(bl)
                per_day_expected_var = np.abs(auto[i] / stack.dt / stack.df / stack.get_nsamples(bl))
                per_day_expected_var[gf] = np.inf
                wgts_arr = np.where(gf, 0, per_day_expected_var**-1)

                out[i] = np.sum(wgts_arr, axis=0) ** -1 * (self.stats.days_binned[bl] - 1)
        return self._cls(out)

    def night_to_night_excess_variance(self, stacks: list[LSTStack]) -> DataContainer:
        varpred = self.night_to_night_variance_predicted(stacks)
        return self._cls({bl: self.stats.std[bl]**2 / varpred[bl] for bl in self.stats.bls()})

    def freq_to_freq_variance(self) -> DataContainer:
        return self._cls({
            bl: noise.interleaved_noise_variance_estimate(
                np.atleast_2d(np.where(self.stats.flags[bl], np.nan, self.stats.mean[bl])),
                kernel=[[1, -2, 1]]
            )[0] for bl in self.stats.bls
        })


@attrs.define(slots=False)
class LSTBinMetricsReduceFreq(LSTBinMetricsCalculator):
    """Class with methods to compute metrics that reduce over frequencies."""
    def reduced_square_zscore(
        self,
        stacks: list[LSTStack],
        channels: np.ndarray | None,
        use_mad: bool = False,
        reducers: list[str] = ['mean', 'max']
    ) -> dict[str, DataContainer]:
        if channels is None:
            channels = slice(None)

        rdcfuncs = {
            'mean': np.nanmean,
            'max': np.nanmax,
        }

        out = {rdc: {} for rdc in reducers}
        for bl in self.stats.bls:
            for rdc in reducers:
                out[rdc][bl] = np.zeros((len(stacks), stacks[0].Ntimes))

            for i, stack in enumerate(stacks):
                if use_mad:
                    zscores = (stack.get_data(bl)[:, channels] - self.stats.median[bl][i, channels]) / self.stats.mad[bl][i, channels]
                else:
                    zscores = (stack.get_data(bl)[:, channels] - self.stats.mean[bl][i, channels]) / self.stats.std[bl][i, channels]

                for rdc in reducers:
                    out[rdc][bl][i] = rdcfuncs[rdc](zscores**2, axis=1)

        return {rdc: self._cls(out[rdc]) for rdc in reducers}


@attrs.define(slots=False)
class LSTBinMetricsReduceBaseline(LSTBinMetricsCalculator):
    """Class with methods to compute metrics that reduce over frequencies."""
    def reduced_square_zscore(
        self,
        stacks: list[LSTStack],
        use_mad: bool = False,
        reducers: list[str] = ['mean', 'max']
    ) -> dict[str, np.ndarray]:

        rdcfuncs = {
            'mean': np.nanmean,
            'max': np.nanmax,
        }

        zscores = np.zeros((len(self.stats.bls), len(stacks), stacks[0].Ntimes))

        for j, bl in enumerate(self.stats.bls):
            for i, stack in enumerate(stacks):
                if use_mad:
                    zscores[j, i] = (stack.get_data(bl) - self.stats.median[bl]) / self.stats.mad[bl]
                else:
                    zscores[j, i] = (stack.get_data(bl) - self.stats.mean[bl]) / self.stats.std[bl]

        out = {}
        for rdc in reducers:
            out[rdc] = rdcfuncs[rdc](zscores**2, axis=0)

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
        [night_tonight_excess_var_distribution(nn) for nn in unique_days_binned],
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
