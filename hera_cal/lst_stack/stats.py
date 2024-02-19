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


@attrs.define(slots=False)
class LSTBinStats:
    days_binned: DataContainer = attrs.field()
    n2n_var_obs: DataContainer = attrs.field()
    lstavg_var_obs: DataContainer = attrs.field()
    lstavg_var_pred: DataContainer = attrs.field()
    per_night_var_pred: DataContainer = attrs.field()

    @classmethod
    def from_data(
        cls,
        *,
        lstbin_data: DataContainer,
        lstbin_nsamples: DataContainer,
        lstbin_flags: DataContainer,
        std_data: DataContainer,
        dt: float,
        df: float,
        data: DataContainer = None,
        nsamples: DataContainer = None,
        flags: DataContainer = None,
    ):
        """Get the observed and predicted variance metrics from observations in a particular LST bin."""
        days_binned = {}
        all_obs_var = {}
        all_predicted_var = {}
        all_interleaved_var = {}
        all_predicted_binned_var = {}
        excess_binned_var = {}
        excess_interleaved_var = {}
        per_night_var_pred = {}

        # Make sure we output correct types
        dcls = lstbin_data.__class__  # Either DataContainer or RedDataContainer
        REDAVG = dcls == RedDataContainer

        if REDAVG:
            dcls = partial(dcls, reds=data.reds)

        if REDAVG and (data is None or nsamples is None or flags is None):
            raise ValueError(
                "If data is redundantly-averaged, you must provide data, nsamples and flags"
            )

        for bl in lstbin_data.bls():
            lstd = lstbin_data[bl][0]
            lstn = lstbin_nsamples[bl][0]
            lstf = lstbin_flags[bl][0]
            stdd = std_data[bl][0]

            if np.all(lstf):
                continue

            splbl = utils.split_bl(bl)
            if splbl[0] == splbl[1]:  # don't use autos
                continue

            # Observed variances.
            all_obs_var[bl] = np.abs(np.where(lstf, np.nan, stdd**2))
            all_interleaved_var[bl] = noise.interleaved_noise_variance_estimate(
                np.atleast_2d(np.where(lstf, np.nan, lstd)), kernel=[[1, -2, 1]]
            )[0]
            # Set first and last frequency to NaN
            all_interleaved_var[bl][[0, -1]] = np.nan

            if REDAVG:
                # In the redundantly-averaged case we need to know the
                # nsamples (and autos) on each night, because they all have
                # different nsamples.

                # Ensure flagged data has zero samples
                gd = data[bl]
                gn = nsamples[bl].copy()
                gf = flags[bl]

                gn[gf] = 0

                # This might be slighly wrong because it gets a different variance
                # each night not just from the Nsamples but also the autos. In the
                # sample variance calculation that goes in to the STD files, we
                # use only the nsamples.
                per_day_expected_var = noise.predict_noise_variance_from_autos(
                    bl, data, dt=dt, df=df, nsamples=nsamples
                )
                per_day_expected_var[gf] = np.inf
                per_night_var_pred[bl] = per_day_expected_var

                wgts_arr = np.where(gf, 0, per_day_expected_var**-1)

                # compute ancillary statistics, see math above
                days_binned[bl] = np.sum(gn > 0, axis=0)

                all_predicted_binned_var[bl] = np.sum(wgts_arr, axis=0) ** -1
                all_predicted_var[bl] = (
                    days_binned[bl] - 1
                ) * all_predicted_binned_var[bl]
            else:
                # Although the above code WOULD work for non-redundantly-averaged
                # data, it is highly inefficient, because we don't need to know
                # the nsamples every night (since we know they're all uniform).
                expected_var = noise.predict_noise_variance_from_autos(
                    bl,
                    lstbin_data,
                    dt=dt,
                    df=df,
                )[0]
                expected_var[lstf] = np.inf
                days_binned[bl] = lstn
                all_predicted_binned_var[bl] = expected_var / lstn
                all_predicted_var[bl] = all_predicted_binned_var[bl] * (lstn - 1)
                per_night_var_pred[bl] = expected_var[None, :]

            excess_binned_var[bl] = all_obs_var[bl] / all_predicted_var[bl]
            excess_interleaved_var[bl] = (
                all_interleaved_var[bl] / all_predicted_binned_var[bl]
            )

        return cls(
            days_binned=dcls(days_binned),
            n2n_var_obs=dcls(all_obs_var),
            lstavg_var_obs=dcls(all_interleaved_var),
            lstavg_var_pred=dcls(all_predicted_binned_var),
            per_night_var_pred=dcls(per_night_var_pred),
        )

    @cached_property
    def _cls(self):
        if isinstance(self.days_binned, RedDataContainer):
            return partial(RedDataContainer, reds=self.days_binned.reds)
        else:
            return DataContainer

    @cached_property
    def n2n_var_pred(self) -> DataContainer:
        return self._cls(
            {
                bl: self.lstavg_var_pred[bl] * (self.days_binned[bl] - 1)
                for bl in self.bls()
            }
        )

    @cached_property
    def n2n_excess_var(self) -> DataContainer:
        return self._cls(
            {bl: self.n2n_var_obs[bl] / self.n2n_var_pred[bl] for bl in self.bls()}
        )

    @cached_property
    def lstavg_excess_var(self) -> DataContainer:
        return self._cls(
            {
                bl: self.lstavg_var_obs[bl] / self.lstavg_var_pred[bl]
                for bl in self.bls()
            }
        )

    @classmethod
    def n2n_excess_var_distribution(cls, ndays_binned: int):
        return gamma(a=(ndays_binned - 1) / 2, scale=2 / (ndays_binned - 1))

    def n2n_excess_var_pred_dist(
        self, bls, freq_inds=slice(None), min_n: int = 1
    ) -> rv_continuous:
        """Get a scipy distribution representing the theoretical distribution of excess variance.

        This will return a MixtureModel -- i.e. it will be the expected distribution of all frequencies
        and baselines asked for (not their average).

        """
        if not hasattr(bls[0], "__len__"):
            bls = [bls]

        all_ns = np.concatenate(tuple(self.days_binned[bl][freq_inds] for bl in bls))
        unique_days_binned, counts = np.unique(all_ns, return_counts=True)
        indx = np.argwhere(unique_days_binned >= min_n)[:, 0]
        unique_days_binned = unique_days_binned[indx]
        counts = counts[indx]

        return MixtureModel(
            [self.n2n_excess_var_distribution(nn) for nn in unique_days_binned],
            weights=counts,
        )

    def n2n_excess_var_avg_pred_dist(self, bls, freq_inds=slice(None), min_n: int = 1):
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
            tuple(self.days_binned[bl][freq_inds] for bl in bls)
        )
        ndays_binned = ndays_binned[ndays_binned >= min_n]

        M = len(ndays_binned)
        ksum = np.sum(M**2 / 2 / np.sum(1 / (ndays_binned - 1)))
        thetasum = 1 / ksum

        return gamma(a=ksum, scale=thetasum)

    def bls(self):
        return self.days_binned.bls()

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
