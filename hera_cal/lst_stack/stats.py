r"""Statistical distributions of visibilities at the LST-Stacked/averaged level.

These statistics are predominantly distributions of Z^2, i.e. the squared Z-score
obtained by subtracting the mean (generally over LSTs) of some set of iid visibilities,
and dividing by the standard deviation of the same set of visibilities (generally
assumed to be predicted by the autos). Note that the distributions here assume that
the mean used to estimate the Z-scores comes from the sample itself, i.e.

    .. math:: Z_i = (V_i - \bar{V}) / (\sigma/\sqrt{2}),

where :math:`\bar{V}` is the sample mean (over LSTs or redundant baselines, or both),
of which :math:`V_i` is a part. These distributions are *not* applicable to cases in
which the mean is estimated from a different, uncorrelated, sample.

The distributions here are derived and discussed in the memo
https://github.com/HERA-Team/H6C-analysis/blob/main/docs/statistics_of_visibilities.ipynb.

Note that the "excess variance" defined in the HERA memo
https://reionization.org/manual_uploads/HERA123_LST_Bin_Statistics-v3.pdf is closely
to the *mean* Z^2 over some set of visibilities (if those visibilities are iid, e.g.
coming from redundant baselines or the same LST). The difference is that the excess
variance is normalized by a factor n/(n-1), in order to ensure that its expected value
is 1. This is implemented in the :func:`excess_variance` function.
"""
from __future__ import annotations
from scipy.stats import rv_continuous
import numpy as np
from scipy.stats import chi2, gamma


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
        if any(w < 0 for w in weights):
            raise ValueError(f"MixtureModel weights must be non-negative. Got {weights}.")

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

    def _rvs(self, *args, size=1, random_state: np.random.Generator | None, **kwargs):
        if random_state is None:
            random_state = np.random.default_rng()

        submodel_choices = random_state.choice(
            len(self.submodels), size=size, p=self.weights
        )
        submodel_samples = [submodel.rvs(*args, size=size, **kwargs) for submodel in self.submodels]
        return np.choose(submodel_choices, submodel_samples)


def zsquare(absolute: bool = True) -> rv_continuous:
    r"""Return the distribution of Z^2 of a visibility.

    If absolute is True, this is the distribution of |Z|^2, otherwise it is the distribution
    of the real/imaginary component of Z^2.

    Parameters
    ----------
    absolute : bool
        Whether to return the distribution of |Z|^2, or Re(Z)^2 (equivalently Im(Z)^2).

    Notes
    -----
    Here Z is defined as

    .. math::  Z_i = \sqrt{\frac{2n_i}{\sigma^2}}\frac{M}{M-n_i}(V_i - \bar{V})

    where M is the sum of nsamples over the redundant set from which the mean is
    estimated, and ni is the nsamples for the particular visibility under consideration.

    Returns
    -------
    dist
        A chi^2(df=1) distribution if absolute is False, or a chi^2(df=2) distribution
        if absolute is True.
    """
    return gamma(a=1 / (1 if absolute else 2), scale=2)


def mean_zsquare_over_redundant_set(n: int, absolute: bool = True) -> rv_continuous:
    r"""
    Return the distribution of the mean of Z^2 over a redundant set.

    In Memo XXX this is called \zeta^2. It is defined as

    .. math:: |\zeta|^2 \equiv \frac{1}{N} \sum_i^N \frac{M-n_i}{M}|Z|^2_i,

    where :math:`|Z|^2_i` has the distribution given in :func:`zsquare`.

    Parameters
    ----------
    n : int
        The number of visibilities in the redundant set (whose nsamples is greater
        than zero).
    """
    return gamma(a=(n - 1) / (1 if absolute else 2), scale=2 / n)


def mean_zsquare_over_independent_set(q: int, absolute: bool = True) -> rv_continuous:
    r"""
    Return the distribution of the mean of Z^2 over independent samples.

    This is different than :func:`mean_zsquare_over_redundant_set` in that it is the
    distribution of the mean of *independent* samples (i.e. from different channels
    or baselines), rather than those that are used as part of the mean when defining Z.

    In Memo XXX it is defined as:

    .. math:: \bar{|Z|^2} \equiv \frac{1}{Q} \sum_j^Q |Z|^2_j,

    where Q is the number of independent visibilities (that have non-zero nsamples).

    Parameters
    ----------
    q : int
        The number of independent visibilities in the mean.
    """
    return gamma(a=q / (1 if absolute else 2), scale=2 / q)


def mean_zsquare_over_redundant_and_independent_sets(
    nsets: int, ntot: int, absolute: bool = True
) -> rv_continuous:
    r"""
    Return the distribution of the mean of Z^2 over both redundant and independent data.

    This is the distribution of the mean of Z^2 over a set of visibilities that are
    both redundant and independent. This is defined in Memo XXX as:

    .. math:: \bar{\zeta^2} \equiv \frac{1}{\sum_j N_j} \sum_j^Q N_j \zeta^2_j.

    where Q is the number of independent visibilities in the mean, and N_j is the number
    of visibilities with non-zero nsamples in the jth redundant set (e.g. the number of
    unflagged nights in an LST-stack).

    Parameters
    ----------
    nsets : int
        The number of independent sets of visibilities (e.g. different channels,
        different baseline types).
    ntot : int
        The total number of unflagged visibilities in the sample (e.g. the sum of the
        number of unflagged nights for each channel/baseline).
    """
    return gamma(a=(ntot - nsets) / (1 if absolute else 2), scale=2 / ntot)


def excess_variance(n: int, absolute: bool = True) -> rv_continuous:
    r"""The expected distribution of the excess variance of n iid visibilities.

    See https://reionization.org/manual_uploads/HERA123_LST_Bin_Statistics-v3.pdf.

    The excess variance, as defined in the above memo, is closely related to the mean
    zsquare. The only difference is that the excess variance is normalized by a factor
    n/(n-1), in order to ensure that its expected value is 1.

    Parameters
    ----------
    n : int
        The number of visibilities used to obtain the variance.
    absolute : bool
        Whether to return the distribution of |gamma|^2, or Re(gamma)^2
        (equivalently Im(gamma)^2).

    Notes
    -----
    The excess variance is defined as the ratio of the empirically-determined variance
    across the visibilities to the expected variance of the visibilities (as predicted,
    for example, by the autos).

    That is:

    .. math:: \gamma = n/(n-1) \frac{1}{\sum_{i=1}^n N_i} \sum_{i=1}^n N_i \mathcal{Re}(V_i - \bar{V})^2 / (\sigma^2/2)

    where :math:`\sigma^2` is the expected/predicted variance of the complex visibilities
    (i.e. :math:`\sigma^2/2` is the variance of the real/imaginary components), :math:`N_i`
    is the number of samples in each visibility (i.e. we have allowed for some pre-averaging)
    and :math:`V_i` is each visibility itself. Note that the resulting excess variance,
    :math:`\gamma`, is a random variable, and this function returns its distribution,
    which does *not* depend on Nsamples.
    """
    return gamma(a=(n - 1) / (1 if absolute else 2), scale=(1 if absolute else 2) / (n - 1))


def mean_zsquare_mixture(
    n: np.ndarray | list[int],
    absolute: bool = True,
    min_n: int = 1
) -> MixtureModel:
    """The distribution of the mean Z^2 over the redundant set of n iid visibilities for a collection of such means.

    This is the distribution of a *collection* of mean Z-square values, each of which may
    be the mean of a different number of z^2.

    Parameters
    ----------
    n : np.ndarray | list[int]
        The number of Z^2 values in each mean. There may be repeated values.
    absolute : bool
        Whether to return the distribution of |Z|^2, or Re(Z)^2
        (equivalently Im(Z)^2).
    min_n : int
        The minimum number of visibilities averaged together in any particular mean
        required to include it in the mixture distribution.

    Returns
    -------
    dist : MixtureModel
        A mixture model representing the distribution of the mean Z^2.
    """
    unique_n, counts = np.unique(n, return_counts=True)
    indx = np.argwhere(unique_n >= min_n)[:, 0]
    unique_n = unique_n[indx]
    counts = counts[indx]

    return MixtureModel([mean_zsquare_over_redundant_set(nn, absolute=absolute) for nn in unique_n], weights=counts)


def excess_variance_mixture(
    n: np.ndarray | list[int],
    absolute: bool = True,
    min_n: int = 1
) -> rv_continuous:
    r"""The expected distribution of a collection of excess variance mesaurements.

    See https://reionization.org/manual_uploads/HERA123_LST_Bin_Statistics-v3.pdf.

    Parameters
    ----------
    n : int
        The number of visibilities used to obtain the variance.
    absolute : bool
        Whether to return the distribution of |gamma|^2, or Re(gamma)^2
        (equivalently Im(gamma)^2).
    min_n : int
        The minimum number of visibilities averaged together in any excess variance
        required to include it in the mixture distribution.
    """
    unique_n, counts = np.unique(n, return_counts=True)
    indx = np.argwhere(unique_n >= min_n)[:, 0]
    unique_n = unique_n[indx]
    counts = counts[indx]

    return MixtureModel([excess_variance(nn, absolute=absolute) for nn in unique_n], weights=counts)
