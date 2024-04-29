r"""Statistical distributions of visibilities at the LST-Stacked/averaged level.

These statistics are predominantly distributions of Z^2, i.e. the squared Z-score
obtained by subtracting the mean (generally over LSTs) of some set of iid visibilities,
and dividing by the standard deviation of the same set of visibilities (generally
assumed to be predicted by the autos). Note that the distributions here assume that
the mean used to estimate the Z-scores comes from the sample itself, i.e.

    .. math:: Z_i = (V_i - \bar{V}) / \sigma,

where :math:`\bar{V}` is the sample mean (over LSTs or redundant baselines, or both),
of which :math:`V_i` is a part. These distributions are *not* applicable to cases in
which the mean is estimated from a different, uncorrelated, sample. Setting n=0 in the
:func:`zsquare` function will give the correct result for the case in which the mean
is the population mean (i.e. the limit as n -> infinity).

Note that the "excess variance" defined in the HERA memo
https://reionization.org/manual_uploads/HERA123_LST_Bin_Statistics-v3.pdf is closely
to the *mean* Z^2 over some set of visibilities (if those visibilities are iid, e.g.
coming from redundant baselines or the same LST). The difference is that the excess
variance is normalized by a factor n/(n-1), in order to ensure that its expected value
is 1. This is implemented in the :func:`excess_variance` function.
"""
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


def zsquare(n: int = 0, absolute: bool = True) -> rv_continuous:
    """The distribution of Z^2 of a single component of a visibility (real/imag).

    If absolute is True, this is the distribution of |Z|^2, otherwise it is the distribution
    of the real/imaginary component of Z^2.

    Parameters
    ----------
    n : int
        The number of visibilities that were used to obtain the mean, from which the
        zscore was calculated. If n=0, assume that the mean is the true population mean
        (i.e. the limit as n -> infinity).
    absolute : bool
        Whether to return the distribution of |Z|^2, or Re(Z)^2 (equivalently Im(Z)^2).

    Returns
    -------
    dist
        A chi^2(df=1) distribution if absolute is False, or a chi^2(df=2) distribution
        if absolute is True.
    """
    factor = 1 if n == 0 else (n - 1) / n
    return gamma(a=1 / (1 if absolute else 2), scale=2 * factor)


def mean_zsquare(n: int, absolute: bool = True) -> rv_continuous:
    """The distribution of the mean Z^2 of n iid visibilities..

    Note that this result is independent of whether the mean is weighted.

    Parameters
    ----------
    n : int
        The number of z^2 values in the mean.
    absolute : bool
        Whether to return the distribution of <|Z|^2>, or <Re(Z)^2>
        (equivalently <Im(Z)^2>).

    Returns
    -------
    dist
        A Gamma distribution, with a=(n-1)/2 and scale=2/(n-1) if absolute is False,
        or scale=1/(n-1) if absolute is True.
    """
    return gamma(a=(n - 1) / (1 if absolute else 2), scale=2 / n)


def mean_zsquare_mixture(
    n: np.ndarray | list[int],
    absolute: bool = True,
    min_n: int = 1
) -> MixtureModel:
    """The distribution of the mean Z^2 of n iid visibilities, per-component (real/imag).

    This is the distribution of a *collection* of mean Z-square values, each of which may
    be the mean of a different number of z^2.
    This is a mixture model of gamma distributions with rate=(n-1)/2 and scale=2/(n-1).

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

    return MixtureModel([mean_zsquare(nn, absolute=absolute) for nn in unique_n], weights=counts)


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
