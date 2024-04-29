"""Test of hera_cal.lst_stack.stats module."""
import pytest
from hera_cal.lst_stack import stats
import numpy as np
from scipy.stats import gamma, chi2, norm


def get_samples(
    ndays: int = 10,
    nvars: int = 200000,
    absolute: bool = True,
    mean: bool = False,
    weighted: bool = False,
    allow_zeros: bool = False
):
    rng = np.random.default_rng(1)

    if weighted:
        weights = rng.integers(low=0 if allow_zeros else 1, high=10, size=(ndays, nvars))
        weights[0] = 1  # Ensure at least one day is included
    else:
        weights = np.ones((ndays, nvars))

    x = rng.normal(size=(ndays, nvars), scale=np.where(weights == 0, 1, 1 / np.sqrt(weights)))
    if absolute:
        y = rng.normal(size=(ndays, nvars), scale=np.where(weights == 0, 1, 1 / np.sqrt(weights)))

    zsq = weights * (x - np.average(x, axis=0, weights=weights))**2
    if absolute:
        zsq += weights * (y - np.average(y, axis=0, weights=weights))**2

    if mean:
        zsq = np.mean(zsq, axis=0)

    n_averaged = np.sum(weights > 0, axis=0)

    return zsq, n_averaged


def get_excess_variance(
    ndays: int = 10,
    nvars: int = 200000,
    absolute: bool = True,
    weighted: bool = False,
    allow_zeros: bool = False
):
    rng = np.random.default_rng(1)

    if weighted:
        weights = rng.integers(low=0 if allow_zeros else 1, high=10, size=(ndays, nvars))
    else:
        weights = np.ones((ndays, nvars))

    x = rng.normal(size=(ndays, nvars), scale=np.where(weights == 0, 1, 1 / np.sqrt(weights)))
    if absolute:
        y = rng.normal(size=(ndays, nvars), scale=np.where(weights == 0, 1, 1 / np.sqrt(weights)))

    mean = np.average(x, axis=0, weights=weights)
    var = np.average((x - mean)**2, axis=0, weights=weights)

    if absolute:
        ymean = np.average(y, axis=0, weights=weights)
        yvar = np.average((y - ymean)**2, axis=0, weights=weights)
        var += yvar

    excess_var = var * np.sum(weights, axis=0) / (ndays - 1)  # Bessel's correction, true var is 1

    if absolute:
        excess_var /= 2

    n_averaged = np.sum(weights > 0, axis=0)

    return excess_var, n_averaged


@pytest.mark.parametrize("absolute", [True, False])
@pytest.mark.parametrize("ndays", [2, 3, 10, 50])
@pytest.mark.parametrize("weighted", [True, False])
def test_zsquare(absolute, ndays, weighted):
    zsq, _ = get_samples(absolute=absolute, mean=False, ndays=ndays, weighted=weighted)
    dist = stats.zsquare(absolute=absolute, n=ndays)

    # Test statistics...
    np.testing.assert_allclose(dist.mean(), np.mean(zsq), atol=1e-2)
    np.testing.assert_allclose(dist.var(), np.var(zsq), rtol=0.5 / ndays)


@pytest.mark.parametrize("absolute", [True, False])
@pytest.mark.parametrize("ndays", [2, 3, 10, 50])
@pytest.mark.parametrize("weighted", [True, False])
def test_zsquare_mean(absolute, ndays, weighted):
    zsq, _ = get_samples(absolute=absolute, mean=True, ndays=ndays, weighted=weighted)
    dist = stats.mean_zsquare(absolute=absolute, n=ndays)

    # Test statistics...
    np.testing.assert_allclose(dist.mean(), np.mean(zsq), atol=1e-2)
    np.testing.assert_allclose(dist.var(), np.var(zsq), rtol=0.1)


@pytest.mark.parametrize("absolute", [True, False])
@pytest.mark.parametrize("ndays", [2, 3, 10, 50])
@pytest.mark.parametrize("weighted", [True, False])
def test_excess_var(absolute, ndays, weighted):
    zsq, _ = get_excess_variance(absolute=absolute, ndays=ndays, weighted=weighted)
    dist = stats.excess_variance(absolute=absolute, n=ndays)

    # Test statistics...
    np.testing.assert_allclose(dist.mean(), np.mean(zsq), atol=1e-2)
    np.testing.assert_allclose(dist.var(), np.var(zsq), rtol=0.2)


@pytest.mark.parametrize("absolute", [True, False])
def test_mean_zsquare_mixture(absolute):
    zsq, navg = get_samples(absolute=absolute, mean=True, ndays=10, nvars=10000, weighted=True, allow_zeros=True)
    dist = stats.mean_zsquare_mixture(absolute=absolute, n=navg)

    # Test statistics...
    np.testing.assert_allclose(dist.mean(), np.mean(zsq), rtol=0.15)
    np.testing.assert_allclose(dist.var(), np.var(zsq), rtol=0.21)


@pytest.mark.parametrize("absolute", [True, False])
def test_excess_variance_mixture(absolute):
    zsq, navg = get_excess_variance(absolute=absolute, ndays=10, weighted=True, nvars=10000, allow_zeros=True)
    dist = stats.excess_variance_mixture(absolute=absolute, n=navg)

    # Test statistics...
    np.testing.assert_allclose(dist.mean(), np.mean(zsq), rtol=0.15)
    np.testing.assert_allclose(dist.var(), np.var(zsq), rtol=0.25)


# Happy Path Tests
class TestMixtureModel:
    @pytest.mark.parametrize("n", [1, 2, 5])
    @pytest.mark.parametrize("model", [gamma(a=1.), chi2(df=2), norm()])
    def test_non_uniform_weights_same_model(self, n, model):
        """Test that a mixture model of many of the same distributions is just the distribution itself."""
        rng = np.random.default_rng(0)

        # Arrange
        mixture = stats.MixtureModel([model] * n, weights=rng.uniform(size=n))

        # Act
        x = np.linspace(1, 5, 10)
        np.testing.assert_allclose(mixture.pdf(x), model.pdf(x))
        np.testing.assert_allclose(mixture.sf(x), model.sf(x))
        np.testing.assert_allclose(mixture.cdf(x), model.cdf(x))

    def test_default_weights(self):
        # Arrange
        model1 = gamma(a=1.)
        model2 = chi2(df=2)
        model3 = norm()
        mixture = stats.MixtureModel([model1, model2, model3])
        assert mixture.weights == [1 / 3, 1 / 3, 1 / 3]

    def test_bad_weights_length(self):
        # Arrange
        model1 = gamma(a=1.)
        model2 = chi2(df=2)
        model3 = norm()

        # Act & Assert
        with pytest.raises(ValueError, match="There are 3 submodels and 2 weights"):
            stats.MixtureModel([model1, model2, model3], weights=[0.5, 0.5])

    def test_negative_weights(self):
        with pytest.raises(ValueError, match="weights must be non-negative"):
            stats.MixtureModel([gamma(a=1.0)], weights=[-0.5])

    def test_gaussian_mixture(self):
        # Arrange
        model1 = norm(loc=1, scale=1)
        model2 = norm(loc=5, scale=2)
        mixture = stats.MixtureModel([model1, model2], weights=[1, 9])

        # Act
        x = np.linspace(-5, 10, 100)
        pdf = mixture.pdf(x)
        cdf = mixture.cdf(x)
        sf = mixture.sf(x)

        # Assert
        assert np.allclose(pdf, 0.1 * model1.pdf(x) + 0.9 * model2.pdf(x))
        assert np.allclose(cdf, 0.1 * model1.cdf(x) + 0.9 * model2.cdf(x))
        assert np.allclose(sf, 0.1 * model1.sf(x) + 0.9 * model2.sf(x))

        # Check random samples
        rng = np.random.default_rng(0)
        samples = mixture.rvs(size=10000, random_state=rng)
        assert np.isclose(np.mean(samples), 0.1 * 1 + 0.9 * 5, atol=0.1)
        assert np.isclose(np.var(samples), 0.1 * 1**2 + 0.9 * 2**2 + 0.1 * 0.9 * (5 - 1)**2, atol=0.1)
