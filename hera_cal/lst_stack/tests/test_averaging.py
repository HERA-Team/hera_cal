import numpy as np
import pytest
from .. import averaging as avg
from ...tests import mock_uvdata as mockuvd
from hera_cal.lst_stack import LSTStack
from hera_filters.dspec import dpss_operator
from functools import partial
from astropy import units as un


class TestGetMaskedData:
    def setup_class(self):
        self.shape = (2, 3, 4, 5)

        self.data = np.ones(self.shape, dtype=complex)
        self.data_with_nans = self.data.copy()
        self.data_with_nans[0, 0, 0, 0] = np.nan

        self.nsamples = np.ones(self.shape, dtype=float)
        self.nsamples_with_negatives = self.nsamples.copy()
        self.nsamples_with_negatives[0, 0, 0, 0] = -1

        self.noflags = np.zeros(self.shape, dtype=bool)
        self.flags = np.zeros_like(self.noflags)
        self.flags[0, 0, 0, 0] = True
        self.ten_flags = np.zeros_like(self.noflags)
        self.ten_flags[0, 0, [1, 2], :] = True

    def test_equality_for_different_modes_when_no_flags(self):
        d, f, n = avg.get_masked_data(
            self.data, self.noflags, self.nsamples, inpainted_mode=False
        )
        di, fi, ni = avg.get_masked_data(
            self.data, self.noflags, self.nsamples, inpainted_mode=True
        )

        assert np.all(d == di)
        assert np.all(f == fi)
        assert np.all(n == ni)

    def test_inpainted_data_gets_counted_in_inpainted_mode(self):
        d, f, n = avg.get_masked_data(
            self.data, self.noflags, self.nsamples_with_negatives, inpainted_mode=True
        )

        assert np.all(n.mask == d.mask)  # always true
        # the mask of data/nsamples is what tells the averager what to include
        # in the mean, so nothing should be flagged here (since there are no
        # non-inpainted flags).
        assert not np.any(n.mask)

        # however, the output flags array tells us what to include in the final
        # tallied nsamples, so it should have some flags (the 'inpainted' ones)
        assert np.sum(f) == 1

    def test_inpainted_data_not_counted_in_direct_mode(self):
        d, f, n = avg.get_masked_data(
            self.data, self.noflags, self.nsamples_with_negatives, inpainted_mode=False
        )

        assert np.all(n.mask == d.mask)
        assert np.all(n.mask == f)
        assert np.sum(f) == 1

    @pytest.mark.parametrize("inpaint", (True, False))
    def test_inpainted_data_with_nans(self, inpaint):
        # If there are nans, we can never count that data, even if it is
        # supposedly inpainted and unflagged
        d, f, n = avg.get_masked_data(
            self.data_with_nans,
            self.noflags,
            self.nsamples_with_negatives,
            inpainted_mode=inpaint,
        )

        assert np.all(n.mask == d.mask)
        assert np.all(n.mask == f)
        assert np.sum(f) == 1

    @pytest.mark.parametrize("inpaint", (True, False))
    def test_inpainted_and_flagged(self, inpaint):
        # This shouldn't ever happen, but if data is marked as flagged, and
        # has negative nsamples, even though it is inpainted it should be counted
        # as flagged (as if it is re-flagged even after inpainting, suggesting that
        # the inpainting was not successful).
        d, f, n = avg.get_masked_data(
            self.data, self.flags, self.nsamples_with_negatives, inpainted_mode=inpaint
        )
        assert np.all(n.mask == d.mask)
        assert np.all(n.mask == f)
        assert np.sum(f) == 1

    @pytest.mark.parametrize("inpaint", (True, False))
    def test_flagged_and_not_inpainted(self, inpaint):
        d, f, n = avg.get_masked_data(
            self.data, self.ten_flags, self.nsamples, inpainted_mode=inpaint
        )
        assert np.all(n.mask == d.mask)
        assert np.all(n.mask == f)
        assert np.sum(f) == 10


class TestLSTAverage:
    def setup_class(self):
        self.shape = (2, 3, 4, 5)
        rng = np.random.default_rng(42)
        self.data = (
            rng.standard_normal(self.shape) + rng.standard_normal(self.shape) * 1j
        )

        self.data_with_nans = self.data.copy()
        self.data_with_nans[0, 0, 0, 0] = np.nan

        self.nsamples = np.ones(self.shape, dtype=float)
        self.nsamples_with_negatives = self.nsamples.copy()
        self.nsamples_with_negatives[0, 0, 0, 0] = -1

        self.nsamples_full_input = np.ones(self.shape, dtype=float)
        self.nsamples_full_input[:, 0, 0, 0] = -1

        self.noflags = np.zeros(self.shape, dtype=bool)
        self.flags = np.zeros_like(self.noflags)
        self.flags[0, 0, 0, 0] = True

        self.all_flags = np.zeros_like(self.noflags)
        self.all_flags[:, 0, 0, 0] = True

    def test_average_repeated(self):
        shape = (7, 8, 9)
        _data = np.random.random(shape) + np.random.random(shape) * 1j

        data = np.array([_data, _data, _data])
        nsamples = np.ones_like(data)
        flags = np.zeros_like(data, dtype=bool)

        _d, _f, _n = avg.get_masked_data(data, flags, nsamples, inpainted_mode=False)
        data_n, flg_n, std_n, norm_n, db = avg.lst_average(_d, _n, _f)

        assert np.allclose(data_n, _data)
        assert not np.any(flg_n)
        assert np.allclose(std_n, 0.0)
        assert np.allclose(norm_n, 3.0)

        # Now flag the last "night"
        flags[-1] = True

        _d, _f, _n = avg.get_masked_data(data, flags, nsamples, inpainted_mode=False)
        data_n, flg_n, std_n, norm_n, db = avg.lst_average(_d, _n, _f)

        assert np.allclose(data_n, _data)
        assert not np.any(flg_n)
        assert np.allclose(std_n, 0.0)
        assert np.allclose(norm_n, 2.0)

    def test_std_simple(self):
        shape = (5000, 1, 2, 2)  # 1000 nights, doesn't matter what the other axis is.

        std = 2.0
        data = (
            np.random.normal(scale=std, size=shape)
            + np.random.normal(scale=std, size=shape) * 1j
        )
        nsamples = np.ones_like(data, dtype=float)
        flags = np.zeros_like(data, dtype=bool)

        _d, _f, _n = avg.get_masked_data(data, flags, nsamples, inpainted_mode=False)
        data_n, flg_n, std_n, norm_n, db = avg.lst_average(_d, _n, _f)

        # Check the averaged data is within 6 sigma of the population mean
        np.testing.assert_allclose(data_n, 0.0, atol=std * 6 / np.sqrt(shape[0]))

        # Check the standard deviation is within 20% of the true value
        np.testing.assert_allclose(std_n, std + std * 1j, rtol=0.2)

        assert not np.any(flg_n)

    @pytest.mark.parametrize("nsamples", ("ones", "random"))
    @pytest.mark.parametrize("flags", ("zeros", "random"))
    def test_std(self, nsamples, flags):
        shape = (5000, 1, 10, 2)  # 1000 nights, doesn't matter what the other axis is.

        std = 2.0
        if nsamples == "ones":
            warn = False
            nsamples = np.ones(shape)
        else:
            warn = True
            rng = np.random.default_rng(42)
            nsamples = rng.integers(1, 10, size=shape).astype(float)

        std = std / np.sqrt(nsamples)

        if flags == "zeros":
            flags = np.zeros(shape, dtype=bool)
        else:
            flags = np.random.random(shape) > 0.1

        data = np.random.normal(scale=std) + np.random.normal(scale=std) * 1j

        flags = np.zeros(data.shape, dtype=bool)

        _d, _f, _n = avg.get_masked_data(data, flags, nsamples, inpainted_mode=False)

        if warn:
            with pytest.warns(
                UserWarning, match="Nsamples is not uniform across frequency"
            ):
                data_n, flg_n, std_n, _, _ = avg.lst_average(_d, _n, _f)
        else:
            data_n, flg_n, std_n, _, _ = avg.lst_average(_d, _n, _f)

        # Check the averaged data is within 6 sigma of the population mean
        assert np.allclose(data_n, 0.0, atol=std * 6 / np.sqrt(shape[0]))

        # In reality the std is infinity where flags is True
        std[flags] = np.inf
        w = 1 / np.sum(1.0 / std**2, axis=0)

        sample_var_expectation = sve = w * (shape[0] - 1)
        # Check the standard deviation is within 20% of the true value
        np.testing.assert_allclose(std_n, np.sqrt(sve) + np.sqrt(sve) * 1j, rtol=0.2)

        assert not np.any(flg_n)

    @pytest.mark.filterwarnings("ignore:invalid value encountered")
    @pytest.mark.parametrize("with_flags", (True, False))
    def test_inpaint_mode_does_nothing_for_unpainted(self, with_flags: bool):
        # This tests that if there is nothing inpainted (i.e. nsamples is all non-negative),
        # then inpainted mode changes nothing -- whether we're flagging data or not.
        if with_flags:
            flg = self.flags
        else:
            flg = self.noflags

        # First test -- no flags should mean inpainted_mode does nothing.
        _d, _f, _n = avg.get_masked_data(
            self.data, flg, self.nsamples, inpainted_mode=False
        )
        df, ff, stdf, nf, dbf = avg.lst_average(data=_d, nsamples=_n, flags=_f)

        _d, _f, _n = avg.get_masked_data(
            self.data, flg, self.nsamples, inpainted_mode=True
        )
        di, fi, stdi, ni, dbi = avg.lst_average(data=_d, nsamples=_n, flags=_f)

        np.testing.assert_allclose(df, di)
        np.testing.assert_allclose(ff, fi)
        np.testing.assert_allclose(stdf, stdi)
        np.testing.assert_allclose(nf, ni)
        np.testing.assert_allclose(dbf, dbi)

    @pytest.mark.filterwarnings("ignore:invalid value encountered")
    def test_inpainted_data_differences_between_modes(self):
        # This one tests that if we have flagged but inpainted data, then inpainted
        # mode makes a difference. Nsamples with negatives implies inpainting, and
        # we use "noflags" because lst_average assumes that anything that we want to use
        # that's inpainted will have be unflagged.
        _d, _f, _n = avg.get_masked_data(
            self.data, self.noflags, self.nsamples_with_negatives, inpainted_mode=False
        )
        df, ff, stdf, nf, dbf = avg.lst_average(_d, _n, _f)

        _d, _f, _n = avg.get_masked_data(
            self.data, self.noflags, self.nsamples_with_negatives, inpainted_mode=True
        )
        di, fi, stdi, ni, dbi = avg.lst_average(_d, _n, _f)

        # The data and std in the fully-flagged bin should be different, but
        # Nsamples, Flags and Days Binned should be the same.
        # Flags are the same because the whole bin is not flagged -- just one night.
        assert not np.allclose(df.flatten()[0], di.flatten()[0])
        np.testing.assert_allclose(df.flatten()[1:], di.flatten()[1:])
        assert not np.allclose(stdf.flatten()[0], stdi.flatten()[0])
        np.testing.assert_allclose(stdf.flatten()[1:], stdi.flatten()[1:])

        np.testing.assert_allclose(ff, fi)
        np.testing.assert_allclose(nf, ni)
        np.testing.assert_allclose(dbf, dbi)

    @pytest.mark.filterwarnings("ignore:invalid value encountered")
    def test_fullbin_inpaint_differences_between_modes(self):
        # This one tests that if we have flagged but inpainted data, then inpainted
        # mode makes a difference. Nsamples with negatives implies inpainting, and
        # we use "noflags" because lst_average assumes that anything that we want to use
        # that's inpainted will have be unflagged.
        _d, _f, _n = avg.get_masked_data(
            self.data, self.noflags, self.nsamples_full_input, inpainted_mode=False
        )
        df, ff, stdf, nf, dbf = avg.lst_average(_d, _n, _f)

        _d, _f, _n = avg.get_masked_data(
            self.data, self.noflags, self.nsamples_full_input, inpainted_mode=True
        )

        di, fi, stdi, ni, dbi = avg.lst_average(_d, _n, _f)

        # The data, flags and std in the fully-flagged bin should be different, but
        # Nsamples, and Days Binned should be the same (zero).
        # Flags are NOT the same because the whole bin is flagged
        assert not np.allclose(df.flatten()[0], di.flatten()[0])
        np.testing.assert_allclose(df.flatten()[1:], di.flatten()[1:])
        assert not np.allclose(ff.flatten()[0], fi.flatten()[0])
        np.testing.assert_allclose(ff.flatten()[1:], fi.flatten()[1:])
        assert not np.allclose(stdf.flatten()[0], stdi.flatten()[0])
        np.testing.assert_allclose(stdf.flatten()[1:], stdi.flatten()[1:])

        np.testing.assert_allclose(nf, ni)
        np.testing.assert_allclose(dbf, dbi)
        assert nf[0, 0, 0] == 0
        assert dbf[0, 0, 0] == 0


def test_get_std():
    data = np.linspace(0, 10, 3 * 4 * 5).reshape(3, 4, 5)
    nsamples = np.ones_like(data)
    flags = np.zeros_like(data, dtype=bool)
    flags[0, 0, 0] = True

    _d, _f, _n = avg.get_masked_data(data, flags, nsamples, inpainted_mode=False)

    mean = np.mean(_d, axis=0)

    std = avg.compute_std(_d, _n, mean=mean)[0]
    std2 = avg.compute_std(_d, _n)[0]
    assert np.all(std == std2)


class TestReduceLSTBins:
    @classmethod
    def get_input_data(
        cls,
        nfreqs: int = 3,
        npols: int = 1,
        nbls: int = 6,
        ntimes: int = 4,
    ):
        data = np.random.random((nbls, nfreqs, npols))

        # Make len(ntimes) LST bins, each with ntimes[i] time-entries, all the same
        # data.
        data = np.array([data] * ntimes)
        flags = np.zeros(data.shape, dtype=bool)
        nsamples = np.ones(data.shape, dtype=float)

        return data, flags, nsamples

    def test_one_point_per_bin(self):
        d, f, n = self.get_input_data(ntimes=1)
        rdc = avg.reduce_lst_bins(data=d, flags=f, nsamples=n)

        assert (
            rdc["data"].shape
            == rdc["flags"].shape
            == rdc["std"].shape
            == rdc["nsamples"].shape
        )

        np.testing.assert_allclose(rdc["data"], d[0])
        assert not np.any(rdc["flags"])
        np.testing.assert_allclose(rdc["nsamples"], 1.0)

    @pytest.mark.filterwarnings("ignore:invalid value encountered")
    def test_zerosize_bin(self):
        d, f, n = self.get_input_data(ntimes=0)
        rdc = avg.reduce_lst_bins(data=d, flags=f, nsamples=n, get_mad=True)

        assert np.all(np.isnan(rdc["data"]))
        assert np.all(rdc["flags"])
        assert np.all(rdc["nsamples"] == 0.0)
        assert np.all(np.isinf(rdc["mad"]))
        assert np.all(np.isnan(rdc["median"]))

    def test_multi_points_per_bin_flagged(self):
        d, f, n = self.get_input_data(ntimes=4)
        f[2:] = True
        d[2:] = 1000.0
        rdc = avg.reduce_lst_bins(data=d, flags=f, nsamples=n)

        assert (
            rdc["data"].shape
            == rdc["flags"].shape
            == rdc["std"].shape
            == rdc["nsamples"].shape
        )

        np.testing.assert_allclose(rdc["data"], d[0])
        assert not np.any(rdc["flags"])
        np.testing.assert_allclose(rdc["nsamples"], 2.0)

    def test_get_med_mad(self):
        d, f, n = self.get_input_data(ntimes=4)
        rdc = avg.reduce_lst_bins(data=d, flags=f, nsamples=n, get_mad=True)

        assert np.all(rdc["median"] == rdc["data"])

    def test_bad_input(self):
        d, f, n = self.get_input_data(ntimes=4)
        with pytest.raises(
            ValueError, match="data, flags, and nsamples must all be provided"
        ):
            avg.reduce_lst_bins(data=d, flags=f)


class TestAverageInpaintSimultaneouslySingleBl:
    """
    Testing at a single-bl level makes it easier to test more cases, so we use this
    class to do a bunch of precision tests.
    """

    def setup_class(self):
        self.rng = np.random.default_rng(42)

    def create_data(
        self,
        nnights=14,
        nfreqs=1536,
        add_tones: bool = False,
        add_noise: bool = True,
        gain_spread: float = 0.0,
        nsamples_func: callable = np.ones,
        flag_func: callable = partial(np.zeros, dtype=bool),
    ):
        freqs = mockuvd.PHASEII_FREQS[:nfreqs]

        basis = dpss_operator(
            freqs,
            filter_centers=[0],
            filter_half_widths=[200e-9],
            eigenval_cutoff=[1e-9],
        )[0].real

        ncoeff = basis.shape[-1]

        def gauss_noise(size, scale=1.0):
            return scale * (
                self.rng.normal(size=size) + 1j * self.rng.normal(size=size)
            )

        coeffs_mean = gauss_noise(ncoeff, 10)  # avg dpss coeffs
        coeffs = coeffs_mean + gauss_noise(
            (nnights, ncoeff), 0.01
        )  # daily variation in dpss coeffs
        d_true = np.einsum("nc,fc->nf", coeffs, basis)

        if add_tones:
            tones = gauss_noise((nnights, 1), 0.1) * np.exp(
                2j * np.pi * freqs[None, :] * 190e-9
            )  # a ripple
            d_true += tones

        # daily variation in gain
        gains = (
            1 + gain_spread * self.rng.uniform(size=d_true.shape[0]) - gain_spread / 2
        )
        d_true *= gains[:, None]

        nsamples = nsamples_func(d_true.shape)
        flags = flag_func(d_true.shape)

        if add_noise:
            n_true = gauss_noise(d_true.shape, 0.04) / nsamples**0.5
            d_true += n_true

        nsamples[flags] *= -1

        return freqs, d_true, flags, nsamples

    def random_nsamples(self, shape):
        n = self.rng.integers(1, 10, size=shape[0]).astype(float)
        return n[:, None] * np.ones(shape)

    def test_no_flags_no_nsamples(self):
        freqs, d, f, n = self.create_data()

        inp_mean, ff, model = avg.average_and_inpaint_simultaneously_single_bl(
            freqs=freqs,
            stackd=d,
            stackf=f,
            stackn=n,
            base_noise_var=0.04**2 * np.ones(d.shape),
            df=(freqs[1] - freqs[0]) * un.Hz,
            filter_half_widths=[200e-9],
            eigenval_cutoff=[1e-9],
        )

        assert np.all(inp_mean == np.mean(d, axis=0))

    @pytest.mark.parametrize("gain_spread", [0.0, 0.1, 0.5])
    @pytest.mark.parametrize("add_tones", [False, True])
    def test_no_flags_with_nsamples(self, gain_spread, add_tones):
        freqs, d, f, n = self.create_data(
            nsamples_func=self.random_nsamples,
            gain_spread=gain_spread,
            add_tones=add_tones,
        )

        inp_mean, ff, model = avg.average_and_inpaint_simultaneously_single_bl(
            freqs=freqs,
            stackd=d,
            stackf=f,
            stackn=n,
            base_noise_var=0.04**2 * np.ones(d.shape),
            df=(freqs[1] - freqs[0]) * un.Hz,
            filter_half_widths=[200e-9],
            eigenval_cutoff=[1e-9],
        )

        assert np.allclose(inp_mean, np.average(d, axis=0, weights=n))

    @pytest.mark.parametrize("gain_spread", [0.0, 0.3])
    @pytest.mark.parametrize("add_tones", [False, True])
    @pytest.mark.parametrize("gap_size", [1, 3])
    @pytest.mark.parametrize("nnights_flagged", [1, 2, 7, 13, 14])
    def test_small_flag_gap(self, gain_spread, add_tones, gap_size, nnights_flagged):
        freqs, d, f, n = self.create_data(gain_spread=gain_spread, add_tones=add_tones)
        slc = slice(750, 750 + gap_size)
        f[:nnights_flagged, slc] = True

        inp_mean, ff, model = avg.average_and_inpaint_simultaneously_single_bl(
            freqs=freqs,
            stackd=d,
            stackf=f,
            stackn=n,
            base_noise_var=0.04**2 * np.ones(d.shape),
            df=(freqs[1] - freqs[0]) * un.Hz,
            filter_half_widths=[200e-9],
            eigenval_cutoff=[1e-9],
        )

        np.testing.assert_allclose(
            inp_mean[slc],
            np.mean(d, axis=0)[slc],
            atol=5 * 0.04 / d.shape[0] ** 0.5,  # 5-sigma
        )

    @pytest.mark.parametrize("gain_spread", [0.0, 0.3])
    @pytest.mark.parametrize("add_tones", [False, True])
    @pytest.mark.parametrize("gap_size", [10, 20])
    @pytest.mark.parametrize("nnights_flagged", [1, 2, 7])
    def test_large_flag_gap(self, gain_spread, add_tones, gap_size, nnights_flagged):
        freqs, d, f, n = self.create_data(gain_spread=gain_spread, add_tones=add_tones)
        slc = slice(750, 750 + gap_size)
        f[:nnights_flagged, slc] = True

        inp_mean, ff, model = avg.average_and_inpaint_simultaneously_single_bl(
            freqs=freqs,
            stackd=d,
            stackf=f,
            stackn=n,
            base_noise_var=0.04**2 * np.ones(d.shape),
            df=(freqs[1] - freqs[0]) * un.Hz,
            filter_half_widths=[200e-9],
            eigenval_cutoff=[1e-9],
        )

        np.testing.assert_allclose(
            inp_mean[slc],
            np.mean(d, axis=0)[slc],
            atol=5 * 0.04 / np.sqrt(d.shape[0]),  # 5-sigma
        )

    @pytest.mark.parametrize("gain_spread", [0.3])
    @pytest.mark.parametrize("add_tones", [True])
    @pytest.mark.parametrize("gap_size", [1, 5, 10])
    @pytest.mark.parametrize("nnights_flagged", [1, 2, 7])
    @pytest.mark.parametrize("bias", [1.5, 3.0])
    def test_biased_flags(
        self, gain_spread, add_tones, gap_size, nnights_flagged, bias
    ):
        if bias > 1.5 and nnights_flagged > 5:
            pytest.xfail(
                "Expected failure of simultaneous inpainting with large bias in large gaps"
            )

        freqs, d, f, n = self.create_data(gain_spread=gain_spread, add_tones=add_tones)
        slc = slice(750, 750 + gap_size)
        f[:nnights_flagged, slc] = True

        # Do the bias
        d[:nnights_flagged] *= bias

        inp_mean, ff, model = avg.average_and_inpaint_simultaneously_single_bl(
            freqs=freqs,
            stackd=d,
            stackf=f,
            stackn=n,
            base_noise_var=0.04**2 * np.ones(d.shape),
            df=(freqs[1] - freqs[0]) * un.Hz,
            filter_half_widths=[200e-9],
            eigenval_cutoff=[1e-9],
        )

        np.testing.assert_allclose(
            inp_mean[slc],
            np.mean(d, axis=0)[slc],
            atol=5 * 0.04 / np.sqrt(d.shape[0]),  # 5-sigma
        )

    @pytest.mark.parametrize("gain_spread", [0.3])
    @pytest.mark.parametrize("add_tones", [True])
    @pytest.mark.parametrize("gap_size", [1, 5, 15])
    @pytest.mark.parametrize("nnights_flagged", [1, 2, 7])
    @pytest.mark.parametrize("bias", [1.0, 1.5])
    def test_uneven_flags(
        self, gain_spread, add_tones, gap_size, nnights_flagged, bias
    ):
        freqs, d, f, n = self.create_data(gain_spread=gain_spread, add_tones=add_tones)

        slc = slice(750 - gap_size, 750 + gap_size)
        for i in range(nnights_flagged):
            start = 750 + np.random.randint(-gap_size // 2, gap_size // 2)
            end = start + gap_size
            _slc = slice(start, end)
            f[i, _slc] = True

        # Do the bias
        d[:nnights_flagged] *= bias

        inp_mean, ff, model = avg.average_and_inpaint_simultaneously_single_bl(
            freqs=freqs,
            stackd=d,
            stackf=f,
            stackn=n,
            base_noise_var=0.04**2 * np.ones(d.shape),
            df=(freqs[1] - freqs[0]) * un.Hz,
            filter_half_widths=[200e-9],
            eigenval_cutoff=[1e-9],
        )

        np.testing.assert_allclose(
            inp_mean[slc],
            np.mean(d, axis=0)[slc],
            atol=5 * 0.04 / np.sqrt(d.shape[0]),  # 5-sigma
        )

    @pytest.mark.parametrize("gain_spread", [0.3])
    @pytest.mark.parametrize("add_tones", [True])
    @pytest.mark.parametrize("gap_size", [1, 5, 15])
    @pytest.mark.parametrize("nnights_flagged", [1, 2, 7])
    @pytest.mark.parametrize("bias", [1.0, 1.5])
    def test_band_edge(self, gain_spread, add_tones, gap_size, nnights_flagged, bias):
        if gap_size > 1:
            pytest.xfail("Expected failure of simultaneous inpainting at band edge")

        freqs, d, f, n = self.create_data(gain_spread=gain_spread, add_tones=add_tones)

        slc = slice(0, gap_size)
        f[:nnights_flagged, slc] = True

        # Do the bias
        d[:nnights_flagged] *= bias

        inp_mean, ff, model = avg.average_and_inpaint_simultaneously_single_bl(
            freqs=freqs,
            stackd=d,
            stackf=f,
            stackn=n,
            base_noise_var=0.04**2 * np.ones(d.shape),
            df=(freqs[1] - freqs[0]) * un.Hz,
            filter_half_widths=[200e-9],
            eigenval_cutoff=[1e-9],
        )

        np.testing.assert_allclose(
            inp_mean[slc],
            np.mean(d, axis=0)[slc],
            atol=5 * 0.04 / np.sqrt(d.shape[0]),  # 5-sigma
        )

    def test_non_uniform_nsamples(self):
        freqs, d, f, n = self.create_data()
        n[0, 1] = 25.0

        with pytest.raises(
            ValueError, match="assumes that nsamples is constant over frequency"
        ):
            avg.average_and_inpaint_simultaneously_single_bl(
                freqs=freqs,
                stackd=d,
                stackf=f,
                stackn=n,
                base_noise_var=0.04**2 * np.ones(d.shape),
                df=(freqs[1] - freqs[0]) * un.Hz,
                filter_half_widths=[200e-9],
            )

    def test_fully_flagged(self):
        freqs, d, f, n = self.create_data()
        f[:] = True

        data, flg, m = avg.average_and_inpaint_simultaneously_single_bl(
            freqs=freqs,
            stackd=d,
            stackf=f,
            stackn=n,
            base_noise_var=0.04**2 * np.ones(d.shape),
            df=(freqs[1] - freqs[0]) * un.Hz,
            filter_half_widths=[200e-9],
        )

        assert np.all(np.isnan(data))
        assert np.all(flg)

    def test_too_long_flag_gap(self):
        freqs, d, f, n = self.create_data()
        f[:, 100:200] = True

        data, flg, m = avg.average_and_inpaint_simultaneously_single_bl(
            freqs=freqs,
            stackd=d,
            stackf=f,
            stackn=n,
            base_noise_var=0.04**2 * np.ones(d.shape),
            df=(freqs[1] - freqs[0]) * un.Hz,
            filter_half_widths=[200e-9],
            max_gap_factor=1,
        )

        assert np.all(np.isnan(data))
        assert np.all(flg)

    def test_single_night_corner_case(self):
        freqs, d, f, n = self.create_data(nnights=1)
        data, flg, m = avg.average_and_inpaint_simultaneously_single_bl(
            freqs=freqs,
            stackd=d,
            stackf=f,
            stackn=n,
            base_noise_var=0.04**2 * np.ones(d.shape),
            df=(freqs[1] - freqs[0]) * un.Hz,
            filter_half_widths=[200e-9],
        )

        assert np.all(data == d)


class TestAverageInpaintSimultaneously:
    def setup_class(self):
        self.uvd = mockuvd.create_uvd_identifiable(
            integration_time=24 * 3600,
            ntimes=20,
            jd_start=2459844.0,
            antpairs=[(0, 1), (0, 2)],
            time_axis_faster_than_bls=False,
        )
        self.stack = LSTStack(self.uvd)
        self.stack.data[1:] = self.stack.data[0]  # All nights exactly the same

        self.auto_uvd = mockuvd.create_uvd_identifiable(
            integration_time=24 * 3600,
            ntimes=20,
            jd_start=2459844.0,
            antpairs=[(0, 0)],
            time_axis_faster_than_bls=False,
        )
        self.auto_stack = LSTStack(self.auto_uvd)
        self.auto_stack.data[1:] = self.auto_stack.data[
            0
        ]  # All nights exactly the same

    def test_no_flags(self):
        lstavg, models = avg.average_and_inpaint_simultaneously(
            self.stack, self.auto_stack, return_models=True
        )

        # Since there were no flags at all, there should be no models at all.
        assert len(models) == 0

        np.testing.assert_allclose(lstavg["data"], self.stack.data[0])

    def test_all_flagged(self):
        self.stack.flags[:] = True

        lstavg, models = avg.average_and_inpaint_simultaneously(
            self.stack, self.auto_stack, return_models=True
        )
        self.stack.flags[:] = False

        assert len(models) == 0
        assert np.all(np.isnan(lstavg["data"]))

    def test_fully_flagged_channel(self):
        self.stack.flags[:, 0, self.stack.Nfreqs // 2, 0] = True

        lstavg, models = avg.average_and_inpaint_simultaneously(
            self.stack, self.auto_stack, return_models=True
        )
        self.stack.flags[:] = False

        assert (
            len(models) == 1
        )  # only one baseline actually gets a model, others are fully determined by data
        assert not np.any(np.isnan(lstavg["data"]))
        np.testing.assert_allclose(
            lstavg["data"][0, self.stack.Nfreqs // 2, 0],
            self.stack.data[0, 0, self.stack.Nfreqs // 2, 0],
            rtol=1e-4,
        )
        assert lstavg["nsamples"][0, self.stack.Nfreqs // 2, 0] == 0.0
        assert not lstavg["flags"][0, self.stack.Nfreqs // 2, 0]

    def test_fully_flagged_integration(self):
        self.stack.flags[0, 0, :, 0] = True

        lstavg, models = avg.average_and_inpaint_simultaneously(
            self.stack, self.auto_stack, return_models=True
        )
        self.stack.flags[:] = False

        assert len(models) == 1
        assert not np.any(np.isnan(lstavg["data"]))
        np.testing.assert_allclose(
            lstavg["data"][0, :, 0],
            self.stack.data[0, 0, :, 0],
        )
        assert np.all(lstavg["nsamples"][0, :, 0] == len(self.stack.nights) - 1)
        assert not np.any(lstavg["flags"][0, :, 0])

    def test_nonred_data(self):
        auto_uvd = mockuvd.create_uvd_identifiable(
            integration_time=24 * 3600,
            ntimes=20,
            jd_start=2459844.0,
            antpairs=[(0, 0), (1, 1)],
            time_axis_faster_than_bls=False,
        )
        auto_stack = LSTStack(auto_uvd)

        with pytest.raises(
            NotImplementedError,
            match="This code only works with redundantly averaged data",
        ):
            avg.average_and_inpaint_simultaneously(self.stack, auto_stack)
