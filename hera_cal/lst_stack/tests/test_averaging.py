import numpy as np
from ...tests import mock_uvdata as mockuvd
import pytest
from .. import averaging as avg


class Test_LSTAlign:
    @classmethod
    def get_lst_align_data(
        cls,
        ntimes: int = 10,
        nfreqs: int = 5,
        npols: int = 4,
        nants: int = 4,
        ndays: int = 1,
        creator: callable = mockuvd.create_uvd_identifiable,
    ):
        uvds = mockuvd.make_dataset(
            ndays=ndays,
            nfiles=1,
            creator=creator,
            pols=("xx", "yy", "xy", "yx")[:npols],
            freqs=np.linspace(100e6, 200e6, nfreqs),
            ants=np.arange(nants),
            ntimes=ntimes,
            time_axis_faster_than_bls=False,
        )
        uvds = [uvd[0] for uvd in uvds]  # flatten to single list

        data = np.concatenate(
            [uvd.data_array.reshape((ntimes, -1, nfreqs, npols)) for uvd in uvds],
            axis=0,
        )
        freq_array = uvds[0].freq_array
        data_lsts = np.concatenate(
            [
                np.unique(uvds[0].lst_array),
            ]
            * len(uvds)
        )
        antpairs = uvds[0].get_antpairs()
        antpos = uvds[0].antenna_positions

        # Ensure that each LST is in its own bin
        lsts = np.sort(np.unique(data_lsts))

        lst_bin_edges = np.concatenate(
            (
                [lsts[0] - (lsts[1] - lsts[0]) / 2],
                (lsts[1:] + lsts[:-1]) / 2,
                [lsts[-1] + (lsts[-1] - lsts[-2]) / 2],
            )
        )

        return dict(
            data=data,
            data_lsts=data_lsts,
            antpairs=antpairs,
            lst_bin_edges=lst_bin_edges,
            freq_array=freq_array,
            antpos=antpos,
        )

    def test_bad_inputs(self):
        # Test that we get the right errors for bad inputs
        kwargs = self.get_lst_align_data()

        def lst_align_with(**kw):
            return avg.lst_align(**{**kwargs, **kw})

        data = np.ones(kwargs["data"].shape[:-1] + (5,))
        with pytest.raises(ValueError, match="data has more than 4 pols"):
            lst_align_with(data=data)

        with pytest.raises(ValueError, match="data should have shape"):
            lst_align_with(freq_array=kwargs["freq_array"][:-1])

        with pytest.raises(ValueError, match="flags should have shape"):
            lst_align_with(flags=np.ones(13, dtype=bool))

        # Make a wrong-shaped nsample array.
        with pytest.raises(ValueError, match="nsamples should have shape"):
            lst_align_with(nsamples=np.ones(13))

        # Use only one bin edge
        with pytest.raises(
            ValueError, match="lst_bin_edges must have at least 2 elements"
        ):
            lst_align_with(lst_bin_edges=np.ones(1))

        # Try rephasing without freq_array or antpos
        with pytest.raises(
            ValueError, match="freq_array and antpos is needed for rephase"
        ):
            lst_align_with(rephase=True, antpos=None)

    def test_increasing_lsts_one_per_bin(self, benchmark):
        kwargs = self.get_lst_align_data(ntimes=6)
        bins, d, f, n, inp = benchmark(avg.lst_align, rephase=False, **kwargs)

        # We should not be changing the data at all.
        d = np.squeeze(np.asarray(d))
        f = np.squeeze(np.asarray(f))
        n = np.squeeze(np.asarray(n))

        np.testing.assert_allclose(d, kwargs["data"])
        assert not np.any(f)
        assert np.all(n == 1.0)
        assert len(bins) == 6

    def test_multi_days_one_per_bin(self, benchmark):
        kwargs = self.get_lst_align_data(ndays=2, ntimes=7)
        bins, d, f, n, inp = benchmark(avg.lst_align, rephase=False, **kwargs)

        # We should not be changing the data at all.
        d = np.squeeze(np.asarray(d))
        f = np.squeeze(np.asarray(f))
        n = np.squeeze(np.asarray(n))

        np.testing.assert_allclose(d[:, 0], kwargs["data"][:7])
        assert not np.any(f)
        assert np.all(n == 1.0)
        assert len(bins) == 7

    def test_multi_days_with_flagging(self, benchmark):
        kwargs = self.get_lst_align_data(ndays=2, ntimes=7)

        # Flag everything after the first day, and make the data there crazy.
        flags = np.zeros_like(kwargs["data"], dtype=bool)
        flags[7:] = True
        kwargs["data"][7:] = 1000.0

        bins, d, f, n, inp = benchmark(
            avg.lst_align, rephase=False, flags=flags, **kwargs
        )

        d = np.squeeze(np.asarray(d))
        f = np.squeeze(np.asarray(f))
        n = np.squeeze(np.asarray(n))

        np.testing.assert_allclose(d[:, 0], kwargs["data"][:7])
        assert not np.any(f[:, 0])
        assert np.all(f[:, 1])
        assert len(bins) == 7

    def test_multi_days_with_nsamples_zero(self, benchmark):
        kwargs = benchmark(self.get_lst_align_data, ndays=2, ntimes=7)

        # Flag everything after the first day, and make the data there crazy.
        nsamples = np.ones_like(kwargs["data"], dtype=float)
        nsamples[7:] = 0.0
        kwargs["data"][7:] = 1000.0

        bins, d, f, n, inp = avg.lst_align(
            rephase=False, nsamples=nsamples, **kwargs
        )

        d = np.squeeze(np.asarray(d))
        f = np.squeeze(np.asarray(f))
        n = np.squeeze(np.asarray(n))

        np.testing.assert_allclose(d[:, 0], kwargs["data"][:7])
        assert not np.any(f)
        assert np.all(n[:, 0] == 1.0)
        assert np.all(n[:, 1] == 0.0)
        assert len(bins) == 7

    def test_rephase(self, benchmark):
        """Test that rephasing where each bin is already at center does nothing."""
        kwargs = self.get_lst_align_data(ntimes=7)

        bins0, d0, f0, n0, inp = benchmark(
            avg.lst_align, rephase=True, **kwargs
        )
        bins, d, f, n, inp = avg.lst_align(rephase=False, **kwargs)
        np.testing.assert_allclose(d, d0, rtol=1e-6)
        np.testing.assert_allclose(f, f0, rtol=1e-6)
        np.testing.assert_allclose(n, n0, rtol=1e-6)
        assert len(bins) == len(bins0)

    def test_lstbinedges_modulus(self, benchmark):
        kwargs = self.get_lst_align_data(ntimes=7)
        edges = kwargs.pop("lst_bin_edges")

        lst_bin_edges = edges.copy()
        lst_bin_edges -= 4 * np.pi

        bins, d0, f0, n0, inp = benchmark(
            avg.lst_align, lst_bin_edges=lst_bin_edges, **kwargs
        )

        lst_bin_edges = edges.copy()
        lst_bin_edges += 4 * np.pi

        bins, d, f, n, inp = avg.lst_align(
            lst_bin_edges=lst_bin_edges, **kwargs
        )

        np.testing.assert_allclose(d, d0)
        np.testing.assert_allclose(f, f0)
        np.testing.assert_allclose(n, n0)

        with pytest.raises(
            ValueError, match="lst_bin_edges must be monotonically increasing."
        ):
            avg.lst_align(lst_bin_edges=lst_bin_edges[::-1], **kwargs)


class Test_ReduceLSTBins:
    @classmethod
    def get_input_data(
        cls, nfreqs: int = 3, npols: int = 1, nbls: int = 6, ntimes: tuple[int] = (4,)
    ):
        data = np.random.random((nbls, nfreqs, npols))

        # Make len(ntimes) LST bins, each with ntimes[i] time-entries, all the same
        # data.
        data = [
            np.array(
                [
                    data,
                ]
                * nt
            ).reshape((nt,) + data.shape)
            * (i + 1)
            for i, nt in enumerate(ntimes)
        ]
        flags = [np.zeros(d.shape, dtype=bool) for d in data]
        nsamples = [np.ones(d.shape, dtype=float) for d in data]

        return data, flags, nsamples

    def test_one_point_per_bin(self, benchmark):
        d, f, n = self.get_input_data(ntimes=(1,))
        rdc = benchmark(avg.reduce_lst_bins, d, f, n)

        assert (
            rdc["data"].shape
            == rdc["flags"].shape
            == rdc["std"].shape
            == rdc["nsamples"].shape
        )

        # reduce_data swaps the order of bls/times
        dd = rdc["data"].swapaxes(0, 1)
        ff = rdc["flags"].swapaxes(0, 1)
        nn = rdc["nsamples"].swapaxes(0, 1)

        np.testing.assert_allclose(dd[0], d[0][0])
        assert not np.any(ff)
        np.testing.assert_allclose(nn, 1.0)

    @pytest.mark.filterwarnings("ignore:invalid value encountered")
    def test_zerosize_bin(self):
        d, f, n = self.get_input_data(ntimes=(0, 1))
        rdc = avg.reduce_lst_bins(d, f, n, get_mad=True)

        assert rdc["data"].shape[1] == 2  # 2 LST bins
        assert np.all(np.isnan(rdc["data"][:, 0]))
        assert np.all(rdc["flags"][:, 0])
        assert np.all(rdc["nsamples"][:, 0] == 0.0)
        assert np.all(np.isinf(rdc["mad"][:, 0]))
        assert np.all(np.isnan(rdc["median"][:, 0]))

    @pytest.mark.parametrize("ntimes", [(4,), (5, 4)])
    def test_multi_points_per_bin(self, ntimes, benchmark):
        d, f, n = self.get_input_data(ntimes=ntimes)
        rdc = benchmark(avg.reduce_lst_bins, d, f, n)

        assert (
            rdc["data"].shape
            == rdc["flags"].shape
            == rdc["std"].shape
            == rdc["nsamples"].shape
        )

        # reduce_data swaps the order of bls/times
        dd = rdc["data"].swapaxes(0, 1)
        ff = rdc["flags"].swapaxes(0, 1)
        nn = rdc["nsamples"].swapaxes(0, 1)

        assert not np.any(ff)
        for lst in range(len(ntimes)):
            np.testing.assert_allclose(dd[lst], d[lst][0])
            np.testing.assert_allclose(nn[lst], ntimes[lst])

    def test_multi_points_per_bin_flagged(self):
        d, f, n = self.get_input_data(ntimes=(4,))
        f[0][2:] = True
        d[0][2:] = 1000.0
        rdc = avg.reduce_lst_bins(d, f, n)

        assert (
            rdc["data"].shape
            == rdc["flags"].shape
            == rdc["std"].shape
            == rdc["nsamples"].shape
        )

        # reduce_data swaps the order of bls/times
        dd = rdc["data"].swapaxes(0, 1)
        ff = rdc["flags"].swapaxes(0, 1)
        nn = rdc["nsamples"].swapaxes(0, 1)

        np.testing.assert_allclose(dd[0], d[0][0])
        assert not np.any(ff)
        np.testing.assert_allclose(nn, 2.0)

    def test_get_med_mad(self):
        d, f, n = self.get_input_data(ntimes=(4,))
        rdc = avg.reduce_lst_bins(d, f, n, get_mad=True)

        assert np.all(rdc["median"] == rdc["data"])


class Test_LSTAverage:
    def test_sigma_clip_without_outliers(self, benchmark):
        shape = (7, 8, 9)
        np.random.seed(42)
        data = np.random.normal(size=shape) + np.random.normal(size=shape) * 1j
        nsamples = np.ones_like(data)
        flags = np.zeros_like(data, dtype=bool)

        data_n, flg_n, std_n, norm_n, daysbinned = avg.lst_average(
            data=data,
            nsamples=nsamples,
            flags=flags,
            sigma_clip_thresh=None,
        )

        data, flg, std, norm, daysbinned = benchmark(
            avg.lst_average,
            data=data,
            nsamples=nsamples,
            flags=flags,
            sigma_clip_thresh=20.0,
        )

        assert data.shape == flg.shape == std.shape == norm.shape == nsamples.shape[1:]
        np.testing.assert_allclose(data, data_n)

    def test_average_repeated(self):
        shape = (7, 8, 9)
        _data = np.random.random(shape) + np.random.random(shape) * 1j

        data = np.array([_data, _data, _data])
        nsamples = np.ones_like(data)
        flags = np.zeros_like(data, dtype=bool)

        data_n, flg_n, std_n, norm_n, db = avg.lst_average(
            data=data,
            nsamples=nsamples,
            flags=flags,
        )

        assert np.allclose(data_n, _data)
        assert not np.any(flg_n)
        assert np.allclose(std_n, 0.0)
        assert np.allclose(norm_n, 3.0)

        # Now flag the last "night"
        flags[-1] = True
        data_n, flg_n, std_n, norm_n, db = avg.lst_average(
            data=data,
            nsamples=nsamples,
            flags=flags,
        )

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

        data_n, flg_n, std_n, norm_n, db = avg.lst_average(
            data=data,
            nsamples=nsamples,
            flags=flags,
        )

        # Check the averaged data is within 6 sigma of the population mean
        np.testing.assert_allclose(data_n, 0.0, atol=std * 6 / np.sqrt(shape[0]))

        # Check the standard deviation is within 20% of the true value
        np.testing.assert_allclose(std_n, std + std * 1j, rtol=0.2)

        assert not np.any(flg_n)

    @pytest.mark.parametrize("nsamples", ("ones", "random"))
    @pytest.mark.parametrize("flags", ("zeros", "random"))
    def test_std(self, nsamples, flags):
        shape = (5000, 1, 2, 2)  # 1000 nights, doesn't matter what the other axis is.

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

        if warn:
            with pytest.warns(UserWarning, match='Nsamples is not uniform across frequency'):
                data_n, flg_n, std_n, _, _ = avg.lst_average(
                    data=data,
                    nsamples=nsamples,
                    flags=flags,
                )
        else:
            data_n, flg_n, std_n, _, _ = avg.lst_average(
                data=data,
                nsamples=nsamples,
                flags=flags,
            )

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
    def test_inpaint_mode(self):
        shape = (3, 2, 4, 1)  # nights, bls, freqs, pols
        _data = np.random.random(shape) + np.random.random(shape) * 1j
        nsamples = np.ones(_data.shape, dtype=float)
        flags = np.zeros(_data.shape, dtype=bool)

        # First test -- no flags should mean inpainted_mode does nothing.
        df, ff, stdf, nf, dbf = avg.lst_average(
            data=_data, nsamples=nsamples, flags=flags, inpainted_mode=False
        )

        di, fi, stdi, ni, dbi = avg.lst_average(
            data=_data, nsamples=nsamples, flags=flags, inpainted_mode=True
        )

        assert np.allclose(df, di)
        assert np.allclose(ff, fi)
        np.testing.assert_allclose(stdf, stdi)
        assert np.allclose(nf, ni)
        assert np.allclose(dbf, dbi)

        # Now test with a whole LST bin flagged for a single bl-chan-pol (but inpainted)
        flags[:, 0, 0, 0] = True
        df, ff, stdf, nf, dbf = avg.lst_average(
            data=_data, nsamples=nsamples, flags=flags, inpainted_mode=False
        )

        di, fi, stdi, ni, dbi = avg.lst_average(
            data=_data, nsamples=nsamples, flags=flags, inpainted_mode=True
        )

        # The data, flags and std in the fully-flagged bin should be different, but
        # Nsamples and Flags should be the same.
        assert not np.allclose(df[0, 0, 0], di[0, 0, 0])
        assert np.allclose(df[1:], di[1:])
        assert not np.allclose(ff, fi)
        assert not np.allclose(stdf[0, 0, 0], stdi[0, 0, 0])
        assert np.allclose(stdf[1:], stdi[1:])
        assert np.allclose(nf, ni)
        assert np.allclose(dbf, dbi)

        # Now test with a whole spectrum flagged for one night
        flags[:] = False
        flags[0, 0, :, 0] = True
        df, ff, stdf, nf, dbf = avg.lst_average(
            data=_data, nsamples=nsamples, flags=flags, inpainted_mode=False
        )

        di, fi, stdi, ni, dbi = avg.lst_average(
            data=_data, nsamples=nsamples, flags=flags, inpainted_mode=True
        )

        # This should give exactly the same results either way, because the full-flagged
        # blt is considered to NOT be inpainted by default.
        assert np.allclose(df, di)
        assert np.allclose(ff, fi)
        np.testing.assert_allclose(stdf, stdi)
        assert np.allclose(nf, ni)
        assert np.allclose(dbf, dbi)

        # However, if we had explicitly told the routine that the blt was inpainted,
        # we'd get a different result...
        _d, _f = averaging.get_masked_data(
            _data, nsamples, flags, inpainted=np.ones_like(flags), inpainted_mode=False
        )
        df, ff, stdf, nf, dbf = avg.lst_average(
            data=_d,
            nsamples=nsamples,
            flags=_f,
            inpainted_mode=False,
        )

        _d, _f = averaging.get_masked_data(
            _data, nsamples, flags, inpainted=np.ones_like(flags), inpainted_mode=True
        )
        di, fi, stdi, ni, dbi = avg.lst_average(
            data=_d,
            nsamples=nsamples,
            flags=_f,
            inpainted_mode=True,
        )

        # The LST-binned data will be different for blt=0, pol=0:
        assert not np.allclose(df[0, :, 0], di[0, :, 0])
        assert np.allclose(df[1:], di[1:])
        assert np.allclose(ff, fi)
        assert not np.allclose(stdf[0, :, 0], stdi[0, :, 0])
        assert np.allclose(stdf[1:], stdi[1:])
        assert np.allclose(nf, ni)
        assert np.allclose(dbf, dbi)

    def test_flag_below_min_N(self):
        shape = (7, 8, 9, 2)
        _data = np.random.random(shape) + np.random.random(shape) * 1j
        nsamples = np.ones(_data.shape, dtype=float)
        flags = np.zeros(_data.shape, dtype=bool)

        # No samples have more than min_N, so they should all be flagged.
        data_n, flg_n, std_n, norm_n, db = avg.lst_average(
            data=_data,
            nsamples=nsamples,
            flags=flags,
            sigma_clip_min_N=8,
            flag_below_min_N=True,
        )

        assert np.all(flg_n)
        assert np.all(norm_n == 7)  # Even though they're flagged, we track the nsamples
        assert not np.all(np.isinf(std_n))
        assert not np.all(np.isnan(data_n))

        # this time, there's enough samples, but too many are flagged...
        flags[:5] = True
        data_n, flg_n, std_n, norm_n, db = avg.lst_average(
            data=_data,
            nsamples=nsamples,
            flags=flags,
            sigma_clip_min_N=5,
            flag_below_min_N=True,
        )

        assert np.all(flg_n)
        # nsamples is zero because all are flagged.
        assert np.all(norm_n == 0)
        assert np.all(np.isinf(std_n))

        # this time, only one column is flagged too much...
        flags[:] = False
        flags[:5, 0] = True
        data_n, flg_n, std_n, norm_n, db = avg.lst_average(
            data=_data,
            nsamples=nsamples,
            flags=flags,
            sigma_clip_min_N=5,
            flag_below_min_N=True,
        )

        assert np.all(flg_n[0])
        assert np.all(norm_n[0] == 0)
        assert np.all(np.isinf(std_n[0]))

        assert not np.any(flg_n[1:])
        assert np.all(norm_n[1:] == 7)
        assert not np.any(np.isinf(std_n[1:]))
        assert not np.any(np.isnan(data_n[1:]))

    def test_sigma_clip_and_inpainted_warning(self):
        """Test that a warning is raised if doing inpainted_mode as well as sigma-clip."""
        shape = (7, 8, 9, 2)
        _data = np.random.random(shape) + np.random.random(shape) * 1j
        nsamples = np.ones(_data.shape, dtype=float)
        flags = np.zeros(_data.shape, dtype=bool)

        with pytest.warns(UserWarning, match="Direct-mode sigma-clipping in in-painted mode"):
            avg.lst_average(
                data=_data,
                nsamples=nsamples,
                flags=flags,
                sigma_clip_thresh=2.0,
                inpainted_mode=True,
            )
