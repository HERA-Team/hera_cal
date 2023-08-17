from __future__ import annotations

from . import mock_uvdata as mockuvd
import pytest
from pathlib import Path
from pyuvdata import UVCal
from itertools import combinations_with_replacement
import numpy as np
from hera_cal import lstbin_simple as lstbin_simple
import pytest
import numpy as np
from pyuvdata import UVCal, UVData
from .. import io, utils, lstbin_simple, noise
from hera_cal import apply_cal
from pyuvdata import utils as uvutils
from hera_cal.red_groups import RedundantGroups
from astropy import units
from functools import partial

try:
    benchmark
except NameError:
    @pytest.fixture(scope='module')
    def benchmark():
        def fnc(wrapped, *args, **kwargs):
            return wrapped(*args, **kwargs)
        return fnc

class Test_LSTAlign:
    @classmethod
    def get_lst_align_data(
        cls,
        ntimes: int = 10,
        nfreqs: int = 5,
        npols: int = 4,
        nants: int  = 4,
        ndays: int = 1,
        creator: callable = mockuvd.create_uvd_identifiable,
    ):

        uvds = mockuvd.make_dataset(
            ndays=ndays,
            nfiles=1,
            creator=creator,
            pols=('xx', 'yy', 'xy', 'yx')[:npols],
            freqs = np.linspace(100e6, 200e6, nfreqs),
            ants = np.arange(nants),
            ntimes=ntimes,
            time_axis_faster_than_bls=False,
        )
        uvds = [uvd[0] for uvd in uvds]  # flatten to single list

        data = np.concatenate([uvd.data_array.reshape((ntimes, -1, nfreqs, npols)) for uvd in uvds], axis=0)
        freq_array = uvds[0].freq_array
        data_lsts = np.concatenate([np.unique(uvds[0].lst_array),]*len(uvds))
        antpairs = uvds[0].get_antpairs()
        antpos = uvds[0].antenna_positions

        # Ensure that each LST is in its own bin
        lsts = np.sort(np.unique(data_lsts))

        lst_bin_edges = np.concatenate((
            [lsts[0] - (lsts[1] - lsts[0])/2],
            (lsts[1:] + lsts[:-1])/2,
            [lsts[-1] + (lsts[-1] - lsts[-2])/2]
        ))

        return dict(
            data = data,
            data_lsts = data_lsts,
            antpairs=antpairs,
            lst_bin_edges=lst_bin_edges,
            freq_array=freq_array,
            antpos=antpos,
        )


    def test_bad_inputs(self):
        # Test that we get the right errors for bad inputs
        kwargs = self.get_lst_align_data()

        def lst_align_with(**kw):
            return lstbin_simple.lst_align(**{**kwargs, **kw})

        data = np.ones(kwargs['data'].shape[:-1] + (5,))
        with pytest.raises(ValueError, match="data has more than 4 pols"):
            lst_align_with(data=data)

        with pytest.raises(ValueError, match="data should have shape"):
            lst_align_with(freq_array=kwargs['freq_array'][:-1])

        with pytest.raises(ValueError, match="flags should have shape"):
            lst_align_with(flags=np.ones(13, dtype=bool))

        # Make a wrong-shaped nsample array.
        with pytest.raises(ValueError, match="nsamples should have shape"):
            lst_align_with(nsamples=np.ones(13))

        # Use only one bin edge
        with pytest.raises(ValueError, match="lst_bin_edges must have at least 2 elements"):
            lst_align_with(lst_bin_edges=np.ones(1))

        # Try rephasing without freq_array or antpos
        with pytest.raises(ValueError, match="freq_array and antpos is needed for rephase"):
            lst_align_with(rephase=True, antpos=None)

    def test_increasing_lsts_one_per_bin(self, benchmark):
        kwargs = self.get_lst_align_data(ntimes=6)
        bins, d, f, n, inp = benchmark(lstbin_simple.lst_align, rephase=False, **kwargs)

        # We should not be changing the data at all.
        d = np.squeeze(np.asarray(d))
        f = np.squeeze(np.asarray(f))
        n = np.squeeze(np.asarray(n))

        np.testing.assert_allclose(d, kwargs['data'])
        assert not np.any(f)
        assert np.all(n == 1.0)
        assert len(bins) == 6

    def test_multi_days_one_per_bin(self, benchmark):
        kwargs = self.get_lst_align_data(ndays=2, ntimes=7)
        bins, d, f, n, inp = benchmark(lstbin_simple.lst_align, rephase=False, **kwargs)

        # We should not be changing the data at all.
        d = np.squeeze(np.asarray(d))
        f = np.squeeze(np.asarray(f))
        n = np.squeeze(np.asarray(n))

        np.testing.assert_allclose(d[:, 0], kwargs['data'][:7])
        assert not np.any(f)
        assert np.all(n == 1.0)
        assert len(bins) == 7

    def test_multi_days_with_flagging(self, benchmark):
        kwargs = self.get_lst_align_data(ndays=2, ntimes=7)

        # Flag everything after the first day, and make the data there crazy.
        flags = np.zeros_like(kwargs['data'], dtype=bool)
        flags[7:] = True
        kwargs['data'][7:] = 1000.0

        bins, d, f, n, inp = benchmark(lstbin_simple.lst_align, rephase=False, flags=flags, **kwargs)

        d = np.squeeze(np.asarray(d))
        f = np.squeeze(np.asarray(f))
        n = np.squeeze(np.asarray(n))

        np.testing.assert_allclose(d[:, 0], kwargs['data'][:7])
        assert not np.any(f[:, 0])
        assert np.all(f[:, 1])
        assert len(bins) == 7

    def test_multi_days_with_nsamples_zero(self, benchmark):
        kwargs = benchmark(self.get_lst_align_data, ndays=2, ntimes=7)

        # Flag everything after the first day, and make the data there crazy.
        nsamples = np.ones_like(kwargs['data'], dtype=float)
        nsamples[7:] = 0.0
        kwargs['data'][7:] = 1000.0

        bins, d, f, n, inp = lstbin_simple.lst_align(rephase=False, nsamples=nsamples, **kwargs)

        d = np.squeeze(np.asarray(d))
        f = np.squeeze(np.asarray(f))
        n = np.squeeze(np.asarray(n))

        np.testing.assert_allclose(d[:, 0], kwargs['data'][:7])
        assert not np.any(f)
        assert np.all(n[:, 0] == 1.0)
        assert np.all(n[:, 1] == 0.0)
        assert len(bins) == 7


    def test_rephase(self, benchmark):
        """Test that rephasing where each bin is already at center does nothing."""
        kwargs = self.get_lst_align_data(ntimes=7)

        bins0, d0, f0, n0, inp = benchmark(lstbin_simple.lst_align, rephase=True, **kwargs)
        bins, d, f, n, inp = lstbin_simple.lst_align(rephase=False, **kwargs)
        np.testing.assert_allclose(d, d0, rtol=1e-6)
        np.testing.assert_allclose(f, f0, rtol=1e-6)
        np.testing.assert_allclose(n, n0, rtol=1e-6)
        assert len(bins) == len(bins0)

    def test_lstbinedges_modulus(self, benchmark):

        kwargs = self.get_lst_align_data(ntimes=7)
        edges = kwargs.pop("lst_bin_edges")

        lst_bin_edges = edges.copy()
        lst_bin_edges -= 4*np.pi

        bins, d0, f0, n0, inp = benchmark(lstbin_simple.lst_align, lst_bin_edges=lst_bin_edges, **kwargs)

        lst_bin_edges = edges.copy()
        lst_bin_edges += 4*np.pi

        bins, d, f, n, inp = lstbin_simple.lst_align(lst_bin_edges=lst_bin_edges, **kwargs)

        np.testing.assert_allclose(d, d0)
        np.testing.assert_allclose(f, f0)
        np.testing.assert_allclose(n, n0)

        with pytest.raises(ValueError, match="lst_bin_edges must be monotonically increasing."):
            lstbin_simple.lst_align(lst_bin_edges=lst_bin_edges[::-1], **kwargs)

def test_argparser_returns():
    args = lstbin_simple.lst_bin_arg_parser()
    assert args is not None

class Test_ReduceLSTBins:
    @classmethod
    def get_input_data(
        cls,
        nfreqs: int=3, npols: int = 1, nbls: int = 6, ntimes: tuple[int] = (4, )
    ):
        data = np.random.random((nbls, nfreqs, npols))

        # Make len(ntimes) LST bins, each with ntimes[i] time-entries, all the same
        # data.
        data = [np.array([data, ]*nt).reshape((nt,) + data.shape)*(i+1) for i, nt in enumerate(ntimes)]
        flags = [np.zeros(d.shape, dtype=bool) for d in data]
        nsamples = [np.ones(d.shape, dtype=float) for d in data]

        return data, flags, nsamples

    def test_one_point_per_bin(self, benchmark):
        d, f, n = self.get_input_data(ntimes=(1,))
        rdc= benchmark(lstbin_simple.reduce_lst_bins, d, f, n)

        assert rdc['data'].shape == rdc['flags'].shape == rdc['std'].shape == rdc['nsamples'].shape

        # reduce_data swaps the order of bls/times
        dd = rdc['data'].swapaxes(0, 1)
        ff = rdc['flags'].swapaxes(0, 1)
        nn = rdc['nsamples'].swapaxes(0, 1)

        np.testing.assert_allclose(dd[0], d[0][0])
        assert not np.any(ff)
        np.testing.assert_allclose(nn, 1.0)

    def test_zerosize_bin(self):
        d, f, n = self.get_input_data(ntimes=(0, 1))
        print(d[0].shape, len(d))
        rdc = lstbin_simple.reduce_lst_bins(d, f, n)

        assert rdc['data'].shape[1] == 2  # 2 LST bins
        assert np.all(np.isnan(rdc['data'][:, 0]))
        assert np.all(rdc['flags'][:, 0])
        assert np.all(rdc['nsamples'][:, 0] == 0.0)

    @pytest.mark.parametrize("ntimes", [(4, ), (5, 4)])
    def test_multi_points_per_bin(self, ntimes, benchmark):
        d, f, n = self.get_input_data(ntimes=ntimes)
        rdc = benchmark(lstbin_simple.reduce_lst_bins, d, f, n)

        assert rdc['data'].shape == rdc['flags'].shape == rdc['std'].shape == rdc['nsamples'].shape

        # reduce_data swaps the order of bls/times
        dd = rdc['data'].swapaxes(0, 1)
        ff = rdc['flags'].swapaxes(0, 1)
        nn = rdc['nsamples'].swapaxes(0, 1)

        assert not np.any(ff)
        for lst in range(len(ntimes)):
            np.testing.assert_allclose(dd[lst], d[lst][0])
            np.testing.assert_allclose(nn[lst], ntimes[lst])

    def test_multi_points_per_bin_flagged(self):
        d, f, n = self.get_input_data(ntimes=(4,))
        f[0][2:] = True
        d[0][2:] = 1000.0
        rdc = lstbin_simple.reduce_lst_bins(d,f,n)

        assert rdc['data'].shape == rdc['flags'].shape == rdc['std'].shape == rdc['nsamples'].shape

        # reduce_data swaps the order of bls/times
        dd = rdc['data'].swapaxes(0, 1)
        ff = rdc['flags'].swapaxes(0, 1)
        nn = rdc['nsamples'].swapaxes(0, 1)


        np.testing.assert_allclose(dd[0], d[0][0])
        assert not np.any(ff)
        np.testing.assert_allclose(nn, 2.0)

    def test_get_med_mad(self):
        d, f, n = self.get_input_data(ntimes=(4,))
        rdc = lstbin_simple.reduce_lst_bins(d, f, n, get_mad=True)

        assert np.all(rdc['median'] == rdc['data'])

def test_apply_calfile_rules(tmpdir_factory):
    direc = tmpdir_factory.mktemp("test_apply_calfile_rules")

    datas = [Path(direc / f"data{i}.uvh5") for i in range(3)]
    for d in datas:
        d.touch()

    cals = [Path(direc / f"data{i}.calfile") for i in range(3)]
    for c in cals:
        c.touch()

    data_files, calfiles = lstbin_simple.apply_calfile_rules(
        [[str(d) for d in datas]],
        calfile_rules = [('.uvh5', '.calfile')],
        ignore_missing=False
    )
    assert len(data_files[0]) == 3
    assert len(calfiles[0]) == 3

    cals[-1].unlink()
    with pytest.raises(IOError, match="does not exist"):
        lstbin_simple.apply_calfile_rules(
            [[str(d) for d in datas]],
            calfile_rules = [('.uvh5', '.calfile')],
            ignore_missing=False
        )

    data_files, calfiles = lstbin_simple.apply_calfile_rules(
        [[str(d) for d in datas]],
        calfile_rules = [('.uvh5', '.calfile')],
        ignore_missing=True
    )
    assert len(data_files[0]) == 2
    assert len(calfiles[0]) == 2


class Test_LSTAverage:

    def test_sigma_clip_without_outliers(self, benchmark):
        shape = (7,8,9)
        np.random.seed(42)
        data = np.random.normal(size=shape) + np.random.normal(size=shape)*1j
        nsamples = np.ones_like(data)
        flags = np.zeros_like(data, dtype=bool)

        data_n, flg_n, std_n, norm_n, daysbinned =lstbin_simple.lst_average(
            data=data,
            nsamples=nsamples,
            flags=flags,
            sigma_clip_thresh=None,
        )

        data, flg, std, norm, daysbinned = benchmark(
            lstbin_simple.lst_average,
            data=data,
            nsamples=nsamples,
            flags=flags,
            sigma_clip_thresh=20.0,
        )

        assert data.shape == flg.shape == std.shape == norm.shape == nsamples.shape[1:]
        np.testing.assert_allclose(data,data_n)

    def test_average_repeated(self):
        shape = (7,8,9)
        _data = np.random.random(shape) + np.random.random(shape)*1j

        data = np.array([_data, _data, _data])
        nsamples = np.ones_like(data)
        flags = np.zeros_like(data, dtype=bool)

        data_n, flg_n, std_n, norm_n, db = lstbin_simple.lst_average(
            data=data, nsamples=nsamples, flags=flags,
        )

        assert np.allclose(data_n, _data)
        assert not np.any(flg_n)
        assert np.allclose(std_n, 0.0)
        assert np.allclose(norm_n, 3.0)

        # Now flag the last "night"
        flags[-1] = True
        data_n, flg_n, std_n, norm_n, db = lstbin_simple.lst_average(
            data=data, nsamples=nsamples, flags=flags,
        )

        assert np.allclose(data_n, _data)
        assert not np.any(flg_n)
        assert np.allclose(std_n, 0.0)
        assert np.allclose(norm_n, 2.0)

    def test_std_simple(self):
        shape = (5000, 1, 2, 2)  # 1000 nights, doesn't matter what the other axis is.

        std = 2.0
        data = np.random.normal(scale=std, size=shape) + np.random.normal(scale=std, size=shape)*1j
        nsamples = np.ones_like(data, dtype=float)
        flags = np.zeros_like(data, dtype=bool)

        data_n, flg_n, std_n, norm_n, db = lstbin_simple.lst_average(
            data=data, nsamples=nsamples, flags=flags,
        )

        # Check the averaged data is within 6 sigma of the population mean
        np.testing.assert_allclose(data_n, 0.0, atol=std*6/np.sqrt(shape[0]))

        # Check the standard deviation is within 20% of the true value
        np.testing.assert_allclose(std_n, std + std*1j, rtol=0.2)

        assert not np.any(flg_n)

    @pytest.mark.parametrize("nsamples", ("ones", "random"))
    @pytest.mark.parametrize("flags", ("zeros", "random"))
    def test_std(self, nsamples, flags):
        shape = (5000, 1, 2, 2)  # 1000 nights, doesn't matter what the other axis is.

        std = 2.0
        if nsamples == "ones":
            nsamples = np.ones(shape)
        else:
            nsamples = np.random.random_integers(1, 10, size=shape).astype(float)

        std = std / np.sqrt(nsamples)

        if flags == "zeros":
            flags = np.zeros(shape, dtype=bool)
        else:
            flags = np.random.random(shape) > 0.1

        data = np.random.normal(scale=std) + np.random.normal(scale=std)*1j

        flags = np.zeros(data.shape, dtype=bool)

        data_n, flg_n, std_n, norm_n, db = lstbin_simple.lst_average(
            data=data, nsamples=nsamples, flags=flags,
        )

        # Check the averaged data is within 6 sigma of the population mean
        assert np.allclose(data_n, 0.0, atol=std*6/np.sqrt(shape[0]))

        # In reality the std is infinity where flags is True
        std[flags] = np.inf
        w = 1 / np.sum(1.0 / std**2, axis=0)

        sample_var_expectation = sve = w * (shape[0] - 1)
        # Check the standard deviation is within 20% of the true value
        np.testing.assert_allclose(std_n, np.sqrt(sve) + np.sqrt(sve)*1j, rtol=0.2)

        assert not np.any(flg_n)

    def test_inpaint_mode(self):
        shape = (3,2,4, 1)  # nights, bls, freqs, pols
        _data = np.random.random(shape) + np.random.random(shape)*1j
        nsamples = np.ones(_data.shape, dtype=float)
        flags = np.zeros(_data.shape, dtype=bool)

        # First test -- no flags should mean inpainted_mode does nothing.
        df, ff, stdf, nf, dbf = lstbin_simple.lst_average(
            data=_data, nsamples=nsamples, flags=flags, inpainted_mode=False
        )

        di, fi, stdi, ni, dbi = lstbin_simple.lst_average(
            data=_data, nsamples=nsamples, flags=flags, inpainted_mode=True
        )

        assert np.allclose(df, di)
        assert np.allclose(ff, fi)
        np.testing.assert_allclose(stdf, stdi)
        assert np.allclose(nf, ni)
        assert np.allclose(dbf, dbi)


        # Now test with a whole LST bin flagged for a single bl-chan-pol (but inpainted)
        flags[:, 0, 0, 0] = True
        df, ff, stdf, nf, dbf = lstbin_simple.lst_average(
            data=_data, nsamples=nsamples, flags=flags, inpainted_mode=False
        )

        di, fi, stdi, ni, dbi = lstbin_simple.lst_average(
            data=_data, nsamples=nsamples, flags=flags, inpainted_mode=True
        )

        # The data and std in the fully-flagged bin should be different, but Nsamples
        # and Flags should be the same.
        assert not np.allclose(df[0,0,0], di[0,0,0])
        assert np.allclose(df[1:], di[1:])
        assert np.allclose(ff, fi)
        assert not np.allclose(stdf[0,0,0], stdi[0,0,0])
        assert np.allclose(stdf[1:], stdi[1:])
        assert np.allclose(nf, ni)
        assert np.allclose(dbf, dbi)

        # Now test with a whole spectrum flagged for one night
        flags[:] = False
        flags[0, 0, :, 0] = True
        df, ff, stdf, nf, dbf = lstbin_simple.lst_average(
            data=_data, nsamples=nsamples, flags=flags, inpainted_mode=False
        )

        di, fi, stdi, ni, dbi = lstbin_simple.lst_average(
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
        _d, _f = lstbin_simple.get_masked_data(_data, nsamples, flags, inpainted=np.ones_like(flags),inpainted_mode=False)
        df, ff, stdf, nf, dbf = lstbin_simple.lst_average(
            data=_d, nsamples=nsamples, flags=_f, inpainted_mode=False,
        )

        _d, _f = lstbin_simple.get_masked_data(_data, nsamples, flags, inpainted=np.ones_like(flags),inpainted_mode=True)
        di, fi, stdi, ni, dbi = lstbin_simple.lst_average(
            data=_d, nsamples=nsamples, flags=_f, inpainted_mode=True,
        )

        # The LST-binned data will be different for blt=0, pol=0:
        assert not np.allclose(df[0,:,0], di[0,:,0])
        assert np.allclose(df[1:], di[1:])
        assert np.allclose(ff, fi)
        assert not np.allclose(stdf[0,:,0], stdi[0,:,0])
        assert np.allclose(stdf[1:], stdi[1:])
        assert np.allclose(nf, ni)
        assert np.allclose(dbf, dbi)

        


    def test_flag_below_min_N(self):
        shape = (7,8,9, 2)
        _data = np.random.random(shape) + np.random.random(shape)*1j
        nsamples = np.ones(_data.shape, dtype=float)
        flags = np.zeros(_data.shape, dtype=bool)

        # No samples have more than min_N, so they should all be flagged.
        data_n, flg_n, std_n, norm_n, db = lstbin_simple.lst_average(
            data=_data, nsamples=nsamples, flags=flags, sigma_clip_min_N=8,
            flag_below_min_N=True
        )

        assert np.all(flg_n)
        assert np.all(norm_n==7)  # Even though they're flagged, we track the nsamples
        assert not np.all(np.isinf(std_n))
        assert not np.all(np.isnan(data_n))

        # this time, there's enough samples, but too many are flagged...
        flags[:5] = True
        data_n, flg_n, std_n, norm_n, db = lstbin_simple.lst_average(
            data=_data, nsamples=nsamples, flags=flags, sigma_clip_min_N=5,
            flag_below_min_N=True
        )

        assert np.all(flg_n)
        # just because we're flagging it, doesn't mean we need to set nsamples=0
        # or the std to inf. We have info there, we're just choosing not to use it.
        assert np.all(norm_n==2)
        assert not np.any(np.isinf(std_n))
        
        # this time, only one column is flagged too much...
        # this time, there's enough samples, but too many are flagged...
        flags[:] = False
        flags[:5, 0] = True
        data_n, flg_n, std_n, norm_n, db = lstbin_simple.lst_average(
            data=_data, nsamples=nsamples, flags=flags, sigma_clip_min_N=5,
            flag_below_min_N=True
        )

        assert np.all(flg_n[0])
        assert np.all(norm_n[0]==2)
        assert not np.any(np.isinf(std_n[0]))

        print(np.sum(flg_n[1:]), flg_n[1:].size)
        assert not np.any(flg_n[1:])
        assert np.all(norm_n[1:]==7)
        assert not np.any(np.isinf(std_n[1:]))
        assert not np.any(np.isnan(data_n[1:]))
        

def create_small_array_uvd(identifiable: bool = False, **kwargs):
    kwargs.update(
        freqs=np.linspace(150e6, 160e6, 100),
        ants=[0,1,2,127,128],
        antpairs=[(0,0), (0,1), (0,2), (1, 1), (1,2), (2, 2)],
        pols=('xx', 'yy')
    )
    if identifiable:
        return mockuvd.create_uvd_identifiable(**kwargs)
    else:
        return mockuvd.create_uvd_ones(**kwargs)

@pytest.fixture(scope="function")
def uvd():
    return create_small_array_uvd()

@pytest.fixture(scope="function")
def uvd_redavg():
    return create_small_array_uvd(redundantly_averaged=True)

@pytest.fixture(scope="function")
def uvc(uvd):
    return UVCal.initialize_from_uvdata(
        uvd,
        cal_style = "redundant",
        gain_convention = "multiply",
        jones_array = "linear",
        cal_type = "gain",
        metadata_only=False,
    )

@pytest.fixture(scope="function")
def uvd_file(uvd, tmpdir_factory) -> Path:
    # Write to file, so we can run lst_bin_files
    tmp = Path(tmpdir_factory.mktemp("test_partial_times"))
    mock = tmp / 'mock.uvh5'
    uvd.write_uvh5(str(mock), clobber=True)
    return mock

@pytest.fixture(scope="function")
def uvd_redavg_file(uvd_redavg, tmpdir_factory) -> Path:
    # Write to file, so we can run lst_bin_files
    tmp = Path(tmpdir_factory.mktemp("test_partial_times"))
    mock = tmp / 'mock.uvh5'
    uvd_redavg.write_uvh5(str(mock), clobber=True)
    return mock


@pytest.fixture(scope="function")
def uvc_file(uvc, uvd_file: Path) -> Path:
    # Write to file, so we can run lst_bin_files
    tmp = uvd_file.parent
    fl = f'{tmp}/mock.calfits'
    uvc.write_calfits(str(fl), clobber=True)
    return fl

class Test_LSTBinFilesForBaselines:
    def test_defaults(self, uvd, uvd_file):
        lstbins, d0, f0, n0, inpflg, times0 = lstbin_simple.lst_bin_files_for_baselines(
            data_files=[uvd_file],
            lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max()+0.01],
            antpairs=uvd.get_antpairs(),
            rephase=False
        )

        lstbins, d, f, n, inpflg, times = lstbin_simple.lst_bin_files_for_baselines(
            data_files=[uvd_file],
            lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max()+0.01],
            antpairs=uvd.get_antpairs(),
            freqs=uvd.freq_array,
            pols=uvd.polarization_array,
            time_idx = [np.ones(uvd.Ntimes, dtype=bool)],
            time_arrays=[np.unique(uvd.time_array)],
            lsts = np.unique(uvd.lst_array),
            rephase=False
        )

        np.testing.assert_allclose(d0, d)
        np.testing.assert_allclose(f0, f)
        np.testing.assert_allclose(n0, n)
        np.testing.assert_allclose(times0, times)

    def test_empty(self, uvd, uvd_file):
        lstbins, d0, f0, n0, inpflg, times0 = lstbin_simple.lst_bin_files_for_baselines(
            data_files=[uvd_file],
            lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max()],
            antpairs=[(127, 128)],
            rephase=True
        )

        assert np.all(f0)

    def test_extra(self, uvd, uvd_file):
        # Providing baselines that don't exist in the file is fine, they're just ignored.
        lstbins, d0, f0, n0, inpflg, times0 = lstbin_simple.lst_bin_files_for_baselines(
            data_files=[uvd_file],
            lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max()],
            antpairs=uvd.get_antpairs() + [(127, 128)],
            rephase=True
        )

        assert np.all(f0[0][:, -1])  # last baseline is the extra one that's all flagged.

    def test_freqrange(self, uvd, uvd_file, uvc_file):
        """Test that providing freq_range works."""
        bins, data, flags, nsamples,  inpflg, times = lstbin_simple.lst_bin_files_for_baselines(
            data_files=[uvd_file],
            lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max()],
            cal_files = [uvc_file],
            freq_min=153e6,
            freq_max=158e6,
            antpairs=uvd.get_antpairs(),
        )

        assert data[0].shape[-2] < uvd.Nfreqs

    def test_bad_pols(self, uvd, uvd_file):
        with pytest.raises(KeyError, match='7'):
            lstbin_simple.lst_bin_files_for_baselines(
                data_files=[uvd_file],
                lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max()],
                pols=[3.0, 7, -1],
                antpairs=uvd.get_antpairs(),
            )

    def test_incorrect_red_input(self, uvd, uvd_file, uvc_file):
        with pytest.raises(ValueError, match='reds must be provided if redundantly_averaged is True'):
            lstbin_simple.lst_bin_files_for_baselines(
                data_files=[uvd_file],
                lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max()],
                redundantly_averaged=True,
                antpairs=uvd.get_antpairs(),
            )

        with pytest.raises(ValueError, match="Cannot apply calibration if redundantly_averaged is True"):
            lstbin_simple.lst_bin_files_for_baselines(
                data_files=[uvd_file],
                lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max()],
                cal_files=[uvc_file],
                redundantly_averaged=True,
                reds=RedundantGroups.from_antpos(
                    dict(zip(uvd.antenna_numbers, uvd.antenna_positions)),
                    pols=uvutils.polnum2str(uvd.polarization_array, x_orientation=uvd.x_orientation),
                ),
                antpairs=uvd.get_antpairs(),
            )

    def test_simple_redundant_averaged_file(self, uvd_redavg, uvd_redavg_file):
        lstbins, d0, f0, n0, inpflg, times0 = lstbin_simple.lst_bin_files_for_baselines(
            data_files=[uvd_redavg_file],
            lst_bin_edges=[uvd_redavg.lst_array.min()-0.1, uvd_redavg.lst_array.max()+0.1],
            redundantly_averaged=True,
            rephase=False,
            antpairs=uvd_redavg.get_antpairs(),
            reds=RedundantGroups.from_antpos(
                dict(zip(uvd_redavg.antenna_numbers, uvd_redavg.antenna_positions)),
            ),
        )

        assert len(d0) == 1
        assert d0[0].shape == (uvd_redavg.Ntimes, uvd_redavg.Nbls, uvd_redavg.Nfreqs, uvd_redavg.Npols)

def test_make_lst_grid():
    lst_grid = lstbin_simple.make_lst_grid(0.01, begin_lst=None)
    assert len(lst_grid) == 628
    assert np.isclose(lst_grid[0], 0.0050025360725952121)
    lst_grid = lstbin_simple.make_lst_grid(0.01, begin_lst=np.pi)
    assert len(lst_grid) == 628
    assert np.isclose(lst_grid[0], 3.1365901175171982)
    lst_grid = lstbin_simple.make_lst_grid(0.01, begin_lst=-np.pi)
    assert len(lst_grid) == 628
    assert np.isclose(lst_grid[0], 3.1365901175171982)

class Test_GetAllUnflaggedBaselines:
    @pytest.mark.parametrize("redundantly_averaged", [True, False])
    @pytest.mark.parametrize("only_last_file_per_night", [True, False])
    def test_get_all_baselines(self, tmp_path_factory, redundantly_averaged, only_last_file_per_night):
        tmp = tmp_path_factory.mktemp("get_all_baselines")
        uvds = mockuvd.make_dataset(
            ndays=3,
            nfiles=4,
            ntimes=2,
            ants=np.arange(10),
            creator=create_small_array_uvd,
            redundantly_averaged=redundantly_averaged,
        )
        data_files = mockuvd.write_files_in_hera_format(uvds, tmp)

        antpairs, pols = lstbin_simple.get_all_unflagged_baselines(
            data_files,
            redundantly_averaged=redundantly_averaged,
            only_last_file_per_night=only_last_file_per_night,
        )

        assert len(antpairs) == len(uvds[0][0].get_antpairs())

    def test_bad_inputs(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("get_all_baselines")
        uvds = mockuvd.make_dataset(
            ndays=3,
            nfiles=4,
            ntimes=2,
            ants=np.arange(10),
            creator=create_small_array_uvd,
            redundantly_averaged=True,
        )
        data_files = mockuvd.write_files_in_hera_format(uvds, tmp)

        with pytest.raises(ValueError, match='Cannot ignore antennas if the files are redundantly averaged'):
            lstbin_simple.get_all_unflagged_baselines(data_files, ignore_ants=[0, 1])

        with pytest.raises(ValueError, match='Cannot exclude antennas if the files are redundantly averaged'):
            lstbin_simple.get_all_unflagged_baselines(
                data_files,
                ex_ant_yaml_files=['non-existent-file.yaml'],
            )

        uvds_different_xorient = mockuvd.make_dataset(
            ndays=1,
            nfiles=4,
            ntimes=2,
            ants=np.arange(10),
            creator=create_small_array_uvd,
            x_orientation='east',
            redundantly_averaged=True,
        )

        data_files = mockuvd.write_files_in_hera_format(uvds + uvds_different_xorient, tmp)

        with pytest.raises(ValueError, match='Not all files have the same xorientation!'):
            lstbin_simple.get_all_unflagged_baselines(data_files)

class Test_LSTBinFiles:
    def test_golden_data(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("lstbin_golden_data")
        uvds = mockuvd.make_dataset(
            ndays=3, nfiles=4, ntimes=2, identifiable=True,
            creator=create_small_array_uvd, time_axis_faster_than_bls=True
        )
        data_files = mockuvd.write_files_in_hera_format(uvds, tmp)
        print(len(uvds))
        cfl = tmp / "lstbin_config_file.yaml"
        print(cfl)
        lstbin_simple.make_lst_bin_config_file(
            cfl, data_files, ntimes_per_file=2,
        )

        out_files = lstbin_simple.lst_bin_files(
            config_file=cfl, rephase=False,
            golden_lsts=uvds[0][1].lst_array.min() + 0.0001
        )

        assert len(out_files) == 4
        assert out_files[1]['GOLDEN']
        assert not out_files[0]["GOLDEN"]
        assert not out_files[2]["GOLDEN"]
        assert not out_files[3]["GOLDEN"]

        uvd = UVData()
        uvd.read(out_files[1]['GOLDEN'])


        # Read the Golden File
        golden_hd = io.HERAData(out_files[1]['GOLDEN'])
        gd, gf, gn = golden_hd.read()

        assert gd.shape[0] == 3  # ndays
        assert len(gd.antpairs()) == 6
        assert gd.shape[1] == uvds[0][0].freq_array.size
        assert len(gd.pols()) == 2

        assert len(gd.keys()) == 12

        # Check that autos are all the same
        assert np.all(gd[(0,0,'ee')] == gd[(1, 1,'ee')])
        assert np.all(gd[(0,0,'ee')] == gd[(2, 2,'ee')])

        # Since each day is at exactly the same LST by construction, the golden data
        # over time should be the same.
        np.testing.assert_allclose(gd.lsts, gd.lsts[0], atol=1e-6)

        for key, data in gd.items():
            for day in data:
                np.testing.assert_allclose(data[0], day, atol=1e-6)

        assert not np.allclose(gd[(0, 1, 'ee')][0], gd[(0, 2, 'ee')][0])
        assert not np.allclose(gd[(1, 2, 'ee')][0], gd[(0, 2, 'ee')][0])
        assert not np.allclose(gd[(1, 2, 'ee')][0], gd[(0, 1, 'ee')][0])


    def test_save_chans(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("lstbin_golden_data")
        uvds = mockuvd.make_dataset(ndays=3, nfiles=4, ntimes=2, identifiable=True, creator=create_small_array_uvd)
        data_files = mockuvd.write_files_in_hera_format(uvds, tmp)

        cfl = tmp / "lstbin_config_file.yaml"
        lstbin_simple.make_lst_bin_config_file(
            cfl, data_files, ntimes_per_file=2,
        )

        out_files = lstbin_simple.lst_bin_files(
            config_file=cfl, save_channels=[50], rephase=False
        )

        assert len(out_files) == 4
        # Ensure there's a REDUCEDCHAN file for each output LST
        for fl in out_files:
            assert fl['REDUCEDCHAN']

            # Read the Golden File
            hd = io.HERAData(fl['REDUCEDCHAN'])
            gd, gf, gn = hd.read()

            assert gd.shape[0] == 3  # ndays
            assert len(gd.antpairs()) == 6
            assert gd.shape[1] == 1  # single frequency
            assert len(gd.pols()) == 2

            assert len(gd.keys()) == 12

            # Check that autos are all the same
            assert np.all(gd[(0,0,'ee')] == gd[(1, 1,'ee')])
            assert np.all(gd[(0,0,'ee')] == gd[(2, 2,'ee')])

            # Since each day is at exactly the same LST by construction, the golden data
            # over time should be the same.
            for key, data in gd.items():
                for day in data:
                    np.testing.assert_allclose(data[0], day, rtol=1e-6)

            assert not np.allclose(gd[(0, 1, 'ee')][0], gd[(0, 2, 'ee')][0])
            assert not np.allclose(gd[(1, 2, 'ee')][0], gd[(0, 2, 'ee')][0])
            assert not np.allclose(gd[(1, 2, 'ee')][0], gd[(0, 1, 'ee')][0])

    def test_baseline_chunking(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("baseline_chunking")
        uvds = mockuvd.make_dataset(
            ndays=3, nfiles=4, ntimes=2,
            creator=mockuvd.create_uvd_identifiable,
            antpairs = [(i,j) for i in range(10) for j in range(i, 10)],  # 55 antpairs
            pols = ['xx', 'yy'],
            freqs=np.linspace(140e6, 180e6, 12),
        )
        data_files = mockuvd.write_files_in_hera_format(uvds, tmp)

        cfl = tmp / "lstbin_config_file.yaml"
        config_info = lstbin_simple.make_lst_bin_config_file(
            cfl, data_files, ntimes_per_file=2,
        )

        out_files = lstbin_simple.lst_bin_files(
            config_file=cfl, fname_format="zen.{kind}.{lst:7.5f}.uvh5",
            write_med_mad=True
        )
        out_files_chunked = lstbin_simple.lst_bin_files(
            config_file=cfl, fname_format="zen.{kind}.{lst:7.5f}.chunked.uvh5",
            Nbls_to_load=10,
            write_med_mad=True
        )

        for flset, flsetc in zip(out_files, out_files_chunked):
            assert flset[('LST', False)] != flsetc[('LST', False)]
            uvdlst = UVData()
            uvdlst.read(flset[('LST', False)])

            uvdlstc = UVData()
            uvdlstc.read(flsetc[('LST', False)])

            assert uvdlst == uvdlstc

            assert flset[('MED', False)] != flsetc[('MED', False)]
            uvdlst = UVData()
            uvdlst.read(flset[('MED', False)])

            uvdlstc = UVData()
            uvdlstc.read(flsetc[('MED', False)])

            assert uvdlst == uvdlstc
             
    def test_compare_nontrivial_cal(
        self, tmp_path_factory
    ):
        tmp = tmp_path_factory.mktemp("nontrivial_cal")
        uvds = mockuvd.make_dataset(
            ndays=3, nfiles=4, ntimes=2,
            creator=mockuvd.create_uvd_identifiable,
            antpairs = [(i,j) for i in range(7) for j in range(i, 7)],  # 55 antpairs
            pols = ('xx', 'yy'),
            freqs=np.linspace(140e6, 180e6, 3),
        )
        uvcs = [
            [mockuvd.make_uvc_identifiable(d) for d in uvd ] for uvd in uvds
        ]

        for night in uvds:
            print([np.unique(night[0].lst_array)])

        data_files = mockuvd.write_files_in_hera_format(uvds, tmp)
        cal_files = mockuvd.write_cals_in_hera_format(uvcs, tmp)
        decal_files = [[df.replace(".uvh5", ".decal.uvh5") for df in dfl] for dfl in data_files]

        for flist, clist, ulist in zip(data_files, cal_files, decal_files):
            for df, cf, uf in zip(flist, clist, ulist):
                apply_cal.apply_cal(
                    df, uf, cf,
                    gain_convention='divide',  # go the wrong way
                    clobber=True,
                )

        # First, let's go the other way to check if we get the same thing back directly
        recaled_files = [[df.replace(".uvh5", ".recal.uvh5") for df in dfl] for dfl in data_files]
        for flist, clist, ulist in zip(recaled_files, cal_files, decal_files):
            for df, cf, uf in zip(flist, clist, ulist):
                apply_cal.apply_cal(
                    uf, df, cf,
                    gain_convention='multiply',  # go the wrong way
                    clobber=True,
                )

        for flset, flsetc in zip(data_files, recaled_files):
            for fl, flc in zip(flset, flsetc):
                uvdlst = UVData()
                uvdlst.read(fl)

                uvdlstc = UVData()
                uvdlstc.read(flc)
                np.testing.assert_allclose(uvdlst.data_array, uvdlstc.data_array)

        cfl = tmp / "lstbin_config_file.yaml"
        lstbin_simple.make_lst_bin_config_file(
            cfl, decal_files, ntimes_per_file=2,
        )

        out_files_recal = lstbin_simple.lst_bin_files(
            config_file=cfl, calfile_rules=[(".decal.uvh5", ".calfits")],
            fname_format="zen.{kind}.{lst:7.5f}.recal.uvh5",
            Nbls_to_load=10,
            rephase=False
        )

        lstbin_simple.make_lst_bin_config_file(
            cfl, data_files, ntimes_per_file=2, clobber=True,
        )
        out_files = lstbin_simple.lst_bin_files(
            config_file=cfl, fname_format="zen.{kind}.{lst:7.5f}.uvh5",
            Nbls_to_load=11,
            rephase=False
        )

        for flset, flsetc in zip(out_files, out_files_recal):
            assert flset[('LST', False)] != flsetc[('LST', False)]
            uvdlst = UVData()
            uvdlst.read(flset[('LST', False)])

            uvdlstc = UVData()
            uvdlstc.read(flsetc[('LST', False)])
            print(np.unique(uvdlstc.lst_array))
            expected = mockuvd.identifiable_data_from_uvd(uvdlst, reshape=False)

            strpols = utils.polnum2str(uvdlst.polarization_array)
            for i, ap in enumerate(uvdlst.get_antpairs()):
                for j, pol in enumerate(strpols):
                    print(f"Baseline {ap + (pol,)}")

                    # We only check where the flags are False, because
                    # when we put in flags, we end up setting the data to nan (and
                    # never using it...)
                    np.testing.assert_allclose(
                        np.where(uvdlst.get_flags(ap+(pol,)), 1.0, uvdlstc.get_data(ap+(pol,))),
                        np.where(uvdlst.get_flags(ap+(pol,)), 1.0, expected[i, :, :, j]),
                        rtol=1e-4
                    )

            # Don't worry about history here, because we know they use different inputs
            uvdlst.history = uvdlstc.history
            assert uvdlst == uvdlstc

    @pytest.mark.parametrize(
        "random_ants_to_drop, rephase, sigma_clip_thresh, flag_strategy, pols, freq_range",
        [
            (0, True, 0.0, (0,0,0), ('xx', 'yy'), None),
            (0, True, 0.0, (0,0,0), ('xx', 'yy', 'xy', 'yx'), None),
            (0, True, 0.0, (0,0,0), ('xx', 'yy'), (150e6, 180e6)),
            (0, True, 0.0, (2,1,3), ('xx', 'yy'), None),
            (0, True, 3.0, (0,0,0), ('xx', 'yy'), None),
            (0, False, 0.0, (0,0,0), ('xx', 'yy'), None),
            (3, True, 0.0, (0,0,0), ('xx', 'yy'), None),
        ]
    )
    def test_nontrivial_cal(
        self, tmp_path_factory, random_ants_to_drop: int, rephase: bool,
        sigma_clip_thresh: float, flag_strategy: tuple[int, int, int],
        pols: tuple[str], freq_range: tuple[float, float] | None,
        benchmark
    ):

        tmp = tmp_path_factory.mktemp("nontrivial_cal")
        uvds = mockuvd.make_dataset(
            ndays=3, nfiles=2, ntimes=2, ants=np.arange(7),
            creator=mockuvd.create_uvd_identifiable,
            pols = pols,
            freqs=np.linspace(140e6, 180e6, 3),
            random_ants_to_drop=random_ants_to_drop,
        )

        uvcs = [
            [mockuvd.make_uvc_identifiable(d, *flag_strategy) for d in uvd ] for uvd in uvds
        ]

        data_files = mockuvd.write_files_in_hera_format(uvds, tmp)
        cal_files = mockuvd.write_cals_in_hera_format(uvcs, tmp)
        decal_files = [[df.replace(".uvh5", ".decal.uvh5") for df in dfl] for dfl in data_files]

        for flist, clist, ulist in zip(data_files, cal_files, decal_files):
            for df, cf, uf in zip(flist, clist, ulist):
                apply_cal.apply_cal(
                    df, uf, cf,
                    gain_convention='divide',  # go the wrong way
                    clobber=True,
                )

        cfl = tmp / "lstbin_config_file.yaml"
        lstbin_simple.make_lst_bin_config_file(
            cfl, data_files, ntimes_per_file=2, clobber=True,
        )
        out_files = benchmark(
            lstbin_simple.lst_bin_files,
            config_file=cfl, fname_format="zen.{kind}.{lst:7.5f}.uvh5",
            Nbls_to_load=11, rephase=rephase,
            sigma_clip_thresh=sigma_clip_thresh,
            sigma_clip_min_N=2,
            freq_min=freq_range[0] if freq_range is not None else None,
            freq_max=freq_range[1] if freq_range is not None else None,
            overwrite=True
        )
        assert len(out_files) == 2
        for flset in out_files:
            uvdlst = UVData()
            uvdlst.read(flset[('LST', False)])

            # Don't worry about history here, because we know they use different inputs
            expected = mockuvd.identifiable_data_from_uvd(uvdlst, reshape=False)

            strpols = utils.polnum2str(uvdlst.polarization_array)
            for i, ap in enumerate(uvdlst.get_antpairs()):
                for j, pol in enumerate(strpols):
                    print(f"Baseline {ap + (pol,)}")

                    # Unfortunately, we don't have LSTs for the files that exactly align
                    # with bin centres, so some rephasing will happen -- we just have to
                    # live with it and change the tolerance
                    # Furthermore, we only check where the flags are False, because
                    # when we put in flags, we end up setting the data to 1.0 (and
                    # never using it...)
                    print(uvdlst.get_data(ap))
                    print(expected[i])
                    np.testing.assert_allclose(
                        np.where(uvdlst.get_flags(ap+(pol,)), 1.0, uvdlst.get_data(ap+(pol,))),
                        np.where(uvdlst.get_flags(ap+(pol,)), 1.0, expected[i, :, :, j]),
                        rtol=1e-4 if (not rephase or (ap[0] == ap[1] and pol[0]==pol[1])) else 1e-3
                    )


    @pytest.mark.parametrize("tell_it", (True, False))
    def test_redundantly_averaged(
        self, tmp_path_factory, tell_it
    ):

        tmp = tmp_path_factory.mktemp("nontrivial_cal")
        uvds = mockuvd.make_dataset(
            ndays=3, nfiles=2, ntimes=2,
            creator=mockuvd.create_uvd_identifiable,
            antpairs = [(i,j) for i in range(7) for j in range(i, 7)],  # 55 antpairs
            pols = ('xx', 'yx'),
            freqs=np.linspace(140e6, 180e6, 3),
            redundantly_averaged=True,
        )

        data_files = mockuvd.write_files_in_hera_format(uvds, tmp)

        cfl = tmp / "lstbin_config_file.yaml"
        config_info = lstbin_simple.make_lst_bin_config_file(
            cfl, data_files, ntimes_per_file=2, clobber=True,
        )
        out_files = lstbin_simple.lst_bin_files(
            config_file=cfl, fname_format="zen.{kind}.{lst:7.5f}.uvh5",
            Nbls_to_load=11, rephase=False,
            sigma_clip_thresh=0.0,
            sigma_clip_min_N=2,
            redundantly_averaged=True if tell_it else None,
        )

        assert len(out_files) == 2

        for flset in out_files:
            uvdlst = UVData()
            uvdlst.read(flset[('LST', False)])

            # Don't worry about history here, because we know they use different inputs
            expected = mockuvd.identifiable_data_from_uvd(uvdlst, reshape=False)

            strpols = utils.polnum2str(uvdlst.polarization_array)

            for i, ap in enumerate(uvdlst.get_antpairs()):
                for j, pol in enumerate(strpols):
                    print(f"Baseline {ap + (pol,)}")

                    # Unfortunately, we don't have LSTs for the files that exactly align
                    # with bin centres, so some rephasing will happen -- we just have to
                    # live with it and change the tolerance
                    # Furthermore, we only check where the flags are False, because
                    # when we put in flags, we end up setting the data to 1.0 (and
                    # never using it...)
                    np.testing.assert_allclose(
                        uvdlst.get_data(ap+(pol,)),
                        np.where(uvdlst.get_flags(ap+(pol,)), 1.0, expected[i, :, :, j]),
                        rtol=1e-4
                    )

    def test_output_file_select(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("output_file_select")
        uvds = mockuvd.make_dataset(
            ndays=3, nfiles=4, ntimes=2,
            creator=mockuvd.create_uvd_identifiable,
            antpairs = [(i,j) for i in range(4) for j in range(i, 4)],  # 55 antpairs
            pols = ('xx', 'yx'),
            freqs=np.linspace(140e6, 180e6, 3),
        )

        data_files = mockuvd.write_files_in_hera_format(uvds, tmp)

        cfl = tmp / "lstbin_config_file.yaml"
        config_info = lstbin_simple.make_lst_bin_config_file(
            cfl, data_files, ntimes_per_file=2, clobber=True,
        )
        lstbf = partial(
            lstbin_simple.lst_bin_files, config_file=cfl, 
            fname_format="zen.{kind}.{lst:7.5f}.uvh5",
            Nbls_to_load=11, rephase=False,
        )

        out_files = lstbf(output_file_select=0)
        assert len(out_files) == 1


        out_files = lstbf(output_file_select=(1, 2))
        assert len(out_files) == 2

        with pytest.raises(ValueError, match='output_file_select must be less than the number of output files'):
            lstbf(output_file_select=100)


    def test_inpaint_mode_no_flags(self, tmp_path_factory):
        """Test that using inpaint mode does nothing when there's no flags."""
        tmp = tmp_path_factory.mktemp("inpaint_no_flags")
        uvds = mockuvd.make_dataset(
            ndays=3, nfiles=1, ntimes=2,
            creator=mockuvd.create_uvd_identifiable,
            antpairs = [(i,j) for i in range(3) for j in range(i, 3)],  # 55 antpairs
            pols = ('xx', 'yx'),
            freqs=np.linspace(140e6, 180e6, 3),
            redundantly_averaged=True,
        )

        data_files = mockuvd.write_files_in_hera_format(uvds, tmp)

        cfl = tmp / "lstbin_config_file.yaml"
        config_info = lstbin_simple.make_lst_bin_config_file(
            cfl, data_files, ntimes_per_file=2, clobber=True,
        )
        out_files = lstbin_simple.lst_bin_files(
            config_file=cfl, fname_format="zen.{kind}.{lst:7.5f}{inpaint_mode}.uvh5",
            rephase=False,
            sigma_clip_thresh=None,
            sigma_clip_min_N=2,
            output_flagged=True,
            output_inpainted=True
        )

        assert len(out_files) == 1

        for flset in out_files:
            flagged = UVData.from_file(flset[('LST', False)])
            inpainted = UVData.from_file(flset[('LST', True)])

            assert flagged == inpainted
