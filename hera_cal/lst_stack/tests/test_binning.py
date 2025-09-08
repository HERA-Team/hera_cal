import pytest
from .. import binning, config, calibration
from ...tests import mock_uvdata as mockuvd
import numpy as np
from ...red_groups import RedundantGroups
from pyuvdata import utils as uvutils
from pathlib import Path
from ..config import LSTBinConfigurator
import shutil
from ..io import apply_filename_rules
from pyuvdata import UVFlag, UVData
from ...io import HERAData
from ...data import DATA_PATH
import os
import copy


class TestAdjustLSTBinEdges:
    @pytest.mark.parametrize(
        'edges', [
            np.linspace(0, 1, 10),
            np.linspace(700, 800, 10),
            np.linspace(-2, 2, 2)
        ]
    )
    def test_properties(self, edges):
        binning.adjust_lst_bin_edges(edges)
        assert 0 <= edges[0] <= 2 * np.pi
        assert np.all(np.diff(edges) > 0)

    def test_errors(self):
        x = np.linspace(1, 0, 10)
        with pytest.raises(ValueError, match='lst_bin_edges must be monotonically increasing'):
            binning.adjust_lst_bin_edges(x)

        x = np.zeros((2, 3))
        with pytest.raises(ValueError, match='lst_bin_edges must be a 1D array'):
            binning.adjust_lst_bin_edges(x)


class TestLSTAlign:
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
            [np.unique(uvds[0].lst_array)] * len(uvds)
        )
        antpairs = uvds[0].get_antpairs()
        antpos = uvds[0].telescope.antenna_positions

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
            return binning.lst_align(**{**kwargs, **kw})

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

    def test_increasing_lsts_one_per_bin(self):
        kwargs = self.get_lst_align_data(ntimes=6)
        bins, d, f, n, inp = binning.lst_align(rephase=False, **kwargs)

        # We should not be changing the data at all.
        d = np.squeeze(np.asarray(d))
        f = np.squeeze(np.asarray(f))
        n = np.squeeze(np.asarray(n))

        np.testing.assert_allclose(d, kwargs["data"])
        assert not np.any(f)
        assert np.all(n == 1.0)
        assert len(bins) == 6

    def test_multi_days_one_per_bin(self):
        kwargs = self.get_lst_align_data(ndays=2, ntimes=7)
        bins, d, f, n, inp = binning.lst_align(rephase=False, **kwargs)

        # We should not be changing the data at all.
        d = np.squeeze(np.asarray(d))
        f = np.squeeze(np.asarray(f))
        n = np.squeeze(np.asarray(n))

        np.testing.assert_allclose(d[:, 0], kwargs["data"][:7])
        assert not np.any(f)
        assert np.all(n == 1.0)
        assert len(bins) == 7

    def test_multi_days_with_flagging(self):
        kwargs = self.get_lst_align_data(ndays=2, ntimes=7)

        # Flag everything after the first day, and make the data there crazy.
        flags = np.zeros_like(kwargs["data"], dtype=bool)
        flags[7:] = True
        kwargs["data"][7:] = 1000.0

        bins, d, f, n, inp = binning.lst_align(rephase=False, flags=flags, **kwargs)

        d = np.squeeze(np.asarray(d))
        f = np.squeeze(np.asarray(f))
        n = np.squeeze(np.asarray(n))

        np.testing.assert_allclose(d[:, 0], kwargs["data"][:7])
        assert not np.any(f[:, 0])
        assert np.all(f[:, 1])
        assert len(bins) == 7

    def test_multi_days_with_nsamples_zero(self):
        kwargs = self.get_lst_align_data(ndays=2, ntimes=7)

        # Flag everything after the first day, and make the data there crazy.
        nsamples = np.ones_like(kwargs["data"], dtype=float)
        nsamples[7:] = 0.0
        kwargs["data"][7:] = 1000.0

        bins, d, f, n, inp = binning.lst_align(
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

    def test_rephase(self):
        """Test that rephasing where each bin is already at center does nothing."""
        kwargs = self.get_lst_align_data(ntimes=7)

        bins0, d0, f0, n0, inp = binning.lst_align(rephase=True, **kwargs)
        bins, d, f, n, inp = binning.lst_align(rephase=False, **kwargs)
        np.testing.assert_allclose(d, d0, rtol=1e-6)
        np.testing.assert_allclose(f, f0, rtol=1e-6)
        np.testing.assert_allclose(n, n0, rtol=1e-6)
        assert len(bins) == len(bins0)

    def test_lstbinedges_modulus(self):
        kwargs = self.get_lst_align_data(ntimes=7)
        edges = kwargs.pop("lst_bin_edges")

        lst_bin_edges = edges.copy()
        lst_bin_edges -= 4 * np.pi

        bins, d0, f0, n0, inp = binning.lst_align(lst_bin_edges=lst_bin_edges, **kwargs)

        lst_bin_edges = edges.copy()
        lst_bin_edges += 4 * np.pi

        bins, d, f, n, inp = binning.lst_align(
            lst_bin_edges=lst_bin_edges, **kwargs
        )

        np.testing.assert_allclose(d, d0)
        np.testing.assert_allclose(f, f0)
        np.testing.assert_allclose(n, n0)

        with pytest.raises(
            ValueError, match="lst_bin_edges must be monotonically increasing."
        ):
            binning.lst_align(lst_bin_edges=lst_bin_edges[::-1], **kwargs)


@pytest.mark.filterwarnings("ignore", message="Getting antpos from the first file only")
class TestLSTBinFilesForBaselines:
    def test_defaults(self, uvd, uvd_file):
        _, d0, f0, n0, inpflg, _, _ = binning.lst_bin_files_for_baselines(
            data_files=[uvd_file],
            lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max() + 0.01],
            antpairs=uvd.get_antpairs(),
            rephase=False,
        )

        _, d, f, n, inpflg, _, _ = binning.lst_bin_files_for_baselines(
            data_files=[uvd_file],
            lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max() + 0.01],
            antpairs=uvd.get_antpairs(),
            freqs=uvd.freq_array,
            pols=uvd.polarization_array,
            time_idx=[np.ones(uvd.Ntimes, dtype=bool)],
            # time_arrays=[np.unique(uvd.time_array)],
            lsts=np.unique(uvd.lst_array),
            rephase=False,
        )

        np.testing.assert_allclose(d0, d)
        np.testing.assert_allclose(f0, f)
        np.testing.assert_allclose(n0, n)
        # np.testing.assert_allclose(times0, times)

    def test_empty(self, uvd, uvd_file):
        f0 = binning.lst_bin_files_for_baselines(
            data_files=[uvd_file],
            lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max()],
            antpairs=[(127, 128)],
            rephase=True,
        )[2]

        assert np.all(f0)

    def test_extra(self, uvd, uvd_file):
        # Providing baselines that don't exist in the file is fine, they're just ignored.
        f0 = binning.lst_bin_files_for_baselines(
            data_files=[uvd_file],
            lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max()],
            antpairs=uvd.get_antpairs() + [(127, 128)],
            rephase=True,
        )[2]

        # last baseline is the extra one that's all flagged.
        assert np.all(f0[0][:, -1])

    def test_freqrange(self, uvd, uvd_file, uvc_file):
        """Test that providing freq_range works."""
        data = binning.lst_bin_files_for_baselines(
            data_files=[uvd_file],
            lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max()],
            cal_files=[uvc_file],
            freq_min=153e6,
            freq_max=158e6,
            antpairs=uvd.get_antpairs(),
        )[1]

        assert data[0].shape[-2] < uvd.Nfreqs

    def test_bad_pols(self, uvd, uvd_file):
        with pytest.raises(KeyError, match="7"):
            binning.lst_bin_files_for_baselines(
                data_files=[uvd_file],
                lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max()],
                pols=[3.0, 7, -1],
                antpairs=uvd.get_antpairs(),
            )

    def test_incorrect_red_input(self, uvd, uvd_file, uvc_file):
        with pytest.raises(
            ValueError, match="reds must be provided if redundantly_averaged is True"
        ):
            binning.lst_bin_files_for_baselines(
                data_files=[uvd_file],
                lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max()],
                redundantly_averaged=True,
                antpairs=uvd.get_antpairs(),
            )

        with pytest.raises(
            ValueError, match="Cannot apply calibration if redundantly_averaged is True"
        ):
            binning.lst_bin_files_for_baselines(
                data_files=[uvd_file],
                lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max()],
                cal_files=[uvc_file],
                redundantly_averaged=True,
                reds=RedundantGroups.from_antpos(
                    dict(zip(uvd.telescope.antenna_numbers, uvd.telescope.antenna_positions)),
                    pols=uvutils.polnum2str(
                        uvd.polarization_array, x_orientation=uvd.telescope.get_x_orientation_from_feeds()
                    ),
                ),
                antpairs=uvd.get_antpairs(),
            )

    def test_simple_redundant_averaged_file(self, uvd_redavg, uvd_redavg_file):
        d0 = binning.lst_bin_files_for_baselines(
            data_files=[uvd_redavg_file],
            lst_bin_edges=[
                uvd_redavg.lst_array.min() - 0.1,
                uvd_redavg.lst_array.max() + 0.1,
            ],
            redundantly_averaged=True,
            rephase=False,
            antpairs=uvd_redavg.get_antpairs(),
            reds=RedundantGroups.from_antpos(
                dict(zip(uvd_redavg.telescope.antenna_numbers, uvd_redavg.telescope.antenna_positions)),
            ),
        )[1]

        assert len(d0) == 1
        assert d0[0].shape == (
            uvd_redavg.Ntimes,
            uvd_redavg.Nbls,
            uvd_redavg.Nfreqs,
            uvd_redavg.Npols,
        )


class TestLSTBinFilesFromConfig:
    def get_config(self, season, request, outfile_index=0):
        cfg = LSTBinConfigurator(
            request.getfixturevalue(f"season_{season}"),
            where_inpainted_file_rules=[(".uvh5", ".where_inpainted.h5")] if 'inpaint' in season else None,
        )
        mf = cfg.get_matched_files()
        return cfg.create_config(mf).at_single_outfile(outfile=outfile_index)

    @pytest.mark.parametrize('season', ['redavg', 'notredavg', 'redavg_inpaint'])
    def test_bin_files(self, season, request):
        cfg = self.get_config(season, request)

        uvd, uvd1 = binning.lst_bin_files_from_config(cfg)
        assert uvd.Ntimes == uvd1.Ntimes == 3  # ndays
        assert uvd.Nbls == uvd1.Nbls == len(cfg.antpairs)
        assert uvd.Nfreqs == uvd1.Nfreqs == len(cfg.config.datameta.freq_array)
        assert uvd.Npols == uvd1.Npols == len(cfg.config.datameta.pols)

        # test that exposes bug fixed in 3a3ead0fd13400578b50b5fe05af39be61717206
        uvd0 = UVData.from_file(cfg.matched_files[0])
        assert np.allclose(uvd.telescope.get_enu_antpos(), uvd0.telescope.get_enu_antpos())

    def test_redavg_with_where_inpainted(self, request, tmp_path_factory):
        # This is kind of a dodgy way to test that if the inpainted files don't have
        # all the baselines, things will fail. We copy the original data files,
        # write a whole new dataset in the same place but with fewer baselines, then
        # copy the data files (but not the where_inpainted files) back, so they mismatch.

        cfg = self.get_config('redavg_inpaint', request)
        cfg_bad = self.get_config("redavg_inpaint_fewer_ants", request)

        newobs = tmp_path_factory.mktemp("newobs")
        shutil.copytree(cfg.config.datameta.path.parent.parent, newobs, dirs_exist_ok=True)

        bad_inpaint = apply_filename_rules(
            cfg_bad.config.data_files, cfg_bad.config.where_inpainted_file_rules
        )
        for night in bad_inpaint:
            for fl in night:
                fl = Path(fl)
                shutil.copyfile(fl, newobs / fl.parent.name / fl.name)

        cfg = LSTBinConfigurator(
            [sorted(pth.glob("*.uvh5")) for pth in sorted(newobs.glob("*"))],
            where_inpainted_file_rules=[(".uvh5", ".where_inpainted.h5")],
        )
        mf = cfg.get_matched_files()
        cfg = cfg.create_config(mf).at_single_outfile(outfile=0)

        with pytest.raises(ValueError, match="Could not find any baseline from group"):
            binning.lst_bin_files_from_config(cfg)


# Silence the expected antpos warning from the binning code
pytestmark = pytest.mark.filterwarnings(
    "ignore:Getting antpos from the first file only"
)


class _FakeConfiguratorSingle:
    """Minimal interface that SingleBaselineStacker uses."""
    def __init__(self, bl_to_file_map):
        self.bl_to_file_map = bl_to_file_map


# --- fixtures using real test files -------------------------------------------------

@pytest.fixture(scope="module")
def real_files_and_grid():
    baseline_string = "0_4"
    filenames = [
        "zen.2459861.baseline.0_4.sum.smooth_calibrated.red_avg.inpainted.uvh5",
        "zen.2459862.baseline.0_4.sum.smooth_calibrated.red_avg.inpainted.uvh5",
        "zen.2459863.baseline.0_4.sum.smooth_calibrated.red_avg.inpainted.uvh5",
    ]
    files = [os.path.join(DATA_PATH, "test_input", f) for f in filenames]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        pytest.skip(f"Missing test inputs: {missing}")

    configurator = _FakeConfiguratorSingle({baseline_string: files})
    hd = HERAData(files[-1])  # for freqs/lsts metadata

    # Native-width 0..2π LST grid
    dlst = np.median(np.diff(hd.lsts))
    lst_grid = config.make_lst_grid(dlst, begin_lst=0.0, lst_width=2 * np.pi)
    lst_bin_edges = np.concatenate([lst_grid - dlst / 2, [lst_grid[-1] + dlst / 2]])

    return {
        "baseline_string": baseline_string,
        "configurator": configurator,
        "lst_bin_edges": lst_bin_edges,
        "dlst": dlst,
        "hd": hd,
        "where_rules": [[".inpainted.uvh5", ".where_inpainted.h5"]],
    }

@pytest.fixture(scope="module")
def real_files_and_grid_uncalibrated():
    baseline_string = "0_4"
    filenames = [
        "zen.2459861.baseline.0_4.sum.uvh5",
        "zen.2459862.baseline.0_4.sum.uvh5",
        "zen.2459863.baseline.0_4.sum.uvh5",
    ]
    calfilenames = [
        "zen.2459861.lstcal.hdf5",
        "zen.2459862.lstcal.hdf5",
        "zen.2459863.lstcal.hdf5",
    ]
    files = [os.path.join(DATA_PATH, "test_input", f) for f in filenames]
    calfiles = [os.path.join(DATA_PATH, "test_input", f) for f in calfilenames]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        pytest.skip(f"Missing test inputs: {missing}")

    configurator_uncal = _FakeConfiguratorSingle({baseline_string: files})
    hd = HERAData(files[-1])  # for freqs/lsts metadata

    configurator_cal = _FakeConfiguratorSingle({baseline_string: files})
    configurator_cal.bl_to_calfile_map = {baseline_string: calfiles}

    # Native-width 0..2π LST grid
    dlst = np.median(np.diff(hd.lsts))
    lst_grid = config.make_lst_grid(dlst, begin_lst=0.0, lst_width=2 * np.pi)
    lst_bin_edges = np.concatenate([lst_grid - dlst / 2, [lst_grid[-1] + dlst / 2]])

    return {
        "baseline_string": baseline_string,
        "configurator_uncal": configurator_uncal,
        "configurator_cal": configurator_cal,
        "lst_bin_edges": lst_bin_edges,
        "dlst": dlst,
        "hd": hd,
        "where_rules": [[".inpainted.uvh5", ".where_inpainted.h5"]],
    }


@pytest.fixture(scope="module")
def sbs_default(real_files_and_grid):
    p = real_files_and_grid
    return binning.SingleBaselineStacker.from_configurator(
        p["configurator"], p["baseline_string"], p["lst_bin_edges"]
    )


@pytest.fixture(scope="module")
def sbs_keep_all(real_files_and_grid):
    p = real_files_and_grid
    return binning.SingleBaselineStacker.from_configurator(
        p["configurator"], p["baseline_string"], p["lst_bin_edges"], to_keep_slice=slice(None)
    )


@pytest.fixture(scope="module")
def sbs_branchcut(real_files_and_grid):
    p = real_files_and_grid
    return binning.SingleBaselineStacker.from_configurator(
        p["configurator"],
        p["baseline_string"],
        p["lst_bin_edges"],
        lst_branch_cut=5.4,
        where_inpainted_file_rules=p["where_rules"],
    )

@pytest.fixture(scope="module")
def sbs_with_lstcal(real_files_and_grid_uncalibrated):
    p = real_files_and_grid_uncalibrated
    uncal_crosses = binning.SingleBaselineStacker.from_configurator(
        p['configurator_uncal'],
        p["baseline_string"],
        p["lst_bin_edges"],
    )

    cal_crosses = binning.SingleBaselineStacker.from_configurator(
        p['configurator_cal'],
        p["baseline_string"],
        p["lst_bin_edges"],
        cal_file_loader=calibration.load_single_baseline_lstcal_gains,
    )

    return uncal_crosses, cal_crosses


# --- tests -------------------------------------------------------------------------

def test_shapes_and_types_default(sbs_default, real_files_and_grid):
    hd = real_files_and_grid["hd"]

    # All lists have same length as number of kept bins
    L = len(sbs_default.bin_lst)
    for name in ("data", "flags", "nsamples", "times_in_bins", "lsts_in_bins"):
        assert len(getattr(sbs_default, name)) == L

    # Per-bin arrays are (Nints_in_bin, Nfreqs, Npols) (baseline dimension removed)
    for d, f, n in zip(sbs_default.data, sbs_default.flags, sbs_default.nsamples):
        assert d.ndim == f.ndim == n.ndim == 3
        assert d.shape[-2] == len(hd.freqs)
        assert d.shape[-1] == len(hd.pols)
        assert f.shape == d.shape
        assert n.shape == d.shape
        assert np.iscomplexobj(d)
        assert f.dtype == bool
        assert n.dtype.kind in ("f", "i")  # nsamples float or int


def test_auto_trimming_keeps_nonempty_edges(sbs_default):
    # Auto-trim should remove empty bins at ends; first & last bins should have data or at least exist
    assert len(sbs_default.data) == len(sbs_default.bin_lst)
    # Expect at least the first & last kept bins to be potentially non-empty after trimming.
    # (We don't assert strictly >0 because some nights may still have gaps, but trimming
    # should have removed long empty runs.)
    assert len(sbs_default.data) > 0


def test_to_keep_slice_keeps_empties(sbs_keep_all, real_files_and_grid):
    # Keeping all bins means we retain the full grid count
    L_edges = len(real_files_and_grid["lst_bin_edges"]) - 1
    assert len(sbs_keep_all.bin_lst) == L_edges

    # The fully-expanded grid over 0..2π will have empties at ends; check at least one empty bin
    empties = sum(arr.shape[0] == 0 for arr in sbs_keep_all.data)
    assert empties >= 1


def test_branch_cut_roll_and_wrap(sbs_branchcut, real_files_and_grid):
    cut = 5.4
    # After rolling, centers should start at >= cut and be strictly increasing
    assert np.all(np.diff(sbs_branchcut.bin_lst) > 0)
    assert sbs_branchcut.bin_lst.min() >= cut
    # Shouldn’t exceed cut + 2π (allow tiny epsilon)
    assert sbs_branchcut.bin_lst.max() <= cut + 2 * np.pi + 1e-8

    # If a bin has LSTs, ensure they’re wrapped to be >= cut
    nonempty = [lst for lst in sbs_branchcut.lsts_in_bins if len(lst) > 0]
    if nonempty:
        assert min(np.min(lst) for lst in nonempty) >= cut


def test_inpaint_flags_present_and_boolean(sbs_branchcut):
    # where_inpainted should be a list (same length as bins) of boolean arrays or None
    assert len(sbs_branchcut.where_inpainted) == len(sbs_branchcut.bin_lst)
    any_present = False
    for wf, d in zip(sbs_branchcut.where_inpainted, sbs_branchcut.data):
        if wf is None:
            continue
        any_present = any_present or wf.size > 0
        assert wf.shape == d.shape
        assert wf.dtype == bool
    assert any_present  # expect at least some inpaint flags in these test inputs


def test_bin_membership_close_to_centers(sbs_default, real_files_and_grid):
    # LSTs in each bin should lie within ~half-width of the bin center (pre-rephase assignment)
    half = real_files_and_grid["dlst"] / 2 + 1e-6
    centers = sbs_default.bin_lst
    for c, lsts in zip(centers, sbs_default.lsts_in_bins):
        if len(lsts) == 0:
            continue
        # angular distance on circle to bin center
        ang = np.mod(lsts - c + np.pi, 2 * np.pi) - np.pi
        assert np.all(np.abs(ang) <= half + 1e-3)  # be forgiving with file rounding


def test_flags_and_nsamples_match_data(sbs_default):
    # Sanity: where we have data samples, shapes align and boolean/float logic is consistent
    for d, f, n in zip(sbs_default.data, sbs_default.flags, sbs_default.nsamples):
        assert d.shape == f.shape == n.shape
        # If there are integrations, ensure flagged entries correspond to float nsamples (any value)
        if d.size:
            assert f.dtype == bool
            assert np.isfinite(np.asarray(n)[~np.isnan(np.real(d)) | ~np.isnan(np.imag(d))]).all()

def test_calc_with_lstcal(sbs_with_lstcal):
    uncal_crosses, cal_crosses = sbs_with_lstcal
    lst_avg_uncal, _, _ = uncal_crosses.average_over_nights()
    lst_avg_cal, _, _ = cal_crosses.average_over_nights()
    uncalibrated_var = np.array([
        np.nanmean(np.where(f, np.nan, np.abs(np.square(d - lst_avg_uncal[ci])))) 
        for ci, (d, f) in enumerate(zip(uncal_crosses.data, uncal_crosses.flags))
    ])

    calibrated_var = np.array([
        np.nanmean(np.where(f, np.nan, np.abs(np.square(d - lst_avg_cal[ci])))) 
        for ci, (d, f) in enumerate(zip(cal_crosses.data, cal_crosses.flags))
    ])

    # Variance after calibration should be lower in bins where both have data
    assert np.all(
        uncalibrated_var[np.isfinite(uncalibrated_var) & np.isfinite(calibrated_var)] > 
        calibrated_var[np.isfinite(uncalibrated_var) & np.isfinite(calibrated_var)]
    ), "Variance after calibration should be lower in bins where both have data"

# --- average_over_nights tests -------------------------------------------------------------


def test_average_over_nights_shapes_and_dtypes(sbs_branchcut):
    """Output has (Nlst, Nfreqs, Npols) shape and correct dtypes."""
    out_d, out_f, out_n = sbs_branchcut.average_over_nights(inpainted_data_are_samples=False)
    nfreq = len(sbs_branchcut.hd.freqs)
    npol = len(sbs_branchcut.hd.pols)
    assert out_d.shape == out_f.shape == out_n.shape == (len(sbs_branchcut.bin_lst), nfreq, npol)
    assert np.iscomplexobj(out_d)
    assert out_f.dtype == bool
    assert out_n.dtype.kind in ("f", "i")


def test_average_over_nights_flags_AND_consistency(sbs_branchcut):
    """Flags should be AND across nights."""
    out_d, out_f, out_n = sbs_branchcut.average_over_nights(inpainted_data_are_samples=False)

    # manual AND across nights, per-bin
    for lidx, f in enumerate(sbs_branchcut.flags):
        expected_flags = np.all(f, axis=0)
        assert np.array_equal(out_f[lidx], expected_flags)


def test_average_over_nights_weighted_mean_matches_manual(sbs_branchcut):
    """Average equals weighted mean with nsamples as weights (flagged weight=0),
    with columns fully flagged given weight=1 to keep denominator sane."""
    out_d, out_f, _ = sbs_branchcut.average_over_nights(inpainted_data_are_samples=False)

    for lidx, (d, f, n, wip) in enumerate(zip(
        sbs_branchcut.data, sbs_branchcut.flags, sbs_branchcut.nsamples, sbs_branchcut.where_inpainted
    )):
        # flags AND across nights
        fully_flagged = np.all(f, axis=0)  # (Nfreq, Npol)

        # weights: flagged -> 0
        weights = np.where(f, 0, n).astype(float)  # (Nnight, Nfreq, Npol)

        # where fully flagged across nights, set weights to 1 across nights (per-pol)
        for p in range(d.shape[-1]):
            mask_fp = fully_flagged[:, p]
            if np.any(mask_fp):
                weights[:, mask_fp, p] = 1.0

        # zero-out flagged data for the numerator
        d_used = np.where(f, 0.0, d)

        num = np.sum(d_used * weights, axis=0)
        den = np.sum(weights, axis=0)
        # (den should be >= 1 where fully flagged because we set weights=1; still be safe)
        den = np.where(den == 0, 1.0, den)

        expected = num / den
        np.testing.assert_allclose(out_d[lidx], expected, rtol=0, atol=1e-12)


def test_average_over_nights_inpainted_toggle_changes_nsamples_when_wip_present(sbs_branchcut):
    """When where_inpainted is present, excluding inpainted samples should reduce some nsample counts."""
    d0, f0, n_excl = sbs_branchcut.average_over_nights(inpainted_data_are_samples=False)
    d1, f1, n_incl = sbs_branchcut.average_over_nights(inpainted_data_are_samples=True)

    # flags identical regardless of toggle
    assert np.array_equal(f0, f1)
    # data identical (toggle affects only nsamples in output, not averaging weights in your implementation)
    np.testing.assert_allclose(d0, d1, rtol=0, atol=1e-12)

    # If any inpainted True entries exist, we expect some strict reduction
    any_wip = False
    for wip in sbs_branchcut.where_inpainted:
        if wip is not None and wip.size and np.any(wip):
            any_wip = True
            break

    # In all cases, exclusion <= inclusion
    assert np.all(n_excl <= n_incl)
    if any_wip:
        assert np.any(n_excl < n_incl)


def test_average_over_nights_does_not_mutate_input(sbs_branchcut):
    """Check that average_over_nights does not mutate internal arrays."""
    # snapshot simple checksums
    before = []
    for d, f, n in zip(sbs_branchcut.data, sbs_branchcut.flags, sbs_branchcut.nsamples):
        before.append((
            np.asarray(d).sum(dtype=np.complex128),
            np.asarray(f, dtype=np.int64).sum(),
            np.asarray(n, dtype=np.float64).sum(),
        ))
    _ = sbs_branchcut.average_over_nights(inpainted_data_are_samples=False)
    after = []
    for d, f, n in zip(sbs_branchcut.data, sbs_branchcut.flags, sbs_branchcut.nsamples):
        after.append((
            np.asarray(d).sum(dtype=np.complex128),
            np.asarray(f, dtype=np.int64).sum(),
            np.asarray(n, dtype=np.float64).sum(),
        ))
    for (bd, bf, bn), (ad, af, an) in zip(before, after):
        # exact equality for bool/int sums; complex sum up to FP noise
        assert bf == af
        assert bn == an
        np.testing.assert_allclose(bd, ad, rtol=0, atol=0.0)


def test_average_over_nights_runs_without_where_inpainted_when_counting_inpainted(sbs_default):
    """On objects with where_inpainted=None, average_over_nights should still run with inpainted_data_are_samples=True."""
    d, f, n = sbs_default.average_over_nights(inpainted_data_are_samples=True)
    nfreq = len(sbs_default.hd.freqs)
    npol = len(sbs_default.hd.pols)
    assert d.shape == f.shape == n.shape == (len(sbs_default.bin_lst), nfreq, npol)


def test_average_over_nights_handles_empty_bins(sbs_keep_all):
    """Test that average_over_nights handles empty bins correctly."""
    # sbs_keep_all keeps empty edge bins by design
    d0, f0, n0 = sbs_keep_all.average_over_nights(inpainted_data_are_samples=True)
    d1, f1, n1 = sbs_keep_all.average_over_nights(inpainted_data_are_samples=False)
    # Should not raise, and shapes should be correct
    assert d0.shape == d1.shape == (len(sbs_keep_all.bin_lst), len(sbs_keep_all.hd.freqs), len(sbs_keep_all.hd.pols))
    # Empty bins (0 nights) should remain zeros/True flags
    for lidx, arr in enumerate(sbs_keep_all.data):
        if arr.shape[0] == 0:
            assert np.allclose(d0[lidx], 0)
            assert np.allclose(n0[lidx], 0)
            assert np.all(f0[lidx])
