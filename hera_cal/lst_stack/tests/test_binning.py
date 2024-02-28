import pytest
from .. import binning
from ...tests import mock_uvdata as mockuvd
import numpy as np
from ...red_groups import RedundantGroups
from pyuvdata import utils as uvutils
from pathlib import Path


@pytest.mark.filterwarnings("ignore", message="Getting antpos from the first file only")
class Test_LSTBinFilesForBaselines:
    def test_defaults(self, uvd, uvd_file):
        lstbins, d0, f0, n0, inpflg, times0 = binning.lst_bin_files_for_baselines(
            data_files=[uvd_file],
            lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max() + 0.01],
            antpairs=uvd.get_antpairs(),
            rephase=False,
        )

        lstbins, d, f, n, inpflg, times = binning.lst_bin_files_for_baselines(
            data_files=[uvd_file],
            lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max() + 0.01],
            antpairs=uvd.get_antpairs(),
            freqs=uvd.freq_array,
            pols=uvd.polarization_array,
            time_idx=[np.ones(uvd.Ntimes, dtype=bool)],
            time_arrays=[np.unique(uvd.time_array)],
            lsts=np.unique(uvd.lst_array),
            rephase=False,
        )

        np.testing.assert_allclose(d0, d)
        np.testing.assert_allclose(f0, f)
        np.testing.assert_allclose(n0, n)
        np.testing.assert_allclose(times0, times)

    def test_empty(self, uvd, uvd_file):
        lstbins, d0, f0, n0, inpflg, times0 = binning.lst_bin_files_for_baselines(
            data_files=[uvd_file],
            lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max()],
            antpairs=[(127, 128)],
            rephase=True,
        )

        assert np.all(f0)

    def test_extra(self, uvd, uvd_file):
        # Providing baselines that don't exist in the file is fine, they're just ignored.
        lstbins, d0, f0, n0, inpflg, times0 = binning.lst_bin_files_for_baselines(
            data_files=[uvd_file],
            lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max()],
            antpairs=uvd.get_antpairs() + [(127, 128)],
            rephase=True,
        )

        assert np.all(
            f0[0][:, -1]
        )  # last baseline is the extra one that's all flagged.

    def test_freqrange(self, uvd, uvd_file, uvc_file):
        """Test that providing freq_range works."""
        (
            bins,
            data,
            flags,
            nsamples,
            inpflg,
            times,
        ) = binning.lst_bin_files_for_baselines(
            data_files=[uvd_file],
            lst_bin_edges=[uvd.lst_array.min(), uvd.lst_array.max()],
            cal_files=[uvc_file],
            freq_min=153e6,
            freq_max=158e6,
            antpairs=uvd.get_antpairs(),
        )

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
                    dict(zip(uvd.antenna_numbers, uvd.antenna_positions)),
                    pols=uvutils.polnum2str(
                        uvd.polarization_array, x_orientation=uvd.x_orientation
                    ),
                ),
                antpairs=uvd.get_antpairs(),
            )

    def test_simple_redundant_averaged_file(self, uvd_redavg, uvd_redavg_file):
        lstbins, d0, f0, n0, inpflg, times0 = binning.lst_bin_files_for_baselines(
            data_files=[uvd_redavg_file],
            lst_bin_edges=[
                uvd_redavg.lst_array.min() - 0.1,
                uvd_redavg.lst_array.max() + 0.1,
            ],
            redundantly_averaged=True,
            rephase=False,
            antpairs=uvd_redavg.get_antpairs(),
            reds=RedundantGroups.from_antpos(
                dict(zip(uvd_redavg.antenna_numbers, uvd_redavg.antenna_positions)),
            ),
        )

        assert len(d0) == 1
        assert d0[0].shape == (
            uvd_redavg.Ntimes,
            uvd_redavg.Nbls,
            uvd_redavg.Nfreqs,
            uvd_redavg.Npols,
        )

    def test_redavg_with_where_inpainted(self, tmp_path):
        uvds = mockuvd.make_dataset(
            ndays=2,
            nfiles=3,
            ntimes=2,
            ants=np.arange(7),
            creator=mockuvd.create_uvd_identifiable,
            freqs=mockuvd.PHASEII_FREQS[:25],
            pols=['xx', 'xy'],
            redundantly_averaged=True,
        )

        uvd_files = mockuvd.write_files_in_hera_format(
            uvds, tmp_path, add_where_inpainted_files=True
        )

        ap = uvds[0][0].get_antpairs()
        reds = RedundantGroups.from_antpos(
            dict(zip(uvds[0][0].antenna_numbers, uvds[0][0].antenna_positions)),
        )
        lstbins, d0, f0, n0, inpflg, times0 = binning.lst_bin_files_for_baselines(
            data_files=sum(uvd_files, []),  # flatten the list-of-lists
            lst_bin_edges=[0, 1.9 * np.pi],
            redundantly_averaged=True,
            rephase=False,
            antpairs=ap,
            reds=reds,
            where_inpainted_files=[str(Path(f).with_suffix(".where_inpainted.h5")) for f in sum(uvd_files, [])],
        )
        assert len(lstbins) == 1

        # Also test that if a where_inpainted file has missing baselines, an error is
        # raised.
        # This is kind of a dodgy way to test it: copy the original data files,
        # write a whole new dataset in the same place but with fewer baselines, then
        # copy the data files (but not the where_inpainted files) back, so they mismatch.
        for flist in uvd_files:
            for fl in flist:
                fl = Path(fl)
                fl.rename(fl.parent / f"{fl.with_suffix('.bk')}")

                winp = fl.with_suffix(".where_inpainted.h5")
                winp.unlink()

        uvds = mockuvd.make_dataset(
            ndays=2,
            nfiles=3,
            ntimes=2,
            ants=np.arange(5),  # less than the original
            creator=mockuvd.create_uvd_identifiable,
            freqs=mockuvd.PHASEII_FREQS[:25],
            pols=['xx', 'xy'],
            redundantly_averaged=True,
        )

        uvd_files = mockuvd.write_files_in_hera_format(
            uvds, tmp_path, add_where_inpainted_files=True
        )

        # Move back the originals.
        for flist in uvd_files:
            for fl in flist:
                fl = Path(fl)
                fl.unlink()
                (fl.parent / f"{fl.with_suffix('.bk')}").rename(fl)

        with pytest.raises(ValueError, match="Could not find any baseline from group"):
            binning.lst_bin_files_for_baselines(
                data_files=sum(uvd_files, []),  # flatten the list-of-lists
                lst_bin_edges=[0, 1.9 * np.pi],
                redundantly_averaged=True,
                rephase=False,
                antpairs=ap,
                reds=reds,
                where_inpainted_files=[str(Path(f).with_suffix(".where_inpainted.h5")) for f in sum(uvd_files, [])],
            )
