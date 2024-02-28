from ...tests import mock_uvdata as mockuvd
from .conftest import create_small_array_uvd
from pyuvdata import UVData, UVFlag
import numpy as np
from ... import io
from ... import apply_cal
from pyuvdata import utils
import pytest
from functools import partial
from .. import wrappers
from ..io import apply_filename_rules


def test_argparser_returns():
    args = wrappers.lst_bin_arg_parser()
    assert args is not None


class Test_LSTBinFiles:
    def test_golden_data(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("lstbin_golden_data")
        uvds = mockuvd.make_dataset(
            ndays=3,
            nfiles=4,
            ntimes=2,
            identifiable=True,
            creator=create_small_array_uvd,
            time_axis_faster_than_bls=True,
        )
        data_files = mockuvd.write_files_in_hera_format(uvds, tmp)
        print(len(uvds))
        cfl = tmp / "lstbin_config_file.yaml"
        print(cfl)
        lstbin.make_lst_bin_config_file(
            cfl,
            data_files,
            ntimes_per_file=2,
        )

        out_files = wrappers.lst_bin_files(
            config_file=cfl,
            rephase=False,
            golden_lsts=uvds[0][1].lst_array.min() + 0.0001,
        )

        assert len(out_files) == 4
        assert out_files[1]["GOLDEN"]
        assert not out_files[0]["GOLDEN"]
        assert not out_files[2]["GOLDEN"]
        assert not out_files[3]["GOLDEN"]

        uvd = UVData()
        uvd.read(out_files[1]["GOLDEN"], use_future_array_shapes=True)

        # Read the Golden File
        golden_hd = io.HERAData(out_files[1]["GOLDEN"])
        gd, gf, gn = golden_hd.read()

        assert gd.shape[0] == 3  # ndays
        assert len(gd.antpairs()) == 6
        assert gd.shape[1] == uvds[0][0].freq_array.size
        assert len(gd.pols()) == 2

        assert len(gd.keys()) == 12

        # Check that autos are all the same
        assert np.all(gd[(0, 0, "ee")] == gd[(1, 1, "ee")])
        assert np.all(gd[(0, 0, "ee")] == gd[(2, 2, "ee")])

        # Since each day is at exactly the same LST by construction, the golden data
        # over time should be the same.
        np.testing.assert_allclose(gd.lsts, gd.lsts[0], atol=1e-6)

        for key, data in gd.items():
            for day in data:
                np.testing.assert_allclose(data[0], day, atol=1e-6)

        assert not np.allclose(gd[(0, 1, "ee")][0], gd[(0, 2, "ee")][0])
        assert not np.allclose(gd[(1, 2, "ee")][0], gd[(0, 2, "ee")][0])
        assert not np.allclose(gd[(1, 2, "ee")][0], gd[(0, 1, "ee")][0])

    def test_save_chans(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("lstbin_golden_data")
        uvds = mockuvd.make_dataset(
            ndays=3,
            nfiles=4,
            ntimes=2,
            identifiable=True,
            creator=create_small_array_uvd,
        )
        data_files = mockuvd.write_files_in_hera_format(uvds, tmp)

        cfl = tmp / "lstbin_config_file.yaml"
        lstbin.make_lst_bin_config_file(
            cfl,
            data_files,
            ntimes_per_file=2,
        )

        out_files = wrappers.lst_bin_files(
            config_file=cfl, save_channels=[50], rephase=False
        )

        assert len(out_files) == 4
        # Ensure there's a REDUCEDCHAN file for each output LST
        for fl in out_files:
            assert fl["REDUCEDCHAN"]

            # Read the Golden File
            hd = io.HERAData(fl["REDUCEDCHAN"])
            gd, gf, gn = hd.read()

            assert gd.shape[0] == 3  # ndays
            assert len(gd.antpairs()) == 6
            assert gd.shape[1] == 1  # single frequency
            assert len(gd.pols()) == 2

            assert len(gd.keys()) == 12

            # Check that autos are all the same
            assert np.all(gd[(0, 0, "ee")] == gd[(1, 1, "ee")])
            assert np.all(gd[(0, 0, "ee")] == gd[(2, 2, "ee")])

            # Since each day is at exactly the same LST by construction, the golden data
            # over time should be the same.
            for key, data in gd.items():
                for day in data:
                    np.testing.assert_allclose(data[0], day, rtol=1e-6)

            assert not np.allclose(gd[(0, 1, "ee")][0], gd[(0, 2, "ee")][0])
            assert not np.allclose(gd[(1, 2, "ee")][0], gd[(0, 2, "ee")][0])
            assert not np.allclose(gd[(1, 2, "ee")][0], gd[(0, 1, "ee")][0])

    def test_baseline_chunking(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("baseline_chunking")
        uvds = mockuvd.make_dataset(
            ndays=3,
            nfiles=4,
            ntimes=2,
            creator=mockuvd.create_uvd_identifiable,
            antpairs=[(i, j) for i in range(10) for j in range(i, 10)],  # 55 antpairs
            pols=["xx", "yy"],
            freqs=np.linspace(140e6, 180e6, 12),
        )
        data_files = mockuvd.write_files_in_hera_format(uvds, tmp)

        cfl = tmp / "lstbin_config_file.yaml"
        config_info = lstbin.make_lst_bin_config_file(
            cfl,
            data_files,
            ntimes_per_file=2,
        )

        out_files = wrappers.lst_bin_files(
            config_file=cfl,
            fname_format="zen.{kind}.{lst:7.5f}.uvh5",
            write_med_mad=True,
        )
        out_files_chunked = wrappers.lst_bin_files(
            config_file=cfl,
            fname_format="zen.{kind}.{lst:7.5f}.chunked.uvh5",
            Nbls_to_load=10,
            write_med_mad=True,
        )

        for flset, flsetc in zip(out_files, out_files_chunked):
            assert flset[("LST", False)] != flsetc[("LST", False)]
            uvdlst = UVData()
            uvdlst.read(flset[("LST", False)], use_future_array_shapes=True)

            uvdlstc = UVData()
            uvdlstc.read(flsetc[("LST", False)], use_future_array_shapes=True)

            assert uvdlst == uvdlstc

            assert flset[("MED", False)] != flsetc[("MED", False)]
            uvdlst = UVData()
            uvdlst.read(flset[("MED", False)], use_future_array_shapes=True)

            uvdlstc = UVData()
            uvdlstc.read(flsetc[("MED", False)], use_future_array_shapes=True)

            assert uvdlst == uvdlstc

    def test_compare_nontrivial_cal(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("nontrivial_cal")
        uvds = mockuvd.make_dataset(
            ndays=3,
            nfiles=4,
            ntimes=2,
            creator=mockuvd.create_uvd_identifiable,
            antpairs=[(i, j) for i in range(7) for j in range(i, 7)],  # 55 antpairs
            pols=("xx", "yy"),
            freqs=np.linspace(140e6, 180e6, 3),
        )
        uvcs = [[mockuvd.make_uvc_identifiable(d) for d in uvd] for uvd in uvds]

        for night in uvds:
            print([np.unique(night[0].lst_array)])

        data_files = mockuvd.write_files_in_hera_format(uvds, tmp)
        cal_files = mockuvd.write_cals_in_hera_format(uvcs, tmp)
        decal_files = [
            [df.replace(".uvh5", ".decal.uvh5") for df in dfl] for dfl in data_files
        ]

        for flist, clist, ulist in zip(data_files, cal_files, decal_files):
            for df, cf, uf in zip(flist, clist, ulist):
                apply_cal.apply_cal(
                    df,
                    uf,
                    cf,
                    gain_convention="divide",  # go the wrong way
                    clobber=True,
                )

        # First, let's go the other way to check if we get the same thing back directly
        recaled_files = [
            [df.replace(".uvh5", ".recal.uvh5") for df in dfl] for dfl in data_files
        ]
        for flist, clist, ulist in zip(recaled_files, cal_files, decal_files):
            for df, cf, uf in zip(flist, clist, ulist):
                apply_cal.apply_cal(
                    uf,
                    df,
                    cf,
                    gain_convention="multiply",  # go the wrong way
                    clobber=True,
                )

        for flset, flsetc in zip(data_files, recaled_files):
            for fl, flc in zip(flset, flsetc):
                uvdlst = UVData()
                uvdlst.read(fl, use_future_array_shapes=True)

                uvdlstc = UVData()
                uvdlstc.read(flc, use_future_array_shapes=True)
                np.testing.assert_allclose(uvdlst.data_array, uvdlstc.data_array)

        cfl = tmp / "lstbin_config_file.yaml"
        lstbin.make_lst_bin_config_file(
            cfl,
            decal_files,
            ntimes_per_file=2,
        )

        out_files_recal = wrappers.lst_bin_files(
            config_file=cfl,
            calfile_rules=[(".decal.uvh5", ".calfits")],
            fname_format="zen.{kind}.{lst:7.5f}.recal.uvh5",
            Nbls_to_load=10,
            rephase=False,
        )

        lstbin.make_lst_bin_config_file(
            cfl,
            data_files,
            ntimes_per_file=2,
            clobber=True,
        )
        out_files = wrappers.lst_bin_files(
            config_file=cfl,
            fname_format="zen.{kind}.{lst:7.5f}.uvh5",
            Nbls_to_load=11,
            rephase=False,
        )

        for flset, flsetc in zip(out_files, out_files_recal):
            assert flset[("LST", False)] != flsetc[("LST", False)]
            uvdlst = UVData()
            uvdlst.read(flset[("LST", False)], use_future_array_shapes=True)

            uvdlstc = UVData()
            uvdlstc.read(flsetc[("LST", False)], use_future_array_shapes=True)
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
                        np.where(
                            uvdlst.get_flags(ap + (pol,)),
                            1.0,
                            uvdlstc.get_data(ap + (pol,)),
                        ),
                        np.where(
                            uvdlst.get_flags(ap + (pol,)), 1.0, expected[i, :, :, j]
                        ),
                        rtol=1e-4,
                    )

            # Don't worry about history here, because we know they use different inputs
            uvdlst.history = uvdlstc.history
            assert uvdlst == uvdlstc

    @pytest.mark.parametrize(
        "random_ants_to_drop, rephase, sigma_clip_thresh, flag_strategy, pols, freq_range",
        [
            (0, True, 0.0, (0, 0, 0), ("xx", "yy"), None),
            (0, True, 0.0, (0, 0, 0), ("xx", "yy", "xy", "yx"), None),
            (0, True, 0.0, (0, 0, 0), ("xx", "yy"), (150e6, 180e6)),
            (0, True, 0.0, (2, 1, 3), ("xx", "yy"), None),
            (0, True, 3.0, (0, 0, 0), ("xx", "yy"), None),
            (0, False, 0.0, (0, 0, 0), ("xx", "yy"), None),
            (3, True, 0.0, (0, 0, 0), ("xx", "yy"), None),
        ],
    )
    def test_nontrivial_cal(
        self,
        tmp_path_factory,
        random_ants_to_drop: int,
        rephase: bool,
        sigma_clip_thresh: float,
        flag_strategy: tuple[int, int, int],
        pols: tuple[str],
        freq_range: tuple[float, float] | None,
        benchmark,
    ):
        tmp = tmp_path_factory.mktemp("nontrivial_cal")
        uvds = mockuvd.make_dataset(
            ndays=3,
            nfiles=2,
            ntimes=2,
            ants=np.arange(7),
            creator=mockuvd.create_uvd_identifiable,
            pols=pols,
            freqs=np.linspace(140e6, 180e6, 3),
            random_ants_to_drop=random_ants_to_drop,
        )

        uvcs = [
            [mockuvd.make_uvc_identifiable(d, *flag_strategy) for d in uvd]
            for uvd in uvds
        ]

        data_files = mockuvd.write_files_in_hera_format(uvds, tmp)
        cal_files = mockuvd.write_cals_in_hera_format(uvcs, tmp)
        decal_files = [
            [df.replace(".uvh5", ".decal.uvh5") for df in dfl] for dfl in data_files
        ]

        for flist, clist, ulist in zip(data_files, cal_files, decal_files):
            for df, cf, uf in zip(flist, clist, ulist):
                apply_cal.apply_cal(
                    df,
                    uf,
                    cf,
                    gain_convention="divide",  # go the wrong way
                    clobber=True,
                )

        cfl = tmp / "lstbin_config_file.yaml"
        lstbin.make_lst_bin_config_file(
            cfl,
            data_files,
            ntimes_per_file=2,
            clobber=True,
        )
        out_files = benchmark(
            wrappers.lst_bin_files,
            config_file=cfl,
            fname_format="zen.{kind}.{lst:7.5f}.uvh5",
            Nbls_to_load=11,
            rephase=rephase,
            sigma_clip_thresh=sigma_clip_thresh,
            sigma_clip_min_N=2,
            freq_min=freq_range[0] if freq_range is not None else None,
            freq_max=freq_range[1] if freq_range is not None else None,
            overwrite=True,
        )
        assert len(out_files) == 2
        for flset in out_files:
            uvdlst = UVData()
            uvdlst.read(flset[("LST", False)], use_future_array_shapes=True)

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
                        np.where(
                            uvdlst.get_flags(ap + (pol,)),
                            1.0,
                            uvdlst.get_data(ap + (pol,)),
                        ),
                        np.where(
                            uvdlst.get_flags(ap + (pol,)), 1.0, expected[i, :, :, j]
                        ),
                        rtol=1e-4
                        if (not rephase or (ap[0] == ap[1] and pol[0] == pol[1]))
                        else 1e-3,
                    )

    @pytest.mark.parametrize("tell_it", (True, False))
    def test_redundantly_averaged(self, tmp_path_factory, tell_it):
        tmp = tmp_path_factory.mktemp("nontrivial_cal")
        uvds = mockuvd.make_dataset(
            ndays=3,
            nfiles=2,
            ntimes=2,
            creator=mockuvd.create_uvd_identifiable,
            antpairs=[(i, j) for i in range(7) for j in range(i, 7)],  # 55 antpairs
            pols=("xx", "yx"),
            freqs=np.linspace(140e6, 180e6, 3),
            redundantly_averaged=True,
        )

        data_files = mockuvd.write_files_in_hera_format(uvds, tmp)

        cfl = tmp / "lstbin_config_file.yaml"
        config_info = lstbin.make_lst_bin_config_file(
            cfl,
            data_files,
            ntimes_per_file=2,
            clobber=True,
        )
        out_files = wrappers.lst_bin_files(
            config_file=cfl,
            fname_format="zen.{kind}.{lst:7.5f}.uvh5",
            Nbls_to_load=11,
            rephase=False,
            sigma_clip_thresh=0.0,
            sigma_clip_min_N=2,
            redundantly_averaged=True if tell_it else None,
        )

        assert len(out_files) == 2

        for flset in out_files:
            uvdlst = UVData()
            uvdlst.read(flset[("LST", False)], use_future_array_shapes=True)

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
                        uvdlst.get_data(ap + (pol,)),
                        np.where(
                            uvdlst.get_flags(ap + (pol,)), 1.0, expected[i, :, :, j]
                        ),
                        rtol=1e-4,
                    )

    def test_output_file_select(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("output_file_select")
        uvds = mockuvd.make_dataset(
            ndays=3,
            nfiles=4,
            ntimes=2,
            creator=mockuvd.create_uvd_identifiable,
            antpairs=[(i, j) for i in range(4) for j in range(i, 4)],  # 55 antpairs
            pols=("xx", "yx"),
            freqs=np.linspace(140e6, 180e6, 3),
        )

        data_files = mockuvd.write_files_in_hera_format(uvds, tmp)

        cfl = tmp / "lstbin_config_file.yaml"
        config_info = lstbin.make_lst_bin_config_file(
            cfl,
            data_files,
            ntimes_per_file=2,
            clobber=True,
        )
        lstbf = partial(
            wrappers.lst_bin_files,
            config_file=cfl,
            fname_format="zen.{kind}.{lst:7.5f}.uvh5",
            Nbls_to_load=11,
            rephase=False,
        )

        out_files = lstbf(output_file_select=0)
        assert len(out_files) == 1

        out_files = lstbf(output_file_select=(1, 2))
        assert len(out_files) == 2

        with pytest.raises(
            ValueError,
            match="output_file_select must be less than the number of output files",
        ):
            lstbf(output_file_select=100)

    def test_inpaint_mode_no_flags(self, tmp_path_factory):
        """Test that using inpaint mode does nothing when there's no flags."""
        tmp = tmp_path_factory.mktemp("inpaint_no_flags")
        uvds = mockuvd.make_dataset(
            ndays=3,
            nfiles=1,
            ntimes=2,
            creator=mockuvd.create_uvd_identifiable,
            antpairs=[(i, j) for i in range(3) for j in range(i, 3)],  # 55 antpairs
            pols=("xx", "yx"),
            freqs=np.linspace(140e6, 180e6, 3),
            redundantly_averaged=True,
        )

        data_files = mockuvd.write_files_in_hera_format(uvds, tmp)

        cfl = tmp / "lstbin_config_file.yaml"
        config_info = lstbin.make_lst_bin_config_file(
            cfl,
            data_files,
            ntimes_per_file=2,
            clobber=True,
        )

        # Additionally try fname format with leading / which should be removed
        # automatically in the writing.
        out_files = wrappers.lst_bin_files(
            config_file=cfl,
            fname_format="/zen.{kind}.{lst:7.5f}.{inpaint_mode}.uvh5",
            rephase=False,
            sigma_clip_thresh=None,
            sigma_clip_min_N=2,
            output_flagged=True,
            output_inpainted=True,
        )

        assert len(out_files) == 1

        for flset in out_files:
            flagged = UVData.from_file(flset[("LST", False)], use_future_array_shapes=True)
            inpainted = UVData.from_file(flset[("LST", True)], use_future_array_shapes=True)

            assert flagged == inpainted

    def test_inpaint_mode_no_flags_where_inpainted(self, tmp_path_factory):
        """Test that ."""
        tmp = tmp_path_factory.mktemp("inpaint_no_flags")
        uvds = mockuvd.make_dataset(
            ndays=3,
            nfiles=1,
            ntimes=2,
            creator=mockuvd.create_uvd_identifiable,
            antpairs=[(i, j) for i in range(3) for j in range(i, 3)],  # 55 antpairs
            pols=("xx", "yx"),
            freqs=np.linspace(140e6, 180e6, 3),
            redundantly_averaged=True,
        )

        data_files = mockuvd.write_files_in_hera_format(
            uvds, tmp, add_where_inpainted_files=True
        )

        cfl = tmp / "lstbin_config_file.yaml"
        config_info = lstbin.make_lst_bin_config_file(
            cfl,
            data_files,
            ntimes_per_file=2,
            clobber=True,
        )
        out_files = wrappers.lst_bin_files(
            config_file=cfl,
            fname_format="zen.{kind}.{lst:7.5f}{inpaint_mode}.uvh5",
            rephase=False,
            sigma_clip_thresh=None,
            sigma_clip_min_N=2,
            output_flagged=True,
            output_inpainted=True,
            where_inpainted_file_rules=[(".uvh5", ".where_inpainted.h5")],
        )

        assert len(out_files) == 1

        for flset in out_files:
            flagged = UVData.from_file(flset[("LST", False)], use_future_array_shapes=True)
            inpainted = UVData.from_file(flset[("LST", True)], use_future_array_shapes=True)

            assert flagged == inpainted

    def test_where_inpainted_not_baseline_type(self, tmp_path_factory):
        """Test that proper error is raised when using incorrect inpainted files."""
        tmp = tmp_path_factory.mktemp("inpaint_not_baseline_type")
        uvds = mockuvd.make_dataset(
            ndays=3,
            nfiles=1,
            ntimes=2,
            creator=mockuvd.create_uvd_identifiable,
            antpairs=[(i, j) for i in range(3) for j in range(i, 3)],  # 55 antpairs
            pols=("xx", "yx"),
            freqs=np.linspace(140e6, 180e6, 3),
            redundantly_averaged=True,
        )

        data_files = mockuvd.write_files_in_hera_format(
            uvds, tmp, add_where_inpainted_files=True
        )

        # Now create dodgy where_inpainted files
        inp = apply_filename_rules(
            data_files, [(".uvh5", ".where_inpainted.h5")]
        )
        for fllist in inp:
            for fl in fllist:
                uvf = UVFlag()
                uvf.read(fl, use_future_array_shapes=True)
                uvf.to_waterfall()
                uvf.to_flag()
                uvf.write(fl.replace(".h5", ".waterfall.h5"), clobber=True)

        cfl = tmp / "lstbin_config_file.yaml"
        lstbin.make_lst_bin_config_file(
            cfl,
            data_files,
            ntimes_per_file=2,
            clobber=True,
        )
        with pytest.raises(ValueError, match="to be a DataContainer"):
            wrappers.lst_bin_files(
                config_file=cfl,
                fname_format="zen.{kind}.{lst:7.5f}{inpaint_mode}.uvh5",
                output_flagged=False,
                where_inpainted_file_rules=[(".uvh5", ".where_inpainted.waterfall.h5")],
            )

    def test_sigma_clip_use_autos(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("test_sigma_clip_use_autos")
        uvds = mockuvd.make_dataset(
            ndays=3,
            nfiles=4,
            ntimes=2,
            creator=mockuvd.create_uvd_identifiable,
            antpairs=[(i, j) for i in range(10) for j in range(i, 10)],  # 55 antpairs
            pols=["xx", "yy"],
            freqs=np.linspace(140e6, 180e6, 12),
        )
        data_files = mockuvd.write_files_in_hera_format(uvds, tmp)

        cfl = tmp / "lstbin_config_file.yaml"
        lstbin.make_lst_bin_config_file(
            cfl,
            data_files,
            ntimes_per_file=2,
        )

        out_files = wrappers.lst_bin_files(
            config_file=cfl,
            fname_format="zen.{kind}.{lst:7.5f}.uvh5",
            write_med_mad=False,
            sigma_clip_thresh=10.0,
            sigma_clip_use_autos=True,
        )

        for flset in out_files:
            uvdlst = UVData()
            # Just making sure it ran...
            uvdlst.read(flset[("LST", False)], use_future_array_shapes=True)
