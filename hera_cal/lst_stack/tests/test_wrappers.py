from __future__ import annotations
from ...tests import mock_uvdata as mockuvd
from pyuvdata import UVData
import numpy as np
from ... import apply_cal
from pyuvdata import utils
import pytest
from functools import partial
from .. import wrappers
from hera_cal.lst_stack.config import LSTBinConfigurator


def test_argparser_returns():
    args = wrappers.lst_bin_arg_parser()
    assert args is not None


class TestLSTBinFiles:
    def check_outputfile_equiv(
        self, list1: list[dict], list2: list[dict], ignore=()
    ):
        for flset1, flset2 in zip(list1, list2):
            for key in flset1:
                assert flset1[key] != flset2[key]  # not the same filenames
                uvdlst1 = UVData()
                uvdlst1.read(flset1[key], use_future_array_shapes=True)

                uvdlst2 = UVData()
                uvdlst2.read(flset2[key], use_future_array_shapes=True)

                for key in ignore:
                    setattr(uvdlst1, key, getattr(uvdlst2, key))

                assert uvdlst1 == uvdlst2

    def check_identifiable_data_equiv(self, files):
        for flset in files:
            uvdlst = UVData()
            uvdlst.read(flset[("LST", False)], use_future_array_shapes=True)
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
                            uvdlst.get_data(ap + (pol,)),
                        ),
                        np.where(
                            uvdlst.get_flags(ap + (pol,)), 1.0, expected[i, :, :, j]
                        ),
                        rtol=1e-4,
                    )

    def test_baseline_chunking(self, season_notredavg):
        config = LSTBinConfigurator(season_notredavg, nlsts_per_file=2)
        mf = config.get_matched_files()
        config = config.create_config(mf)

        out_files = wrappers.lst_bin_files(
            config_file=config,
            fname_format="zen.{kind}.{lst:7.5f}.uvh5",
            write_med_mad=True,
        )
        out_files_chunked = wrappers.lst_bin_files(
            config_file=config,
            fname_format="zen.{kind}.{lst:7.5f}.chunked.uvh5",
            bl_chunk_size=10,
            write_med_mad=True,
        )

        self.check_outputfile_equiv(out_files, out_files_chunked)

    def test_compare_nontrivial_cal(self, season_notredavg):
        decal_files = [
            [df.replace(".uvh5", ".decal.uvh5") for df in dfl] for dfl in season_notredavg
        ]
        config = LSTBinConfigurator(decal_files, nlsts_per_file=2, calfile_rules=[(".decal.uvh5", ".calfits")])
        mf = config.get_matched_files()
        config = config.create_config(mf)

        out_files_recal = wrappers.lst_bin_files(
            config_file=config,
            fname_format="zen.{kind}.{lst:7.5f}.recal.uvh5",
            write_med_mad=False,
            overwrite=True
        )

        config = LSTBinConfigurator(
            season_notredavg, nlsts_per_file=2
        )
        mf = config.get_matched_files()
        config = config.create_config(mf)

        out_files = wrappers.lst_bin_files(
            config_file=config,
            fname_format="zen.{kind}.{lst:7.5f}.uvh5",
            write_med_mad=False,
            overwrite=True
        )

        self.check_identifiable_data_equiv(out_files_recal)
        self.check_outputfile_equiv(out_files, out_files_recal, ignore=('history',))

    @pytest.mark.parametrize(
        "random_ants_to_drop, rephase, flag_strategy, pols, freq_range",
        [
            (0, True, (0, 0, 0), ("xx", "yy"), None),
            (0, True, (0, 0, 0), ("xx", "yy", "xy", "yx"), None),
            (0, True, (0, 0, 0), ("xx", "yy"), (150e6, 180e6)),
            (0, True, (2, 1, 3), ("xx", "yy"), None),
            (0, False, (0, 0, 0), ("xx", "yy"), None),
            (3, True, (0, 0, 0), ("xx", "yy"), None),
        ],
    )
    def test_nontrivial_cal(
        self,
        tmp_path_factory,
        random_ants_to_drop: int,
        rephase: bool,
        flag_strategy: tuple[int, int, int],
        pols: tuple[str],
        freq_range: tuple[float, float] | None,
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

        config = LSTBinConfigurator(decal_files, nlsts_per_file=2, calfile_rules=[(".decal.uvh5", ".calfits")])
        mf = config.get_matched_files()
        config = config.create_config(mf)

        out_files = wrappers.lst_bin_files(
            config_file=config,
            fname_format="zen.{kind}.{lst:7.5f}.uvh5",
            bl_chunk_size=11,
            rephase=rephase,
            freq_min=freq_range[0] if freq_range is not None else None,
            freq_max=freq_range[1] if freq_range is not None else None,
            overwrite=True,
        )

        assert len(out_files) == 2
        self.check_identifiable_data_equiv(out_files)

        # for flset in out_files:
        #     uvdlst = UVData()
        #     uvdlst.read(flset[("LST", False)], use_future_array_shapes=True)

        #     # Don't worry about history here, because we know they use different inputs
        #     expected = mockuvd.identifiable_data_from_uvd(uvdlst, reshape=False)

        #     strpols = utils.polnum2str(uvdlst.polarization_array)
        #     for i, ap in enumerate(uvdlst.get_antpairs()):
        #         for j, pol in enumerate(strpols):
        #             print(f"Baseline {ap + (pol,)}")

        #             # Unfortunately, we don't have LSTs for the files that exactly align
        #             # with bin centres, so some rephasing will happen -- we just have to
        #             # live with it and change the tolerance
        #             # Furthermore, we only check where the flags are False, because
        #             # when we put in flags, we end up setting the data to 1.0 (and
        #             # never using it...)
        #             print(uvdlst.get_data(ap))
        #             print(expected[i])
        #             np.testing.assert_allclose(
        #                 np.where(
        #                     uvdlst.get_flags(ap + (pol,)),
        #                     1.0,
        #                     uvdlst.get_data(ap + (pol,)),
        #                 ),
        #                 np.where(
        #                     uvdlst.get_flags(ap + (pol,)), 1.0, expected[i, :, :, j]
        #                 ),
        #                 rtol=1e-4
        #                 if (not rephase or (ap[0] == ap[1] and pol[0] == pol[1]))
        #                 else 1e-3,
        #             )

    def test_redundantly_averaged(self, season_redavg):
        config = LSTBinConfigurator(season_redavg, nlsts_per_file=2)
        mf = config.get_matched_files()
        config = config.create_config(mf)

        out_files = wrappers.lst_bin_files(
            config_file=config,
            fname_format="zen.{kind}.{lst:7.5f}.uvh5",
            bl_chunk_size=11,
            rephase=False,
            overwrite=True
        )

        assert len(out_files) == 4
        self.check_identifiable_data_equiv(out_files)
        # for flset in out_files:
        #     uvdlst = UVData()
        #     uvdlst.read(flset[("LST", False)], use_future_array_shapes=True)

        #     # Don't worry about history here, because we know they use different inputs
        #     expected = mockuvd.identifiable_data_from_uvd(uvdlst, reshape=False)

        #     strpols = utils.polnum2str(uvdlst.polarization_array)

        #     for i, ap in enumerate(uvdlst.get_antpairs()):
        #         for j, pol in enumerate(strpols):
        #             print(f"Baseline {ap + (pol,)}")

        #             # Unfortunately, we don't have LSTs for the files that exactly align
        #             # with bin centres, so some rephasing will happen -- we just have to
        #             # live with it and change the tolerance
        #             # Furthermore, we only check where the flags are False, because
        #             # when we put in flags, we end up setting the data to 1.0 (and
        #             # never using it...)
        #             np.testing.assert_allclose(
        #                 uvdlst.get_data(ap + (pol,)),
        #                 np.where(
        #                     uvdlst.get_flags(ap + (pol,)), 1.0, expected[i, :, :, j]
        #                 ),
        #                 rtol=1e-4,
        #             )

    def test_output_file_select(self, season_redavg_inpaint):
        config = LSTBinConfigurator(season_redavg_inpaint, nlsts_per_file=2, where_inpainted_file_rules=[(".uvh5", ".where_inpainted.h5")])
        mf = config.get_matched_files()
        config = config.create_config(mf)

        lstbf = partial(
            wrappers.lst_bin_files,
            config_file=config,
            fname_format="zen.{kind}.{lst:7.5f}.uvh5",
            bl_chunk_size=11,
            rephase=False,
            overwrite=True,
        )

        out_files = lstbf(output_file_select=0)
        assert len(out_files) == 1

        out_files = lstbf(output_file_select=(1, 2))
        assert len(out_files) == 2

        with pytest.raises(
            ValueError,
            match="output_file_select must be",
        ):
            lstbf(output_file_select=100)

    # def test_inpaint_mode_no_flags(self, tmp_path_factory):
    #     """Test that using inpaint mode does nothing when there's no flags."""
    #     tmp = tmp_path_factory.mktemp("inpaint_no_flags")
    #     uvds = mockuvd.make_dataset(
    #         ndays=3,
    #         nfiles=1,
    #         ntimes=2,
    #         creator=mockuvd.create_uvd_identifiable,
    #         antpairs=[(i, j) for i in range(3) for j in range(i, 3)],  # 55 antpairs
    #         pols=("xx", "yx"),
    #         freqs=np.linspace(140e6, 180e6, 3),
    #         redundantly_averaged=True,
    #     )

    #     data_files = mockuvd.write_files_in_hera_format(uvds, tmp)

    #     cfl = tmp / "lstbin_config_file.yaml"
    #     config_info = lstbin.make_lst_bin_config_file(
    #         cfl,
    #         data_files,
    #         ntimes_per_file=2,
    #         clobber=True,
    #     )

    #     # Additionally try fname format with leading / which should be removed
    #     # automatically in the writing.
    #     out_files = wrappers.lst_bin_files(
    #         config_file=cfl,
    #         fname_format="/zen.{kind}.{lst:7.5f}.{inpaint_mode}.uvh5",
    #         rephase=False,
    #         sigma_clip_thresh=None,
    #         sigma_clip_min_N=2,
    #         output_flagged=True,
    #         output_inpainted=True,
    #     )

    #     assert len(out_files) == 1

    #     for flset in out_files:
    #         flagged = UVData.from_file(flset[("LST", False)], use_future_array_shapes=True)
    #         inpainted = UVData.from_file(flset[("LST", True)], use_future_array_shapes=True)

    #         assert flagged == inpainted

    # def test_inpaint_mode_no_flags_where_inpainted(self, tmp_path_factory):
    #     """Test that ."""
    #     tmp = tmp_path_factory.mktemp("inpaint_no_flags")
    #     uvds = mockuvd.make_dataset(
    #         ndays=3,
    #         nfiles=1,
    #         ntimes=2,
    #         creator=mockuvd.create_uvd_identifiable,
    #         antpairs=[(i, j) for i in range(3) for j in range(i, 3)],  # 55 antpairs
    #         pols=("xx", "yx"),
    #         freqs=np.linspace(140e6, 180e6, 3),
    #         redundantly_averaged=True,
    #     )

    #     data_files = mockuvd.write_files_in_hera_format(
    #         uvds, tmp, add_where_inpainted_files=True
    #     )

    #     cfl = tmp / "lstbin_config_file.yaml"
    #     config_info = lstbin.make_lst_bin_config_file(
    #         cfl,
    #         data_files,
    #         ntimes_per_file=2,
    #         clobber=True,
    #     )
    #     out_files = wrappers.lst_bin_files(
    #         config_file=cfl,
    #         fname_format="zen.{kind}.{lst:7.5f}{inpaint_mode}.uvh5",
    #         rephase=False,
    #         sigma_clip_thresh=None,
    #         sigma_clip_min_N=2,
    #         output_flagged=True,
    #         output_inpainted=True,
    #         where_inpainted_file_rules=[(".uvh5", ".where_inpainted.h5")],
    #     )

    #     assert len(out_files) == 1

    #     for flset in out_files:
    #         flagged = UVData.from_file(flset[("LST", False)], use_future_array_shapes=True)
    #         inpainted = UVData.from_file(flset[("LST", True)], use_future_array_shapes=True)

    #         assert flagged == inpainted

    # def test_where_inpainted_not_baseline_type(self, tmp_path_factory):
    #     """Test that proper error is raised when using incorrect inpainted files."""
    #     tmp = tmp_path_factory.mktemp("inpaint_not_baseline_type")
    #     uvds = mockuvd.make_dataset(
    #         ndays=3,
    #         nfiles=1,
    #         ntimes=2,
    #         creator=mockuvd.create_uvd_identifiable,
    #         antpairs=[(i, j) for i in range(3) for j in range(i, 3)],  # 55 antpairs
    #         pols=("xx", "yx"),
    #         freqs=np.linspace(140e6, 180e6, 3),
    #         redundantly_averaged=True,
    #     )

    #     data_files = mockuvd.write_files_in_hera_format(
    #         uvds, tmp, add_where_inpainted_files=True
    #     )

    #     # Now create dodgy where_inpainted files
    #     inp = apply_filename_rules(
    #         data_files, [(".uvh5", ".where_inpainted.h5")]
    #     )
    #     for fllist in inp:
    #         for fl in fllist:
    #             uvf = UVFlag()
    #             uvf.read(fl, use_future_array_shapes=True)
    #             uvf.to_waterfall()
    #             uvf.to_flag()
    #             uvf.write(fl.replace(".h5", ".waterfall.h5"), clobber=True)

    #     cfl = tmp / "lstbin_config_file.yaml"
    #     lstbin.make_lst_bin_config_file(
    #         cfl,
    #         data_files,
    #         ntimes_per_file=2,
    #         clobber=True,
    #     )
    #     with pytest.raises(ValueError, match="to be a DataContainer"):
    #         wrappers.lst_bin_files(
    #             config_file=cfl,
    #             fname_format="zen.{kind}.{lst:7.5f}{inpaint_mode}.uvh5",
    #             output_flagged=False,
    #             where_inpainted_file_rules=[(".uvh5", ".where_inpainted.waterfall.h5")],
    #         )

    # def test_sigma_clip_use_autos(self, tmp_path_factory):
    #     tmp = tmp_path_factory.mktemp("test_sigma_clip_use_autos")
    #     uvds = mockuvd.make_dataset(
    #         ndays=3,
    #         nfiles=4,
    #         ntimes=2,
    #         creator=mockuvd.create_uvd_identifiable,
    #         antpairs=[(i, j) for i in range(10) for j in range(i, 10)],  # 55 antpairs
    #         pols=["xx", "yy"],
    #         freqs=np.linspace(140e6, 180e6, 12),
    #     )
    #     data_files = mockuvd.write_files_in_hera_format(uvds, tmp)

    #     cfl = tmp / "lstbin_config_file.yaml"
    #     lstbin.make_lst_bin_config_file(
    #         cfl,
    #         data_files,
    #         ntimes_per_file=2,
    #     )

    #     out_files = wrappers.lst_bin_files(
    #         config_file=cfl,
    #         fname_format="zen.{kind}.{lst:7.5f}.uvh5",
    #         write_med_mad=False,
    #         sigma_clip_thresh=10.0,
    #         sigma_clip_use_autos=True,
    #     )

    #     for flset in out_files:
    #         uvdlst = UVData()
    #         # Just making sure it ran...
    #         uvdlst.read(flset[("LST", False)], use_future_array_shapes=True)
