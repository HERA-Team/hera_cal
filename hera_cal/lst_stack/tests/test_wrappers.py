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
                uvdlst1.read(flset1[key])

                uvdlst2 = UVData()
                uvdlst2.read(flset2[key])

                for key in ignore:
                    setattr(uvdlst1, key, getattr(uvdlst2, key))

                assert uvdlst1 == uvdlst2

    def check_identifiable_data_equiv(self, files):
        for flset in files:
            uvdlst = UVData()
            uvdlst.read(flset[("LST", False)])
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
