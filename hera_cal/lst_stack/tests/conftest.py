import pytest
from pyuvdata import UVCal
from ...tests import mock_uvdata as mockuvd
from pathlib import Path
import numpy as np
from hera_cal import apply_cal


def create_small_array_uvd(identifiable: bool = False, **kwargs):
    kw = {
        **dict(
            freqs=np.linspace(150e6, 160e6, 30),
            ants=[0, 1, 2, 127, 128],
            antpairs=[(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)],
            pols=("xx", "yy"),
        ),
        **kwargs
    }

    if identifiable:
        return mockuvd.create_uvd_identifiable(**kw)
    else:
        return mockuvd.create_uvd_ones(**kw)


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
        cal_style="redundant",
        gain_convention="multiply",
        jones_array="linear",
        cal_type="gain",
        metadata_only=False,
    )


@pytest.fixture(scope="function")
def uvd_file(uvd, tmpdir_factory) -> Path:
    # Write to file, so we can run lst_bin_files
    tmp = Path(tmpdir_factory.mktemp("test_partial_times"))
    mock = tmp / "mock.uvh5"
    uvd.write_uvh5(str(mock), clobber=True)
    return mock


@pytest.fixture(scope="function")
def uvd_redavg_file(uvd_redavg, tmpdir_factory) -> Path:
    # Write to file, so we can run lst_bin_files
    tmp = Path(tmpdir_factory.mktemp("test_partial_times"))
    mock = tmp / "mock.uvh5"
    uvd_redavg.write_uvh5(str(mock), clobber=True)
    return mock


@pytest.fixture(scope="function")
def uvc_file(uvc, uvd_file: Path) -> Path:
    # Write to file, so we can run lst_bin_files
    tmp = uvd_file.parent
    fl = f"{tmp}/mock.calfits"
    uvc.write_calfits(str(fl), clobber=True)
    return fl

# Below, define some different "season-like" datasets that we can
# use throughout tests.


def _make_season(
    pthfac,
    with_cals: bool = False,
    flag_full_ant: int = 0,
    flag_ant_time: int = 0,
    flag_ant_freq: int = 0,
    inpaint: bool = False,
    **kwargs
) -> list[list[str]]:
    tmp = pthfac.mktemp("season")
    uvds = mockuvd.make_dataset(
        ndays=kwargs.pop('ndays', 3),
        nfiles=kwargs.pop('nfiles', 4),
        ntimes=kwargs.pop('ntimes', 2),
        integration_time=kwargs.pop("integration_time", 3600.0),
        creator=create_small_array_uvd,
        identifiable=True,

        redundantly_averaged=kwargs.pop("redundantly_averaged", True),
        **kwargs
    )

    data_files = mockuvd.write_files_in_hera_format(
        uvds, tmp, add_where_inpainted_files=inpaint
    )

    if with_cals:
        uvcs = [
            [
                mockuvd.make_uvc_identifiable(
                    d,
                    flag_full_ant=flag_full_ant,
                    flag_ant_time=flag_ant_time,
                    flag_ant_freq=flag_ant_freq,
                ) for d in uvd
            ] for uvd in uvds
        ]
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

    return data_files


@pytest.fixture(scope="session")
def season_redavg(tmp_path_factory):
    return _make_season(tmp_path_factory)


@pytest.fixture(scope="session")
def season_redavg_irregular(tmp_path_factory):
    files = _make_season(tmp_path_factory)
    files[1] = files[1][1:]
    files[2] = files[2][2:]
    return files


@pytest.fixture(scope="session")
def season_redavg_inpaint(tmp_path_factory):
    return _make_season(tmp_path_factory, inpaint=True)


@pytest.fixture(scope="session")
def season_redavg_inpaint_fewer_ants(tmp_path_factory):
    return _make_season(tmp_path_factory, inpaint=True, antpairs=[(0, 0), (0, 1)])


@pytest.fixture(scope="session")
def season_notredavg(tmp_path_factory):
    return _make_season(
        tmp_path_factory, redundantly_averaged=False, with_cals=True,
        antpairs=[(i, j) for i in range(10) for j in range(i, 10)],  # 55 antpairs
    )


@pytest.fixture(scope='session')
def season_nonredavg_with_noise(tmp_path_factory):
    return _make_season(
        tmp_path_factory, with_cals=False, with_noise=True,
        redundantly_averaged=False, ndays=100, nfiles=1,
    )
