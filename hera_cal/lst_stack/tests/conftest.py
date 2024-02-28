import pytest
from pyuvdata import UVCal
from ...tests import mock_uvdata as mockuvd
from pathlib import Path
import numpy as np


def create_small_array_uvd(identifiable: bool = False, **kwargs):
    kwargs.update(
        freqs=np.linspace(150e6, 160e6, 100),
        ants=[0, 1, 2, 127, 128],
        antpairs=[(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)],
        pols=("xx", "yy"),
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
