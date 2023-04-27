"""Functions for mocking UVdata objects for testing purposes."""

from pyuvdata import UVData
from pyuvdata.uvdata import FastUVH5Meta
from hera_cal import utils
import numpy as np
from astropy.coordinates import EarthLocation
import yaml
from hera_cal.data import DATA_PATH
from pathlib import Path

HERA_LOC = EarthLocation.from_geodetic(
    lat=-30.721526120690307,
    lon=21.428303826863015,
    height=1051.690000070259,
)

with open(f"{DATA_PATH}/hera_antpos.yaml", "r") as fl:
    HERA_ANTPOS = yaml.safe_load(fl)

def create_mock_hera_obs(
    jdint: int = 2459855, 
    integration_time=10.7,
    lst_start=0.1, 
    jd_start: float | None = None,
    ntimes: int=2,
    freqs: np.ndarray = np.linspace(45e6, 250e6, 1500),
    pols: list[str] = ["xx", "yy", "xy", "yx"],
    ants: list[int] | None = None,
    antpairs: list[tuple[int, int]] | None = None,
    empty: bool = False,
    time_axis_faster_than_bls: bool = True,
) -> UVData:
    tint = integration_time / (24 * 3600)
    dlst = tint * 2 * np.pi
    if jd_start is None:
        lsts = np.arange(lst_start, ntimes*dlst + lst_start, dlst)
        times = utils.LST2JD(lsts, start_jd=jdint, allow_other_jd=False)
    else:
        if jd_start < 2000000:
            jd_start += jdint
        times = np.arange(jd_start, ntimes*tint + jd_start, tint)[:ntimes]

    if ants is None:
        ants = list(HERA_ANTPOS.keys())

    antpos = {k: v for k,v in HERA_ANTPOS.items() if k in ants}

    # build a UVData object that knows about 5 antennas, but only uses 3 of them 
    # in baselines.
    uvd = UVData.new(
        freq_array= freqs,
        polarization_array=pols,
        antenna_positions= antpos,
        antpairs=antpairs,
        telescope_location= HERA_LOC,
        telescope_name= "HERA",
        times= times,
        x_orientation='n',
        empty=empty,
        time_axis_faster_than_bls=time_axis_faster_than_bls,
        do_blt_outer=True
    )
    return uvd


def create_uvd_ones(**kwargs) -> UVData:
    uvd = create_mock_hera_obs(
        empty=True,
        **kwargs
    )
    uvd.data_array += 1.0
    return uvd

def create_uvd_identifiable(**kwargs) -> UVData:
    """Make a UVData object with identifiable data.
    
    Each baseline, pol, LST and freq channel should be identifiable in the data with the
    following pattern:

        data = lst * np.cos(freq*a) + I * np.sin(freq*b)
    """
    ones = create_uvd_ones(**kwargs)
    lsts = np.unique(ones.lst_array)
    normfreqs = np.linspace(0, 2*np.pi, ones.freq_array.size)

    data = ones.data_array
    orig_shape = data.shape
    # in-place reshape of data to make it easier for us.
    data.shape = (len(ones.get_antpairs()), ones.Ntimes, ones.Nfreqs, len(ones.polarization_array))
    
    for i, ap in enumerate(ones.get_antpairs()):
        for j, p in enumerate(ones.polarization_array):
            if ap[0] == ap[1]:
                d = np.outer(lsts, np.ones_like(normfreqs))
            else:
                d = np.outer(
                    lsts, np.cos(normfreqs*ap[0]) + np.sin(normfreqs*ap[1])*1j
                )

            data[i, :, :, j] = d
            
    data.shape = orig_shape
    return ones

def write_files_in_hera_format(
    uvds: list[UVData] | list[list[UVData]], 
    tmpdir: Path,
    fmt: str = "zen.{jd:.5f}.sum.uvh5",
    in_jdint_dir: bool = True,
) -> list[list[str]]:

    if not tmpdir.exists():
        tmpdir.mkdir()

    single_level = False
    if not isinstance(uvds[0], list):
        single_level = True
        uvds = [uvds]

    fls = []
    for uvdlist in uvds:
        if in_jdint_dir:
            jdint = int(uvdlist[0].time_array.min())
            daydir = (tmpdir / str(jdint))
            daydir.mkdir()
        else:
            daydir = tmpdir

        _fls = []
        for obj in uvdlist:
            fl = daydir / fmt.format(jd=obj.time_array.min())
            if obj.metadata_only:
                obj.initialize_uvh5_file(fl, clobber=True)
            else:
                obj.write_uvh5(fl, clobber=True)
            _fls.append(str(fl))
        fls.append(_fls)

    if single_level:
        return fls[0]
    else:
        return fls

def make_day(nfiles: int, creator: callable = create_mock_hera_obs, **kwargs) -> list[UVData]:
    """Make a day of UVData objects."""

    uvds = []
    lst_start = kwargs.pop("lst_start", 0.1)

    for i in range(nfiles):
        uvds.append(creator(lst_start=lst_start, **kwargs))
        if i ==0:
            lsts = np.unique(uvds[0].lst_array)
            dlst = lsts[1] - lsts[0]
        lst_start = uvds[-1].lst_array[-1] + dlst
        kwargs['jd_start'] = None  # only use for first file, after, use lst_start
    return uvds

def make_dataset(
    ndays: int, nfiles: int, start_jdint: int = 2459855, creator: callable = create_mock_hera_obs, **kwargs
) -> list[list[UVData]]:
    """Make a dataset of UVData objects."""
    uvds = []
    for i in range(ndays):
        uvds.append(make_day(nfiles, jdint=start_jdint + i, creator=creator, **kwargs))
    
    return uvds