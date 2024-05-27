"""Functions for mocking UVdata objects for testing purposes."""
from __future__ import annotations

from pyuvdata import UVData, UVCal, UVFlag

from hera_cal import utils
from hera_cal import io
from hera_cal import noise
from hera_cal.lst_stack.config import make_lst_grid
import numpy as np
import yaml
from hera_cal.data import DATA_PATH
from pathlib import Path
from hera_cal.red_groups import RedundantGroups
from astropy import units

try:
    from pyuvdata import known_telescope_location
    HERA_LOC = known_telescope_location("HERA")
except ImportError:
    # this can go away when we require pyuvdata >= 3.0
    from pyuvdata import get_telescope
    from astropy.coordinates import EarthLocation
    hera_tel = get_telescope("HERA")
    HERA_LOC = EarthLocation.from_geocentric(*hera_tel.telescope_location * units.m)

with open(f"{DATA_PATH}/hera_antpos.yaml", "r") as fl:
    HERA_ANTPOS = yaml.safe_load(fl)

PHASEII_FREQS = np.arange(
    46920776.3671875, 234298706.0546875 + 10.0, 122070.3125
)


def create_mock_hera_obs(
    jdint: int = 2459855,
    integration_time=9.663677215576172,
    lst_start=0.1,
    jd_start: float | None = None,
    ntimes: int = 2,
    freqs: np.ndarray = PHASEII_FREQS,
    pols: list[str] = ["xx", "yy", "xy", "yx"],
    ants: list[int] | None = None,
    antpairs: list[tuple[int, int]] | None = None,
    empty: bool = False,
    time_axis_faster_than_bls: bool = True,
    redundantly_averaged: bool = False,
    x_orientation: str = "n",
) -> UVData:
    tint = integration_time / (24 * 3600)
    dlst = tint * 2 * np.pi

    if jd_start is None:
        # We ensure that the LSTs align exactly with the LST-grid that would be LST-binned.
        lst_start = make_lst_grid(dlst, begin_lst=lst_start)[0]
        lsts = np.arange(lst_start, ntimes * dlst + lst_start, dlst)[:ntimes]
        times = utils.LST2JD(lsts, start_jd=jdint, allow_other_jd=False)
    else:
        if jd_start < 2000000:
            jd_start += jdint
        times = np.arange(jd_start, ntimes * tint + jd_start, tint)[:ntimes]

    if ants is None:
        ants = list(HERA_ANTPOS.keys())

    antpos = {k: v for k, v in HERA_ANTPOS.items() if k in ants}

    reds = RedundantGroups.from_antpos(antpos)

    if antpairs is None:
        antpairs = [(i, j) for i in ants for j in ants if i <= j]
    else:
        antpairs = [tuple(ap) for ap in antpairs]

    antpairs = [ap for ap in antpairs if ap in reds]

    if redundantly_averaged:
        antpairs = list({reds.get_ubl_key(ap) for ap in antpairs})

    # build a UVData object
    uvd = UVData.new(
        freq_array=freqs,
        polarization_array=pols,
        antenna_positions=antpos,
        antpairs=np.array(antpairs),
        antenna_diameters=np.ones(len(antpos)) * 14.6,
        telescope_location=HERA_LOC,
        telescope_name="HERA",
        times=times,
        x_orientation=x_orientation,
        empty=empty,
        time_axis_faster_than_bls=time_axis_faster_than_bls,
        do_blt_outer=True,
        channel_width=np.diff(freqs)[0] if len(freqs) > 1 else np.diff(PHASEII_FREQS)[0],
    )
    uvd.polarization_array = np.array(uvd.polarization_array)
    uvd.blts_are_rectangular = True
    uvd.time_axis_faster_than_bls = time_axis_faster_than_bls
    return uvd


def create_uvd_ones(**kwargs) -> UVData:
    uvd = create_mock_hera_obs(empty=True, **kwargs)
    uvd.data_array += 1.0
    return uvd


def identifiable_data_from_uvd(
    ones: UVData,
    reshape: bool = True,
) -> np.ndarray:
    lsts = np.unique(ones.lst_array)

    normfreqs = 2 * np.pi * (ones.freq_array - 100e6) / 100e6

    data = np.zeros_like(ones.data_array)

    orig_shape = data.shape
    # in-place reshape of data to make it easier for us.
    data.shape = (
        len(ones.get_antpairs()),
        ones.Ntimes,
        ones.Nfreqs,
        len(ones.polarization_array),
    )

    for i, ap in enumerate(ones.get_antpairs()):
        for j, p in enumerate(ones.polarization_array):
            if ap[0] == ap[1] and p in (-5, -6):
                d = np.outer(-p * lsts * 1000, (ones.freq_array / 75e6) ** -2)
            else:
                d = np.outer(
                    p * lsts, np.cos(normfreqs * ap[0]) + np.sin(normfreqs * ap[1]) * 1j
                )

            data[i, :, :, j] = d
    if reshape:
        data.shape = orig_shape
    return data


def create_uvd_identifiable(
    with_noise: bool = False,
    autos_noise: bool = False,
    flag_frac: float = 0.0,
    time_axis_faster_than_bls: bool = True,
    **kwargs,
) -> UVData:
    """Make a UVData object with identifiable data.

    Each baseline, pol, LST and freq channel should be identifiable in the data with the
    following pattern:

        data = pol* lst * [np.cos(freq*a) + I * np.sin(freq*b)]
    """
    uvd = create_mock_hera_obs(empty=True, time_axis_faster_than_bls=time_axis_faster_than_bls, **kwargs)
    uvd.data_array = identifiable_data_from_uvd(uvd)

    if not time_axis_faster_than_bls:
        uvd.data_array.shape = (uvd.Nbls, uvd.Ntimes, uvd.Nfreqs, uvd.Npols)
        uvd.data_array = np.transpose(uvd.data_array, (1, 0, 2, 3))
        uvd.data_array = uvd.data_array.reshape((uvd.Nbls * uvd.Ntimes, uvd.Nfreqs, uvd.Npols))

    if with_noise:
        add_noise_to_uvd(uvd, autos=autos_noise)

    if flag_frac > 0:
        add_flags_to_uvd(uvd, flag_frac=flag_frac)
    return uvd


def add_noise_to_uvd(uvd, autos: bool = False):
    hd = io.to_HERAData(uvd)

    data, flags, nsamples = hd.read()
    dt = (data.times[1] - data.times[0]) * units.si.day.in_units(units.si.s)
    df = data.freqs[1] - data.freqs[0]

    for bl in data.bls():
        if bl[0] == bl[1] and bl[2][0] == bl[2][1] and not autos:
            continue

        variance = noise.predict_noise_variance_from_autos(
            bl, data, dt=dt, df=df, nsamples=nsamples
        )
        data[bl] += np.random.normal(
            scale=np.sqrt(variance / 2)
        ) + 1j * np.random.normal(scale=np.sqrt(variance / 2))

    hd.update(data=data)


def add_flags_to_uvd(uvd, flag_frac: float = 0.1):
    hd = io.to_HERAData(uvd)
    data, flags, nsamples = hd.read()

    for bl in data.bls():
        if bl[0] == bl[1] and bl[2][0] == bl[2][1]:
            continue
        flags[bl] = np.random.uniform(size=flags[bl].shape) < flag_frac
    hd.update(flags=flags)


def write_files_in_hera_format(
    uvds: list[UVData] | list[list[UVData]],
    tmpdir: Path,
    fmt: str = "zen.{jd:.5f}.sum.uvh5",
    in_jdint_dir: bool = True,
    add_where_inpainted_files: bool = False,
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
            daydir = tmpdir / str(jdint)
            if not daydir.exists():
                daydir.mkdir()
        else:
            daydir = tmpdir

        _fls = []
        for obj in uvdlist:
            fl = daydir / fmt.format(jd=np.mean(obj.time_array))
            if obj.metadata_only:
                obj.initialize_uvh5_file(fl, clobber=True)
            else:
                obj.write_uvh5(fl, clobber=True)

            if add_where_inpainted_files:
                flg = UVFlag()
                flg.from_uvdata(
                    obj, copy_flags=True, waterfall=False, mode="flag",
                    use_future_array_shapes=True
                )
                flg.flag_array[:] = True  # Everything inpainted.
                flgfile = fl.with_suffix(".where_inpainted.h5")
                flg.write(flgfile, clobber=True)

            _fls.append(str(fl))
        fls.append(_fls)

    if single_level:
        return fls[0]
    else:
        return fls


def make_day(
    nfiles: int, creator: callable = create_mock_hera_obs, **kwargs
) -> list[UVData]:
    """Make a day of UVData objects."""

    uvds = []
    lst_start = kwargs.pop("lst_start", 0.1)

    for i in range(nfiles):
        uvds.append(creator(lst_start=lst_start, **kwargs))
        if i == 0:
            lsts = np.unique(uvds[0].lst_array)
            dlst = lsts[1] - lsts[0]
        lst_start = uvds[-1].lst_array[-1] + dlst
        kwargs["jd_start"] = None  # only use for first file, after, use lst_start
    return uvds


def make_dataset(
    ndays: int,
    nfiles: int,
    start_jdint: int = 2459855,
    creator: callable = create_mock_hera_obs,
    random_ants_to_drop: int = 0,
    **kwargs,
) -> list[list[UVData]]:
    """Make a dataset of UVData objects."""

    if random_ants_to_drop > 0:
        default = creator(jdint=start_jdint, **kwargs)
        antpairs = default.get_antpairs()
        data_ants = list(set([ap[0] for ap in antpairs] + [ap[1] for ap in antpairs]))
        if "antpairs" in kwargs:
            del kwargs["antpairs"]

    uvds = []
    for i in range(ndays):
        if random_ants_to_drop > 0:
            drop_ants = np.random.choice(data_ants, random_ants_to_drop, replace=False)
            _antpairs = [
                ap
                for ap in antpairs
                if ap[0] not in drop_ants and ap[1] not in drop_ants
            ]
            kwargs["antpairs"] = _antpairs
        uvds.append(make_day(nfiles, jdint=start_jdint + i, creator=creator, **kwargs))

    return uvds


def make_uvc_ones(
    uvd: UVData, flag_full_ant: int = 0, flag_ant_time: int = 0, flag_ant_freq: int = 0
):
    uvc = UVCal.initialize_from_uvdata(
        uvd,
        cal_style="redundant",
        gain_convention="multiply",
        jones_array="linear",
        cal_type="gain",
        metadata_only=False,
    )

    if flag_full_ant > 0:
        badants = np.random.choice(
            np.arange(uvc.Nants_data), flag_full_ant, replace=False
        )
        uvc.flag_array[badants] = True
    if flag_ant_time > 0:
        badants = np.random.choice(
            np.arange(uvc.Nants_data), flag_ant_time, replace=False
        )
        badtime = np.random.randint(0, uvc.Ntimes)
        uvc.flag_array[badants, :, badtime] = True
    if flag_ant_freq > 0:
        badants = np.random.choice(
            np.arange(uvc.Nants_data), flag_ant_freq, replace=False
        )
        badfreq = np.random.randint(0, uvc.Nfreqs)
        uvc.flag_array[badants, badfreq, :] = True
    return uvc


def identifiable_gains_from_uvc(uvc: UVCal):
    gains = np.zeros_like(uvc.gain_array)
    for i, ant in enumerate(uvc.ant_array):
        for j in range(uvc.Njones):
            gains[i, :, :, j] = np.outer(
                np.exp(2j * np.pi * uvc.freq_array * (j + 1)),
                np.ones_like(uvc.time_array) * (ant + 1),
            )
    return gains


def make_uvc_identifiable(
    uvd: UVData, flag_full_ant: int = 0, flag_ant_time: int = 0, flag_ant_freq: int = 0
):
    uvc = make_uvc_ones(
        uvd,
        flag_full_ant=flag_full_ant,
        flag_ant_time=flag_ant_time,
        flag_ant_freq=flag_ant_freq,
    )
    uvc.gain_array = identifiable_gains_from_uvc(uvc)
    return uvc


def write_cals_in_hera_format(
    uvds: list[UVCal] | list[list[UVCal]],
    tmpdir: Path,
    fmt: str = "zen.{jd:.5f}.sum.calfits",
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
            daydir = tmpdir / str(jdint)
            if not daydir.exists():
                daydir.mkdir()
        else:
            daydir = tmpdir

        _fls = []
        for obj in uvdlist:
            fl = daydir / fmt.format(jd=np.mean(obj.time_array))
            obj.write_calfits(fl, clobber=True)
            _fls.append(str(fl))
        fls.append(_fls)

    if single_level:
        return fls[0]
    else:
        return fls
