from __future__ import annotations
import numpy as np
from pathlib import Path
from pyuvdata.uvdata import FastUVH5Meta
from ..red_groups import RedundantGroups
import logging
from hera_qm.metrics_io import read_a_priori_ant_flags
from .. import io, utils
from typing import Any
import yaml
import attrs
from functools import cached_property
from astropy import units
import h5py

logger = logging.getLogger(__name__)


def make_lst_grid(
    dlst: float,
    begin_lst: float | None = None,
    lst_width: float = 2 * np.pi,
) -> np.ndarray:
    """
    Make a uniform grid in local sidereal time.

    By default, this grid will span 2pi radians, starting at zero radians. Even if
    the ``lst_width`` is not 2pi, we enforce that the grid equally divides 2pi, so
    that it can wrap around if later the width is increased and the same dlst is used.

    Parameters:
    -----------
    dlst :
        The width of a single LST bin in radians. 2pi must be equally divisible
        by dlst. If not, will default to the closest dlst that satisfies this criterion that
        is also greater than the input dlst. There is a minimum allowed dlst of 6.283e-6 radians,
        or .0864 seconds.
    begin_lst
        Beginning point for lst_grid. ``begin_lst`` must fall exactly on an LST bin
        given a dlst, within 0-2pi. If not, it is replaced with the closest bin.
        Default is zero radians.
    lst_width
        The width of the LST grid (including all bins) in radians.
        Default is 2pi radians. Note that regardless of this value, the final grid
        will always be equally divisible by 2pi.

    Output:
    -------
    lst_grid
        Uniform LST grid marking the center of each LST bin.
    """
    assert dlst >= 6.283e-6, "dlst must be greater than 6.283e-6 radians, or .0864 seconds."
    assert dlst < 2 * np.pi, "dlst must be less than 2pi radians, or 24 hours."

    # check 2pi is equally divisible by dlst
    if not (
        np.isclose((2 * np.pi / dlst) % 1, 0.0, atol=1e-5)
        or np.isclose((2 * np.pi / dlst) % 1, 1.0, atol=1e-5)
    ):
        # generate array of appropriate dlsts
        dlsts = 2 * np.pi / np.arange(1, 1000000)

        # get dlsts closest to dlst, but also greater than dlst
        dlst_diff = dlsts - dlst
        dlst_diff[dlst_diff < 0] = 10
        new_dlst = dlsts[np.argmin(dlst_diff)]
        logger.warning(
            f"2pi is not equally divisible by input dlst ({dlst:.16f}) at 1 part in 1e7.\n"
            f"Using {new_dlst:.16f} instead."
        )
        dlst = new_dlst

    # make an lst grid from [0, 2pi), with the first bin having a left-edge at 0 radians.
    lst_grid = np.arange(0, 2 * np.pi - 1e-7, dlst) + dlst / 2

    # shift grid by begin_lst
    if begin_lst is not None:
        # enforce begin_lst to be within 0-2pi
        if begin_lst < 0 or begin_lst >= 2 * np.pi:
            logger.warning("begin_lst was < 0 or >= 2pi, taking modulus with (2pi)")
            begin_lst = begin_lst % (2 * np.pi)
        begin_lst = lst_grid[np.argmin(np.abs(lst_grid - begin_lst))] - dlst / 2
        lst_grid += begin_lst
    else:
        begin_lst = 0.0

    lst_grid = lst_grid[lst_grid < (begin_lst + lst_width)]

    return lst_grid


def get_all_antpairs(
    data_files: list[list[str | Path | FastUVH5Meta]],
    ex_ant_yaml_files: list[str] | None = None,
    include_autos: bool = True,
    ignore_ants: tuple[int] = (),
    only_last_file_per_night: bool = False,
    redundantly_averaged: bool | None = None,
    reds: RedundantGroups | None = None,
    blts_are_rectangular: bool | None = None,
    time_axis_faster_than_bls: bool | None = None,
) -> tuple[list[tuple[int, int]], list[str]]:
    """Generate a set of all antpairs that have at least one un-flagged entry.

    This is performed over a list of nights, each of which consists of a list of
    individual uvh5 files. Each UVH5 file is *assumed* to have the same set of times
    for each baseline internally (different nights obviously have different times).

    If ``reds`` is provided, then any baseline found is mapped back to the first
    baseline in the redundant group it appears in. This *must* be set if

    Returns
    -------
    all_baselines
        The set of all antpairs in all files in the given list.
    all_pols
        A list of all polarizations in the files in the given list, as strings like
        'ee' and 'nn' (i.e. with x_orientation information).
    """
    if blts_are_rectangular is None and not isinstance(data_files[0][0], FastUVH5Meta):
        meta0 = FastUVH5Meta(data_files[0][0])
        blts_are_rectangular = meta0.blts_are_rectangular
        time_axis_faster_than_bls = meta0.time_axis_faster_than_bls

    data_files = [
        [
            (
                fl
                if isinstance(fl, FastUVH5Meta)
                else FastUVH5Meta(
                    fl,
                    blts_are_rectangular=blts_are_rectangular,
                    time_axis_faster_than_bls=time_axis_faster_than_bls,
                )
            )
            for fl in fl_list
        ]
        for fl_list in data_files
    ]

    all_baselines = set()
    all_pols = set()

    meta0 = data_files[0][0]
    x_orientation = meta0.get_transactional("x_orientation")

    # reds will contain all of the redundant groups for the whole array, because
    # all the antenna positions are included in every file.
    if reds is None:
        reds = RedundantGroups.from_antpos(
            antpos=dict(zip(meta0.antenna_numbers, meta0.antpos_enu)),
            include_autos=True,
        )

    if redundantly_averaged is None:
        # Try to work out if the files are redundantly averaged.
        # just look at the middle file from each night.
        for fl_list in data_files:
            meta = fl_list[len(fl_list) // 2]
            antpairs = meta.get_transactional("antpairs")
            ubls = {reds.get_ubl_key(ap) for ap in antpairs}
            if len(ubls) != len(antpairs):
                # At least two of the antpairs are in the same redundant group.
                redundantly_averaged = False
                logger.info("Inferred that files are not redundantly averaged.")
                break
        else:
            redundantly_averaged = True
            logger.info("Inferred that files are redundantly averaged.")

    if redundantly_averaged:
        if ignore_ants:
            raise ValueError(
                "Cannot ignore antennas if the files are redundantly averaged."
            )
        if ex_ant_yaml_files:
            raise ValueError(
                "Cannot exclude antennas if the files are redundantly averaged."
            )

    for night, fl_list in enumerate(data_files):
        if ex_ant_yaml_files:
            a_priori_antenna_flags = read_a_priori_ant_flags(
                ex_ant_yaml_files[night], ant_indices_only=True
            )
        else:
            a_priori_antenna_flags = set()

        if only_last_file_per_night:
            # Actually, use first AND last, just to be cautious
            fl_list = [fl_list[0], fl_list[-1]]

        for meta in fl_list:
            antpairs = meta.antpairs
            all_pols.update(set(meta.pols))
            this_xorient = meta.x_orientation

            # Clear the cache to save memory.
            meta.close()
            del meta.antpairs

            if this_xorient != x_orientation:
                raise ValueError(
                    f"Not all files have the same xorientation! The x_orientation in {meta.path} "
                    f"is {this_xorient}, but in {meta0.path} it is {x_orientation}."
                )

            for a1, a2 in antpairs:
                if redundantly_averaged:
                    a1, a2 = reds.get_ubl_key((a1, a2))

                if (
                    (a1, a2) not in all_baselines
                    and (a2, a1) not in all_baselines  # Do this first because after the
                    and a1 not in ignore_ants  # first file it often triggers.
                    and a2 not in ignore_ants
                    and (include_autos or a1 != a2)
                    and a1 not in a_priori_antenna_flags
                    and a2 not in a_priori_antenna_flags
                ):
                    all_baselines.add((a1, a2))

    return sorted(all_baselines), sorted(all_pols)


def _subsorted_list(x: Sequence[Sequence[Any]]) -> list[list[Any]]:
    return [sorted(y) for y in x]


def _fix_dlst(dlst: float) -> float:
    """Fix dlst to equally divide 2pi in less than 1 million sub-divisions."""
    dlsts = 2 * np.pi / np.arange(1, 1000000)

    if dlst < np.min(dlsts):
        raise ValueError(
            f"dlst must be more than {np.min(dlsts):1.5e}, the smallest possible value."
        )

    if dlst > np.max(dlsts):
        raise ValueError(
            f"dlst must be less than {np.max(dlsts):1.5e}, the largest possible value."
        )

    # get dlsts closest to dlst, but also greater than dlst
    dlst_diff = dlsts - dlst
    return dlsts[dlst_diff > 0][0]


@attrs.define
class LSTBinConfiguration:
    """
    LST-bin configuration specification.

    This class is meant to be used to specify a configuration for LST-binning, and has
    methods for searching for and matching files to LST bins. It can be used to create
    an :class:`LSTConfig` object, which is a static configuration object that holds
    the output of many of the methods of this class.

    The algorithm for matching files to LST bins is only approximate, but is conservative
    (i.e. it includes *at least* all files that should be included in the LST bin).

    Parameters
    ----------
    data_files : list of lists of str or FastUVH5Meta
        List of lists of data files to consider for LST-binning. The outer list
        is a list of nights, and the inner list is a list of files for that night.
    dlst : float, optional
        The approximate width of each LST bin in radians. Default is integration time of
        the first file on the first night. This is approximate because the final bin
        width is always equally divided into 2pi.
    atol : float, optional
        Absolute tolerance for matching LSTs to LST bins. Default is 1e-10.
    lst_start : float, optional
        The starting LST for the LST grid. Default is the lowest LST in the first file
        on the first night.
    lst_end : float, optional
        The ending LST for the LST grid. Default is lst_start + 2pi.
    ntimes_per_file : int, optional
        The number of LST bins to include in each output file. Default is 60.
    jd_regex : str, optional
        Regex to use to extract the JD from the file name. Set to None or empty
        to force the LST-matching to use the LSTs in the metadata within the file.
    """
    data_files: list[list[str | Path | FastUVH5Meta]] = attrs.field(converter=_subsorted_list)
    nlsts_per_file: int = attrs.field(converter=int, validator=attrs.validators.gt(0))
    dlst: float = attrs.field(
        converter=_fix_dlst
    )
    atol: float = attrs.field(
        default=1e-10,
        converter=float,
        validator=(attrs.validators.gt(0), attrs.validators.lt(0.1))
    )
    lst_start: float = attrs.field(default=0.0, converter=float)
    lst_end: float = attrs.field(converter=float)
    jd_regex: str = attrs.field(default=r"zen\.(\d+\.\d+)\.")

    @cached_property
    def datameta(self):
        return FastUVH5Meta(self.data_files[0][0])

    @nlsts_per_file.default
    def _nlsts_per_file_default(self) -> int:
        return self.datameta.Ntimes

    @dlst.default
    def _dlst_default(self) -> float:
        df0 = self.datameta
        if len(df0.lsts) > 1:
            dlst = df0.lsts[1] - df0.lsts[0]
        else:
            dlst = np.min(df0.integration_time) * 2 * np.pi / (units.sday.to("s"))
        return dlst

    @lst_end.default
    def _lst_end_default(self):
        return self.lst_start + 2 * np.pi

    @lst_start.validator
    @lst_end.validator
    @dlst.validator
    def _lst_start_end_validator(self, attribute, value):
        if value < 0 or value >= 2 * np.pi:
            raise ValueError("LST must be between 0 and 2pi")

    @cached_property
    def lst_grid(self) -> np.ndarray:
        return make_lst_grid(
            self.dlst,
            begin_lst=self.lst_start,
            lst_width=self.lst_end - self.lst_start
        )

    @cached_property
    def lst_grid_edges(self) -> np.ndarray:
        return np.concatenate([self.lst_grid - self.dlst / 2, [self.lst_grid[-1] + self.dlst / 2]])

    @property
    def nfiles(self) -> int:
        return int(np.ceil(len(self.lst_grid) / self.nlsts_per_file))

    @cached_property
    def reds(self) -> RedundantGroups:
        return RedundantGroups.from_antpos(
            antpos=dict(zip(self.datameta.antenna_numbers, self.datameta.antpos_enu)),
            include_autos=True,
        )

    @cached_property
    def is_redundantly_averaged(self) -> bool:
        # Try to work out if the files are redundantly averaged.
        # just look at the middle file from each night.
        for fl_list in self.data_files:
            meta = fl_list[len(fl_list) // 2]
            antpairs = meta.get_transactional("antpairs")
            ubls = {self.reds.get_ubl_key(ap) for ap in antpairs}
            if len(ubls) != len(antpairs):
                # At least two of the antpairs are in the same redundant group.
                return False

        return True

    def get_file_lst_edges(self) -> np.ndarray:
        last_edge = self.lst_grid_edges[-1]
        lst_edges = self.lst_grid_edges[::self.nlsts_per_file]
        if len(lst_edges) < self.nfiles + 1:
            lst_edges = np.concatenate([lst_edges, [last_edge]])
        return lst_edges

    def get_matched_files(self) -> list[list[FastUVH5Meta]]:
        """
        Configure data for LST binning.

        Make a 24 hour lst grid, starting LST and output files given
        input data files and LSTbin params.
        """
        lst_edges = self.get_file_lst_edges()

        matched_files = [[] for _ in self.lst_grid]
        for fllist in self.data_files:
            # matched here is a list of lists of FastUVH5Meta objects.
            # Each list is for a single lst-bin.
            matched = utils.match_files_to_lst_bins(
                lst_edges=lst_edges,
                file_list=fllist,
                files_sorted=True,
                jd_regex=self.jd_regex,
                blts_are_rectangular=self.datameta.blts_are_rectangular,
                time_axis_faster_than_bls=self.datameta.time_axis_faster_than_bls,
                atol=self.atol,
            )
            for i, m in enumerate(matched):
                matched_files[i].append(m)

        return matched_files

    def create_config(
        self,
        matched_files: list[list[FastUVH5Meta]],
        lst_branch_cut: float | None = None,
        ignore_ants: tuple[int] = (),
        only_last_file_per_night: bool = True,
    ) -> LSTConfig:
        """
        Create an LSTConfig object from the given matched files.

        Parameters
        ----------
        matched_files : list[list[FastUVH5Meta]]
            The matched files to use for LST binning. This is the output of
            :meth:`get_matched_files`.
        lst_branch_cut
            The LST at which to branch cut the LST grid for file writing. The JDs in the
            output LST-binned files will be *lowest* at the lst_branch_cut, and all file
            names will have LSTs that are higher than lst_branch_cut. If None, this will
            be determined automatically by finding the largest gap in LSTs and starting
            AFTER it.
        ignore_ants
            Antennas to ignore when creating the antpair list.
        only_last_file_per_night
            If True, only use the last file from each night when finding the full list
            of antpairs.
        """
        lst_grid = self.lst_grid.copy()
        if nextra := self.lst_grid.size % self.nfiles > 0:
            lst_grid = np.concatenate(lst_grid, [np.nan] * (self.nlsts_per_file - nextra))

        lst_grid = lst_grid.reshape((self.nfiles, self.nlsts_per_file))

        file_mask = np.array([any(len(mff) > 0 for mff in mf) for mf in matched_files])
        lst_grid = lst_grid[file_mask]
        matched_files = [mf for mm, mf in zip(file_mask, matched_files) if mm]

        # Get the best lst_branch_cut by finding the largest gap in LSTs and starting
        # AFTER it
        if lst_branch_cut is None:
            lst_branch_cut = float(utils.get_best_lst_branch_cut(np.concatenate(lst_grid)))

        matched_files = [
            [[str(m.path) for m in night] for night in outfiles]
            for outfiles in matched_files
        ]

        antpairs, pols = get_all_antpairs(
            data_files=[sum(fls, start=[]) for fls in matched_files],
            include_autos=True,
            ignore_ants=ignore_ants,
            only_last_file_per_night=only_last_file_per_night,
            redundantly_averaged=self.is_redundantly_averaged,
            reds=self.reds,
            blts_are_rectangular=self.datameta.blts_are_rectangular,
            time_axis_faster_than_bls=self.datameta.time_axis_faster_than_bls,
        )

        return LSTConfig(
            config=self,
            lst_grid=lst_grid,
            matched_files=matched_files,
            antpairs=[tuple(ap) for ap in antpairs if ap[0] != ap[1]],
            autos=[tuple(ap) for ap in antpairs if ap[0] == ap[1]],
            pols=pols,
            properties={
                "lst_branch_cut": lst_branch_cut,
                "blts_are_rectangular": self.datameta.blts_are_rectangular,
                "time_axis_faster_than_bls": self.datameta.time_axis_faster_than_bls,
                "x_orientation": self.datameta.x_orientation,
            }
        )

    def write(self, group: h5py.Group):
        dct = attrs.asdict(self)

        for k, v in dct.items():
            if k == 'data_files':
                for night, files in enumerate(v):
                    group.create_dataset(f"night_{night}", data=[str(f) for f in files])
            else:
                group.attrs[k] = v

    @classmethod
    def read(cls, group: h5py.Group):
        dct = {}
        for k, v in group.attrs.items():
            dct[k] = v

        n_nights = len([k for k in group.keys() if k.startswith("night_")])
        dct["data_files"] = []
        for night in range(n_nights):
            dct['data_files'].append([Path(f) for f in group[f"night_{night}"][()]])

        return cls(**dct)


@attrs.define(slots=False, frozen=False)
class LSTConfig:
    config: LSTBinConfiguration = attrs.field()
    lst_grid: np.ndarray = attrs.field(converter=np.asarray)
    matched_files: list[list[list[FastUVH5Meta]]] = attrs.field()
    autos: list[tuple[int, int]] = attrs.field()
    antpairs: list[tuple[int, int]] = attrs.field()
    pols: list[str] = attrs.field()
    properties: dict = attrs.field()

    @lst_grid.validator
    def _lst_grid_validator(self, attribute, value):
        if value.ndim != 2:
            raise ValueError("lst_grid must be a 2D array, with shape (nfiles, nlsts_per_file)")

    @property
    def dlst(self):
        return self.config.dlst

    @cached_property
    def lst_grid_edges(self) -> np.ndarray:
        return np.concatenate(
            [self.lst_grid - self.dlst / 2, [self.lst_grid[-1] + self.dlst / 2]]
        )

    def write(self, fname: str | Path):
        with h5py.File(fname, "w") as fl:
            self.config.write(fl.create_group("config"))
            fl.create_dataset("lst_grid", data=self.lst_grid)
            fl.create_dataset("matched_files", data=self.matched_files)
            fl.create_dataset("antpairs", data=self.antpairs)
            fl.create_dataset("autos", data=self.autos)
            fl.create_dataset("pols", data=self.pols)
            for k, v in self.properties.items():
                fl.attrs[k] = v

    @classmethod
    def from_file(cls, config_file: str | Path) -> LSTConfig:
        with h5py.File(config_file, "r") as fl:
            config = LSTBinConfiguration.read(fl["config"])
            lst_grid = fl["lst_grid"][()]
            matched_files = fl["matched_files"][()]
            antpairs = fl["antpairs"][()]
            autos = fl["autos"][()]
            pols = fl["pols"][()]
            properties = {k: v for k, v in fl.attrs.items()}

        return cls(
            config=config,
            lst_grid=lst_grid,
            matched_files=[[[FastUVH5Meta(fl) for fl in night] for night in outfile] for outfile in matched_files],
            properties=properties,
            antpairs=antpairs,
            autos=autos,
            pols=pols,
        )
