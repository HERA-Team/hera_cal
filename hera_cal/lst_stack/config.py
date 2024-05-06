"""
Configuration objects for LST-binning.

These objects are used to specify the configuration for LST-binning, and to read/write
that configuration to/from file. The intended workflow for using this configuration is
as follows::

    >>> from hera_cal.lststack import config

First, create an LSTBinConfigurator object from a TOML file, which contains the relevant
parameters for constructing an LST grid, and specifications of where to find datafiles::

    >>> cfg = config.LSTBinConfigurator.from_toml("config.toml")

Now, we can match the datafiles to LST bins::

    >>> matched_files = cfg.get_matched_files()

This ``matched_files`` is a list of lists of lists of files, where the outer list is
the LST bins, the next list is the nights, and the innermost list is the files that
match that LST bin on that night. This is pretty unwieldy, and ultimately we want
an object that holds these matched files and knows how to slice into them. To get this,
we create an LSTConfig object::

    >>> lst_cfg = cfg.create_config(matched_files)

This object can be written to disk, and loaded from disk::

    >>> lst_cfg.write("lst_config.h5")
    >>> new_cfg = config.LSTConfig.from_file("lst_config.h5")

This object is useful, but often we want to deal with a single LST bin, or the LST bins
corresponding to one intended output file. We can get this in the following way::

    >>> this_cfg = new_cfg.at_single_outfile(lst=1.3)

This object is a slice of the original LSTConfig object, and can be used to read the
datafiles that correspond to that LST bin, and to write the output file for that LST bin.
For example, this object has the ``matched_files`` attribute, which is a list of paths
to the files that correspond to that LST bin. It also has the ``matched_lsts`` attribute
which is the specific LSTs in each file that fall into the bin (note that not all LSTs
in this list will be in the bin -- at least one LST in each file will be in the bin).
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from pyuvdata.uvdata import FastUVH5Meta
from ..red_groups import RedundantGroups
import logging
from hera_qm.metrics_io import read_a_priori_ant_flags
from .. import utils
from typing import Any, Sequence
import attrs
from functools import cached_property
from astropy import units
import h5py
from .io import apply_filename_rules, filter_required_files_by_times
from abc import ABC
import toml

logger = logging.getLogger(__name__)


def _fix_dlst(dlst: float) -> float:
    """Fix dlst to equally divide 2pi in less than 1 million sub-divisions."""
    dlsts = 2 * np.pi / np.arange(1000000, 0, -1)

    if dlst < np.min(dlsts):
        raise ValueError(
            f"dlst must be more than {np.min(dlsts):1.5e}, the smallest possible value."
        )

    if dlst > np.max(dlsts):
        raise ValueError(
            f"dlst must be less than {np.max(dlsts):1.5e}, the largest possible value."
        )

    # get dlsts closest to dlst, but also greater than dlst
    return dlsts[dlsts >= dlst - 1e-12][0]


def make_lst_grid(
    dlst: float,
    begin_lst: float = 0.0,
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
    dlst = _fix_dlst(dlst)

    if lst_width <= dlst:
        raise ValueError("lst_width must be greater than dlst")

    # make an lst grid from [0, 2pi), with the first bin having a left-edge at 0 radians.
    lst_grid = np.arange(0, 2 * np.pi - 1e-7, dlst) + dlst / 2

    # shift grid by begin_lst
    # enforce begin_lst to be within 0-2pi
    begin_lst %= 2 * np.pi
    begin_lst = lst_grid[np.argmin(np.abs(lst_grid - begin_lst))] - dlst / 2
    lst_grid += begin_lst

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
    """Generate the set of all antpairs over a list of files.

    This is performed over a list of nights, each of which consists of a list of
    individual uvh5 files. Each UVH5 file is *assumed* to have the same set of times
    for each baseline internally (different nights obviously have different times).

    If ``reds`` is provided, then any baseline found is mapped back to the first
    baseline in the redundant group it appears in.

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


@attrs.define(slots=False, frozen=True)
class LSTBinConfigurator:
    r"""
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
        Absolute tolerance for matching LSTs to LST bins. Default is 1e-10. Units rad.
    lst_start : float, optional
        The starting LST for the LST grid. Default is the lowest LST in the first file
        on the first night. I radians.
    lst_end : float, optional
        The ending LST for the LST grid. Default is lst_start + 2pi. In radians.
    ntimes_per_file : int, optional
        The number of LST bins to include in each output file. Default is 60.
    jd_regex : str, optional
        Regex to use to extract the JD from the file name. Set to None or empty
        to force the LST-matching to use the LSTs in the metadata within the file.
        The default is "zen\.(\d+\.\d+)\.", which would match names like
        "zen.2458042.12345.uvh5" with the JD being 2458042.12345.
    calfile_rules
        A list of tuples of strings. Each tuple is a pair of strings that are used to
        replace the first string with the second string in the data file name to get
        the calibration file name. For example, providing [(".uvh5", ".calfits")] will
        generate a list of calfiles that have the same basename as the data files, but
        with the extension ".calfits" instead of ".uvh5". Multiple entries to the list
        are allowed, and the replacements are applied in order. If the resulting calfile
        name does not exist, the data file is ignored.
    where_inpainted_file_rules
        Rules to transform the input data file names into the corresponding "where
        inpainted" files (which should be in UVFlag format). If provided, this indicates
        that the data itself is in-painted. These files should specify which data is
        in-painted in the associated data file (which may be different than the in-situ
        flags of the data object).
    ignore_ants
        A list of antennas to ignore when binning data.
    antpairs_from_last_file_each_night : bool, optional
        If True, only the last file from each night is used to infer the observed
        antpairs. Setting to False can be very slow for large data sets, and is almost
        never necessary, as the antpairs observed are generally set per-night.
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
    calfile_rules: list[tuple[str, str]] | None = attrs.field(default=None)
    where_inpainted_file_rules: list[tuple[str, str]] | None = attrs.field(default=None)
    ignore_ants: tuple[int] = attrs.field(
        default=(),
        converter=tuple,
        validator=attrs.validators.deep_iterable(
            attrs.validators.instance_of(int)
        )
    )
    antpairs_from_last_file_each_night: bool = attrs.field(default=True)

    @cached_property
    def datameta(self):
        return FastUVH5Meta(self.data_files[0][0])

    def get_earliest_jd_in_set(self) -> float:
        """Assuming that each sub-list of datafiles is a night, return the earliest JD."""
        first_files = [FastUVH5Meta(fl[0]) for fl in self.data_files]
        return min(fl.times[0] for fl in first_files)

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
        if value < 0 or value > 2 * np.pi:
            raise ValueError("LST must be between 0 and 2pi")

    @calfile_rules.validator
    @where_inpainted_file_rules.validator
    def _rules_validator(self, attribute, value):
        if value is not None and not all(
            isinstance(v, (list, tuple))
            and len(v) == 2
            and all(isinstance(vv, str) for vv in v) for v in value
        ):
            raise ValueError(f"{attribute.name} must be a list of tuples of length 2.")

    @cached_property
    def lst_grid(self) -> np.ndarray:
        return make_lst_grid(
            self.dlst,
            begin_lst=self.lst_start,
            lst_width=self.lst_end - self.lst_start
        )

    @cached_property
    def lst_grid_edges(self) -> np.ndarray:
        return np.append(self.lst_grid - self.dlst / 2, self.lst_grid[-1] + self.dlst / 2)

    @property
    def nfiles(self) -> int:
        return int(np.ceil(len(self.lst_grid) / self.nlsts_per_file))

    @property
    def n_nights(self) -> int:
        return len(self.data_files)

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
            meta = FastUVH5Meta(fl_list[len(fl_list) // 2])
            antpairs = meta.get_transactional("antpairs")
            ubls = {self.reds.get_ubl_key(ap) for ap in antpairs}
            if len(ubls) != len(antpairs):
                # At least two of the antpairs are in the same redundant group.
                return False

        return True

    @classmethod
    def from_toml(cls, toml_file: str | Path | dict) -> LSTBinConfigurator:
        if isinstance(toml_file, dict):
            dct = toml_file
        elif isinstance(toml_file, (str, Path)) and Path(toml_file).exists():
            dct = toml.load(toml_file)
        elif isinstance(toml_file, str):
            dct = toml.loads(toml_file)
        else:
            raise ValueError("toml_file must be a valid path, toml-serialized string, or a dictionary.")

        _datafiles = dct.pop("datafiles", None)
        if _datafiles is None:
            raise ValueError("datafiles must be specified.")
        elif isinstance(_datafiles, dict):
            datafiles = cls.find_datafiles(**_datafiles)
        elif isinstance(_datafiles, list):
            datadir = Path(dct.pop("datadir"))
            datafiles = [sorted(datadir.glob(fl)) for fl in _datafiles]

        if len(datafiles) == 0 or all(len(fls) == 0 for fls in datafiles):
            raise ValueError(f"No data files found! Check config: {_datafiles}")

        return cls(data_files=datafiles, **dct)

    @staticmethod
    def find_datafiles(
        datadir: str | Path,
        nights: list[str],
        fileglob: str = "{night}/zen.{night}.*.sum.uvh5",
    ) -> list[list[Path]]:
        """Determine the datafiles from specifications.

        This will search for files with the following glob:

            ``{datadir}/{fileglob}``

        where fileglob may contain instances of "{night}" which will be replaced by
        each of the nights in ``nights``.

        Parameters
        ----------
        datadir : str or Path
            The top-level directory where the data files are stored.
        nights : list of str
            The nights to consider.
        fileglob
            A string glob-pattern to search for files. This string may contain instances
            of "{night}" which will be replaced by each of the nights in ``nights``.

        Returns
        -------
        data_files : list[list[Path | FastUVH5Meta]]
            The list of data files for each night.
        """

        datadir = Path(datadir)
        out = []
        for night in nights:
            this = fileglob.format(night=night)
            out.append(sorted(datadir.glob(this)))
        return out

    def get_file_lst_edges(self) -> np.ndarray:
        """Get the LST edges for each *output* file.

        Returns
        -------
        lst_edges : np.ndarray
            A 1D array of LST edges, with length nfiles + 1. The first entry is the
            lower edge of the first file, and the last entry is the upper edge of the
            last file.
        """
        last_edge = self.lst_grid_edges[-1]
        lst_edges = self.lst_grid_edges[::self.nlsts_per_file]
        if len(lst_edges) < self.nfiles + 1:
            lst_edges = np.concatenate([lst_edges, [last_edge]])
        return lst_edges

    def get_matched_files(self, as_paths: bool = False) -> list[list[list[FastUVH5Meta | Path]]]:
        """
        Find the files that are matched to each LST-bin.

        The output is a triple-nested list. The first list is for each output file,
        the second list is for each night, and the third list is for files within that
        night that might overlap with any LST-bin in that outfile file.
        The elements of the third list are FastUVH5Meta objects or Path objects
        depending on ``as_paths``.

        Note that there is no distinction made between LST bins within each output file,
        as it is expected that all of the files will be read in any case.

        Here, the lists are all unfiltered -- there can be many empty lists for LST
        bins that are never observed in a given set of raw files.

        Parameters
        ----------
        as_paths : bool, optional
            If True, return the paths of the files instead of the FastUVH5Meta objects.

        Returns
        -------
        matched_files : list[list[list[FastUVH5Meta | Path]]]
            The list of matched files. See above for the nesting structure.
        """
        lst_edges = self.get_file_lst_edges()

        matched_files = [[] for _ in lst_edges[:-1]]
        for fllist in self.data_files:
            # matched here is a list of lists of FastUVH5Meta objects.
            # Each list is for a single output LST bin file.
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
        matched_files: list[list[list[FastUVH5Meta]]],
    ) -> LSTConfig:
        """
        Create an LSTConfig object from the given matched files.

        Parameters
        ----------
        matched_files : list[list[list[FastUVH5Meta]]]
            The matched files to use for LST binning. This is the output of
            :meth:`get_matched_files`.
        """
        lst_grid = self.lst_grid.copy()
        if (nextra := self.lst_grid.size % self.nlsts_per_file) > 0:
            lst_grid = np.concatenate((lst_grid, [np.nan] * (self.nlsts_per_file - nextra)))

        lst_grid = lst_grid.reshape((self.nfiles, self.nlsts_per_file))

        file_mask = np.array([any(len(mff) > 0 for mff in mf) for mf in matched_files])
        lst_grid = lst_grid[file_mask]
        matched_files = [mf for mm, mf in zip(file_mask, matched_files) if mm]

        # Get the best lst_branch_cut by finding the largest gap in LSTs and starting
        # AFTER it
        lst_branch_cut = float(utils.get_best_lst_branch_cut(np.concatenate(lst_grid)))

        # Turn matched_files back into a list of lists of strings.
        matched_files = [
            [[str(m.path) for m in night] for night in outfiles]
            for outfiles in matched_files
        ]

        antpairs, pols = get_all_antpairs(
            data_files=self.data_files,
            include_autos=True,
            ignore_ants=self.ignore_ants,
            only_last_file_per_night=self.antpairs_from_last_file_each_night,
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
            autopairs=[tuple(ap) for ap in antpairs if ap[0] == ap[1]],
            pols=pols,
            inpaint_files=apply_filename_rules(
                matched_files, self.where_inpainted_file_rules, missing='raise'
            ) if self.where_inpainted_file_rules else None,
            calfiles=apply_filename_rules(
                matched_files, self.calfile_rules, missing='raise'
            ) if self.calfile_rules else None,
            properties={
                "lst_branch_cut": lst_branch_cut,
                "blts_are_rectangular": self.datameta.blts_are_rectangular,
                "time_axis_faster_than_bls": self.datameta.time_axis_faster_than_bls,
                "x_orientation": self.datameta.x_orientation,
                "first_jd": self.get_earliest_jd_in_set(),
            }
        )

    def _write(self, group: h5py.Group):
        """Write the current object into a HDF5 group.

        Note that this method is not meant to be called by users. The File associated
        with the Group must be open and writeable.
        """
        dct = attrs.asdict(self)

        for k, v in dct.items():
            if k == 'data_files':
                for night, files in enumerate(v):
                    group.create_dataset(f"night_{night}", data=[str(f) for f in files])
            elif k in ('calfile_rules', 'where_inpainted_file_rules'):
                if v:
                    group.create_dataset(k, data=v)
            else:
                group.attrs[k] = v

    @classmethod
    def _read(cls, group: h5py.Group):
        """Read an LSTBinConfigurator object from an open HDF5 group.

        Note that this method is not meant to be called by users. The File associated
        with the Group must be open and readable.
        """
        dct = dict(group.attrs.items())
        n_nights = len([k for k in group.keys() if k.startswith("night_")])
        dct["data_files"] = []
        for night in range(n_nights):
            dct['data_files'].append([f.decode() for f in group[f"night_{night}"][()]])

        for k in ("calfile_rules", "where_inpainted_file_rules"):
            if k in group:
                dct[k] = [tuple(x.decode() for x in rule) for rule in group[k][()]]
            else:
                dct[k] = None

        return cls(**dct)


def _nested_list_of(cls):
    """
    Recursively process a nested list-like structure and returns a new nested list.

    Parameters
    ----------
    x : array_like
        The input nested list-like structure to be processed.

    Returns
    -------
    array_like
        A new nested list with the same structure as the input, but potentially with modified elements.
    """
    def get_nested_list(x):
        if x is None:
            return None

        if isinstance(x, (Path, str, FastUVH5Meta)):
            return cls(x)

        return [get_nested_list(xx) for xx in x]
    return get_nested_list


def _to_antpairs(x) -> list[tuple]:
    return [tuple(int(a) for a in xx) for xx in x]


@attrs.define(slots=False, frozen=False, kw_only=True)
class _LSTConfigBase(ABC):
    """A base class for LSTConfig objects.

    This has two subclasses -- LSTConfig and LSTConfigSingle, which can be used by
    users.
    """
    config: LSTBinConfigurator = attrs.field()
    lst_grid: np.ndarray = attrs.field(converter=np.asarray, eq=attrs.cmp_using(eq=np.allclose))
    matched_files: list[list[list[Path]]] = attrs.field(converter=_nested_list_of(Path))
    autopairs: list[tuple[int, int]] = attrs.field(converter=_to_antpairs)
    antpairs: list[tuple[int, int]] = attrs.field(converter=_to_antpairs)
    pols: list[str] = attrs.field()
    properties: dict = attrs.field(factory=dict)
    calfiles: list[list[list[Path]]] | None = attrs.field(
        default=None, converter=_nested_list_of(Path),
    )
    inpaint_files: list[list[list[Path]]] | None = attrs.field(
        default=None, converter=_nested_list_of(Path),
    )

    @calfiles.validator
    @inpaint_files.validator
    def _extra_files_validator(self, attribute, value):
        """Validate that extra files are the same shape as matched_files.

        This is an attrs validator, called at object-creation time.

        Parameters
        ----------
        attribute : attrs.Attribute
            The attribute being validated (either calfiles or inpaint_files).
        value : list[list[list[Path]]] | None
            The value of the attribute being validated. Should be a triple-nested list
            of paths, in the same shape as matched_files.
        """
        if value is None:
            return

        def validate_sublist(this, that):
            if len(this) != len(that):
                raise ValueError(f"{attribute.name} must have the same shape as matched_files.")

            for this_sub, that_sub in zip(this, that):
                if isinstance(this_sub, list) and isinstance(that_sub, list):
                    validate_sublist(this_sub, that_sub)
                elif isinstance(this_sub, Path) and isinstance(that_sub, Path):
                    if not this_sub.exists():
                        raise ValueError(f"{this_sub} does not exist.")
                else:
                    raise ValueError(f"{attribute.name} has a different shape than matched_files.")

        validate_sublist(value, self.matched_files)

    @lst_grid.validator
    def _lst_grid_validator(self, attribute, value):
        if value.ndim not in (1, 2):
            raise ValueError("lst_grid must be a 1D or 2D array.")

        if value.ndim == 2 and value.shape[1] != self.config.nlsts_per_file:
            raise ValueError(
                "lst_grid must have shape (n_output_files, nlsts_per_file). "
                f"Got {value.shape} instead of (..., {self.config.nlsts_per_file})."
            )

    @property
    def n_output_files(self) -> int:
        return len(self.lst_grid)

    @property
    def n_nights(self) -> int:
        return self.config.n_nights

    @matched_files.validator
    def _matched_files_validator(self, attribute, value):
        """Validate the matched_files attribute.

        This is an attrs validator, called at object-creation time.

        Parameters
        ----------
        attribute : attrs.Attribute
            The attribute being validated (unused in this particular validator).
        value : list[list[list[Path]]]
            The value of the attribute being validated. Should be either a triple-nested
            list of paths (if lst_grid is shape ``(noutfiles, nlsts_per_file)``), or a
            single list of Paths (if lst_grid is shape ``(nlsts,)``).
        """
        if self.lst_grid.ndim == 2:
            if len(value) != self.n_output_files:
                raise ValueError(f"matched_files must be a list with one entry per output file: {self.n_output_files}")
            if len(value[0]) > self.n_nights:
                # Any particular outfile might have _less_ than n_nights, since not all nights
                # will contribute to any particular output file.
                raise ValueError(f"each list in matched_files should be n_nights long: {self.n_nights}")
        elif not all(isinstance(pth, Path) for pth in value):
            raise ValueError("matched_files must be a list of Path objects.")

    @cached_property
    def matched_metas(self) -> list[list[list[FastUVH5Meta]]]:
        """Matched files represented as FastUVH5Meta objects."""
        return _nested_list_of(FastUVH5Meta)(self.matched_files)

    @autopairs.validator
    @antpairs.validator
    def _antpairs_validator(self, attribute, value):
        if any(len(v) != 2 for v in value):
            raise ValueError(f"{attribute.name} must be a list of tuples of length 2.")

    @pols.validator
    def _pols_validator(self, attribute, value):
        if not all(isinstance(v, str) for v in value):
            raise ValueError(f"{attribute.name} must be a list of strings.")
        if len(value) > 4:
            raise ValueError(f"{attribute.name} must have at most 4 elements.")

    @autopairs.validator
    def _autos_validator(self, attribute, value):
        if any(a != b for a, b in value):
            raise ValueError("Autos must have the same antenna number on both sides.")

    @property
    def dlst(self):
        """The width of each LST bin in radians."""
        return self.config.dlst


def _write_irregular_list_of_paths_hdf5(fl, name, value):
    if value is None:
        return

    max_files_per_night = max(
        len(night) for outfile in value for night in outfile
    )

    regular = [
        [
            [str(m) for m in night] + [''] * (max_files_per_night - len(night))
            for night in outfile
        ] for outfile in value
    ]

    fl.create_dataset(name, data=regular)


def _read_irregular_list_of_paths_hdf5(fl, name):
    if name not in fl:
        return None

    value = fl[name][()]

    irregular = [
        [
            [Path(fl.decode()) for fl in night if fl]
            for night in outfile
        ] for outfile in value
    ]
    return irregular


@attrs.define(slots=False, frozen=False, kw_only=True)
class LSTConfig(_LSTConfigBase):
    """
    LST-bin configuration object.

    This object holds the configuration for a full run of LST-stacking, including the
    datafiles matched to an LST grid. It is envisaged that objects of this type will
    be created via the ``LSTBinConfigurator.get_config()`` method.

    Parameters
    ----------
    config : LSTBinConfigurator
        The LSTBinConfigurator object that was used to create this LSTConfig object.
    lst_grid : np.ndarray
        The LST grid for the LST-binning. This is a 2D array of shape
        ``(nfiles, nlsts_per_file)``.
    matched_files : list[list[list[Path]]]
        The matched files for each LST bin. This is a triple-nested list of paths: the
        outer list is over output files, the next list is over nights, and the inner
        list is over files within each night.
    calfiles : list[list[list[Path]]] | None
        The calibration files corresponding to each of the matched_files. Optional.
    inpaint_files : list[list[list[Path]]] | None
        The inpainted files corresponding to each of the matched_files. Optional.
    autos : list[tuple[int, int]]
        The list of all auto antpairs that appear in any of the matched files.
    antpairs : list[tuple[int, int]]
        The list of all cross antpairs that appear in any of the matched files.
    pols : list[str]
        The list of polarizations that appear in the matched files.
    properties : dict
        Properties of the files in the dataset being binned. Entries are arbitrary, but
        typically include "lst_branch_cut", "blts_are_rectangular",
        "time_axis_faster_than_bls", "x_orientation", "first_jd".
    """
    @cached_property
    def lst_grid_edges(self) -> np.ndarray:
        return np.concatenate(
            [self.lst_grid - self.dlst / 2, self.lst_grid[:, [-1]] + self.dlst / 2],
            axis=1
        )

    def write(self, fname: str | Path):
        """Write the object to a HDF5 file."""
        with h5py.File(fname, "w") as fl:
            self.config._write(fl.create_group("config"))
            fl.create_dataset("lst_grid", data=self.lst_grid)
            fl.create_dataset("antpairs", data=self.antpairs, dtype=int)
            fl.create_dataset("autopairs", data=self.autopairs, dtype=int)
            fl.create_dataset("pols", data=self.pols)
            _write_irregular_list_of_paths_hdf5(fl, "matched_files", self.matched_files)
            _write_irregular_list_of_paths_hdf5(fl, "calfiles", self.calfiles)
            _write_irregular_list_of_paths_hdf5(fl, "inpaint_files", self.inpaint_files)

            for k, v in self.properties.items():
                if v is None:
                    fl.attrs[k] = 'none'
                else:
                    try:
                        fl.attrs[k] = v
                    except TypeError as e:

                        raise ValueError(f"Cannot write attribute {k} of type {type(v)}.") from e

    @classmethod
    def from_file(cls, config_file: str | Path) -> LSTConfig:
        """Construct an LSTConfig object by reading it from file."""
        with h5py.File(config_file, "r") as fl:
            config = LSTBinConfigurator._read(fl["config"])
            lst_grid = fl["lst_grid"][()]
            mfs = _read_irregular_list_of_paths_hdf5(fl, "matched_files")
            calfiles = _read_irregular_list_of_paths_hdf5(fl, "calfiles")
            inpaint_files = _read_irregular_list_of_paths_hdf5(fl, "inpaint_files")
            antpairs = fl["antpairs"][()]
            autos = fl["autopairs"][()]
            pols = fl["pols"][()]
            properties = dict(fl.attrs.items())
            for k in properties:
                if properties[k] == 'none':
                    properties[k] = None

        return cls(
            config=config,
            lst_grid=lst_grid,
            matched_files=mfs,
            properties=properties,
            antpairs=[(a, b) for a, b in antpairs],
            autopairs=[(a, b) for a, b in autos],
            pols=[p.decode() for p in pols],
            calfiles=calfiles,
            inpaint_files=inpaint_files,
        )

    def _get_single_config(self, outfile, lstindex: int | None) -> LSTConfigSingle:
        """Return a single LSTConfigSingle object.

        This method is used to create a LSTConfigSingle object from the current
        LSTConfig object. It is used by the at_single_outfile and at_single_bin
        methods. The output LSTConfigSingle object represents the files and config
        required for a single LST bin or output file (which might contain a few
        LST bins).
        """
        lst_grid = self.lst_grid[outfile]
        grid_edges = self.lst_grid_edges[outfile]

        if lstindex is not None:
            lst_grid = lst_grid[[lstindex]]
            grid_edges = grid_edges[lstindex:lstindex + 2]

        tinds, _, matched_files, cals, inp = filter_required_files_by_times(
            lst_range=(grid_edges[0], grid_edges[-1]),
            data_metas=self.matched_metas[outfile],
            cal_files=self.calfiles[outfile] if self.calfiles else None,
            where_inpainted_files=self.inpaint_files[outfile] if self.inpaint_files else None,
        )

        kw = attrs.asdict(self, recurse=False)
        kw['lst_grid'] = lst_grid
        kw['matched_files'] = [m.path for m in matched_files]
        kw['calfiles'] = cals
        kw['inpaint_files'] = inp

        return LSTConfigSingle(time_indices=tinds, **kw)

    def at_single_outfile(
        self,
        outfile: int | None = None,
        lst: float | None = None,
    ) -> LSTConfigSingle:
        """Return an LSTConfigSingle object for a single output file.

        Parameters
        ----------
        outfile : int, optional
            The index of the output file to return. If not specified, ``lst`` must be.
        lst : float, optional
            The LST to return the output file for. If not specified, ``outfile`` must be.
            If set, the returned object will contain all LST bins in the output file that
            contains this particular LST. If you want a single LST bin, use the
            ``at_single_bin`` method instead.
        """
        if lst is None and outfile is None:
            raise ValueError("Either lst or outfile must be specified.")

        if lst is not None and outfile is not None:
            raise ValueError("Only one of lst or outfile can be specified.")

        if lst is not None:
            outfile = np.searchsorted(self.lst_grid_edges[:, 0], lst, side='right') - 1

        return self._get_single_config(outfile, None)

    def at_single_bin(
        self,
        bin_index: int | None = None,
        lst: float | None = None,
    ) -> LSTConfigSingle:
        """Return an LSTConfigSingle object for a single LST bin.

        Parameters
        ----------
        bin_index : int, optional
            The index of the LST bin to return. If not specified, ``lst`` must be.
        lst : float, optional
            The LST to return the LST bin for. If not specified, ``bin_index`` must be.
            If set, the returned object will contain only the LST bin that
            contains this particular LST. If you want a full output file, use the
            ``at_single_outfile`` method instead.
        """
        if lst is None and bin_index is None:
            raise ValueError("Either lst or bin_index must be specified.")

        if lst is not None and bin_index is not None:
            raise ValueError("Only one of lst or bin_index can be specified.")

        edges = self.lst_grid_edges[:, :-1].flatten()

        if lst is not None:
            bin_index = np.searchsorted(edges, lst, side='right') - 1

        fl_index = bin_index // self.config.nlsts_per_file
        bin_index = bin_index % self.config.nlsts_per_file

        return self._get_single_config(fl_index, bin_index)


@attrs.define(slots=False, kw_only=True)
class LSTConfigSingle(_LSTConfigBase):
    """
    LST-bin configuration object for a single output file or LST bin.

    This object holds the lstbin configuration for a particular set of LSTs -- generally
    from a single output file, or at a single LST bin, including the
    datafiles matched to the LST grid. It is envisaged that objects of this type will
    be created via the ``LSTBinConfigurator.get_config().at_single_outfile()`` method.

    Parameters
    ----------
    config : LSTBinConfigurator
        The LSTBinConfigurator object that was used to create this LSTConfig object.
    lst_grid : np.ndarray
        The LST grid for the LST-binning. This is a 1D array of shape ``(nlsts,)``.
    matched_files : list[Path]
        All the files matched to the LST bins in this object. This is a list of paths.
    calfiles : list[Path] | None
        The calibration files corresponding to each of the matched_files. Optional.
    inpaint_files : list[Path] | None
        The inpainted files corresponding to each of the matched_files. Optional.
    autos : list[tuple[int, int]]
        The list of all auto antpairs that appear in any of the matched files.
    antpairs : list[tuple[int, int]]
        The list of all cross antpairs that appear in any of the matched files.
    pols : list[str]
        The list of polarizations that appear in the matched files.
    properties : dict
        Properties of the files in the dataset being binned. Entries are arbitrary, but
        typically include "lst_branch_cut", "blts_are_rectangular",
        "time_axis_faster_than_bls", "x_orientation", "first_jd".
    time_indices : list[np.ndarray]
        The time indices of the matched files that fall into these LST bins.
        This will be determined by reading the metadata in the matched files if
        necessary.
    """
    time_indices: list[np.ndarray] = attrs.field(
        eq=attrs.cmp_using(lambda x, y: all(np.allclose(xx, yy) for xx, yy in zip(x, y)))
    )

    @time_indices.default
    def _time_indices_default(self):
        lstmin = self.lst_grid[0] - self.dlst / 2
        lstmax = self.lst_grid[-1] + self.dlst / 2

        tinds = []
        for meta in self.matched_metas:
            lsts = meta.lsts % (2 * np.pi)
            lsts[lsts < lstmin] += 2 * np.pi

            tind = np.argwhere((lsts >= lstmin) & (lsts < lstmax)).flatten()
            tinds.append(tind)

        return tinds

    @time_indices.validator
    def _time_indices_validator(self, attribute, value):
        if len(value) != len(self.matched_metas):
            raise ValueError("time_indices must have the same length as matched_metas.")

        for tind, meta in zip(value, self.matched_metas):
            if tind.dtype.kind != 'i':
                raise ValueError("time_indices must be integer arrays.")
            if len(tind) > len(meta.lsts) or tind.max() >= len(meta.lsts):
                raise ValueError("time_indices must be shorter than the LSTs in the file.")

    @cached_property
    def lst_grid_edges(self) -> np.ndarray:
        """The LST grid edges for the LST bins in this object."""
        return np.concatenate(
            [self.lst_grid - self.dlst / 2, [self.lst_grid[-1] + self.dlst / 2]],
        )

    @property
    def n_lsts(self) -> int:
        """The number of LST bins in this object."""
        return self.lst_grid.size

    def get_lsts(self) -> tuple[np.ndarray]:
        """Return the LSTs of the observations that fall into this LST bin/outfile.

        Returns
        -------
        tuple[np.ndarray]
            The LSTs of the observations that fall into this LST bin/outfile.
            Each element of the tuple represents a single input data file, and is
            an array of LSTs from that file that fall into the LST bin/outfile.
        """
        return tuple(meta.lsts[tind] % (2 * np.pi) for meta, tind in zip(self.matched_metas, self.time_indices))
