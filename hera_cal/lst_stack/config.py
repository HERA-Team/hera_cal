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


def get_all_unflagged_baselines(
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


def config_lst_bin_files(
    data_files: list[list[str | FastUVH5Meta]],
    dlst: float | None = None,
    atol: float = 1e-10,
    lst_start: float | None = None,
    lst_width: float = 2 * np.pi,
    blts_are_rectangular: bool | None = None,
    time_axis_faster_than_bls: bool | None = None,
    ntimes_per_file: int | None = None,
    jd_regex: str = r"zen\.(\d+\.\d+)\.",
):
    """
    Configure data for LST binning.

    Make a 24 hour lst grid, starting LST and output files given
    input data files and LSTbin params.

    Parameters
    ----------
    data_files : list of lists
        nested set of lists, with each nested list containing paths to
        data files from a particular night. Frequency axis of each file must be identical.
    dlst : float
        LST bin width. If None, will get this from the first file in data_files.
    atol : float
        absolute tolerance for LST bin float comparison
    lst_start : float
        starting LST for binner as it sweeps from lst_start to lst_start + 2pi.
        Default is first LST of the first file of the first night.
    lst_width : float
        How much LST to bin.
    ntimes_per_file : int
        number of LST bins in a single output file

    Returns
    -------
    lst_grid : float ndarray holding LST bin centers.
    matched_files : list of lists of files, one list for each output file.
    """
    logger.info("Configuring lst_grid")

    data_files = [sorted(df) for df in data_files]

    df0 = data_files[0][0]

    # get dlst from first data file if None
    if dlst is None:
        dlst = io.get_file_times(df0, filetype="uvh5")[0]

    # Get rectangularity of blts from first file if None
    meta = FastUVH5Meta(
        df0,
        blts_are_rectangular=blts_are_rectangular,
        time_axis_faster_than_bls=time_axis_faster_than_bls,
    )

    if blts_are_rectangular is None:
        blts_are_rectangular = meta.blts_are_rectangular
    if time_axis_faster_than_bls is None:
        time_axis_faster_than_bls = meta.time_axis_faster_than_bls

    if ntimes_per_file is None:
        ntimes_per_file = meta.Ntimes

    # Get the initial LST as the lowest LST in any of the first files on each night
    first_files = [
        FastUVH5Meta(
            df[0],
            blts_are_rectangular=blts_are_rectangular,
            time_axis_faster_than_bls=time_axis_faster_than_bls,
        )
        for df in data_files
    ]

    # get begin_lst from lst_start or from the first JD in the data_files
    if lst_start is None:
        lst_start = np.min([ff.lsts[0] for ff in first_files])
    begin_lst = lst_start

    # make LST grid that divides into 2pi
    lst_grid = make_lst_grid(dlst, begin_lst=begin_lst, lst_width=lst_width)
    dlst = lst_grid[1] - lst_grid[0]

    lst_edges = np.concatenate([lst_grid - dlst / 2, [lst_grid[-1] + dlst / 2]])

    # Now, what we need here is actually the lst_edges of the *files*, not the actual
    # bins.
    nfiles = int(np.ceil(len(lst_grid) / ntimes_per_file))
    last_edge = lst_edges[-1]
    lst_edges = lst_edges[::ntimes_per_file]
    if len(lst_edges) < nfiles + 1:
        lst_edges = np.concatenate([lst_edges, [last_edge]])

    matched_files = [[] for _ in lst_grid]
    for fllist in data_files:
        matched = utils.match_files_to_lst_bins(
            lst_edges=lst_edges,
            file_list=fllist,
            files_sorted=True,
            jd_regex=jd_regex,
            blts_are_rectangular=blts_are_rectangular,
            time_axis_faster_than_bls=time_axis_faster_than_bls,
            atol=atol,
        )
        for i, m in enumerate(matched):
            matched_files[i].append(m)

    nfiles = int(np.ceil(len(lst_grid) / ntimes_per_file))
    lst_grid = [
        lst_grid[ntimes_per_file * i: ntimes_per_file * (i + 1)] for i in range(nfiles)
    ]

    # Only keep output files that have data associated
    lst_grid = [
        lg for lg, mf in zip(lst_grid, matched_files) if any(len(mff) > 0 for mff in mf)
    ]
    matched_files = [mf for mf in matched_files if any(len(mff) > 0 for mff in mf)]
    return lst_grid, matched_files


def make_lst_bin_config_file(
    config_file: str | Path,
    data_files: list[list[str | FastUVH5Meta]],
    clobber: bool = False,
    dlst: float | None = None,
    atol: float = 1e-10,
    lst_start: float | None = None,
    lst_width: float = 2 * np.pi,
    ntimes_per_file: int = 60,
    blts_are_rectangular: bool | None = None,
    time_axis_faster_than_bls: bool | None = None,
    jd_regex: str = r"zen\.(\d+\.\d+)\.",
    lst_branch_cut: float | None = None,
) -> dict[str, Any]:
    """Construct and write a YAML configuration file for lst-binning.

    This determines an LST-grid, and then determines which files should be
    included in each LST-bin. The output is a YAML file that can be used to
    quickly read in raw files that correspond to a particular LST-bin.

    The idea of this function is for it to be run as a separate step (e.g. by hera_opm)
    that needs to run to setup a full LST-binning run on multiple parallel tasks. Each
    task will read the YAML file, and select out the portion appropriate for that
    LST bin file.

    The algorithm for matching files to LST bins is only approximate, but is conservative
    (i.e. it includes *at least* all files that should be included in the LST bin).

    Parameters
    ----------
    config_file : str or Path
        Path to write the YAML configuration file to.
    data_files : list of lists of str or FastUVH5Meta
        List of lists of data files to consider for LST-binning. The outer list
        is a list of nights, and the inner list is a list of files for that night.
    clobber : bool, optional
        If True, overwrite the config_file if it exists. If False, raise an error
        if the config_file exists.
    dlst : float, optional
        The approximate width of each LST bin in radians. Default is integration time of
        the first file on the first night. This is approximate because the final bin
        width is always equally divided into 2pi.
    atol : float, optional
        Absolute tolerance for matching LSTs to LST bins. Default is 1e-10.
    lst_start : float, optional
        The starting LST for the LST grid. Default is the lowest LST in the first file
        on the first night.
    lst_width : float, optional
        The width of the LST grid. Default is 2pi. Note that this is not the width of
        each LST bin, which is given by dlst. Further note that the LST grid is always
        equally divided into 2pi, regardless of `lst_width`.
    ntimes_per_file : int, optional
        The number of LST bins to include in each output file. Default is 60.
    blts_are_rectangular : bool, optional
        If True, assume that the data-layout in the input files is rectangular in
        baseline-times. This will be determined if not given.
    time_axis_faster_than_bls : bool, optional
        If True, assume that the time axis moves faster than the baseline axis in the
        input files. This will be determined if not given.
    jd_regex : str, optional
        Regex to use to extract the JD from the file name. Set to None or empty
        to force the LST-matching to use the LSTs in the metadata within the file.
    lst_branch_cut
        The LST at which to branch cut the LST grid for file writing. The JDs in the
        output LST-binned files will be *lowest* at the lst_branch_cut, and all file
        names will have LSTs that are higher than lst_branch_cut. If None, this will
        be determined automatically by finding the largest gap in LSTs and starting
        AFTER it.

    Returns
    -------
    config : dict
        The configuration dictionary that was written to the YAML file.
    """
    config_file = Path(config_file)
    if config_file.exists() and not clobber:
        raise IOError(f"{config_file} exists and clobber is False")

    lst_grid, matched_files = config_lst_bin_files(
        data_files=data_files,
        dlst=dlst,
        atol=atol,
        lst_start=lst_start,
        lst_width=lst_width,
        blts_are_rectangular=blts_are_rectangular,
        time_axis_faster_than_bls=time_axis_faster_than_bls,
        jd_regex=jd_regex,
        ntimes_per_file=ntimes_per_file,
    )

    # Get the best lst_branch_cut by finding the largest gap in LSTs and starting
    # AFTER it
    if lst_branch_cut is None:
        lst_branch_cut = float(utils.get_best_lst_branch_cut(np.concatenate(lst_grid)))

    dlst = lst_grid[0][1] - lst_grid[0][0]
    # Make it a real list of floats to make the YAML easier to read
    lst_grid = [[float(lst) for lst in lsts] for lsts in lst_grid]

    # now matched files is a list of output files, each containing a list of nights,
    # each containing a list of files
    logger.info("Getting metadata from first file...")

    def get_meta():
        for outfile in matched_files:
            for night in outfile:
                for i, fl in enumerate(night):
                    return fl

    meta = get_meta()

    tint = np.median(meta.integration_time)
    if not np.all(np.abs(np.diff(np.diff(meta.times))) < 1e-6):
        raise ValueError(
            "All integrations must be of equal length (BDA not supported), got diffs: "
            f"{np.diff(meta.times)}"
        )

    matched_files = [
        [[str(m.path) for m in night] for night in outfiles]
        for outfiles in matched_files
    ]

    output = {
        "config_params": {
            "dlst": float(dlst),
            "atol": atol,
            "lst_start": lst_start,
            "lst_width": lst_width,
            "jd_regex": jd_regex,
        },
        "lst_grid": lst_grid,
        "matched_files": matched_files,
        "metadata": {
            "x_orientation": meta.x_orientation,
            "blts_are_rectangular": meta.blts_are_rectangular,
            "time_axis_faster_than_bls": meta.time_axis_faster_than_bls,
            "start_jd": int(meta.times[0]),
            "integration_time": float(tint),
            "lst_branch_cut": lst_branch_cut,
        },
    }

    with open(config_file, "w") as fl:
        yaml.safe_dump(output, fl)

    return output
