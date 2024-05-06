from __future__ import annotations
from pyuvdata.uvdata import FastUVH5Meta
import warnings
import numpy as np
from pathlib import Path
import h5py
from .. import io
from .. import utils
import logging
from pyuvdata import utils as uvutils
import re
from typing import Literal

logger = logging.getLogger(__name__)


def apply_filename_rules_to_file(
    filename: str,
    rules: list[tuple[str, str]],
    missing: Literal["warn", "raise", "ignore"] = "raise",
) -> str:
    """
    Apply a set of rules to convert a data file name to another file name.

    Parameters
    ----------
    filename : str
        Data file name.
    rules : list of tuple of str
        List of rules to apply. Each rule is a tuple of two strings, the first
        is the string to replace, the second is the string to replace it with.
        Each 2-tuple constitutes one "rule" and they are applied in order to each file.
        All rules are applied to all files.
    missing : str, optional
        What to do if a file is missing after applying the rules. Options are:
        "warn" : issue a warning and continue
        "raise" : raise an IOError
        "ignore" : silently continue
        "replace" : replace the missing file with None

    Returns
    -------
    new_filename : str
        New file name.
    """
    new_filename = filename
    for rule in rules:
        new_filename = re.sub(rule[0], rule[1], new_filename)

    if missing in ["warn", "ignore", 'raise'] and not Path(new_filename).exists():
        if missing == "warn":
            warnings.warn(f"File {new_filename} does not exist")
        elif missing == "raise":
            raise IOError(f"File {new_filename} does not exist")
        else:
            new_filename = None

    return new_filename


def apply_filename_rules(
    files: list[list[str]],
    rules: list[tuple[str, str]],
    missing: Literal["warn", "raise", "ignore"] = "raise",
) -> list[list[str]]:
    """
    Apply a set of rules to convert data file names to other file names.

    Parameters
    ----------
    files : arbitrarily-deep nested lists of strings
        Lists (of lists, of lists...) of data file names.
    rules : list of 2-tuples of str
        List of rules to apply. Each rule is a tuple of two strings, the first
        is the string to replace, the second is the string to replace it with.
        Each 2-tuple constitutes one "rule" and they are applied in order to each file.
        All rules are applied to all files. Each string can be a regex pattern.
    missing : str, optional
        What to do if a file is missing after applying the rules. Options are:
        "warn" : issue a warning and continue
        "raise" : raise an IOError
        "ignore" : silently continue
        "replace" : replace the missing file with None

    Returns
    -------
    output_files : arbitrarily-deep nested lists of strings
        List of lists of calibration file names. Each inner list is a night of data.
        Any calibration files that were missing are not included.
    """
    if isinstance(files, str):
        return apply_filename_rules_to_file(files, rules, missing=missing)

    elif len(files) == 0:
        return files

    if not isinstance(files[0], str):
        return [apply_filename_rules(f, rules, missing=missing) for f in files]

    return [apply_filename_rules_to_file(f, rules, missing) for f in files]


def filter_required_files_by_times(
    lst_range: tuple[float, float],
    data_metas: list[list[FastUVH5Meta]],
    cal_files: list[list[str]] | None = None,
    where_inpainted_files: list[list[str]] | None = None,
) -> tuple[list, list, list, list, list, list]:
    lstmin, lstmax = lst_range
    lstmin %= 2 * np.pi
    lstmax %= 2 * np.pi
    if lstmin > lstmax:
        lstmax += 2 * np.pi

    have_calfiles = cal_files is not None
    if not cal_files:
        cal_files = [[None for _ in dm] for dm in data_metas]

    have_inp = where_inpainted_files is not None
    if not where_inpainted_files:
        where_inpainted_files = [[None for _ in dm] for dm in data_metas]

    tinds = []
    all_lsts = []
    file_list = []
    cals = []
    where_inpainted = []

    # This loop just gets the number of times that we'll be reading.
    # Even though the input files should already have been configured to be those
    # that fall within the output LST range, we still need to read them in to
    # check exactly which time integrations we require.

    for night, callist, inplist in zip(data_metas, cal_files, where_inpainted_files):
        for meta, cal, inp in zip(night, callist, inplist):
            lsts = meta.lsts % (2 * np.pi)
            lsts[lsts < lstmin] += 2 * np.pi

            tind = np.argwhere((lsts > lstmin) & (lsts < lstmax)).flatten()

            if len(tind) > 0:
                tinds.append(tind)
                all_lsts.append(lsts[tind])
                file_list.append(meta)
                cals.append(cal)
                where_inpainted.append(inp)

    if not have_calfiles:
        cals = None
    if not have_inp:
        where_inpainted = None

    return tinds, all_lsts, file_list, cals, where_inpainted


def _configure_inpainted_mode(
    output_flagged, output_inpainted, where_inpainted_files
) -> list[bool]:
    # Sort out defaults for inpaint/flagging mode
    if output_inpainted is None:
        output_inpainted = bool(where_inpainted_files)

    if not output_inpainted and not output_flagged:
        raise ValueError(
            "Both output_inpainted and output_flagged are False. One must be True."
        )

    inpaint_modes = [True] * output_inpainted + [False] * output_flagged

    return inpaint_modes


def format_outfile_name(
    lst: float,
    pols: list[str],
    fname_format: str = "zen.{kind}.{lst:7.5f}.{inpaint_mode}.uvh5",
    inpaint_mode: bool | None = None,
    kind: str | None = None,
    lst_branch_cut: float = 0.0,
):
    if lst < lst_branch_cut:
        lst += 2 * np.pi

    kind = kind or "{kind}"
    return fname_format.format(
        lst=lst,
        pol="".join(pols),
        inpaint_mode=(
            "inpaint" if inpaint_mode else ("flagged" if inpaint_mode is False else "")
        ),
        kind=kind
    )


def create_empty_uvd(
    pols: list[str],
    file_list: list[FastUVH5Meta],
    start_jd: float,
    times: np.ndarray | None = None,
    lsts: np.ndarray | None = None,
    history: str = "",
    antpairs: list[tuple[int, int]] | None = None,
    freq_min: float | None = None,
    freq_max: float | None = None,
    channels: np.ndarray | list[int] | None = None,
    vis_units: str = "Jy",
    lst_branch_cut: float = 0.0,
):
    # update history
    file_list_str = "-".join(ff.path.name for ff in file_list)
    file_history = f"{history} Input files: {file_list_str}"
    _history = file_history + utils.history_string()

    freqs = np.squeeze(file_list[0].freq_array)
    if freq_min:
        freqs = freqs[freqs >= freq_min]
    if freq_max:
        freqs = freqs[freqs <= freq_max]
    if channels:
        channels = list(channels)
        freqs = freqs[channels]

    uvd_template = io.uvdata_from_fastuvh5(
        meta=file_list[0],
        antpairs=antpairs,
        lsts=lsts,
        times=times,
        history=_history,
        start_jd=start_jd,
        time_axis_faster_than_bls=True,
        vis_units=vis_units,
        lst_branch_cut=lst_branch_cut,
    )
    uvd_template.select(frequencies=freqs, polarizations=pols, inplace=True)

    # Need to set the polarization array manually because even though the select
    # operation does the down-select, it doesn't re-order the pols.
    uvd_template.polarization_array = np.array(
        uvutils.polstr2num(pols, x_orientation=uvd_template.x_orientation)
    )
    return uvd_template


def create_lstbin_output_file(
    uvd_template: UVData,
    outdir: Path,
    fname: str,
    kind: str | None = None,
    overwrite: bool = False,
) -> Path:
    outdir = Path(outdir)

    if kind:
        fname = fname.format(kind=kind)

    # There's a weird gotcha with pathlib where if you do path / "/file.name"
    # You get just "/file.name" which is in root.
    if fname.startswith("/"):
        fname = fname[1:]
    fname = outdir / fname

    if not fname.parent.exists():
        fname.parent.mkdir(parents=True)

    logger.info(f"Initializing {fname}")

    # check for overwrite
    if fname.exists() and not overwrite:
        raise FileExistsError(f"{fname} exists, not overwriting")

    uvd_template.initialize_uvh5_file(str(fname.absolute()), clobber=overwrite)

    return fname
