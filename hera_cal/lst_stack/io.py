from __future__ import annotations
from pyuvdata.uvdata import FastUVH5Meta
import warnings
import os
import numpy as np
from pathlib import Path
import h5py
from .. import io
from .. import utils
import logging
from pyuvdata import utils as uvutils

logger = logging.getLogger(__name__)


def apply_calfile_rules(
    data_files: list[list[str]],
    calfile_rules: list[tuple[str, str]],
    ignore_missing: bool,
) -> tuple[list[list[str]], list[list[str]]]:
    """
    Apply a set of rules to convert data file names to calibration file names.

    Parameters
    ----------
    data_files : list of list of str
        List of lists of data file names. Each inner list is a night of data.
    calfile_rules : list of tuple of str
        List of rules to apply. Each rule is a tuple of two strings, the first
        is the string to replace, the second is the string to replace it with.
        Each 2-tuple constitutes one "rule" and they are applied in order to each file.
        All rules are applied to all files.
    ignore_missing : bool
        If True, ignore missing calibration files. If False, raise an error.

    Returns
    -------
    data_files : list of list of str
        List of lists of data file names. Each inner list is a night of data.
        Files that were removed due to missing calibration files are removed.
    input_cals : list of list of str
        List of lists of calibration file names. Each inner list is a night of data.
        Any calibration files that were missing are not included.
    """
    input_cals = []
    for night, dflist in enumerate(data_files):
        this = []
        input_cals.append(this)
        missing = []
        for df in dflist:
            cf = df
            for rule in calfile_rules:
                cf = cf.replace(rule[0], rule[1])

            if os.path.exists(cf):
                this.append(cf)
            elif ignore_missing:
                warnings.warn(f"Calibration file {cf} does not exist")
                missing.append(df)
            else:
                raise IOError(f"Calibration file {cf} does not exist")
        data_files[night] = [df for df in dflist if df not in missing]
    return data_files, input_cals


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

    if not cal_files:
        cal_files = [[None for _ in dm] for dm in data_metas]

    if not where_inpainted_files:
        where_inpainted_files = [[None for _ in dm] for dm in data_metas]

    tinds = []
    all_lsts = []
    file_list = []
    time_arrays = []
    cals = []
    where_inpainted = []

    # This loop just gets the number of times that we'll be reading.
    # Even though the input files should already have been configured to be those
    # that fall within the output LST range, we still need to read them in to
    # check exactly which time integrations we require.

    for night, callist, inplist in zip(data_metas, cal_files, where_inpainted_files):
        for meta, cal, inp in zip(night, callist, inplist):
            tarr = meta.times
            lsts = meta.lsts % (2 * np.pi)
            lsts[lsts < lstmin] += 2 * np.pi

            tind = (lsts > lstmin) & (lsts < lstmax)

            if np.any(tind):
                tinds.append(tind)
                time_arrays.append(tarr[tind])
                all_lsts.append(lsts[tind])
                file_list.append(meta)
                cals.append(cal)
                where_inpainted.append(inp)

    return tinds, time_arrays, all_lsts, file_list, cals, where_inpainted


def _get_where_inpainted_files(
    data_files: list[list[str | Path]],
    where_inpainted_file_rules: list[tuple[str, str]] | None,
) -> list[list[str | Path]] | None:
    if where_inpainted_file_rules is None:
        return None

    where_inpainted_files = []
    for dflist in data_files:
        this = []
        where_inpainted_files.append(this)
        for df in dflist:
            wif = str(df)
            for rule in where_inpainted_file_rules:
                wif = wif.replace(rule[0], rule[1])
            if os.path.exists(wif):
                this.append(wif)
            else:
                raise IOError(f"Where inpainted file {wif} does not exist")

    return where_inpainted_files


def _configure_inpainted_mode(output_flagged, output_inpainted, where_inpainted_files):
    # Sort out defaults for inpaint/flagging mode
    if output_inpainted is None:
        output_inpainted = bool(where_inpainted_files)

    if not output_inpainted and not output_flagged:
        raise ValueError(
            "Both output_inpainted and output_flagged are False. One must be True."
        )

    return output_flagged, output_inpainted


def create_lstbin_output_file(
    outdir: Path,
    kind: str,
    lst: float,
    pols: list[str],
    file_list: list[FastUVH5Meta],
    start_jd: float,
    times: np.ndarray | None = None,
    lsts: np.ndarray | None = None,
    history: str = "",
    fname_format: str = "zen.{kind}.{lst:7.5f}.{inpaint_mode}.uvh5",
    overwrite: bool = False,
    antpairs: list[tuple[int, int]] | None = None,
    freq_min: float | None = None,
    freq_max: float | None = None,
    channels: np.ndarray | list[int] | None = None,
    vis_units: str = "Jy",
    inpaint_mode: bool | None = None,
    lst_branch_cut: float = 0.0,
) -> Path:
    outdir = Path(outdir)

    # update history
    file_list_str = "-".join(ff.path.name for ff in file_list)
    file_history = f"{history} Input files: {file_list_str}"
    _history = file_history + utils.history_string()

    if lst < lst_branch_cut:
        lst += 2 * np.pi

    fname = fname_format.format(
        kind=kind,
        lst=lst,
        pol="".join(pols),
        inpaint_mode=(
            "inpaint" if inpaint_mode else ("flagged" if inpaint_mode is False else "")
        ),
    )
    # There's a weird gotcha with pathlib where if you do path / "/file.name"
    # You get just "/file.name" which is in root.
    if fname.startswith("/"):
        fname = fname[1:]
    fname = outdir / fname

    logger.info(f"Initializing {fname}")

    # check for overwrite
    if fname.exists() and not overwrite:
        raise FileExistsError(f"{fname} exists, not overwriting")

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
    uvd_template.initialize_uvh5_file(str(fname.absolute()), clobber=overwrite)

    return fname


def write_baseline_slc_to_file(
    fl: Path, slc: slice, data: np.ndarray, flags: np.ndarray, nsamples: np.ndarray
):
    """Write a baseline slice to a file."""
    with h5py.File(fl, "r+") as f:
        ntimes = int(f["Header"]["Ntimes"][()])
        timefirst = bool(f["Header"]["time_axis_faster_than_bls"][()])
        if not timefirst and ntimes > 1:
            raise NotImplementedError("Can only do time-first files for now.")

        slc = slice(slc.start * ntimes, slc.stop * ntimes, 1)
        f["Data"]["visdata"][slc] = data.reshape((-1, data.shape[2], data.shape[3]))
        f["Data"]["flags"][slc] = flags.reshape((-1, data.shape[2], data.shape[3]))
        f["Data"]["nsamples"][slc] = nsamples.reshape(
            (-1, data.shape[2], data.shape[3])
        )
