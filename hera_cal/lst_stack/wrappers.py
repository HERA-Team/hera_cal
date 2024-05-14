from __future__ import annotations

from pathlib import Path
from typing import Sequence
import argparse
import os
from functools import partial
from . import io
import logging
import numpy as np
from .binning import lst_bin_files_from_config
from .averaging import reduce_lst_bins
from .config import LSTConfig, LSTConfigSingle
from pyuvdata import UVData
from pyuvdata.uvdata.uvh5 import UVH5
import h5py

logger = logging.getLogger(__name__)


def lst_bin_files_single_outfile(
    config: LSTConfigSingle,
    outdir: str | Path | None = None,
    history: str = "",
    fname_format: str = "zen.{kind}.{lst:7.5f}.uvh5",
    overwrite: bool = False,
    rephase: bool = False,
    bl_chunk_size: int | None = None,
    write_kwargs: dict | None = None,
    freq_min: float | None = None,
    freq_max: float | None = None,
    output_inpainted: bool | None = None,
    output_flagged: bool = True,
    write_med_mad: bool = False,
) -> dict[str, Path]:
    """
    Bin data files into LST bins, and write all bins to disk in a single file.

    Note that this is generally not meant to be called directly, but rather through
    the :func:`lst_bin_files` function.

    The mode(s) in which the function does the averaging can be specified via the
    `output_inpainted` and `output_flagged` options.
    Algorithmically, there are two modes of averaging: either flagged data is *ignored*
    in the average (and in Nsamples) or the flagged data is included in the average but
    ignored in Nsamples. These are called "flagged" and "inpainted" modes respectively.
    The latter only makes sense if the data in the flagged regions has been inpainted,
    rather than left as raw data. For delay-filtered data, either mode is equivalent,
    since the flagged data itself is set to zero. By default, this function *only*
    uses the "flagged" mode, *unless* the configuration object has inpainted files
    specified, which indicates that the files are definitely in-painted, and in this
    case it will use *both* modes by default.

    .. note:: The "flagged" mode as implemented in this function is *not* spectrally
              smooth if the input Nsamples are not spectrally uniform.

    Parameters
    ----------
    config
        An LSTConfigSingle object that contains all the information needed to bin the
        data files into LST bins.
    outdir
        The output directory. If not provided, this is set to the lowest-level common
        directory for all data files.
    history
        Custom history string to insert into the output file.
    fname_format
        A formatting string to use to write the output file. This can have the following
        fields: "kind" (which will evaluate to one of 'LST', 'STD', 'GOLDEN' or 'REDUCEDCHAN'),
        "lst" (which will evaluate to the LST of the bin), and "pol" (which will evaluate
        to the polarization of the data). Example: "zen.{kind}.{lst:7.5f}.uvh5"
    overwrite
        If True, overwrite output files.
    rephase
        If True, rephase data points in LST bin to center of bin.
    bl_chunk_size
        The number of baselines to load at a time. If None, load all baselines at once.
    write_kwargs
        Arguments to pass to :func:`create_lstbin_output_file`.
    freq_min
        The minimum frequency to include in the output files. If not provided, this
        is set to the minimum frequency in the first data file on the first night.
    freq_max
        The maximum frequency to include in the output files. If not provided, this
        is set to the maximum frequency in the first data file on the first night.
    output_inpainted
        If True, output data LST-binned in in-painted mode. This mode does *not* flag
        data for the averaging, assuming that data that has flags has been in-painted
        to improve spectral smoothness. It does take the flags into account for the
        LST-binned Nsamples, however.
    output_flagged
        If True, output data LST-binned in flagged mode. This mode *does* apply flags
        to the data before averaging. It will yield the same Nsamples as the in-painted
        mode, but simply ignores flagged data for the average, which can yield less
        spectrally-smooth LST-binned results.
    write_med_mad
        If True, write out the median and MAD of the data in each LST bin.

    Returns
    -------
    out_files
        A dict of output files, keyed by the type of file (e.g. 'LST', 'STD', 'GOLDEN',
        'REDUCEDCHAN').

    Notes
    -----
    It is worth describing in a bit more detail what is actually _in_ the output files.
    The "LST" files contain the LST-binned data, with the data averaged over each LST
    bin. There are two possible modes for this: inpainted and flagged. In inpainted mode,
    the data in flagged bl-channel-pols is used for the averaging, as it is considered
    to be in-painted. This gives the most spectrally-smooth results. In order to ignore
    a particular bl-channel-pol while still using inpaint mode, supply a "where inpainted"
    file, which should be a UVFlag object that specifies which bl-channel-pols are
    inpainted in the associated data file. Anything that's flagged but not inpainted is
    ignored for the averaging. In this inpainted mode, the Nsamples are the number of
    un-flagged integrations (whether they were in-painted or not). The LST-binned flags
    are only set to True if ALL of the nights for a given bl-channel-pol are flagged
    (again, whether they were in-painted or not). In flagged mode, both the Nsamples
    and Flags are the same as inpainted mode. The averaged data, however, ignores any
    flagged data. This can lead to less spectrally-smooth results. The "STD" files
    contain LST-binned "data" that is the standard deviation of the data in each LST-bin.
    This differs between inpainted and flagged modes in the same way as the "LST" files:
    in inpainted mode, the flagged and inpainted data is used for calculating the sample
    variance, while in flagged mode, only the unflagged data is used. The final flags
    in the "STD" files is equivalent to that in the "LST" files. The Nsamples in the
    "STD" files is actually the number of unflagged nights in the LST bin (so, not the
    sum of Nsamples), where "unflagged" really does mean unflagged -- whether inpainted
    or not.

    One note here about what is considered "flagged" vs. "flagged and inpainted" vs
    "flagged and not inpainted". In flagged mode, there are input flags that exist in the
    input files. These are potentially *augmented* by sigma clipping within the LST
    binner, and also by flagging whole LST bins if they have too few unflagged integrations.
    In inpainted mode, input flags are considered as normal flags. However, only
    "non-inpainted" flags are *ignored* for the averaging. By default, all flagged data
    is considered to to be in-painted UNLESS it is a blt-pol that is fully flagged (i.e.
    all channels are flagged for an integration for a single bl and pol). However, you
    can tell the routine that other data is NOT in-painted by supplying a "where inpainted"
    file. Now, integrations in LST-bins that end up having "too few" unflagged
    integrations will be flagged inside the binner, however in inpainted mode, if these
    are considered "inpainted", they will still be used in averaging (this means they
    will have "valid" data for the average, but their average will be flagged).
    On the other hand, flags that are applied by sigma-clipping will be considered
    NOT inpainted, i.e. those data will be ignored in the averaged, just like flagging
    mode. In this case, either choice is bad: to include them in the average is bad
    because even though they may have been actually in-painted, whatever value they have
    is clearly triggering the sigma-clipper and is therefore an outlier. On the other
    hand, to ignore them is bad because the point of in-painting mode is to get
    smoother spectra, and this negates that. So, it's best just to not do sigma-clipping
    in inpainted mode.
    """
    # Break out early if this file has no data
    if all(len(x) == 0 for x in config.time_indices):
        return

    write_kwargs = write_kwargs or {}

    # get outdir
    if outdir is None:
        outdir = os.path.dirname(os.path.commonprefix(config.matched_files))

    inpaint_modes = io._configure_inpainted_mode(
        output_flagged, output_inpainted, config.inpaint_files
    )

    # make it a bit easier to create the outfiles
    uvd_template = io.create_empty_uvd(
        pols=config.pols,
        file_list=config.matched_metas,
        history=history,
        antpairs=config.autopairs + config.antpairs,
        start_jd=config.properties['first_jd'],
        freq_min=freq_min,
        freq_max=freq_max,
        lst_branch_cut=config.properties["lst_branch_cut"],
        lsts=config.lst_grid
    )

    create_file = partial(
        io.create_lstbin_output_file,
        uvd_template=uvd_template,
        outdir=outdir,
        overwrite=overwrite,
        **write_kwargs,
    )

    out_files = {}
    for inpaint_mode in inpaint_modes:
        kinds = ["LST", "STD"]
        if write_med_mad:
            kinds += ["MED", "MAD"]
        for kind in kinds:
            key = (kind, inpaint_mode)

            fname = io.format_outfile_name(
                lst=config.lst_grid_edges[0],
                pols=config.pols,
                fname_format=fname_format,
                inpaint_mode=inpaint_mode,
                lst_branch_cut=config.properties["lst_branch_cut"],
                kind=kind
            )

            out_files[key] = create_file(fname=fname)

    # Split up the baselines into chunks that will be LST-binned together.
    # This is just to save on RAM.
    if bl_chunk_size is None:
        bl_chunk_size = len(config.antpairs)
    else:
        bl_chunk_size = min(bl_chunk_size, len(config.antpairs))

    n_bl_chunks = int(np.ceil(len(config.antpairs) / bl_chunk_size))

    def _process_blchunk(
        bl_chunk: list[tuple[int, int]],
        nbls_so_far: int,
    ):
        """Process a single chunk of baselines."""
        stacks: list[LSTStack] = lst_bin_files_from_config(
            config,
            bl_chunk_to_load=bl_chunk,
            nbl_chunks=n_bl_chunks,
            rephase=rephase,
            freq_min=freq_min,
            freq_max=freq_max,
        )
        chunk_size = stacks[0].Nbls
        slc = slice(nbls_so_far, nbls_so_far + chunk_size)

        dshape = (chunk_size, stacks[0].Nfreqs, stacks[0].Npols)

        for inpainted in inpaint_modes:
            for lstidx, stack in enumerate(stacks):

                rdc = reduce_lst_bins(
                    lststack=stack,
                    inpainted_mode=inpainted,
                    get_mad=write_med_mad,
                )
                write = partial(
                    uvd_template.write_uvh5_part,
                    blt_inds=np.arange(nbls_so_far, nbls_so_far + chunk_size) * len(stacks) + lstidx,
                    flag_array=rdc['flags'],
                )

                write(
                    filename=out_files[("LST", inpainted)],
                    data_array=rdc["data"],
                    nsample_array=rdc["nsamples"],
                )

                write(
                    filename=out_files[("STD", inpainted)],
                    data_array=rdc["std"],
                    nsample_array=rdc["days_binned"],
                )

                if write_med_mad:
                    write(
                        filename=out_files[("MED", inpainted)],
                        data_array=rdc["median"],
                        nsample_array=rdc["nsamples"],
                    )
                    write(
                        filename=out_files[("MAD", inpainted)],
                        data_array=rdc["mad"],
                        nsample_array=rdc["days_binned"],
                    )
        return chunk_size

    nbls_so_far = _process_blchunk('autos', 0)

    for i in range(n_bl_chunks):
        logger.info(f"Baseline Chunk {i + 1} / {n_bl_chunks}")
        nbls_so_far += _process_blchunk(i, nbls_so_far=nbls_so_far)

    return out_files


def lst_bin_files(
    config_file: str | Path | LSTConfig,
    output_file_select: int | Sequence[int] | None = None,
    **kwargs,
) -> list[dict[str, Path]]:
    """
    LST bin a series of UVH5 files.

    This takes a series of UVH5 files where each file has the same frequency bins and
    pols, grids them onto a common LST grid, and then averages all integrations
    that appear in that LST bin. It writes a series of UVH5 files, as configured in the
    `config_file`, including the LST-averaged data, the standard deviation of the data
    in each LST bin, optional full data across nights for each LST-bin with a reduced
    number of frequency channels, and optionally the full data across nights (and all
    channels) for a 'GOLDEN' subset of LST bins.

    Parameters
    ----------
    config_file
        A configuration file to use. This should be a YAML file constructed by
        :func:`~make_lst_bin_config_file`, encoding the configuration of the LST
        grid, and the matching of input data files to LST bins.
    output_file_select
        If provided, this is a list of integers that select which output files to
        write. For example, if this is [0, 2], then only the first and third output
        files will be written. This is useful for parallelizing the LST binning.
    include_autos
        If True, include autocorrelations in the LST binning. If False, ignore them.
    **kwargs
        Additional keyword arguments are passed to :func:`~lstbin.lst_bin_files_single_outfile`.


    Returns
    -------
    list of dicts
        list of dicts -- one for each output file.
        Each dict contains keys that indicate the type of output file (e.g. 'LST', 'STD',
        'REDUCEDCHAN', 'GOLDEN') and values that are the path to that file.
    """
    if not isinstance(config_file, LSTConfig):
        config = LSTConfig.from_file(config_file)
    else:
        config = config_file

    if output_file_select is None:
        # remember config.lst_grid is 2D -- (n_outfiles, nlsts_per_file)
        output_file_select = list(range(len(config.lst_grid)))
    elif isinstance(output_file_select, int):
        output_file_select = [output_file_select]

    output_file_select = [int(i) for i in output_file_select]

    if max(output_file_select) >= len(config.lst_grid):
        raise ValueError(
            "output_file_select must be <= the number of output files"
        )

    output_files = []
    for outfile_index in output_file_select:
        thisconf = config.at_single_outfile(outfile_index)

        output_files.append(
            lst_bin_files_single_outfile(
                config=thisconf,
                **kwargs,
            )
        )

    return output_files


def lst_bin_arg_parser():
    """
    arg parser for lst_bin_files() function.
    """
    a = argparse.ArgumentParser(
        description=(
            "drive script for lstbin.lst_bin_files(). "
            "data_files argument must be quotation-bounded "
            "glob-parsable search strings to nightly data. For example: \n"
            "'2458042/zen.2458042.*.xx.HH.uv' '2458043/zen.2458043.*.xx.HH.uv' \n"
            "Consult lstbin.lst_bin_files() for further details on functionality."
        )
    )
    a.add_argument(
        "lstconfig",
        type=str,
        help="config file produced by lstbin.make_lst_bin_config_file",
    )
    a.add_argument(
        "cfgtoml",
        type=str,
        help="TOML configuration file for lstbin.lst_bin_files_single_outfile",
    )
    a.add_argument(
        "--overwrite", default=False, action="store_true", help="overwrite output files"
    )
    a.add_argument(
        "--output_file_select",
        default=None,
        nargs="*",
        type=int,
        help="list of output file integers to run on. Default is all output files.",
    )

    return a
