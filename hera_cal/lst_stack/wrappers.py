from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence
import argparse
import os
from functools import partial
from ..red_groups import RedundantGroups
from .. import abscal
from pyuvdata.uvdata import FastUVH5Meta
from . import io
import logging
from . import config as cfg
import numpy as np
from .binning import get_lst_bins
import yaml
from .binning import lst_bin_files_for_baselines
from .averaging import reduce_lst_bins

logger = logging.getLogger(__name__)


def lst_bin_files_single_outfile(
    config_opts: dict[str, Any],
    metadata: dict[str, Any],
    lst_bins: np.ndarray,
    data_files: list[list[str]],
    calfile_rules: list[tuple[str, str]] | None = None,
    ignore_missing_calfiles: bool = False,
    outdir: str | Path | None = None,
    reds: RedundantGroups | None = None,
    redundantly_averaged: bool | None = None,
    only_last_file_per_night: bool = False,
    history: str = "",
    fname_format: str = "zen.{kind}.{lst:7.5f}.uvh5",
    overwrite: bool = False,
    rephase: bool = False,
    Nbls_to_load: int | None = None,
    ignore_flags: bool = False,
    include_autos: bool = True,
    ex_ant_yaml_files=None,
    ignore_ants: tuple[int] = (),
    write_kwargs: dict | None = None,
    save_channels: list[int] = (),
    golden_lsts: tuple[float] = (),
    sigma_clip_thresh: float | None = None,
    sigma_clip_min_N: int = 4,
    sigma_clip_subbands: list[tuple[int, int]] | None = None,
    sigma_clip_type: Literal['direct', 'mean', 'median'] = 'direct',
    sigma_clip_use_autos: bool = False,
    flag_below_min_N: bool = False,
    flag_thresh: float = 0.7,
    freq_min: float | None = None,
    freq_max: float | None = None,
    output_inpainted: bool | None = None,
    output_flagged: bool = True,
    where_inpainted_file_rules: list[tuple[str, str]] | None = None,
    sigma_clip_in_inpainted_mode: bool = False,
    write_med_mad: bool = False,
) -> dict[str, Path]:
    """
    Bin data files into LST bins, and write all bins to disk in a single file.

    Note that this is generally not meant to be called directly, but rather through
    the :func:`lst_bin_files` function.

    The mode(s) in which the function does the averaging can be specified via the
    `output_inpainted`, `output_flagged` and `where_inpainted_file_rules` options.
    Algorithmically, there are two modes of averaging: either flagged data is *ignored*
    in the average (and in Nsamples) or the flagged data is included in the average but
    ignored in Nsamples. These are called "flagged" and "inpainted" modes respectively.
    The latter only makes sense if the data in the flagged regions has been inpainted,
    rather than left as raw data. For delay-filtered data, either mode is equivalent,
    since the flagged data itself is set to zero. By default, this function *only*
    uses the "flagged" mode, *unless* the `where_inpainted_file_rules` option is set,
    which indicates that the files are definitely in-painted, and in this case it will
    use *both* modes by default.

    .. note:: Both "flagged" and "inpainted" mode as implemented in this function are
        *not* spectrally smooth if the input Nsamples are not spectrally uniform.

    Parameters
    ----------
    config_opts
        A dictionary of LST-bin configuration options. Exactly the "config_params"
        section of the configuration file produced by :func:`make_lst_bin_config`.
    metadata
        A dictionary of metadata for the LST binning. Exactly the "metadata" section
        of the configuration file produced by :func:`make_lst_bin_config`.
    lst_bins
        An array of LST bin *centres* in radians. These should be *one* of the entries
        of the "lst_bins" section of the configuration file produced by
        :func:`make_lst_bin_config` (which is a list of arrays of LST bin centres, one
        for each output file).
    data_files
        A list of lists of data files to LST bin. Each list of files is treated as coming
        from a single night. These should be *one* of the entries of the "matched_files"
        section of the configuration file produced by :func:`make_lst_bin_config` (which
        is a list of lists of lists of data files, one for each output file).
    calfile_rules
        A list of tuples of strings. Each tuple is a pair of strings that are used to
        replace the first string with the second string in the data file name to get
        the calibration file name. For example, providing [(".uvh5", ".calfits")] will
        generate a list of calfiles that have the same basename as the data files, but
        with the extension ".calfits" instead of ".uvh5". Multiple entries to the list
        are allowed, and the replacements are applied in order. If the resulting calfile
        name does not exist, the data file is ignored.
    ignore_missing_calfiles
        If True, ignore missing calibration files (i.e. just drop the corresponding
        data file from the binning). If False, raise an error if a calfile is missing.
    outdir
        The output directory. If not provided, this is set to the lowest-level common
        directory for all data files.
    reds
        A :class:`RedundantGroups` object describing the redundant groups of the array.
        If not provided, this is calculated from the first data file on the first night.
    redundantly_averaged
        If True, the data are assumed to have been redundantly averaged. If not provided
        this is set to True if the first data file on the first night has been redundantly
        averaged, and False otherwise.
    only_last_file_per_night
        If True, only the last file from each night is used to infer the observed
        antpairs. Setting to False can be very slow for large data sets, and is almost
        never necessary, as the antpairs observed are generally set per-night.
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
    Nbls_to_load
        The number of baselines to load at a time. If None, load all baselines at once.
    ignore_flags
        If True, ignore flags when binning data.
    include_autos
        If True, include autocorrelations when binning data.
    ex_ant_yaml_files
        A list of yaml files that specify which antennas to exclude from each
        input data file.
    ignore_ants
        A list of antennas to ignore when binning data.
    write_kwargs
        Arguments to pass to :func:`create_lstbin_output_file`.
    save_channels
        A list of channels for which to save the a full file of LST-gridded data.
        One REDUCEDCHAN file is saved for each output file, corresponding to the
        first LST-bin in that file. The data in that file will have the shape
        ``(Nbls*Ndays, Nsave_chans, Npols)``. This can be helpful for debugging.
    golden_lsts
        A list of LSTs for which to save a full file of LST-aligned (but not
        averaged) data. One GOLDEN file is saved for each ``golden_lst``, with shape
        ``(Nbls*Ndays, Nfreqs, Npols)`` -- that is, the normal "time" axis of a
        UVData array is replaced by a "night" axis. This is an easy way to load up
        the full data that goes into a particular LST-bin after the fact.
    sigma_clip_thresh
        If provided, this is the threshold for sigma clipping. If this is provided,
        then the data is sigma clipped before being averaged. This is done for each
        (antpair, pol, channel) combination.
    sigma_clip_min_N
        The minimum number of integrations for a particular (antpair, pol, channel)
        within an LST-bin required to perform sigma clipping. If `flag_below_min_N`
        is False, these (antpair,pol,channel) combinations are not flagged by
        sigma-clipping (otherwise they are).
    sigma_clip_subbands
        A list of 2-tuples of integers, specifying the start and end indices of the
        frequency axis to perform sigma clipping over. If None, the entire frequency
        axis is used at once.
    sigma_clip_type
        The type of sigma clipping to perform. If ``direct``, each datum is flagged
        individually. If ``mean`` or ``median``, an entire sub-band of the data is
        flagged if its mean (absolute) zscore is beyond the threshold.
    sigma_clip_use_autos
        If True, use the autos to predict the standard deviation for each baseline
        over nights for use in sigma-clipping. If False, use the median absolute
        deviation over nights for each baseline.
    flag_below_min_N
        If True, flag all (antpair, pol,channel) combinations  for an LST-bin that
        contiain fewer than `flag_below_min_N` unflagged integrations within the bin.
    flag_thresh
        The fraction of integrations for a particular (antpair, pol, channel) combination
        within an LST-bin that can be flagged before that combination is flagged
        in the LST-average.
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
    where_inpainted_file_rules
        Rules to transform the input data file names into the corresponding "where
        inpainted" files (which should be in UVFlag format). If provided, this indicates
        that the data itself is in-painted, and the `output_inpainted` mode will be
        switched on by default. These files should specify which data is in-painted
        in the associated data file (which may be different than the in-situ flags
        of the data object). If not provided, but `output_inpainted` is set to True,
        all data-flags will be considered in-painted except for baseline-times that are
        fully flagged, which will be completely ignored.
    sigma_clip_in_inpainted_mode
        If True, sigma clip the data in inpainted mode (if sigma-clipping is turned on).
        This is generally not a good idea, since the point of inpainting is to get
        smoother spectra, and sigma-clipping creates non-uniform Nsamples, which can
        lead to less smooth spectra. This option is only here to enable sigma-clipping
        to be turned on for flagged mode, and off for inpainted mode.
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
    write_kwargs = write_kwargs or {}

    # Check that that there are the same number of input data files and
    # calibration files each night.
    input_cals = []
    if calfile_rules:
        data_files, input_cals = io.apply_calfile_rules(
            data_files, calfile_rules, ignore_missing=ignore_missing_calfiles
        )

    where_inpainted_files = io._get_where_inpainted_files(
        data_files, where_inpainted_file_rules
    )

    output_flagged, output_inpainted = io._configure_inpainted_mode(
        output_flagged, output_inpainted, where_inpainted_files
    )

    # Prune empty nights (some nights start with files, but have files removed because
    # they have no associated calibration)
    data_files = [df for df in data_files if df]
    input_cals = [cf for cf in input_cals if cf]
    if where_inpainted_files is not None:
        where_inpainted_files = [wif for wif in where_inpainted_files if wif]

    logger.info("Got the following numbers of data files per night:")
    for dflist in data_files:
        logger.info(f"{dflist[0].split('/')[-1]}: {len(dflist)}")

    data_metas = [
        [
            FastUVH5Meta(
                df,
                blts_are_rectangular=metadata["blts_are_rectangular"],
                time_axis_faster_than_bls=metadata["time_axis_faster_than_bls"],
            )
            for df in dflist
        ]
        for dflist in data_files
    ]

    # get outdir
    if outdir is None:
        outdir = os.path.dirname(os.path.commonprefix(abscal.flatten(data_files)))

    start_jd = metadata["start_jd"]

    # get metadata
    logger.info("Getting metadata from first file...")
    meta = data_metas[0][0]

    freq_array = np.squeeze(meta.freq_array)

    # reds will contain all of the redundant groups for the whole array, because
    # all the antenna positions are included in every file.
    antpos = dict(zip(meta.antenna_numbers, meta.antpos_enu))
    if reds is None:
        reds = RedundantGroups.from_antpos(antpos=antpos, include_autos=include_autos)

    if redundantly_averaged is None:
        # Try to work out if the files are redundantly averaged.
        # just look at the middle file from each night.
        for fl_list in data_metas:
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

    logger.info("Compiling all unflagged baselines...")
    all_baselines, all_pols = cfg.get_all_unflagged_baselines(
        data_metas,
        ex_ant_yaml_files,
        include_autos=include_autos,
        ignore_ants=ignore_ants,
        redundantly_averaged=redundantly_averaged,
        reds=reds,
        only_last_file_per_night=only_last_file_per_night,
    )
    nants0 = meta.header["antenna_numbers"].size

    # Do a quick check to make sure all nights at least have the same number of Nants
    for dflist in data_metas:
        _nants = dflist[0].header["antenna_numbers"].size
        dflist[0].close()
        if _nants != nants0:
            raise ValueError(
                f"Not all nights have the same number of antennas! Got {_nants} for "
                f"{dflist[0].path} and {nants0} for {meta.path} for {meta.path}"
            )

    # Split up the baselines into chunks that will be LST-binned together.
    # This is just to save on RAM.
    if Nbls_to_load is None:
        Nbls_to_load = len(all_baselines) + 1

    n_bl_chunks = len(all_baselines) // Nbls_to_load + 1

    # First, separate the auto baselines from the rest. We do this first because
    # we'd potentially like to use the autos to infer the noise, and do some
    # preliminary clipping.
    auto_bls = [bl for bl in all_baselines if bl[0] == bl[1]]
    all_baselines = [bl for bl in all_baselines if bl[0] != bl[1]]

    bl_chunks = [
        all_baselines[i * Nbls_to_load: (i + 1) * Nbls_to_load]
        for i in range(n_bl_chunks)
    ]
    bl_chunks = [blg for blg in bl_chunks if len(blg) > 0]

    dlst = config_opts["dlst"]
    lst_bin_edges = np.array(
        [x - dlst / 2 for x in lst_bins] + [lst_bins[-1] + dlst / 2]
    )

    (
        tinds,
        time_arrays,
        all_lsts,
        file_list,
        cals,
        where_inpainted_files,
    ) = io.filter_required_files_by_times(
        (lst_bin_edges[0], lst_bin_edges[-1]),
        data_metas,
        input_cals,
        where_inpainted_files,
    )

    # If we have no times at all for this file, just return
    if len(all_lsts) == 0:
        return {}

    all_lsts = np.concatenate(all_lsts)

    # The "golden" data is the full data over all days for a small subset of LST
    # bins. This works best if the LST bins are small (similar to the size of the
    # raw integrations). Usually, the length of "bins" will be zero.
    # NOTE: we work under the assumption that the LST bins are small, so that
    # each night only gets one integration in each LST bin. If there are *more*
    # than one integration in the bin, we take the first one only.
    golden_bins, _, mask = get_lst_bins(golden_lsts, lst_bin_edges)
    golden_bins = golden_bins[mask]
    logger.info(
        f"golden_lsts bins in this output file: {golden_bins}, "
        f"lst_bin_edges={lst_bin_edges}"
    )

    # make it a bit easier to create the outfiles
    create_outfile = partial(
        io.create_lstbin_output_file,
        outdir=outdir,
        pols=all_pols,
        file_list=file_list,
        history=history,
        fname_format=fname_format,
        overwrite=overwrite,
        antpairs=auto_bls + all_baselines,
        start_jd=start_jd,
        freq_min=freq_min,
        freq_max=freq_max,
        lst_branch_cut=metadata["lst_branch_cut"],
        **write_kwargs,
    )
    out_files = {}
    for inpaint_mode in [True, False]:
        if inpaint_mode and not output_inpainted:
            continue
        if not inpaint_mode and not output_flagged:
            continue

        kinds = ["LST", "STD"]
        if write_med_mad:
            kinds += ["MED", "MAD"]
        for kind in kinds:
            # Create the files we'll write to
            out_files[(kind, inpaint_mode)] = create_outfile(
                kind=kind,
                lst=lst_bin_edges[0],
                lsts=lst_bins,
                inpaint_mode=inpaint_mode,
            )

    _bin_files_for_bls = partial(
        lst_bin_files_for_baselines,
        data_files=file_list,
        lst_bin_edges=lst_bin_edges,
        freqs=freq_array,
        pols=all_pols,
        cal_files=cals,
        time_arrays=time_arrays,
        time_idx=tinds,
        ignore_flags=ignore_flags,
        rephase=rephase,
        antpos=antpos,
        lsts=all_lsts,
        redundantly_averaged=redundantly_averaged,
        reds=reds,
        freq_min=freq_min,
        freq_max=freq_max,
        where_inpainted_files=where_inpainted_files
    )

    def _process_blchunk(
        bl_chunk: list[tuple[int, int]],
        nbls_so_far: int,
        mean_autos: list[np.ndarray] | None = None,
    ):
        """Process a single chunk of baselines."""
        _, data, flags, nsamples, where_inpainted, binned_times = _bin_files_for_bls(
            antpairs=bl_chunk
        )

        slc = slice(nbls_so_far, nbls_so_far + len(bl_chunk))

        if "GOLDEN" not in out_files:
            # TODO: we're not writing out the where_inpainted data for the GOLDEN
            #       or REDUCEDCHAN files yet -- it looks like we'd have to write out
            #       completely new UVFlag files for this.
            out_files["GOLDEN"] = []
            for nbin in golden_bins:
                out_files["GOLDEN"].append(
                    create_outfile(
                        kind="GOLDEN",
                        lst=lst_bin_edges[nbin],
                        times=binned_times[nbin],
                    )
                )

            if save_channels and len(binned_times[0]) > 0:
                out_files["REDUCEDCHAN"] = create_outfile(
                    kind="REDUCEDCHAN",
                    lst=lst_bin_edges[0],
                    times=binned_times[0],
                    channels=list(save_channels),
                )

        if len(golden_bins) > 0:
            for fl, nbin in zip(out_files["GOLDEN"], golden_bins):
                io.write_baseline_slc_to_file(
                    fl=fl,
                    slc=slc,
                    data=data[nbin].transpose((1, 0, 2, 3)),
                    flags=flags[nbin].transpose((1, 0, 2, 3)),
                    nsamples=nsamples[nbin].transpose((1, 0, 2, 3)),
                )

        if "REDUCEDCHAN" in out_files:
            io.write_baseline_slc_to_file(
                fl=out_files["REDUCEDCHAN"],
                slc=slc,
                data=data[0][:, :, save_channels].transpose((1, 0, 2, 3)),
                flags=flags[0][:, :, save_channels].transpose((1, 0, 2, 3)),
                nsamples=nsamples[0][:, :, save_channels].transpose((1, 0, 2, 3)),
            )

        # Get the sigma clip scale
        if sigma_clip_use_autos and mean_autos is not None:
            dtdf = np.median(np.ediff1d(meta.times)) * (meta.freq_array[1] - meta.freq_array[0])
            predicted_var = [np.abs(auto)**2 / dtdf / ns for ns, auto in zip(nsamples, mean_autos)]
            sigma_clip_scale = [np.sqrt(p) for p in predicted_var]
        else:
            sigma_clip_scale = None

        for inpainted in [True, False]:
            if inpainted and not output_inpainted:
                continue
            if not inpainted and not output_flagged:
                continue

            rdc = reduce_lst_bins(
                data,
                flags,
                nsamples,
                where_inpainted=where_inpainted,
                inpainted_mode=inpainted,
                flag_thresh=flag_thresh,
                sigma_clip_thresh=(
                    None
                    if inpainted and not sigma_clip_in_inpainted_mode
                    else sigma_clip_thresh
                ),
                sigma_clip_min_N=sigma_clip_min_N,
                flag_below_min_N=flag_below_min_N,
                get_mad=write_med_mad,
                sigma_clip_subbands=sigma_clip_subbands,
                sigma_clip_type=sigma_clip_type,
                sigma_clip_scale=sigma_clip_scale,
            )

            io.write_baseline_slc_to_file(
                fl=out_files[("LST", inpainted)],
                slc=slc,
                data=rdc["data"],
                flags=rdc["flags"],
                nsamples=rdc["nsamples"],
            )

            io.write_baseline_slc_to_file(
                fl=out_files[("STD", inpainted)],
                slc=slc,
                data=rdc["std"],
                flags=rdc["flags"],
                nsamples=rdc["days_binned"],
            )

            if write_med_mad:
                io.write_baseline_slc_to_file(
                    fl=out_files[("MED", inpainted)],
                    slc=slc,
                    data=rdc["median"],
                    flags=rdc["flags"],
                    nsamples=rdc["nsamples"],
                )
                io.write_baseline_slc_to_file(
                    fl=out_files[("MAD", inpainted)],
                    slc=slc,
                    data=rdc["mad"],
                    flags=rdc["flags"],
                    nsamples=rdc["days_binned"],
                )
        return data, flags, nsamples, rdc

    auto_data, auto_flags, _, _ = _process_blchunk(auto_bls, 0)

    # Get the mean auto
    autos = [np.ma.masked_array(d, mask=f) for d, f in zip(auto_data, auto_flags)]
    # Now each mean has shape (nnights, 1, nfreqs, npols)
    auto_mean = [np.ma.mean(a, axis=1)[:, None] for a in autos]

    nbls_so_far = len(auto_bls)
    for bi, bl_chunk in enumerate(bl_chunks):
        logger.info(f"Baseline Chunk {bi+1} / {len(bl_chunks)}")
        # data/flags/nsamples are *lists*, with nlst_bins entries, each being an
        # array, with shape (times, bls, freqs, npols)
        _process_blchunk(bl_chunk, nbls_so_far=nbls_so_far, mean_autos=auto_mean)
        nbls_so_far += len(bl_chunk)

    return out_files


def lst_bin_files(
    config_file: str | Path,
    output_file_select: int | Sequence[int] | None = None,
    include_autos: bool = True,
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
    config_files
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
    with open(config_file, "r") as fl:
        configuration = yaml.safe_load(fl)

    config_opts = configuration["config_params"]
    lst_grid = configuration["lst_grid"]
    matched_files = configuration["matched_files"]
    metadata = configuration["metadata"]

    if output_file_select is None:
        output_file_select = list(range(len(matched_files)))
    elif isinstance(output_file_select, int):
        output_file_select = [output_file_select]
    output_file_select = [int(i) for i in output_file_select]

    if max(output_file_select) >= len(matched_files):
        raise ValueError(
            "output_file_select must be less than the number of output files"
        )

    meta = FastUVH5Meta(
        matched_files[0][0][0],
        blts_are_rectangular=metadata["blts_are_rectangular"],
        time_axis_faster_than_bls=metadata["time_axis_faster_than_bls"],
    )

    antpos = dict(zip(meta.antenna_numbers, meta.antpos_enu))
    reds = RedundantGroups.from_antpos(antpos=antpos, include_autos=include_autos)

    output_files = []
    for outfile_index in output_file_select:
        data_files = matched_files[outfile_index]
        output_files.append(
            lst_bin_files_single_outfile(
                config_opts=config_opts,
                metadata=metadata,
                lst_bins=lst_grid[outfile_index],
                data_files=data_files,
                reds=reds,
                include_autos=include_autos,
                **kwargs,
            )
        )

    return output_files


def lst_bin_arg_parser():
    """
    arg parser for lst_bin_files() function. data_files argument must be quotation-bounded
    glob-parsable search strings to nightly data. For example:

    '2458042/zen.2458042.*.xx.HH.uv' '2458043/zen.2458043.*.xx.HH.uv'
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
        "configfile",
        type=str,
        help="config file produced by lstbin.make_lst_bin_config_file",
    )
    a.add_argument(
        "--calfile-rules",
        nargs="*",
        type=str,
        help="rules to convert datafile names to calfile names. A series of two strings where the first will be replaced by the latter",
    )
    a.add_argument(
        "--fname-format",
        type=str,
        default="zen.{kind}.{lst:7.5f}.uvh5",
        help="filename format for output files. See docstring for details.",
    )
    a.add_argument(
        "--outdir", default=None, type=str, help="directory for writing output"
    )
    a.add_argument(
        "--overwrite", default=False, action="store_true", help="overwrite output files"
    )
    a.add_argument(
        "--rephase",
        default=False,
        action="store_true",
        help="rephase data to center of LST bin before binning",
    )
    a.add_argument(
        "--history", default=" ", type=str, help="history to insert into output files"
    )
    a.add_argument(
        "--output_file_select",
        default=None,
        nargs="*",
        type=int,
        help="list of output file integers to run on. Default is all output files.",
    )
    a.add_argument(
        "--vis_units", default="Jy", type=str, help="visibility units of output files."
    )
    a.add_argument(
        "--ignore_flags",
        default=False,
        action="store_true",
        help="Ignore flags in data files, such that all input data is included in binning.",
    )
    a.add_argument(
        "--Nbls_to_load",
        default=None,
        type=int,
        help="Number of baselines to load and bin simultaneously. Default is all.",
    )
    a.add_argument(
        "--ex_ant_yaml_files",
        default=None,
        type=str,
        nargs="+",
        help="list of paths to yamls with lists of antennas from each night to exclude lstbinned data files.",
    )
    a.add_argument(
        "--ignore-ants", default=(), type=int, nargs="+", help="ants to ignore"
    )
    a.add_argument(
        "--ignore-missing-calfiles",
        default=False,
        action="store_true",
        help="if true, any datafile with missing calfile will just be removed from lstbinning.",
    )
    a.add_argument(
        "--write_kwargs",
        default="{}",
        type=str,
        help="json dictionary of arguments to the uvh5 writer",
    )
    a.add_argument(
        "--golden-lsts",
        type=str,
        help="LSTS (rad) to save longitudinal data for, separated by commas",
    )
    a.add_argument(
        "--save-channels",
        type=str,
        help="integer channels separated by commas to save longitudinal data for",
    )
    a.add_argument(
        "--sigma-clip-thresh",
        type=float,
        help="sigma clip threshold for flagging data in an LST bin over time. Zero means no clipping.",
        default=None,
    )
    a.add_argument(
        "--sigma-clip-min-N",
        type=int,
        help="number of unflagged data points over time to require before considering sigma clipping",
        default=4,
    )
    a.add_argument(
        "--sigma-clip-subbands",
        type=str,
        help="Band-edges (as channel number) at which bands are separated for homogeneous sigma clipping, e.g. '0~10,100~500'",
        default=None,
    )
    a.add_argument(
        "--sigma-clip-type",
        type=str,
        default='direct',
        choices=['direct', 'mean', 'median'],
        help="How to threshold the absolute zscores for sigma clipping."
    )
    a.add_argument(
        "--sigma-clip-use-autos",
        action="store_true",
        help="whether to use the autos to predict the variance for sigma-clipping"
    )
    a.add_argument(
        "--flag-below-min-N",
        action="store_true",
        help="if true, flag all data in an LST bin if there are fewer than --sigma-clip-min-N unflagged data points over time",
    )
    a.add_argument(
        "--flag-thresh",
        type=float,
        help="fraction of integrations in an LST bin for a particular (antpair, pol, channel) that must be flagged for the entire bin to be flagged",
        default=0.7,
    )
    a.add_argument(
        "--redundantly-averaged",
        action="store_true",
        default=None,
        help="if true, assume input files are redundantly averaged",
    )
    a.add_argument(
        "--only-last-file-per-night",
        action="store_true",
        default=False,
        help="if true, only use the first and last file every night to obtain antpairs",
    )
    a.add_argument(
        "--freq-min",
        type=float,
        default=None,
        help="minimum frequency to include in lstbinning",
    )
    a.add_argument(
        "--freq-max",
        type=float,
        default=None,
        help="maximum frequency to include in lstbinning",
    )
    a.add_argument(
        "--no-flagged-mode",
        action="store_true",
        help="turn off output of flagged mode LST-binning",
    )
    a.add_argument(
        "--do-inpaint-mode",
        action="store_true",
        default=None,
        help="turn on inpainting mode LST-binning",
    )
    a.add_argument(
        "--where-inpainted-file-rules",
        nargs="*",
        type=str,
        help="rules to convert datafile names to where-inpainted-file names. A series of two strings where the first will be replaced by the latter",
    )
    a.add_argument(
        "--sigma-clip-in-inpainted-mode",
        action="store_true",
        default=False,
        help="allow sigma-clipping in inpainted mode",
    )
    a.add_argument(
        "--write-med-mad",
        action="store_true",
        default=False,
        help="option to write out MED/MAD files in addition to LST/STD files",
    )
    return a
