# -*- coding: utf-8 -*-
# Copyright 2019 the HERA Project
# Licensed under the MIT License

import numpy as np
from collections import OrderedDict as odict
import datetime
from hera_filters import dspec
import argparse
from astropy import constants
import copy
import fnmatch
from scipy import signal
import warnings
from pyuvdata import UVFlag
from pyuvdata import utils as uvutils

from . import io, apply_cal, utils, redcal
from .datacontainer import DataContainer
from .utils import echo
from .flag_utils import factorize_flags


def discard_autocorr_imag(data_container):
    """
    Helper function to discard all imaginary components in autocorrs in a datacontainer.

    Parameters
    ----------
    data_container: io.DataContainer dictionary

    Returns
    -------
    N/A modifies DataContainer in place.
    """
    for k in data_container:
        if k[0] == k[1]:
            data_container[k] = data_container[k].real + 0j


def find_discontinuity_edges(x, xtol=1e-3):
    """Find edges based on discontinuity in x-axis

    This function helps us find discontinuities in the x-axis which exist if we have several non-contiguous subbands in our data (for example, excluding the FM band).

    Parameters
    ----------
    x: array-like
        1d numpy array of x-values.
    xtol: float, optional
        fractional discontinuity in diff (relative to median discontinuity)
        to be used as a threshold for identifying discontinuities in the x-axis.
        positions where the diff is larger then the median diff is identified as a
        discontinuity.

    Returns
    -------
    edgesd: list
        list of 2-tuples of indices of (lower bound inclusive, upper-bound (exclusive))
        of contiguous x-indices.
    Examples:
            x = [0, 1, 4, 9] -> [(0, 2) (2, 3), (3, 4)]
            x = [0, 1, 2, 4, 5, 6, 7, 11, 12] -> [(0, 3), (3, 7), (7, 9)]
    """
    xdiff = np.diff(x)
    discontinuities = np.where(~np.isclose(xdiff, np.min(np.abs(xdiff)) * np.sign(xdiff[0]),
                               rtol=0.0, atol=np.abs(np.min(xdiff)) * xtol))[0]
    if len(discontinuities) == 0:
        edges = [(0, len(x))]
    elif len(discontinuities) == 1:
        edges = [(0, discontinuities[0] + 1), (discontinuities[0] + 1, len(x))]
    else:
        edges = [(0, discontinuities[0] + 1)]
        for i in range(len(discontinuities) - 1):
            edges.append((discontinuities[i] + 1, discontinuities[i + 1] + 1))
        edges.append((discontinuities[-1] + 1, len(x)))
    return edges


def truncate_flagged_edges(data_in, weights_in, x, ax='freq'):
    """
    cut away edge channels and integrations that are completely flagged

    Parameters
    ----------
    data_in : array-like, 2d (Ntimes, Nfreqs)
        data from which to remove edge integrations and channels that are completely flagged.
    weights_in : array-like, 2d
        weights to determine which edge integrations and channels are completely flagged
        (close to zero). Will also be truncated.
    x : array-like, 1-d (or 2-tuple/list of arrays)
        x-values for data axis that we are truncating.
        if ax=='both', should be a 2-tuple or 2-list of 1-d arrays.
    ax : string, optional
        axis to truncate flagged edges from. Should be 'freq' to truncate
        completely flagged edge channels, 'time' to truncate completely flagged
        edge integrations, or 'both' to do both.

    Returns
    -------
    xout : array-like 1d (or 2-list if ax=='both').
        x with completely flagged edges trimmed off.
    dout : array-like 2d.
        data_in with completely flagged edges trimmed off.
    wout : array-like 2d.
        weights_in with completely flagged edges trimmed off.
    edges : list of 2-tuples or 2-list of lists of 2-tuples
        the width of the edges trimmed.
        Example: Suppose we provide data 100 frequency channels and discontinuities
        after channel 37 and after channel 59 and
        suppose that channels [0, 12, 36, 37, 59, 60, 61] are flagged.
        Then edges = [(1, 2), (0, 1), (2, 0)]
        which is the number of edge channels flagged on each contiguous chunk.

        If ax == 'both' then a 2-list of lists of tuples
        first list is time dim, second list is freq dim.
    chunks : list of 2-tuples or 2-list of lists of 2-tuples
        List of start / end indices of each chunk that edges are applied to.
        If ax=='both' should be a 2-tuple or 2-list.
    """
    # if axis == 'time', just use freq mode
    # on transposed arrays.
    if ax == 'time':
        xout, dout, wout, edges, chunks = truncate_flagged_edges(data_in.T, weights_in.T, x)
        dout = dout.T
        wout = wout.T
    else:
        # Identify all contiguous chunks.
        if ax == 'both':
            chunks = find_discontinuity_edges(x[1])
        else:
            chunks = find_discontinuity_edges(x)
        inds_left = []
        inds_right = []
        # Identify edge channels that are flagged.
        for chunk in chunks:
            ind_left = 0  # If there are no unflagged channels, then the chunk should have zero width.
            ind_right = 0
            chunkslice = slice(chunk[0], chunk[1])
            unflagged_chans = np.where(~np.all(np.isclose(weights_in[:, chunkslice], 0.0), axis=0))[0]
            if np.count_nonzero(unflagged_chans) > 0:
                # truncate data to be filtered where appropriate.
                ind_left = np.min(unflagged_chans)
                ind_right = np.max(unflagged_chans) + 1

            inds_left.append(ind_left)
            inds_right.append(ind_right)

        dout = np.hstack([data_in[:, chunk[0] + ind_left: chunk[0] + ind_right] for ind_left, ind_right, chunk in zip(inds_left, inds_right, chunks)])
        wout = np.hstack([weights_in[:, chunk[0] + ind_left: chunk[0] + ind_right] for ind_left, ind_right, chunk in zip(inds_left, inds_right, chunks)])
        edges = [(ind_left, chunk[1] - chunk[0] - ind_right) for ind_left, ind_right, chunk in zip(inds_left, inds_right, chunks)]
        if ax == 'both':
            x1 = np.hstack([x[1][chunk[0] + ind_left: chunk[0] + ind_right] for ind_left, ind_right, chunk in zip(inds_left, inds_right, chunks)])
            x0, dout, wout, e0, c0 = truncate_flagged_edges(dout, wout, x[0], ax='time')
            xout = [x0, x1]
            edges = [e0, edges]
            chunks = [c0, chunks]
        else:
            xout = np.hstack([x[chunk[0] + ind_left: chunk[0] + ind_right] for ind_left, ind_right, chunk in zip(inds_left, inds_right, chunks)])
    return xout, dout, wout, edges, chunks


def restore_flagged_edges(data, chunks, edges, ax='freq'):
    """
    fill in flagged regions of data array produced
    by truncate_flagged_edges with zeros.

    Parameters
    ----------
    data: array-like
        2d array containing data that has been trimmed with
        trunate_flagged_edges (dout or wout)
        dimensions (nf_trimmed, nt_trimmed)
    chunks : list of 2-tuples or 2-list of lists of 2-tuples.
        indices indicating the chunk edges that edge widths are reference too.
        first list is time dim, second list is freq dim.
    edges : list of 2-tuples or 2-list of lists of 2-tuples.
        the width of the edges trimmed.
        must be 2-list of lists if ax=='both'
    ax : str, optional
        axis to restore gaps on.
        default is 'freq'

    Returns
    -------
    data_restored: array-like
        2d array with truncated edges restored.
    """
    if ax == 'time':
        # if axis is time, process everything like its the time axis with 0 <-> 1 reversed.
        # switch everything back later.
        data_restored = restore_flagged_edges(data.T, chunks, edges).T
    else:
        if ax == 'both':
            zgen = zip(chunks[1], edges[1])
        else:
            zgen = zip(chunks, edges)
        data_restored = []
        start_ind = 0
        for chunk, edge in zgen:
            stop_ind = start_ind + chunk[1] - chunk[0] - edge[1] - edge[0]
            data_restored.append(np.pad(data[:, start_ind: stop_ind], [(0, 0), edge]))
            start_ind = stop_ind
        if len(data_restored) > 1:
            data_restored = np.hstack(data_restored)
        else:
            data_restored = data_restored[0]
        if ax == 'both':
            # if axis is both, then process time-axis after freq axis.
            data_restored = restore_flagged_edges(data_restored, chunks[0], edges[0], ax='time')
    return data_restored


def flag_rows_with_flags_within_edge_distance(x, weights_in, min_flag_edge_distance, ax='freq'):
    """
    flag integrations (and/or channels) with flags within min_flag_edge_distance of edge.

    Parameters
    -----------
    x: array-like or 2-list / 2-tuple
    weights_in : array-like, 2d (Ntimes, Nfreqs)
        weights to check for flags within min_edge distance of edge along specified axis.
        will set all weights in each row with flags within min_flag_edge_distance to zero.
    min_flag_edge_distance : integer (or two-tuple / list)
        any row of weights_in with zero weights within min_edge distance
        of edge will be set to zero.
    ax : str, optional
        string specifying which axis to flag edges of.
        valid options include 'freq', 'time', 'both'.
        default is 'freq'

    Returns
    -------
    wout, array-like 2d
        weights with rows or columns with zero weights
        within min_flag_edge_distance set entirely to zero.

    """
    if ax == 'time':
        wout = flag_rows_with_flags_within_edge_distance(x, weights_in.T, min_flag_edge_distance).T
    else:
        if isinstance(x, (tuple, list)) and len(x) == 2 and isinstance(x[0], (list, tuple, np.ndarray)):
            chunks = find_discontinuity_edges(x[1])
        else:
            chunks = find_discontinuity_edges(x)
        wout = copy.deepcopy(weights_in)
        for rownum, wrow in enumerate(wout):
            flagrow = False
            for chunk in chunks:
                if ax == 'both':
                    cslice0 = slice(chunk[0], chunk[0] + min_flag_edge_distance[1] + 1)
                    cslice1 = slice(chunk[1] - min_flag_edge_distance[1] - 1, chunk[1])
                    if np.any(np.isclose(wout[rownum, cslice0], 0.0)) | np.any(np.isclose(wout[rownum, cslice1], 0.0)):
                        flagrow = True
                else:
                    cslice0 = slice(chunk[0], chunk[0] + min_flag_edge_distance + 1)
                    cslice1 = slice(chunk[1] - min_flag_edge_distance - 1, chunk[1])
                    if np.any(np.isclose(wout[rownum, cslice0], 0.0)) | np.any(np.isclose(wout[rownum, cslice1], 0.0)):
                        flagrow = True
            if flagrow:
                wout[rownum, :] = 0.
        if ax == 'both':
            wout = flag_rows_with_flags_within_edge_distance(x[0], wout, min_flag_edge_distance[0], ax='time')
    return wout


def flag_rows_with_contiguous_flags(weights_in, max_contiguous_flag, ax='freq'):
    """
    flag any row or column with contiguous zero-weights over a specified limit.

    Parameters
    ----------
    weights_in : array-like, 2d (Ntimes, Nfreqs)
        weights to check. any row (ax='time') or col (ax='freq')
        with contiguous regions of zero with length greater then max_contiguous_flag
        will be set to zero.
    max_contiguous_flag : integer (or 2-list/tuple if ax='both')
        flag any row or column when any N series of contiguous bins in weights_in
        along the axis are zero, where N = max_contiguous_flags
    ax : str, optional
        axis to perform flagging over. options=['time', 'freq', 'both'], default='freq'
    """
    if ax == 'time':
        wout = flag_rows_with_contiguous_flags(weights_in.T, max_contiguous_flag).T
    else:
        wout = copy.deepcopy(weights_in)
        for rownum, wrow in enumerate(wout):
            max_contiguous = 0  # keeps track of the largest contig flags in integration.
            current_flag_length = 0  # keeps track of current contig flag size.
            on_flag = False  # keep track if currently on a flag.
            # iterate over each channel in integration.
            for wr in wrow:
                on_flag = (wr == 0)  # if weights are zero, on a flag.
                # if on a flag, +1 current flag size.
                if on_flag:
                    current_flag_length += 1
                else:
                    # otherwise, check if the current flag len > max.
                    # update max_contiguous if appropriate.
                    if current_flag_length >= max_contiguous:
                        max_contiguous = current_flag_length
                    current_flag_length = 0
            if ax == 'both':
                if max_contiguous >= max_contiguous_flag[1]:
                    wout[rownum][:] = 0.
            elif max_contiguous >= max_contiguous_flag:
                wout[rownum][:] = 0.
        if ax == 'both':
            wout = flag_rows_with_contiguous_flags(wout, max_contiguous_flag[0], ax='time')
    return wout


def get_max_contiguous_flag_from_filter_periods(x, filter_centers, filter_half_widths):
    """
    determine maximum contiguous flags from filter periods

    Parameters
    ----------
    x : array-like, 1d or 2-tuple
        x (and y) axes of data to determine maximum contiguous flags from.
    filter_centers : list or 2-tuple/list of lists
        centers of filtering-windows.
    filter_half_widths : list or 2-tuple list of lists
        half-widths of filtering windows.

    Returns
    -------
    max_contiguous_flag: int or 2-list containing the width of a region corresponding
        to the largest delay in the filter centers and filter_widths
    """
    if isinstance(x, (tuple, list)) and len(x) == 2 and isinstance(x[0], (list, tuple, np.ndarray)):
        if not len(x[0]) > 1 and len(x[1]) > 1:
            raise ValueError("x-axes with only a single element are not supported.")
        else:
            dx = [np.median(np.diff(x[0])), np.median(np.diff(x[1]))]
        max_filter_freq = [np.max(np.abs(np.hstack([[fc - fw, fc + fw] for fc, fw in zip(filter_centers[0], filter_half_widths[0])]))),
                           np.max(np.abs(np.hstack([[fc - fw, fc + fw] for fc, fw in zip(filter_centers[1], filter_half_widths[1])])))]
        return [int(1. / (max_filter_freq[0] * dx[0])), int(1. / (max_filter_freq[1] * dx[1]))]
    else:
        if len(x) > 1:
            dx = np.median(np.diff(x))
        else:
            raise ValueError("x-axes with only a single element are not supported.")
        max_filter_freq = np.max(np.abs(np.hstack([[fc - fw, fc + fw] for fc, fw in zip(filter_centers, filter_half_widths)])))
        return int(1. / (max_filter_freq * dx))


def flag_model_rms(skipped, d, w, mdl, mdl_w=None, model_rms_threshold=1.1, ax='freq'):
    """
    flag integrations or channels where the RMS of the model > RMS of the data

    Parameters
    ----------
    skipped : array-like 2d, bool
        existing clean_flags
    d : array-like 2d, complex
        the data waterfall.
    w : array-like 2d, float
        data weights waterfall. RMS of data will only be determined
        over voxels where |w| > 0.
    mdl : array-like 2d, complex
        model waterfall.
    mdl_w : array-like 2d, float, optional
        model weights waterfall. RMS of model will be determined
        over voxels where |mdl_w| > 0.
        default is None. When None provided, mdl_w set to 1 everywhere.
    model_rms_threshold : float, optional
        flag integrations and/or channels if RMS of model > RMS of data x model_rms_threshold
        default is 1.1
    ax : str, optional
        axis to flag over.
    """
    if mdl_w is None:
        mdl_w = np.ones_like(w)
    if ax == 'freq' or ax == 'both':
        for i in range(mdl.shape[0]):
            if np.any(~skipped[i]):
                if np.mean(np.abs(mdl[i, ~np.isclose(np.abs(mdl_w[i]), 0.0)]) ** 2.) ** .5 >= model_rms_threshold * np.mean(np.abs(d[i, ~np.isclose(np.abs(w[i]), 0.0)]) ** 2.) ** .5:
                    skipped[i] = True
    if ax == 'time' or ax == 'both':
        for i in range(mdl.shape[1]):
            if np.any(~skipped[:, i]):
                if np.mean(np.abs(mdl[~np.isclose(np.abs(mdl_w[:, i]), 0.0), i]) ** 2.) ** .5 >= model_rms_threshold * np.mean(np.abs(d[~np.isclose(np.abs(w[:, i]), 0.0), i]) ** 2.) ** .5:
                    skipped[:, i] = True
    return skipped


class VisClean(object):
    """
    VisClean object for visibility CLEANing and filtering.
    """
    def __init__(self, input_data, filetype='uvh5', input_cal=None, link_data=True,
                 **read_kwargs):
        """
        Initialize the object.

        Args:
            input_data : string, UVData or HERAData object
                Filepath to a miriad, uvfits or uvh5
                datafile, or a UVData or HERAData object.
            filetype : string, options=['miriad', 'uvh5', 'uvfits']
            input_cal : string, UVCal or HERACal object holding
                gain solutions to apply to DataContainers
                as they are built.
            link_data : bool, if True, attempt to link DataContainers
                from HERAData object, otherwise only link metadata if possible.
            read_kwargs : kwargs to pass to UVData.read (e.g. run_check, check_extra and
                run_check_acceptability). Only used for uvh5 filetype
        """
        # attach HERAData
        self.clear_containers()
        self.hd = io.to_HERAData(input_data, filetype=filetype, **read_kwargs)
        # attach calibration
        if input_cal is not None:
            self.attach_calibration(input_cal)

        # attach data and/or metadata to object if exists
        self.attach_data(link_data=link_data)

    def avg_red_baseline_vectors(self, bl_error_tol=1.0):
        """
        Round individual baseline vectors to the
        average baseline vector of each redundant group

        Args:
            bl_error_tol : float, baseline error tolerance [meters]

        Notes:
            Affects self.blvecs and self.bllens in place
        """
        # get redundancies
        antpos, ants = self.hd.get_ENU_antpos(pick_data_ants=True)
        antpos_dict = dict(list(zip(ants, antpos)))
        reds = redcal.get_pos_reds(antpos_dict, bl_error_tol=bl_error_tol)

        # iterate over redundancies
        for red in reds:
            avg_vec = np.mean([self.blvecs[r] for r in red if r in self.blvecs], axis=0)
            for r in red:
                self.blvecs[r] = avg_vec.copy()
                self.bllens[r] = np.linalg.norm(avg_vec) / constants.c.value

    def soft_copy(self, references=[]):
        """
        Make and return a new object with references (not copies)
        to the data objects in self.

        By default, self.hd, self.data, self.flags and self.nsamples
        are referenced into the new object. Additional attributes
        can be specified by references.

        Args:
            references : list of string
                List of extra attributes to copy references from self to output.
                Accepts wildcard * and ? values.

        Returns:
            VisClean object : A VisClean object with references
                to self.hd, and all attributes specified in references.
        """
        # make a new object w/ only copies of metadata
        newobj = self.__class__(self.hd, link_data=False)
        newobj.hd = self.hd
        newobj.data = self.data
        newobj.flags = self.flags
        newobj.nsamples = self.nsamples

        # iterate through extra attributes
        refs = list(self.__dict__.keys())
        for ref in references:
            atrs = fnmatch.filter(refs, ref)
            for atr in atrs:
                setattr(newobj, atr, getattr(self, atr))

        return newobj

    def attach_data(self, link_data=True):
        """
        Attach DataContainers to self.

        If they exist, attach metadata and/or data from self.hd
        and apply calibration solutions from self.hc if it exists.

        Args:
            link_data : bool, if True, attempt to link DataContainers
                from HERAData object, otherwise only link metadata if possible.
        """
        # link the metadata if they exist
        if self.hd.antenna_numbers is not None:
            mdict = self.hd.get_metadata_dict()
            self.antpos = mdict['antpos']
            self.ants = mdict['ants']
            self.data_ants = mdict['data_ants']
            self.freqs = mdict['freqs']
            self.times = mdict['times']
            self.lsts = mdict['lsts']
            self.pols = mdict['pols']
            self.Nfreqs = len(self.freqs)
            self.Ntimes = len(self.times)  # Does not support BDA for now
            self.dlst = np.median(np.diff(self.lsts))
            self.dtime = np.median(np.diff(self.times)) * 24 * 3600
            self.dnu = np.median(np.diff(self.freqs))
            self.bls = sorted(set(self.hd.get_antpairs()))
            self.blvecs = odict([(bl, self.antpos[bl[0]] - self.antpos[bl[1]]) for bl in self.bls])
            self.bllens = odict([(bl, np.linalg.norm(self.blvecs[bl]) / constants.c.value) for bl in self.bls])
            self.lat = self.hd.telescope_location_lat_lon_alt[0] * 180 / np.pi  # degrees
            self.lon = self.hd.telescope_location_lat_lon_alt[1] * 180 / np.pi  # degrees
            self.Nfreqs = len(self.freqs)
        # link the data if they exist
        if self.hd.data_array is not None and link_data:
            self.hd.select(frequencies=self.freqs)
            data, flags, nsamples = self.hd.build_datacontainers()
            self.data = data
            self.flags = flags
            self.nsamples = nsamples

            # apply calibration solutions if they exist
            if hasattr(self, 'hc'):
                self.apply_calibration(self.hc)

    def clear_containers(self, exclude=[]):
        """
        Clear all DataContainers attached to self.

        Args:
            exclude : list of DataContainer names attached
                to self to exclude from purge.
        """
        keys = list(self.__dict__.keys())
        for key in keys:
            if key in exclude:
                continue
            if isinstance(getattr(self, key), DataContainer):
                setattr(self, key, DataContainer({}))

    def attach_calibration(self, input_cal):
        """
        Attach input_cal to self.

        Attach calibration so-as to apply or unapply
        to visibility data on-the-fly as it
        is piped into DataContainers upon read-in.
        """
        # attach HERACal
        self.hc = io.to_HERACal(input_cal)

    def clear_calibration(self):
        """
        Remove calibration object self.hc to clear memory
        """
        if hasattr(self, 'hc'):
            delattr(self, 'hc')

    def apply_calibration(self, input_cal, unapply=False):
        """
        Apply input_cal self.data.

        Args:
            input_cal : UVCal, HERACal or filepath to calfits file
            unapply : bool, if True, reverse gain convention to
                unapply the gains from the data.
        """
        # ensure its a HERACal
        hc = io.to_HERACal(input_cal)
        # load gains
        cal_gains, cal_flags, cal_quals, cal_tquals = hc.read()
        # get overlapping frequency bins
        cal_freqs_in_data = []
        for f in self.freqs:
            match = np.isclose(hc.freqs, f, rtol=1e-10)
            if True in match:
                cal_freqs_in_data.append(np.argmax(match))
        # assert all frequencies in data are found in uvcal
        assert len(cal_freqs_in_data) == len(self.freqs), "Not all freqs in uvd are in uvc"

        for ant in cal_gains:
            cal_gains[ant] = cal_gains[ant][:, cal_freqs_in_data]
            cal_flags[ant] = cal_flags[ant][:, cal_freqs_in_data]
            cal_quals[ant] = cal_quals[ant][:, cal_freqs_in_data]
        if cal_tquals is not None:
            for pol in cal_tquals:
                cal_tquals[pol] = cal_tquals[pol][:, cal_freqs_in_data]

        # apply calibration solutions to data and flags
        gain_convention = hc.gain_convention
        if unapply:
            if gain_convention == 'multiply':
                gain_convention = 'divide'
            elif gain_convention == 'divide':
                gain_convention = 'multiply'
        apply_cal.calibrate_in_place(self.data, cal_gains, self.flags, cal_flags,
                                     gain_convention=gain_convention)

    def apply_flags(self, external_flags, overwrite_flags=False, filetype='uvflag'):
        """
        Apply external set of flags to self.hd.flag_array (inplace!), and re-attach the data and flags.
        Default is to OR existing flags, unless overwrite_flags is True.

        Parameters
        ----------
        external_flags: str or UVFlag object
            Str or list of strings pointing to flag files to apply.
            flag files should be in a format readable by UVFlag.
        overwrite_flags: bool, optional
            If True, overwrite flags (instead of OR) for baselines that are not entirely flagged.
        filetype : str, optional
            Use 'yaml' if the flags are a yaml file or 'uvflag' if the flags are a UVFlag
            file or object. Default is 'uvflag'.
        """

        # get a census of fully flagged baselines to re-flag if overwrite_flags
        if overwrite_flags:
            full_flags = [bl for bl in self.flags if np.all(self.flags[bl])]

        # apply flags
        if filetype == 'uvflag':
            if isinstance(external_flags, str):
                external_flags = UVFlag(external_flags)
            uvutils.apply_uvflag(self.hd, external_flags, unflag_first=overwrite_flags, inplace=True)
        elif filetype == 'yaml':
            from hera_qm.utils import apply_yaml_flags
            self.hd = apply_yaml_flags(self.hd, external_flags, unflag_first=overwrite_flags)
        else:
            raise ValueError(f"{type} is an invalid type! Must be 'yaml' or 'uvflag'.")
        # re-flag fully flagged baselines if necessary
        if overwrite_flags:
            for bl in full_flags:
                tinds = self.hd.antpair2ind(bl)
                self.hd.flag_array[tinds] = True

        # attach data
        self.attach_data()

    def read(self, **read_kwargs):
        """
        Read from self.hd and attach data and/or metadata to self.

        Args:
            read_kwargs : dictionary
                Keyword arguments to pass to HERAData.read().
        """
        # read data
        self.hd.read(return_data=False, **read_kwargs)

        # attach data
        self.attach_data()

    def write_data(self, data, filename, overwrite=False, flags=None, nsamples=None,
                   times=None, lsts=None, filetype='uvh5', partial_write=False,
                   add_to_history='', verbose=True, extra_attrs={}, **kwargs):
        """
        Write data to file.

        Create a new HERAData and update it with data and write to file. Can only write
        data that has associated metadata in the self.hd HERAData object.

        Args:
            data : DataContainer, holding complex visibility data to write to disk.
            filename : string, output filepath
            overwrite : bool, if True, overwrite output file if it exists
            flags : DataContainer, boolean flag arrays to write to disk with data.
            nsamples : DataContainer, float nsample arrays to write to disk with data.
            times : ndarray, list of Julian Date times to replace in HD
            lsts : ndarray, list of LST times [radian] to replace in HD
            filetype : string, output filetype. ['miriad', 'uvh5', 'uvfits'] supported.
            partial_write : bool, if True, begin (or continue) a partial write to
            the output filename and store file descriptor in self.hd._writers.
            add_to_history : string, string to append to hd history.
            extra_attrs : additional UVData/HERAData attributes to update before writing
            kwargs : extra kwargs to pass to UVData.write_*() call
        """
        # get common keys
        keys = [k for k in self.hd.get_antpairpols() if data.has_key(k)]
        if flags is not None:
            keys = [k for k in keys if flags.has_key(k)]
        if nsamples is not None:
            keys = [k for k in keys if nsamples.has_key(k)]

        # if time_array is fed, select out appropriate times
        if times is not None:
            assert lsts is not None, "Both times and lsts must be fed"
            _times = np.unique(self.hd.time_array)[:len(times)]
        else:
            _times = None

        # select out a copy of hd
        hd = self.hd.select(bls=keys, inplace=False, times=_times, frequencies=self.freqs)
        hd._determine_blt_slicing()
        hd._determine_pol_indexing()

        # update HERAData data arrays
        hd.update(data=data, flags=flags, nsamples=nsamples)

        # update extra blt arrays
        for ap in hd.get_antpairs():
            s = hd._blt_slices[ap]
            if times is not None:
                hd.time_array[s] = times
            if lsts is not None:
                hd.lst_array[s] = lsts

        # add history
        hd.history += utils.history_string(add_to_history)

        # update other extra attrs
        for attribute, value in extra_attrs.items():
            hd.__setattr__(attribute, value)

        # write to disk
        if filetype == 'miriad':
            hd.write_miriad(filename, clobber=overwrite, **kwargs)
        elif filetype == 'uvh5':
            if partial_write:
                hd.partial_write(filename, clobber=overwrite, inplace=True, **kwargs)
                self.hd._writers.update(hd._writers)
            else:
                hd.write_uvh5(filename, clobber=overwrite, **kwargs)
        elif filetype == 'uvfits':
            hd.write_uvfits(filename, **kwargs)
        else:
            raise ValueError("filetype {} not recognized".format(filetype))
        echo("...writing to {}".format(filename), verbose=verbose)

    def vis_clean(self, keys=None, x=None, data=None, flags=None, wgts=None,
                  ax='freq', horizon=1.0, standoff=0.0, cache=None, mode='clean',
                  min_dly=10.0, max_frate=None, output_prefix='clean',
                  skip_wgt=0.1, verbose=False, tol=1e-9,
                  overwrite=False, **filter_kwargs):
        """
        Perform visibility cleaning and filtering given flags.

        Parameters
        -----------
        keys : list of antpair-pol tuples
            keys in data to filter, default is all keys
        x : array-like
            x-values of the axes to be filtered. Numpy array if 1d filter.
            2-list/tuple of numpy arrays if 2d filter.
        data : DataContainer, data to clean. Default is self.data
        flags : Datacontainer, flags to use. Default is self.flags
        wgts : DataContainer, weights to use. Default is None.
            Inverse of flags are multiplied into this dictionary
        ax : str, axis to filter, options=['freq', 'time', 'both']
            Where 'freq' and 'time' are 1d filters and 'both' is a 2d filter.
        horizon: coefficient to bl_len where 1 is the horizon [freq filtering]
        standoff: fixed additional delay beyond the horizon (in nanosec) to filter [freq filtering]
        cache: dictionary containing pre-computed filter products.
        mode: string specifying filtering mode. See fourier_filter for supported modes.
        min_dly: max delay (in nanosec) used for freq filter is never below this.
        max_frate : max fringe rate (in milli-Hz) used for time filtering. See hera_filters.dspec.fourier_filter for options.
        output_prefix : str, attach output model, resid, etc, to self as output_prefix + '_model' etc.
        cache: dict, optional
            dictionary for caching fitting matrices.
        skip_wgt : skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
            Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
            time. Skipped channels are then flagged in self.flags.
            Only works properly when all weights are all between 0 and 1.
        verbose : Lots of outputs
        tol : float, optional. To what level are foregrounds subtracted.
        overwrite : bool, if True, overwrite output modules with the same name
                    if they already exist.
        filter_kwargs : optional dictionary, see fourier_filter **filter_kwargs.
                        Do not pass suppression_factors (non-clean)!
                        instead, use tol to set suppression levels in linear filtering.
        """
        if cache is None and not mode == 'clean':
            cache = {}
        if data is None:
            data = self.data
        if flags is None:
            flags = self.flags
        if keys is None:
            keys = data.keys()
        if wgts is None:
            wgts = DataContainer(dict([(k, np.ones_like(flags[k], dtype=float)) for k in keys]))
        # make sure flagged channels have zero weight, regardless of what user supplied.
        wgts = DataContainer(dict([(k, (~flags[k]).astype(float) * wgts[k]) for k in keys]))
        # convert max_frate to DataContainer
        if max_frate is not None:
            if isinstance(max_frate, (int, np.integer, float, np.floating)):
                max_frate = DataContainer(dict([(k, max_frate) for k in data]))
            if not isinstance(max_frate, DataContainer):
                raise ValueError("If fed, max_frate must be a float, or a DataContainer of floats")
            # convert kwargs to proper units
            max_frate = DataContainer(dict([(k, np.asarray(max_frate[k])) for k in max_frate]))

        for k in keys:
            # get filter properties
            mfrate = max_frate[k] if max_frate is not None else None
            filter_centers, filter_half_widths = gen_filter_properties(ax=ax, horizon=horizon,
                                                                       standoff=standoff, min_dly=min_dly,
                                                                       bl_len=self.bllens[k[:2]], max_frate=mfrate)
            if mode != 'clean':
                suppression_factors = [[tol], [tol]] if ax == 'both' else [tol]
                self.fourier_filter(filter_centers=filter_centers, filter_half_widths=filter_half_widths,
                                    keys=[k], mode=mode, suppression_factors=suppression_factors,
                                    x=x, data=data, flags=flags, wgts=wgts, output_prefix=output_prefix,
                                    ax=ax, cache=cache, skip_wgt=skip_wgt, verbose=verbose,
                                    overwrite=overwrite, **filter_kwargs)
            else:
                self.fourier_filter(filter_centers=filter_centers, filter_half_widths=filter_half_widths,
                                    keys=[k], mode=mode, tol=tol, x=x, data=data, flags=flags, wgts=wgts,
                                    output_prefix=output_prefix, ax=ax, skip_wgt=skip_wgt, verbose=verbose,
                                    overwrite=overwrite, **filter_kwargs)

    def fourier_filter(self, filter_centers, filter_half_widths, mode='clean',
                       x=None, keys=None, data=None, flags=None, wgts=None,
                       output_prefix='clean', zeropad=None, cache=None,
                       ax='freq', skip_wgt=0.1, verbose=False, overwrite=False,
                       skip_flagged_edges=False, filter_spw_ranges=None,
                       skip_contiguous_flags=False, max_contiguous_flag=None,
                       keep_flags=False, clean_flags_in_resid_flags=True,
                       skip_if_flag_within_edge_distance=0,
                       flag_model_rms_outliers=False, model_rms_threshold=1.1,
                       **filter_kwargs):
        """
        Generalized fourier filtering wrapper for hera_filters.dspec.fourier_filter.
        It can filter 1d or 2d data with x-axis(es) x and wgts in fourier domain
        rectangular windows centered at filter_centers or filter_half_widths
        perform filtering along any of 2 dimensions in 2d or 1d!
        the 'dft' and 'dayenu' modes support irregularly sampled data.

        Parameters
        -----------
        filter_centers: array-like
            if not 2dfilter: 1d np.ndarray or list or tuple of floats
            specifying centers of rectangular fourier regions to filter.
            If 2dfilter: should be a 2-list or 2-tuple. Each element
            should be a list or tuple or np.ndarray of floats that include
            centers of rectangular regions to filter.
        filter_half_widths: array-like
            if not 2dfilter: 1d np.ndarray or list of tuples of floats
            specifying the half-widths of rectangular fourier regions to filter.
            if 2dfilter: should be a 2-list or 2-tuple. Each element should
            be a list or tuple or np.ndarray of floats that include centers
            of rectangular bins.
        mode: string, optional
            specify filtering mode. Currently supported are
            'clean', iterative clean
            'dpss_lsq', dpss fitting using scipy.optimize.lsq_linear
            'dft_lsq', dft fitting using scipy.optimize.lsq_linear
            'dpss_matrix', dpss fitting using direct lin-lsq matrix
                           computation. Slower then lsq but provides linear
                           operator that can be used to propagate
                           statistics and the matrix is cached so
                           on average, can be faster for data with
                           many similar flagging patterns.
            'dft_matrix', dft fitting using direct lin-lsq matrix
                          computation. Slower then lsq but provides
                          linear operator that can be used to propagate
                          statistics and the matrix is cached so
                          on average, can be faster for data with
                          many similar flagging patterns.
                          !!!WARNING: In my experience,
                          'dft_matrix' option is numerical unstable.!!!
                          'dpss_matrix' works much better.
            'dayenu', apply dayenu filter to data. Does not
                     deconvolve subtracted foregrounds.
            'dayenu_dft_leastsq', apply dayenu filter to data
                     and deconvolve subtracted foregrounds using
                    'dft_leastsq' method (see above).
            'dayenu_dpss_leastsq', apply dayenu filter to data
                     and deconvolve subtracted foregrounds using
                     'dpss_leastsq' method (see above)
            'dayenu_dft_matrix', apply dayenu filter to data
                     and deconvolve subtracted foregrounds using
                    'dft_matrix' mode (see above).
                    !!!WARNING: dft_matrix mode is often numerically
                    unstable. I don't recommend it!
            'dayenu_dpss_matrix', apply dayenu filter to data
                     and deconvolve subtracted foregrounds using
                     'dpss_matrix' method (see above)
            'dayenu_clean', apply dayenu filter to data. Deconvolve
                     subtracted foregrounds with 'clean'.
        x : array-like, x-values of axes to be filtered. Numpy array if 1d filter.
            2-list/tuple of numpy arrays if 2d filter. Default is freqs [Hz] and/or time [sec].
        keys : list, optional, list of tuple ant-pol pair keys of visibilities to filter.
        data : DataContainer, data to clean. Default is self.data
        flags : Datacontainer, flags to use. Default is self.flags
        wgts : DataContainer, weights to use. Default is None.
        output_prefix : string, prefix for attached filter data containers.
        zeropad : int, number of bins to zeropad on both sides of FFT axis. Provide 2-tuple if axis='both'
        ax : string, optional, string specifying axis to filter.
            Where 'freq' and 'time' are 1d filters and 'both' is a 2d filter.
        skip_wgt : skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
            Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
            time. Skipped channels are then flagged in self.flags.
            Only works properly when all weights are all between 0 and 1.
        max_contiguous_edge_flags : int, optional
            if the number of contiguous samples at the edge is greater then this
            at either side, skip.
        verbose : Lots of outputs.
        overwrite : bool, if True, overwrite output modules with the same name
                    if they already exist.
        skip_flagged_edges : bool, optional
            if true, do not filter over flagged edge times (if ax='time') (filter over sub-region)
            or dont filter over flagged edge freqs (if ax='freq') or dont filter over both (if ax='both')
            defualt is False
        filter_spw_ranges : list of 2-tuples, optional
            list of 2-tuples specifying spw-ranges to filter simultaneously
        skip_contiguous_flags : bool, optional
            if true, skip integrations or channels with gaps that are larger then integer
            specified in max_contiguous_flag
        max_contiguous_flag : int (or 2-tuple), optional
            used if skip_contiguous_flags is True
            gaps larger then this value will be skipped.
        keep_flags : bool, optional
            if true, set the post-filtered flags equal to the original flags plus any skipped integrations / channels.
        clean_flags_in_resid_flags : bool, optional
            if true, include clean flags in residual flags that will be written out in res_outfilename.
            default is True.
        skip_if_flag_within_edge_distance : int (or 2-tuple/list), optional. Units of channels or integrations.
            If there is any flag within skip_if_flag_within_edge_distance
            of the edge of the band, then flag that integration (or channel).
            If performing 2dfilter, this arg should be a 2-tuple or list.
        flag_model_rms_outliers : bool, optional
            if true, flag integrations or channels where the rms of the filter model exceeds the rms of the
            unflagged data by model_rms_threshold.
        model_rms_threshold : float, optional
            factor that rms of model in a channel or integration needs to exceed the rms of unflagged data
            to be flagged. only used if flag_model_rms_outliers is true.
        filter_kwargs: dict. Filtering arguments depending on type of filtering.
            NOTE: Unlike the dspec.fourier_filter function, cache is not passed in filter_kwargs.
            dictionary with options for fitting techniques.
            if filter2d is true, this should be a 2-tuple or 2-list
            of dictionaries. The dictionary for each dimension must
            specify the following for each fitting method.
            Also see hera_filters.dspec.fourier_filter where these kwargs are listed.
                * 'dft':
                    'fundamental_period': float or 2-tuple
                        the fundamental_period of dft modes to fit. The number of
                        modes fit within each window in 'filter_half_widths' will
                        equal fw / fundamental_period where fw is the filter_half_width of the window.
                        if filter2d, must provide a 2-tuple with fundamental_period
                        of each dimension.
                * 'dayenu':
                    No parameters necessary if you are only doing 'dayenu'.
                    For 'dayenu_dpss', 'dayenu_dft', 'dayenu_clean' see below
                    and use the appropriate fitting options for each method.
                    suppression_factors: array-like
                        if not 2dfilter: 1d np.ndarray or list of tuples of floats
                        specifying the fractional residuals of model to leave in the data.
                        For example, 1e-6 means that the filter will leave in 1e-6 of data fitted
                        by the model.
                        if 2dfilter: should be a 2-list or 2-tuple. Each element should
                        be a list or tuple or np.ndarray of floats that include centers
                        of rectangular bins.
                * 'dpss':
                    'eigenval_cutoff': array-like
                        list of sinc_matrix eigenvalue cutoffs to use for included dpss modes.
                    'nterms': array-like
                        list of integers specifying the order of the dpss sequence to use in each
                        filter window.
                    'edge_supression': array-like
                        specifies the degree of supression that must occur to tones at the filter edges
                        to calculate the number of DPSS terms to fit in each sub-window.
                    'avg_suppression': list of floats, optional
                        specifies the average degree of suppression of tones inside of the filter edges
                        to calculate the number of DPSS terms. Similar to edge_supression but instead checks
                        the suppression of a since vector with equal contributions from all tones inside of the
                        filter width instead of a single tone.
                    suppression_factors: array-like
                        if not 2dfilter: 1d np.ndarray or list of tuples of floats
                        specifying the fractional residuals of model to leave in the data.
                        For example, 1e-6 means that the filter will leave in 1e-6 of data fitted
                        by the model.
                        if 2dfilter: should be a 2-list or 2-tuple. Each element should
                        be a list or tuple or np.ndarray of floats that include centers
                        of rectangular bins.
                *'clean':
                     'tol': float,
                        clean tolerance. 1e-9 is standard.
                     'maxiter' : int
                        maximum number of clean iterations. 100 is standard.
                     'pad': int or array-like
                        if filt2d is false, just an integer specifing the number of channels
                        to pad for CLEAN (sets Fourier interpolation resolution).
                        if filt2d is true, specify 2-tuple in both dimensions.
                     'filt2d_mode' : string
                        if 'rect', clean withing a rectangular region of Fourier space given
                        by the intersection of each set of windows.
                        if 'plus' only clean the plus-shaped shape along
                        zero-delay and fringe rate.
                    'edgecut_low' : int, number of bins to consider zero-padded at low-side of the FFT axis,
                        such that the windowing function smoothly approaches zero. For 2D cleaning, can
                        be fed as a tuple specifying edgecut_low for first and second FFT axis.
                    'edgecut_hi' : int, number of bins to consider zero-padded at high-side of the FFT axis,
                        such that the windowing function smoothly approaches zero. For 2D cleaning, can
                        be fed as a tuple specifying edgecut_hi for first and second FFT axis.
                    'add_clean_residual' : bool, if True, adds the CLEAN residual within the CLEAN bounds
                        in fourier space to the CLEAN model. Note that the residual actually returned is
                        not the CLEAN residual, but the residual in input data space.
                    'taper' : window function for filtering applied to the filtered axis.
                        See dspec.gen_window for options. If clean2D, can be fed as a list
                        specifying the window for each axis in data.
                    'skip_wgt' : skips filtering rows with very low total weight (unflagged fraction ~< skip_wgt).
                        Model is left as 0s, residual is left as data, and info is {'skipped': True} for that
                        time. Only works properly when all weights are all between 0 and 1.
                    'gain': The fraction of a residual used in each iteration. If this is too low, clean takes
                        unnecessarily long. If it is too high, clean does a poor job of deconvolving.
                    'alpha': float, if window is 'tukey', this is its alpha parameter.
        """
        # type checks
        if ax == 'both':
            if zeropad is None:
                zeropad = [0, 0]
            filterdim = [1, 0]
            filter2d = True
            if x is None:
                x = [(self.times - np.mean(self.times)) * 3600. * 24., self.freqs]
            if skip_if_flag_within_edge_distance == 0:
                skip_if_flag_within_edge_distance = (0, 0)
        elif ax == 'time':
            filterdim = 0
            filter2d = False
            if x is None:
                x = (self.times - np.mean(self.times)) * 3600. * 24.
            if zeropad is None:
                zeropad = 0
        elif ax == 'freq':
            filterdim = 1
            filter2d = False
            if zeropad is None:
                zeropad = 0
            if x is None:
                x = self.freqs
        else:
            raise ValueError("ax must be one of ['freq', 'time', 'both']")
        if filter_spw_ranges is None:
            filter_spw_ranges = [(0, self.Nfreqs)]
        # total number of frequencies in all spw ranges.
        n_spw_chans_sum = np.sum([spw_range[1] - spw_range[0] for spw_range in filter_spw_ranges])
        # frequencies from spw-ranges concatenated together.
        spw_freqs_concatenated = self.freqs[np.hstack([np.arange(spw_range[0], spw_range[1]).astype(int) for spw_range in filter_spw_ranges])]

        # check that combined spw-freqs are equal to original freq axis
        # first conditional is faster (to short-circuit)
        if len(self.freqs) != n_spw_chans_sum or not np.allclose(self.freqs, spw_freqs_concatenated):
            raise NotImplementedError("Channels detected in original frequency array that do not fall into any of the specified SPWS."
                                      "We currently only support SPWs that together include every channel in the original frequency axis.")
        # initialize containers
        containers = ["{}_{}".format(output_prefix, dc) for dc in ['model', 'resid', 'flags', 'data', 'resid_flags']]
        for i, dc in enumerate(containers):
            if not hasattr(self, dc):
                setattr(self, dc, DataContainer({}))
            containers[i] = getattr(self, dc)
        filtered_model, filtered_resid, filtered_flags, filtered_data, resid_flags = containers
        filtered_info = "{}_{}".format(output_prefix, 'info')
        if not hasattr(self, filtered_info):
            setattr(self, filtered_info, {})
        filtered_info = getattr(self, filtered_info)

        # select DataContainers
        if data is None:
            data = self.data
        if flags is None:
            flags = self.flags

        # get keys
        if keys is None:
            keys = data.keys()

        # get weights
        if wgts is None:
            wgts = DataContainer(dict([(k, (~flags[k]).astype(float)) for k in keys]))
        else:
            # make sure flagged channels have zero weight, regardless of what user supplied.
            wgts = DataContainer(dict([(k, (~flags[k]).astype(float) * wgts[k]) for k in keys]))
        if mode != 'clean':
            if cache is None:
                cache = {}
            filter_kwargs['cache'] = cache
        # iterate over keys
        for k in keys:
            if k not in filtered_info:
                filtered_info[k] = {}
            if k in filtered_model and overwrite is False:
                echo("{} exists in clean_model and overwrite is False, skipping...".format(k), verbose=verbose)
                continue
            echo("Starting fourier filter of {} at {}".format(k, str(datetime.datetime.now())), verbose=verbose)
            for spw_range in filter_spw_ranges:
                if spw_range not in filtered_info[k]:
                    filtered_info[k][spw_range] = {}
                spw_slice = slice(spw_range[0], spw_range[1])
                d = data[k][:, spw_slice]
                f = flags[k][:, spw_slice]
                fw = (~f).astype(float)
                w = fw * wgts[k][:, spw_slice]
                # avoid modifying x in-place with zero-padding.
                xp = copy.deepcopy(x)
                if ax == 'freq':
                    xp = xp[spw_slice]
                    # zeropad the data
                    if zeropad > 0:
                        d, _ = zeropad_array(d, zeropad=zeropad, axis=1)
                        w, _ = zeropad_array(w, zeropad=zeropad, axis=1)
                        xp = np.hstack([xp.min() - (1 + np.arange(zeropad)[::-1]) * np.median(np.diff(xp)), xp,
                                        xp.max() + (1 + np.arange(zeropad)) * np.median(np.diff(xp))])
                elif ax == 'time':
                    # zeropad the data
                    if zeropad > 0:
                        d, _ = zeropad_array(d, zeropad=zeropad, axis=0)
                        w, _ = zeropad_array(w, zeropad=zeropad, axis=0)
                        xp = np.hstack([xp.min() - (1 + np.arange(zeropad)[::-1]) * np.median(np.diff(xp)), xp,
                                        xp.max() + (1 + np.arange(zeropad)) * np.median(np.diff(xp))])
                elif ax == 'both':
                    xp[1] = xp[1][spw_slice]
                    if not isinstance(zeropad, (list, tuple)) or not len(zeropad) == 2:
                        raise ValueError("zeropad must be a 2-tuple or 2-list of integers")
                    if not (isinstance(zeropad[0], (int, np.integer)) and isinstance(zeropad[1], (int, np.integer))):
                        raise ValueError("zeropad values must all be integers. You provided %s" % (zeropad))
                    for m in range(2):
                        if zeropad[m] > 0:
                            d, _ = zeropad_array(d, zeropad=zeropad[m], axis=m)
                            w, _ = zeropad_array(w, zeropad=zeropad[m], axis=m)
                            xp[m] = np.hstack([xp[m].min() - (np.arange(zeropad[m])[::-1] + 1) * np.median(np.diff(xp[m])),
                                               xp[m], xp[m].max() + (1 + np.arange(zeropad[m])) * np.median(np.diff(xp[m]))])
                # if we are not including flagged edges in filtering, skip them here.
                if skip_flagged_edges:
                    xp, din, win, edges, chunks = truncate_flagged_edges(d, w, xp, ax=ax)
                else:
                    din = d
                    win = w
                # skip integrations with contiguous edge flags exceeding desired limit
                # (or precomputed limit) here.
                if skip_contiguous_flags:
                    if max_contiguous_flag is None:
                        max_contiguous_flag = get_max_contiguous_flag_from_filter_periods(x, filter_centers, filter_half_widths)
                    win = flag_rows_with_contiguous_flags(win, max_contiguous_flag, ax=ax)
                # skip integrations with flags within some minimum distance of the edges here.
                if np.any(np.asarray(skip_if_flag_within_edge_distance) > 0):
                    win = flag_rows_with_flags_within_edge_distance(xp, win, skip_if_flag_within_edge_distance, ax=ax)

                mdl, res = np.zeros_like(d), np.zeros_like(d)
                if 0 not in din.shape:
                    mdl, res, info = dspec.fourier_filter(x=xp, data=din, wgts=win, filter_centers=filter_centers,
                                                          filter_half_widths=filter_half_widths,
                                                          mode=mode, filter_dims=filterdim, skip_wgt=skip_wgt,
                                                          **filter_kwargs)
                    # insert back the filtered model if we are skipping flagged edgs.
                    if skip_flagged_edges:
                        mdl = restore_flagged_edges(mdl, chunks, edges, ax=ax)
                        res = restore_flagged_edges(res, chunks, edges, ax=ax)
                    # unzeropad array and put in skip flags.
                    if ax == 'freq':
                        if zeropad > 0:
                            mdl, _ = zeropad_array(mdl, zeropad=zeropad, axis=1, undo=True)
                            res, _ = zeropad_array(res, zeropad=zeropad, axis=1, undo=True)
                    elif ax == 'time':
                        if zeropad > 0:
                            mdl, _ = zeropad_array(mdl, zeropad=zeropad, axis=0, undo=True)
                            res, _ = zeropad_array(res, zeropad=zeropad, axis=0, undo=True)
                    elif ax == 'both':
                        for i in range(2):
                            if zeropad[i] > 0:
                                mdl, _ = zeropad_array(mdl, zeropad=zeropad[i], axis=i, undo=True)
                                res, _ = zeropad_array(res, zeropad=zeropad[i], axis=i, undo=True)
                            _trim_status(info, i, zeropad[i - 1])
                        # need to adjust info based on edges and chunks!
                        # restore indices in info necessary if ax=='both'.
                        if skip_flagged_edges:
                            _adjust_info_indices(xp, info, edges, spw_range[0])
                else:
                    info = {'status': {'axis_0': {}, 'axis_1': {}}}
                    if ax == 'freq' or ax == 'both':
                        info['status']['axis_1'] = {i: 'skipped' for i in range(self.Ntimes)}
                    if ax == 'time' or ax == 'both':
                        info['status']['axis_0'] = {i: 'skipped' for i in range(spw_range[1] - spw_range[0])}
                # flag integrations and channels that were skipped.
                skipped = np.zeros_like(mdl, dtype=bool)
                # this is not the correct thing to do for 2d filtering.
                # For 2d filter, only look at time-axis skips.
                # for 1d filter look at both time and freq axes.
                for dim in range(int(ax.lower() == 'both'), 2):
                    dim = 1 - dim
                    if len(info['status']['axis_%d' % dim]) > 0:
                        for i in info['status']['axis_%d' % dim]:
                            if info['status']['axis_%d' % dim][i] == 'skipped':
                                if dim == 0:
                                    skipped[:, i] = True
                                elif dim == 1:
                                    skipped[i] = True
                # just in case any artifacts make it through after our other flagging rounds
                # flag integrations or channels where the RMS of the model exceeds the RMS of the unflagged data
                # by some threshold.
                if flag_model_rms_outliers:
                    skipped = flag_model_rms(skipped, d, w, mdl, model_rms_threshold=model_rms_threshold, ax=ax)

                # also flag skipped edge channels and integrations.
                if skip_flagged_edges:
                    if ax == 'both':
                        for chunk, edge in zip(chunks[1], edges[1]):
                            cslice0 = slice(chunk[0], chunk[0] + edge[0])
                            cslice1 = slice(chunk[1] - edge[1], chunk[1])
                            skipped[:, cslice0] = True
                            skipped[:, cslice1] = True
                        for chunk, edge in zip(chunks[0], edges[0]):
                            cslice0 = slice(chunk[0], chunk[0] + edge[0])
                            cslice1 = slice(chunk[1] - edge[1], chunk[1])
                            skipped[cslice0, :] = True
                            skipped[cslice1, :] = True
                    else:
                        for chunk, edge in zip(chunks, edges):
                            cslice0 = slice(chunk[0], chunk[0] + edge[0])
                            cslice1 = slice(chunk[1] - edge[1], chunk[1])
                            if ax == 'freq':
                                skipped[:, cslice0] = True
                                skipped[:, cslice1] = True
                            elif ax == 'time':
                                skipped[cslice0, :] = True
                                skipped[cslice1, :] = True
                if k not in filtered_model:
                    filtered_model[k] = np.zeros_like(data[k])
                    filtered_resid[k] = np.zeros_like(data[k])
                    filtered_data[k] = np.zeros_like(data[k])
                    filtered_flags[k] = np.zeros_like(flags[k])
                    resid_flags[k] = np.zeros_like(flags[k])
                filtered_model[k][:, spw_slice] = mdl
                filtered_model[k][:, spw_slice][skipped] = 0.
                filtered_resid[k][:, spw_slice] = res * fw
                filtered_resid[k][:, spw_slice][skipped] = 0.
                filtered_data[k][:, spw_slice] = filtered_model[k][:, spw_slice] + filtered_resid[k][:, spw_slice]
                if not keep_flags:
                    filtered_flags[k][:, spw_slice] = skipped
                else:
                    filtered_flags[k][:, spw_slice] = copy.deepcopy(flags[k][:, spw_slice]) | skipped
                filtered_info[k][spw_range] = info
                if clean_flags_in_resid_flags:
                    resid_flags[k][:, spw_slice] = copy.deepcopy(flags[k][:, spw_slice]) | skipped
                else:
                    resid_flags[k][:, spw_slice] = copy.deepcopy(flags[k][:, spw_slice])
        
        # loop through resids, model, and data and make sure everything is real.
        discard_autocorr_imag(filtered_model)
        discard_autocorr_imag(filtered_resid)
        discard_autocorr_imag(filtered_data)

        if hasattr(data, 'times'):
            filtered_data.times = data.times
            filtered_model.times = data.times
            filtered_resid.times = data.times
            filtered_flags.times = data.times

    def fft_data(self, data=None, flags=None, keys=None, assign='dfft', ax='freq', window='none', alpha=0.1,
                 overwrite=False, edgecut_low=0, edgecut_hi=0, ifft=False, ifftshift=False, fftshift=True,
                 zeropad=0, dtime=None, dnu=None, verbose=True):
        """
        Take FFT of data and attach to self.

        Results are stored as self.assign. Default is self.dfft.
        Take note of the adopted fourier convention via ifft and fftshift kwargs.

        Args:
            data : DataContainer
                Object to pull data to FT from. Default is self.data.
            flags : DataContainer
                Object to pull flags in FT from. Default is no flags.
            keys : list of tuples
                List of keys from clean_data to FFT. Default is all keys.
            assign : str
                Name of DataContainer to attach to self. Default is self.dfft
            ax : str, options=['freq', 'time', 'both']
                Axis along with to take FFT.
            window : str
                Windowing function to apply across frequency before FFT. If ax is 'both',
                can feed as a tuple specifying window for 0th and 1st FFT axis.
            alpha : float
                If window is 'tukey' this is its alpha parameter. If ax is 'both',
                can feed as a tuple specifying alpha for 0th and 1st FFT axis.
            edgecut_low : int, number of bins to consider zero-padded at low-side of the FFT axis,
                such that the windowing function smoothly approaches zero. If ax is 'both',
                can feed as a tuple specifying for 0th and 1st FFT axis.
            edgecut_hi : int, number of bins to consider zero-padded at high-side of the FFT axis,
                such that the windowing function smoothly approaches zero. If ax is 'both',
                can feed as a tuple specifying for 0th and 1st FFT axis.
            ifft : bool, if True, use ifft instead of fft
            ifftshift : bool, if True, ifftshift data along FT axis before FFT.
            fftshift : bool, if True, fftshift along FFT axes.
            zeropad : int, number of zero-valued channels to append to each side of FFT axis.
            dtime : float, time spacing of input data [sec], not necessarily integration time!
                Default is self.dtime.
            dnu : float, frequency spacing of input data [Hz]. Default is self.dnu.
            overwrite : bool
                If dfft[key] already exists, overwrite its contents.
        """
        # type checks
        if ax not in ['freq', 'time', 'both']:
            raise ValueError("ax must be one of ['freq', 'time', 'both']")

        # generate home
        if not hasattr(self, assign):
            setattr(self, assign, DataContainer({}))

        # get home
        dfft = getattr(self, assign)

        # get data
        if data is None:
            data = self.data
        if flags is not None:
            wgts = DataContainer(dict([(k, (~flags[k]).astype(float)) for k in flags]))
        else:
            wgts = DataContainer(dict([(k, np.ones_like(data[k], dtype=float)) for k in data]))

        # get keys
        if keys is None:
            keys = data.keys()
        if len(keys) == 0:
            raise ValueError("No keys found")

        # get delta bin
        if ax == 'freq':
            _, delta_bin = self._get_delta_bin(dtime=dtime, dnu=dnu)
            axis = 1
        elif ax == 'time':
            delta_bin, _ = self._get_delta_bin(dtime=dtime, dnu=dnu)
            axis = 0
        else:
            delta_bin = self._get_delta_bin(dtime=dtime, dnu=dnu)
            axis = (0, 1)

        # iterate over keys
        j = 0
        for k in keys:
            if k not in data:
                echo("{} not in data, skipping...".format(k), verbose=verbose)
                continue
            if k in dfft and not overwrite:
                echo("{} in self.{} and overwrite == False, skipping...".format(k, assign), verbose=verbose)
                continue

            # FFT
            dfft[k], fourier_axes = fft_data(data[k], delta_bin, wgts=wgts[k], axis=axis, window=window,
                                             alpha=alpha, edgecut_low=edgecut_low, edgecut_hi=edgecut_hi,
                                             ifft=ifft, ifftshift=ifftshift, fftshift=fftshift, zeropad=zeropad)
            j += 1

        if j == 0:
            raise ValueError("No FFT run with keys {}".format(keys))

        if hasattr(data, 'times'):
            dfft.times = data.times
        if ax == 'freq':
            self.delays = fourier_axes
            self.delays *= 1e9
        elif ax == 'time':
            self.frates = fourier_axes
            self.frates *= 1e3
        else:
            self.frates, self.delays = fourier_axes
            self.delays *= 1e9
            self.frates *= 1e3

    def factorize_flags(self, keys=None, spw_ranges=None, time_thresh=0.05, inplace=False):
        """
        Factorize self.flags into two 1D time and frequency masks.

        This works by broadcasting flags across time if the fraction of
        flagged times exceeds time_thresh, otherwise flags are broadcasted
        across channels in a spw_range.

        Note: although technically allowed, this function may give unexpected
        results if multiple spectral windows in spw_ranges have overlap.

        Note: it is generally not recommended to set time_thresh > 0.5, which
        could lead to substantial amounts of data being flagged.

        Args:
            keys : list of antpairpol tuples to operate on
            spw_ranges : list of tuples
                list of len-2 spectral window tuples, specifying the start (inclusive)
                and stop (exclusive) index of the freq channels for each spw.
                Default is to use the whole band.

            time_thresh : float
                Fractional threshold of flagged pixels across time needed to flag all times
                per freq channel. It is not recommend to set this greater than 0.5.
                Fully flagged integrations do not count towards triggering time_thresh.

            inplace : bool, if True, edit self.flags in place, otherwise return a copy
        """
        # get flags
        flags = self.flags
        if not inplace:
            flags = copy.deepcopy(flags)

        # get keys
        if keys is None:
            keys = flags.keys()

        # iterate over keys
        for k in keys:
            factorize_flags(flags[k], spw_ranges=spw_ranges, time_thresh=time_thresh, inplace=True)

        if not inplace:
            return flags

    def write_filtered_data(self, res_outfilename=None, CLEAN_outfilename=None, filled_outfilename=None, filetype='uvh5',
                            partial_write=False, clobber=False, add_to_history='', extra_attrs={}, prefix='clean', **kwargs):
        '''
        Method for writing data products.

        Can write filtered residuals, CLEAN models, and/or original data with flags filled
        by CLEAN models where possible. Uses input_data from DelayFilter.load_data() as a template.

        Arguments:
            res_outfilename: path for writing the filtered visibilities with flags
            CLEAN_outfilename: path for writing the CLEAN model visibilities (with the same flags)
            filled_outfilename: path for writing the original data but with flags unflagged and replaced
                with CLEAN models wherever possible
            filetype: file format of output result. Default 'uvh5.' Also supports 'miriad' and 'uvfits'.
            partial_write: use uvh5 partial writing capability (only works when going from uvh5 to uvh5)
            clobber: if True, overwrites existing file at the outfilename
            add_to_history: string appended to the history of the output file
            extra_attrs : additional UVData/HERAData attributes to update before writing
            prefix : string, the prefix for the datacontainers to write.
            kwargs : extra kwargs to pass to UVData.write_*() call
        '''
        if not hasattr(self, 'data'):
            raise ValueError("Cannot write data without first loading")
        if (res_outfilename is None) and (CLEAN_outfilename is None) and (filled_outfilename is None):
            raise ValueError('You must specifiy at least one outfilename.')
        else:
            # loop over the three output modes if a corresponding outfilename is supplied
            for mode, outfilename in zip(['residual', 'CLEAN', 'filled'],
                                         [res_outfilename, CLEAN_outfilename, filled_outfilename]):
                if outfilename is not None:
                    if mode == 'residual':
                        data_out, flags_out = getattr(self, prefix + '_resid'), getattr(self, prefix + '_resid_flags')
                    elif mode == 'CLEAN':
                        data_out, flags_out = getattr(self, prefix + '_model'), getattr(self, prefix + '_flags')
                    elif mode == 'filled':
                        data_out, flags_out = self.get_filled_data(prefix=prefix)
                    if partial_write:
                        # add extra_attrs to kwargs
                        for k in extra_attrs:
                            kwargs[k] = extra_attrs[k]
                        if not ((filetype == 'uvh5') and (getattr(self.hd, 'filetype', None) == 'uvh5')):
                            raise NotImplementedError('Partial writing requires input and output types to be "uvh5".')
                        self.hd.partial_write(outfilename, data=data_out, flags=flags_out, clobber=clobber,
                                              add_to_history=utils.history_string(add_to_history), **kwargs)
                    else:
                        self.write_data(data_out, outfilename, filetype=filetype, overwrite=clobber, flags=flags_out,
                                        add_to_history=add_to_history, extra_attrs=extra_attrs, **kwargs)

    def zeropad_data(self, data, binvals=None, zeropad=0, axis=-1, undo=False):
        """
        Iterate through DataContainer "data" and zeropad it inplace.

        Args:
            data : DataContainer to zero-pad (or un-pad)
            binvals : bin for data (e.g. times or freqs) to also pad out
                by relevant amount. If axis is an iterable, binvals must also be
            zeropad : int, number of bins on each axis to pad
                If axis is an iterable, zeropad must be also be
            axis : int, axis to zeropad. Can be a tuple
                to zeropad mutliple axes.
            undo : If True, remove zero-padded edges along axis.
        """
        # iterate over data
        for k in data:
            data[k], bvals = zeropad_array(data[k], binvals=binvals, zeropad=zeropad, axis=axis, undo=undo)

        data.binvals = bvals

    def _get_delta_bin(self, dtime=None, dnu=None):
        """
        Get visibility time & frequency spacing.

        Defaults are self.dtime and self.dnu

        Args:
            dtime : float, time spacing [sec]
            dnu : float, frequency spacing [Hz]

        Returns:
            (dtime, dnu)
        """
        if dtime is None:
            dtime = self.dtime

        if dnu is None:
            dnu = self.dnu

        return dtime, dnu

    def get_filled_data(self, prefix='clean'):
        """Get data with flagged pixels filled with clean_model.
        Parameters
            prefix : string label for data-containers of filtering outputs to get.
        Returns
            filled_data: DataContainer with original data and flags filled with CLEAN model
            filled_flags: DataContainer with flags set to False unless the time is skipped in delay filter
        """
        assert np.all([hasattr(self, n) for n in [prefix + '_model', prefix + '_flags', 'data', 'flags']]), "self.data, self.flags, self.%s_model and self.%s_flags must all exist to get filled data" % (prefix, prefix)
        # construct filled data and filled flags
        filled_data = copy.deepcopy(getattr(self, prefix + '_model'))
        filled_flags = copy.deepcopy(getattr(self, prefix + '_flags'))

        # iterate over filled_data keys
        for k in filled_data.keys():
            # get original data flags
            f = self.flags[k]
            # replace filled_data with original data at f == False
            filled_data[k][~f] = self.data[k][~f]

        return filled_data, filled_flags


def fft_data(data, delta_bin, wgts=None, axis=-1, window='none', alpha=0.2, edgecut_low=0,
             edgecut_hi=0, ifft=False, ifftshift=False, fftshift=True, zeropad=0):
    """
    FFT data along specified axis.

    Note the fourier convention of ifft and fftshift.

    Args:
        data : complex ndarray
        delta_bin : bin size (seconds or Hz). If axis is a tuple can feed
            as tuple with bin size for time and freq axis respectively.
        wgts : float ndarray of shape (Ntimes, Nfreqs)
        axis : int, FFT axis. Can feed as tuple for 2D fft.
        window : str
            Windowing function to apply across frequency before FFT. If axis is tuple,
            can feed as a tuple specifying window for each FFT axis.
        alpha : float
            If window is 'tukey' this is its alpha parameter. If axis is tuple,
            can feed as a tuple specifying alpha for each FFT axis.
        edgecut_low : int, number of bins to consider zero-padded at low-side of the FFT axis,
            such that the windowing function smoothly approaches zero. If axis is tuple,
            can feed as a tuple specifying for each FFT axis.
        edgecut_hi : int, number of bins to consider zero-padded at high-side of the FFT axis,
            such that the windowing function smoothly approaches zero. If axis is tuple,
            can feed as a tuple specifying for each FFT axis.
        ifft : bool, if True, use ifft instead of fft
        ifftshift : bool, if True, ifftshift data along FT axis before FFT.
        fftshift : bool, if True, fftshift along FT axes after FFT.
        zeropad : int, number of zero-valued channels to append to each side of FFT axis.
            If axis is tuple, can feed as a tuple specifying for each FFT axis.
    Returns:
        dfft : complex ndarray FFT of data
        fourier_axes : fourier axes, if axis is ndimensional, so is this.
    """

    # type checks
    if not isinstance(axis, (tuple, list)):
        axis = [axis]
    if not isinstance(window, (tuple, list)):
        window = [window for i in range(len(axis))]
    if not isinstance(alpha, (tuple, list)):
        alpha = [alpha for i in range(len(axis))]
    if not isinstance(edgecut_low, (tuple, list)):
        edgecut_low = [edgecut_low for i in range(len(axis))]
    if not isinstance(edgecut_hi, (tuple, list)):
        edgecut_hi = [edgecut_hi for i in range(len(axis))]
    if not isinstance(zeropad, (tuple, list)):
        zeropad = [zeropad for i in range(len(axis))]
    if not isinstance(delta_bin, (tuple, list)):
        if len(axis) > 1:
            raise ValueError("delta_bin must have same len as axis")
        delta_bin = [delta_bin]
    else:
        if len(delta_bin) != len(axis):
            raise ValueError("delta_bin must have same len as axis")
    Nax = len(axis)

    # get a copy
    data = data.copy()

    # set fft convention
    fourier_axes = []
    if ifft:
        fft = np.fft.ifft
    else:
        fft = np.fft.fft

    # get wgts
    if wgts is None:
        wgts = np.ones_like(data, dtype=float)
    data *= wgts

    # iterate over axis
    for i, ax in enumerate(axis):
        Nbins = data.shape[ax]

        # generate and apply window
        win = dspec.gen_window(window[i], Nbins, alpha=alpha[i], edgecut_low=edgecut_low[i], edgecut_hi=edgecut_hi[i])
        wshape = np.ones(data.ndim, dtype=int)
        wshape[ax] = Nbins
        win.shape = tuple(wshape)
        data *= win

        # zeropad
        data, _ = zeropad_array(data, zeropad=zeropad[i], axis=ax)

        # ifftshift
        if ifftshift:
            data = np.fft.ifftshift(data, axes=ax)

        # FFT
        data = fft(data, axis=ax)

        # get fourier axis
        fax = np.fft.fftfreq(data.shape[ax], delta_bin[i])

        # fftshift
        if fftshift:
            data = np.fft.fftshift(data, axes=ax)
            fax = np.fft.fftshift(fax)

        fourier_axes.append(fax)

    if len(axis) == 1:
        fourier_axes = fourier_axes[0]

    return data, fourier_axes


def trim_model(clean_model, clean_resid, dnu, keys=None, noise_thresh=2.0, delay_cut=3000,
               kernel_size=None, edgecut_low=0, edgecut_hi=0, polyfit_deg=None, verbose=True):
    """
    Truncate CLEAN model components in delay space below some amplitude threshold.

    Estimate the noise in Fourier space by taking median of high delay
    clean residual above delay_cut, and truncate CLEAN model components
    below a multiplier times this level.

    Args:
        clean_model : DataContainer
            Holds clean_model output of self.vis_clean.
        clean_resid : DataContainer
            Holds clean_resid output of self.vis_clean
        dnu : float
            Frequency channel width [Hz]
        keys : list of antpairpol tuples
            List of keys to operate on
        noise_thresh : float
            Coefficient times noise to truncate model components below
        delay_cut : float
            Minimum |delay| [ns] above which to use in estimating noise
        kernel_size : int
            Time median filter kernel_size. None is no median filter.
        edgecut_low : int
            Edgecut bins to apply to low edge of frequency band
        edgecut_hi : int
            Edgecut bins to apply to high edge of frequency band
        polyfit_deg : int
            Degree of polynomial to fit to noise curve w.r.t. time.
            None is no fitting.
        verbose : bool
            Report feedback to stdout

    Returns:
        model : DataContainer
            Truncated clean_model
        noise : DataContainer
            Per integration noise estimate from clean_resid
    """
    # get keys
    if keys is None:
        keys = [k for k in sorted(set(list(clean_model.keys()) + list(clean_resid.keys()))) if k in clean_model and k in clean_resid]

    # estimate noise in Fourier space by taking amplitude of high delay modes
    # above delay_cut
    model = DataContainer({})
    noise = DataContainer({})
    for k in keys:
        # get rfft
        rfft, delays = fft_data(clean_resid[k], dnu, axis=1, window='none', edgecut_low=edgecut_low, edgecut_hi=edgecut_hi, ifft=False, ifftshift=False, fftshift=False)
        delays *= 1e9

        # get NEB of clean_resid: a top-hat window nulled where resid == 0 (i.e. flag pattern)
        w = (~np.isclose(clean_resid[k], 0.0)).astype(float)
        neb = noise_eq_bandwidth(w[:, None])

        # get time-dependent noise level in Fourier space from FFT at high delays
        noise[k] = np.median(np.abs((rfft * neb)[:, np.abs(delays) > delay_cut]), axis=1)

        # median filter it
        if kernel_size is not None:
            n = noise[k]
            nlen = len(n)
            n = np.pad(n, nlen, 'reflect', reflect_type='odd')
            noise[k] = signal.medfilt(n, kernel_size=kernel_size)[nlen:-nlen]

        # fit a polynomial if desired
        if polyfit_deg is not None:
            x = np.arange(noise[k].size, dtype=float)
            f = ~np.isnan(noise[k]) & ~np.isfinite(noise[k]) & ~np.isclose(noise[k], 0.0)
            # only fit if it is well-conditioned: Ntimes > polyfit_deg + 1
            if f.sum() >= (polyfit_deg + 1):
                fit = np.polyfit(x[f], noise[k][f], deg=polyfit_deg)
                noise[k] = np.polyval(fit, x)
            else:
                # not enough points to fit polynomial
                echo("Need more suitable data points for {} to fit {}-deg polynomial".format(k, polyfit_deg), verbose=verbose)

        # get mfft
        mfft, _ = fft_data(clean_model[k], dnu, axis=1, window='none', edgecut_low=edgecut_low, edgecut_hi=edgecut_hi, ifft=False, ifftshift=False, fftshift=False)

        # set all mfft modes below some threshold to zero
        mfft[np.abs(mfft) < (noise[k][:, None] * noise_thresh)] = 0.0

        # re-fft
        mdl, _ = fft_data(mfft, dnu, axis=1, window='none', edgecut_low=0, edgecut_hi=0, ifft=True, ifftshift=False, fftshift=False)
        model[k] = mdl

    return model, noise


def zeropad_array(data, binvals=None, zeropad=0, axis=-1, undo=False):
    """
    Zeropad data ndarray along axis.

    If data is float, int or complex, zeropads with zero.
    If data is boolean, zeropads with True.

    Args:
        data : ndarray to zero-pad (or un-pad)
        binvals : bin values for data (e.g. times or freqs) to also pad out
            by relevant amount. If axis is an iterable, binvals must also be
        zeropad : int, number of bins on each axis to pad
            If axis is an iterable, zeropad must be also be
        axis : int, axis to zeropad. Can be a tuple
            to zeropad mutliple axes.
        undo : If True, remove zero-padded edges along axis.

    Returns:
        padded_data : zero-padded (or un-padded) data
        padded_bvals : bin array(s) padded (or un-padded) if binvals is fed, otherwise None
    """
    # get data type
    bool_dtype = np.issubdtype(data.dtype, np.bool_)

    # type checks
    if not isinstance(axis, (list, tuple, np.ndarray)):
        axis = [axis]
    binvals = copy.deepcopy(binvals)
    if not isinstance(binvals, (list, tuple)):
        binvals = [binvals]
    if not isinstance(zeropad, (list, tuple, np.ndarray)):
        zeropad = [zeropad]
    if isinstance(axis, (list, tuple, np.ndarray)) and not isinstance(zeropad, (list, tuple, np.ndarray)):
        raise ValueError("If axis is an iterable, so must be zeropad.")
    if len(axis) != len(zeropad):
        raise ValueError("len(axis) must equal len(zeropad)")

    for i, ax in enumerate(axis):
        if zeropad[i] > 0:
            if undo:
                s = [slice(None) for j in range(data.ndim)]
                s[ax] = slice(zeropad[i], -zeropad[i])
                s = tuple(s)
                data = data[s]
                if binvals[i] is not None:
                    binvals[i] = binvals[i][s[i]]

            else:
                zshape = list(data.shape)
                zshape[ax] = zeropad[i]
                if bool_dtype:
                    z = np.ones(zshape, bool)
                else:
                    z = np.zeros(zshape, data.dtype)
                data = np.concatenate([z, data, z], axis=ax)
                if binvals[i] is not None:
                    dx = np.median(np.diff(binvals[i]))
                    Nbin = binvals[i].size
                    z = np.arange(1, zeropad[i] + 1)
                    binvals[i] = np.concatenate([binvals[i][0] - z[::-1] * dx, binvals[i], binvals[i][-1] + z * dx])

    if len(binvals) == 1:
        binvals = binvals[0]

    return data, binvals


def noise_eq_bandwidth(window, axis=-1):
    """
    Calculate the noise equivalent bandwidth (NEB) of a windowing function
    as
        sqrt(window.size * window.max ** 2 / sum(window ** 2))

    See https://analog.intgckts.com/equivalent-noise-bandwidth/

    Args:
        window : float ndarray
        axis : int, axis along which to calculate NEB

    Returns
        neb : float or ndarray
            Noise equivalent bandwidth of the window
    """
    return np.sqrt(window.shape[axis] * np.max(window, axis=axis)**2 / np.sum(window**2, dtype=float, axis=axis))


def gen_filter_properties(ax='freq', horizon=1, standoff=0, min_dly=0, bl_len=None,
                          max_frate=0):
    """
    Convert standard delay and fringe-rate filtering parameters
    into hera_filters.dspec.fourier_filter parameters.
    If ax == 'both', filter properties are returned as (time, freq)

    Args:
        ax : str, options = ['freq', 'time', 'both']
        horizon : float, foreground wedge horizon coefficient
        standoff : float, wedge buffer [nanosec]
        min_dly: float, the CLEAN delay window
            is never below this minimum value (in nanosec)
        bl_len : float, baseline length in seconds (i.e. meters / c)
        max_frate : float, maximum |fringe-rate| to filter [mHz]

    Returns:
        filter_centers
            list of filter centers in units of Hz or sec
        filter_half_widths
            list of filter half widths in units Hz or sec
    """
    if ax == 'freq' or ax == 'both':
        filter_centers_freq = [0.]
        assert bl_len is not None
        # bl_dly in nanosec
        bl_dly = dspec._get_bl_dly(bl_len * 1e9, horizon=horizon, standoff=standoff, min_dly=min_dly)
        filter_half_widths_freq = [bl_dly * 1e-9]
    if ax == 'time' or ax == 'both':
        if max_frate is not None:
            filter_centers_time = [0.]
            filter_half_widths_time = [max_frate * 1e-3]
        else:
            raise AssertionError("must supply max_frate if ax=='time' or ax=='both'!")
    if ax == 'both':
        filter_half_widths = [filter_half_widths_time, filter_half_widths_freq]
        filter_centers = [filter_centers_time, filter_centers_freq]
    elif ax == 'freq':
        filter_centers = filter_centers_freq
        filter_half_widths = filter_half_widths_freq
    elif ax == 'time':
        filter_centers = filter_centers_time
        filter_half_widths = filter_half_widths_time

    return filter_centers, filter_half_widths


def _trim_status(info_dict, axis, zeropad):
    '''
    Trims the info status dictionary for a zero-padded
    filter so that the status of integrations that were
    in the zero-pad region are deleted

    Parameters
    ----------
    info : dict, info dictionary
    axis : integer, index of axis to trim
    zeropad : integer

    Returns
    -------
    Nothing, modifies the provided dictionary in place.
    '''
    # delete statuses in zero-pad region
    statuses = info_dict['status']['axis_%d' % axis]
    nints = len(statuses)
    for i in range(zeropad):
        del statuses[i]
        del statuses[nints - i - 1]
    # now update keys of the dict elements we wish to keep
    nints = len(statuses)
    for i in range(nints):
        statuses[i] = statuses.pop(i + zeropad)


def _adjust_info_indices(x, info_dict, edges, freq_baseind):
    """Adjust indices in info dict to reflect rows inserted by restore_flagged_edges

    Parameters
    ----------
    x: 2-tuple/list
        x-axis of data that has had rows/columns adjoining discontinuities removed.
    info_dict: dictionary
        info dict output by dspec.fourier_filter.
    edges: 2-list of lists of 2-tuples.
        list of 2-tuples indicating how many channels need to be inserted back
        within each discontinuous chunk.
    freq_baseind: int.
        index base for freq dimension
        (needed if processing spw_range that is not at lowest index of data).
    Returns
    -------
    N/A:
        modifies info-dict in place.
    """
    chunks = [find_discontinuity_edges(x[m]) for m in [1, 0]]
    axinds = [0, 1]
    edges = [edges[1], edges[0]]
    baseinds = [freq_baseind, 0]
    for axind, axchunks, axedges in zip(axinds, chunks, edges):
        statuses = info_dict['status']['axis_%d' % axind]
        offset = np.sum(np.hstack(axedges))
        for chunk, edge in zip(chunks[axind][::-1], edges[axind][::-1]):
            offset -= edge[1]
            for ind in range(chunk[1] - 1, chunk[0] - 1, -1):
                statuses[ind + offset + baseinds[axind]] = statuses.pop(ind)
            offset -= edge[0]


# ------------------------------------------
# Here is an argparser with core arguments
# needed for all types of xtalk and delay
# filtering.
# ------------------------------------------


def _filter_argparser():
    """
    Core Arg parser for commandline operation of hera_cal.delay_filter and hera_cal.xtalk_filter
    Parameters:
        None

    Returns:
        Argparser with core (but not complete) functionality that is called by _linear_argparser and
        _clean_argparser.
    """
    def list_of_int_tuples(v):
        if '~' in v:
            v = [tuple([int(_x) for _x in x.split('~')]) for x in v.split(",")]
        else:
            v = [tuple([int(_x) for _x in x.split()]) for x in v.split(",")]
        return v
    ap = argparse.ArgumentParser(description="Perform delay filter of visibility data.")
    ap.add_argument("datafilelist", default=None, type=str, nargs="+", help="list of data files to read in and perform filtering on.")
    ap.add_argument("--mode", type=str, default="clean", help="filtering mode to use. Can be dpss_leastsq, clean, dayenu.")
    ap.add_argument("--filetype_in", type=str, default='uvh5', help='filetype of input data files (default "uvh5")')
    ap.add_argument("--filetype_out", type=str, default='uvh5', help='filetype for output data files (default "uvh5")')
    ap.add_argument("--res_outfilename", default=None, type=str, help="path for writing the filtered visibilities with flags")
    ap.add_argument("--clobber", default=False, action="store_true", help='overwrites existing file at outfile')
    ap.add_argument("--spw_range", type=int, default=None, nargs=2, help="spectral window of data to foreground filter.")
    ap.add_argument("--tol", type=float, default=1e-9, help='Threshold for foreground and xtalk subtraction (default 1e-9)')
    ap.add_argument("--cornerturnfile", type=str, default=None, help="path to visibility data file to use as an index for baseline chunk in cornerturn."
                                                                     "Warning: Providing this file will result in outputs with significantly different structure "
                                                                     "then inputs. Only use it if you know what you are doing. Default is None.")
    ap.add_argument("--zeropad", default=None, type=int, help="number of bins to zeropad on both sides of FFT axis")
    ap.add_argument("--Nbls_per_load", default=None, type=int, help="the number of baselines to load at once (default None means load full data")
    ap.add_argument("--calfilelist", default=None, type=str, nargs="+", help="list of calibration files.")
    ap.add_argument("--CLEAN_outfilename", default=None, type=str, help="path for writing the filtered model visibilities (with the same flags)")
    ap.add_argument("--filled_outfilename", default=None, type=str, help="path for writing the original data but with flags unflagged and replaced with filtered models wherever possible")
    ap.add_argument("--polarizations", default=None, type=str, nargs="+", help="list of polarizations to filter.")
    ap.add_argument("--verbose", default=False, action="store_true", help="Lots of text.")
    ap.add_argument("--filter_spw_ranges", default=None, type=list_of_int_tuples, help="List of spw channel selections to filter independently. Two acceptable formats are "
                                                                                       "Ex1: '200~300,500~650' --> [(200, 300), (500, 650), ...] and "
                                                                                       "Ex2: '200 300, 500 650' --> [(200, 300), (500, 650), ...]")

    # Flagging options
    flag_options = ap.add_argument_group(title="Options relating to flagging.")
    flag_options.add_argument("--skip_wgt", type=float, default=0.1, help='skips filtering and flags times with unflagged fraction ~< skip_wgt (default 0.1)')
    flag_options.add_argument("--factorize_flags", default=False, action="store_true", help="Factorize flags.")
    flag_options.add_argument("--time_thresh", type=float, default=0.05, help="time threshold above which to completely flag channels and below which to flag times with flagged channel.")
    flag_options.add_argument("--external_flags", default=None, type=str, nargs="+", help="path(s) to external flag files that you wish to apply.")
    flag_options.add_argument("--overwrite_flags", default=False, action="store_true", help="overwrite existing flags.")
    flag_options.add_argument("--flag_yaml", default=None, type=str, help="path to a flagging yaml containing apriori antenna, freq, and time flags.")
    flag_options.add_argument("--skip_if_flag_within_edge_distance", type=int, default=0, help="skip integrations channels if there is a flag within this integer distance of edge. Default 0 means do nothing.")
    flag_options.add_argument("--dont_skip_contiguous_flags", default=False, action="store_true", help="Don't skip integrations or channels with gaps that are larger then integer specified in max_contiguous_flag")
    flag_options.add_argument("--max_contiguous_flag", type=int, default=None, help="Used if skip_contiguous_flags is True. Gaps larger then this value will be skipped. Default None uses filter periods.")
    flag_options.add_argument("--dont_skip_flagged_edges", default=False, action="store_true", help="Attept to filter over flagged edge times and/or channels.")
    flag_options.add_argument("--max_contiguous_edge_flags", type=int, default=1, help="Skip integrations with at least this number of contiguous edge flags.")
    flag_options.add_argument("--dont_flag_model_rms_outliers", default=False, action="store_true", help="Do not flag integrations or channels where the rms of the filter model exceeds the rms of the unflagged data.")
    flag_options.add_argument("--model_rms_threshold", type=float, default=1.1, help="Factor that rms of model in a channel or integration needs to exceed the rms of unflagged data to be flagged.")
    flag_options.add_argument("--clean_flags_not_in_resid_flags", default=False, action="store_true", help="Do not include flags from times/channels skipped in the resid flags.")

    # Arguments for CLEAN. Not used in linear filtering methods.
    clean_options = ap.add_argument_group(title='Options for CLEAN (arguments only used if mode=="clean"!)')
    clean_options.add_argument("--window", type=str, default='blackman-harris', help='window function for frequency filtering (default "blackman-harris",\
                              see hera_filters.dspec.gen_window for options')
    clean_options.add_argument("--maxiter", type=int, default=100, help='maximum iterations for aipy.deconv.clean to converge (default 100)')
    clean_options.add_argument("--edgecut_low", default=0, type=int, help="Number of channels to flag on lower band edge and exclude from window function.")
    clean_options.add_argument("--edgecut_hi", default=0, type=int, help="Number of channels to flag on upper band edge and exclude from window function.")
    clean_options.add_argument("--gain", type=float, default=0.1, help="Fraction of residual to use in each iteration.")
    clean_options.add_argument("--alpha", type=float, default=.5, help="If window='tukey', use this alpha parameter (default .5).")

    # Options for caching for linear filtering.
    cache_options = ap.add_argument_group(title='Options for caching (arguments only used if mode!="clean")')
    cache_options.add_argument("--write_cache", default=False, action="store_true", help="if True, writes newly computed filter matrices to cache.")
    cache_options.add_argument("--cache_dir", type=str, default=None, help="directory to store cached filtering matrices in.")
    cache_options.add_argument("--read_cache", default=False, action="store_true", help="If true, read in cache files in directory specified by cache_dir.")

    return ap


def time_chunk_from_baseline_chunks(time_chunk_template, baseline_chunk_files, outfilename, clobber=False, time_bounds=False):
    """Combine multiple waterfall files (with disjoint baseline sets) into time-limited file with all baselines.

    The methods delay_filter.load_delay_filter_and_write_baseline_list and
    xtalk_filter.load_xtalk_filter_and_write_baseline_list convert time-chunk files with all baselines into
    baseline-chunk files with all times. This function takes in a list of baseline-chunk files (baseline_chunk_files)
    and outputs a time-chunk file with all baselines with the same times as templatefile.

    Arguments
    ---------
    time_chunk_template : string
        path to file to use as a template for the time-chunk. Function selects times from time-chunk
        that exist in baseline_chunks for all baseline chunks and combines them into a single file.
        If the frequenies of the baseline_chunk files are a subset of the frequencies in the time_chunk, then
        output will trim the extra frequencies in the time_chunk and write out trimmed freqs. The same is true
        for polarizations.
    baseline_chunk_files : list of strings
        list of paths to baseline-chunk files to select time-chunk file from.
    outfilename : string
        name of the output file to write.
    clobber : bool optional.
        If False, don't overwrite outfilename if it already exists. Default is False.
    time_bounds: bool, optional
        If False, then generate new file with exact times from template.
        If True, generate a new file from the file list that keeps times between min/max of
            the times in the template_file. This is helpful if the times dont match in the reconstituted
            data but we want to use the template files to determine time ranges.
            Main application: reconstituting coherently averaged waterfalls into time-chunks that map to
            the original data files.

    Returns
    -------
        Nothing
    """
    hd_time_chunk = io.HERAData(time_chunk_template)
    hd_baseline_chunk = io.HERAData(baseline_chunk_files[0])
    times = hd_time_chunk.times
    freqs = hd_baseline_chunk.freqs
    polarizations = hd_baseline_chunk.pols
    # read in the template file, but only include polarizations, frequencies
    # from the baseline_chunk_file files.
    if not time_bounds:
        hd_time_chunk.read(times=times, frequencies=freqs, polarizations=polarizations)
        # set the all data to zero, flags to True, and nsamples to zero.
        hd_time_chunk.nsample_array[:] = 0.0
        hd_time_chunk.data_array[:] = 0.0 + 0j
        hd_time_chunk.flag_array[:] = True
        # for each baseline_chunk_file, read in only the times relevant to the templatefile.
        # and update the data, flags, nsamples array of the template file
        # with the baseline_chunk_file data.
        for baseline_chunk_file in baseline_chunk_files:
            hd_baseline_chunk = io.HERAData(baseline_chunk_file)
            # find times that are close.
            tload = []
            # use tolerance in times that is set by the time resolution of the dataset.
            atol = np.median(np.diff(hd_baseline_chunk.times)) / 10.
            all_times = np.unique(hd_baseline_chunk.times)
            for t in all_times:
                if np.any(np.isclose(t, hd_time_chunk.times, atol=atol, rtol=0)):
                    tload.append(t)
            d, f, n = hd_baseline_chunk.read(times=tload, axis='blt')
            hd_time_chunk.update(flags=f, data=d, nsamples=n)
        # now that we've updated everything, we write the output file.
        hd_time_chunk.write_uvh5(outfilename, clobber=clobber)
    else:
        dt_time_chunk = np.median(np.diff(hd_time_chunk.times)) / 2.
        tmax = hd_time_chunk.times.max() + dt_time_chunk
        tmin = hd_time_chunk.times.min() - dt_time_chunk
        hd_combined = io.HERAData(baseline_chunk_files)
        # we only compare centers of baseline files to time limits of time-file.
        # this is to prevent integrations that straddle file boundaries from being dropped.
        # when we perform reconstitution.
        t_select = (hd_baseline_chunk.times >= tmin) & (hd_baseline_chunk.times < tmax)
        if np.any(t_select):
            hd_combined.read(times=hd_baseline_chunk.times[t_select], axis='blt')
            hd_combined.write_uvh5(outfilename, clobber=clobber)
        else:
            warnings.warn("no times selected. No outputfile produced.", RuntimeWarning)


def time_chunk_from_baseline_chunks_argparser():
    """
    Arg parser for file reconstitution.
    """
    a = argparse.ArgumentParser(description="Construct time-chunk file from baseline-chunk files.")
    a.add_argument("time_chunk_template", type=str, help="name of template file.")
    a.add_argument("--baseline_chunk_files", type=str, nargs="+", help="list of file baseline-chunk files to select time-chunk from", required=True)
    a.add_argument("--outfilename", type=str, help="Name of output file. Provide the full path string.", required=True)
    a.add_argument("--clobber", action="store_true", help="Include to overwrite old files.")
    a.add_argument("--time_bounds", action="store_true", default=False, help="read times between min and max times of template, regardless of whether they match.")
    return a
