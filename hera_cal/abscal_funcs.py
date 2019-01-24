# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License

from __future__ import print_function, division, absolute_import

import os
import sys
from collections import OrderedDict as odict
import copy
import argparse
import functools
import numpy as np
import itertools
import operator
from functools import reduce
from six.moves import map, range, zip
from scipy import signal
from scipy import interpolate
from scipy import spatial
from pyuvdata import UVCal, UVData
import linsolve

from . import utils
from . import redcal
from . import io
from . import apply_cal
from .datacontainer import DataContainer
from .utils import polnum2str, polstr2num, jnum2str, jstr2num, reverse_bl, echo, fft_dly, split_pol, split_bl


def abs_amp_logcal(model, data, wgts=None, verbose=True):
    """
    calculate absolute (array-wide) gain amplitude scalar
    with a linear solver using the logarithmically linearized equation:

    ln|V_ij,xy^data / V_ij,xy^model| = eta_x + eta_y

    where {i,j} index antenna numbers and {x,y} index polarizations
    of the i-th and j-th antennas respectively.

    Parameters:
    -----------
    model : visibility data of refence model, type=DataContainer
            keys are antenna-pair + polarization tuples, Ex. (1, 2, 'xx').
            values are complex ndarray visibilities.
            these must be 2D arrays, with [0] axis indexing time
            and [1] axis indexing frequency.

    data : visibility data of measurements, type=DataContainer
           keys are antenna pair + pol tuples (must match model), values are
           complex ndarray visibilities matching shape of model

    wgts : weights of data, type=DataContainer, [default=None]
           keys are antenna pair + pol tuples (must match model), values are real floats
           matching shape of model and data

    verbose : print output, type=boolean, [default=False]

    Output:
    -------
    fit : dictionary with 'eta_{}' key for amplitude scalar for {} polarization,
            which has the same shape as the ndarrays in the model
    """
    echo("...configuring linsolve data for abs_amp_logcal", verbose=verbose)

    # get keys from model and data dictionary
    keys = sorted(set(model.keys()) & set(data.keys()))

    # abs of amplitude ratio is ydata independent variable
    ydata = odict([(k, np.log(np.abs(data[k] / model[k]))) for k in keys])

    # make weights if None
    if wgts is None:
        wgts = odict()
        for i, k in enumerate(keys):
            wgts[k] = np.ones_like(ydata[k], dtype=np.float)

    # fill nans and infs
    fill_dict_nans(ydata, wgts=wgts, nan_fill=0.0, inf_fill=0.0)

    # setup linsolve equations
    # a{} is a dummy variable to prevent linsolve from overwriting repeated measurements
    eqns = odict([(k, "a{}*eta_{}+a{}*eta_{}".format(i, utils.split_pol(k[-1])[0],
                                                     i, utils.split_pol(k[-1])[1])) for i, k in enumerate(keys)])
    ls_design_matrix = odict([("a{}".format(i), 1.0) for i, k in enumerate(keys)])

    # setup linsolve dictionaries
    ls_data = odict([(eqns[k], ydata[k]) for i, k in enumerate(keys)])
    ls_wgts = odict([(eqns[k], wgts[k]) for i, k in enumerate(keys)])

    # setup linsolve and run
    sol = linsolve.LinearSolver(ls_data, wgts=ls_wgts, **ls_design_matrix)
    echo("...running linsolve", verbose=verbose)
    fit = sol.solve()
    echo("...finished linsolve", verbose=verbose)

    return fit


def TT_phs_logcal(model, data, antpos, wgts=None, refant=None, verbose=True, zero_psi=True,
                  four_pol=False):
    """
    calculate overall gain phase and gain phase Tip-Tilt slopes (East-West and North-South)
    with a linear solver applied to the logarithmically linearized equation:

    angle(V_ij,xy^data / V_ij,xy^model) = angle(g_i_x * conj(g_j_y))
                                        = psi_x - psi_y + PHI^ew_x*r_i^ew + PHI^ns_x*r_i^ns
                                          - PHI^ew_y*r_j^ew - PHI^ns_y*r_j^ns

    where psi is the overall gain phase across the array [radians] for x and y polarizations,
    and PHI^ew, PHI^ns are the gain phase slopes across the east-west and north-south axes
    of the array in units of [radians / meter], where x and y denote the pol of the i-th and j-th
    antenna respectively. The phase slopes are polarization independent by default (1pol & 2pol cal),
    but can be merged with the four_pol parameter (4pol cal). r_i is the antenna position vector
    of the i^th antenna.

    Parameters:
    -----------
    model : visibility data of refence model, type=DataContainer
            keys are antenna-pair + polarization tuples, Ex. (1, 2, 'xx').
            values are complex ndarray visibilities.
            these must 2D arrays, with [0] axis indexing time
            and [1] axis indexing frequency.

    data : visibility data of measurements, type=DataContainer
           keys are antenna pair + pol tuples (must match model), values are
           complex ndarray visibilities matching shape of model

    wgts : weights of data, type=DataContainer, [default=None]
           keys are antenna pair + pol tuples (must match model), values are real floats
           matching shape of model and data

    refant : antenna number integer to use as a reference,
        The antenna position coordaintes are centered at the reference, such that its phase
        is identically zero across all frequencies. If None, use the first key in data as refant.

    antpos : antenna position vectors, type=dictionary
          keys are antenna integers, values are 2D
          antenna vectors in meters (preferably centered at center of array),
          with [0] index containing east-west separation and [1] index north-south separation

    zero_psi : set psi to be identically zero in linsolve eqns, type=boolean, [default=False]

    four_pol : type=boolean, even if multiple polarizations are present in data, make free
                variables polarization un-aware: i.e. one solution across all polarizations.
                This is the same assumption as 4-polarization calibration in omnical.

    verbose : print output, type=boolean, [default=False]

    Output:
    -------
    fit : dictionary with psi key for overall gain phase and Phi_ew and Phi_ns array containing
            phase slopes across the EW and NS directions of the array. There is a set of each
            of these variables per polarization.
    """
    echo("...configuring linsolve data for TT_phs_logcal", verbose=verbose)

    # get keys from model dictionary
    keys = sorted(set(model.keys()) & set(data.keys()))
    ants = np.unique(list(antpos.keys()))

    # angle of phs ratio is ydata independent variable
    # angle after divide
    ydata = odict([(k, np.angle(data[k] / model[k])) for k in keys])

    # make weights if None
    if wgts is None:
        wgts = odict()
        for i, k in enumerate(keys):
            wgts[k] = np.ones_like(ydata[k], dtype=np.float)

    # fill nans and infs
    fill_dict_nans(ydata, wgts=wgts, nan_fill=0.0, inf_fill=0.0)

    # center antenna positions about the reference antenna
    if refant is None:
        refant = keys[0][0]
    assert refant in ants, "reference antenna {} not found in antenna list".format(refant)
    antpos = odict(list(map(lambda k: (k, antpos[k] - antpos[refant]), antpos.keys())))

    # setup antenna position terms
    r_ew = odict(list(map(lambda a: (a, "r_ew_{}".format(a)), ants)))
    r_ns = odict(list(map(lambda a: (a, "r_ns_{}".format(a)), ants)))

    # setup linsolve equations
    if four_pol:
        eqns = odict([(k, "psi_{}*a1 - psi_{}*a2 + Phi_ew*{} + Phi_ns*{} - Phi_ew*{} - Phi_ns*{}"
                       "".format(utils.split_pol(k[2])[0], utils.split_pol(k[2])[1], r_ew[k[0]],
                                 r_ns[k[0]], r_ew[k[1]], r_ns[k[1]])) for i, k in enumerate(keys)])
    else:
        eqns = odict([(k, "psi_{}*a1 - psi_{}*a2 + Phi_ew_{}*{} + Phi_ns_{}*{} - Phi_ew_{}*{} - Phi_ns_{}*{}"
                       "".format(utils.split_pol(k[2])[0], utils.split_pol(k[2])[1], utils.split_pol(k[2])[0],
                                 r_ew[k[0]], utils.split_pol(k[2])[0], r_ns[k[0]], utils.split_pol(k[2])[1],
                                 r_ew[k[1]], k[2][1], r_ns[k[1]])) for i, k in enumerate(keys)])

    # set design matrix entries
    ls_design_matrix = odict(list(map(lambda a: ("r_ew_{}".format(a), antpos[a][0]), ants)))
    ls_design_matrix.update(odict(list(map(lambda a: ("r_ns_{}".format(a), antpos[a][1]), ants))))

    if zero_psi:
        ls_design_matrix.update({"a1": 0.0, "a2": 0.0})
    else:
        ls_design_matrix.update({"a1": 1.0, "a2": 1.0})

    # setup linsolve dictionaries
    ls_data = odict([(eqns[k], ydata[k]) for i, k in enumerate(keys)])
    ls_wgts = odict([(eqns[k], wgts[k]) for i, k in enumerate(keys)])

    # setup linsolve and run
    sol = linsolve.LinearSolver(ls_data, wgts=ls_wgts, **ls_design_matrix)
    echo("...running linsolve", verbose=verbose)
    fit = sol.solve()
    echo("...finished linsolve", verbose=verbose)

    return fit


def amp_logcal(model, data, wgts=None, verbose=True):
    """
    calculate per-antenna gain amplitude via the
    logarithmically linearized equation

    ln|V_ij,xy^data / V_ij,xy^model| = ln|g_i_x| + ln|g_j_y|
                                     = eta_i_x + eta_j_y

    where {x,y} represent the polarization of the i-th and j-th antenna
    respectively.

    Parameters:
    -----------
    model : visibility data of refence model, type=DataContainer
            keys are antenna-pair + polarization tuples, Ex. (1, 2, 'xx').
            values are complex ndarray visibilities.
            these must 2D arrays, with [0] axis indexing time
            and [1] axis indexing frequency.

    data : visibility data of measurements, type=DataContainer
           keys are antenna pair + pol tuples (must match model), values are
           complex ndarray visibilities matching shape of model

    wgts : weights of data, type=DataContainer, [default=None]
           keys are antenna pair + pol tuples (must match model), values are real floats
           matching shape of model and data

    Output:
    -------
    fit : dictionary containing eta_i = ln|g_i| for each antenna
    """
    echo("...configuring linsolve data for amp_logcal", verbose=verbose)

    # get keys from model dictionary
    keys = sorted(set(model.keys()) & set(data.keys()))

    # difference of log-amplitudes is ydata independent variable
    ydata = odict([(k, np.log(np.abs(data[k] / model[k]))) for k in keys])

    # make weights if None
    if wgts is None:
        wgts = odict()
        for i, k in enumerate(keys):
            wgts[k] = np.ones_like(ydata[k], dtype=np.float)

    # fill nans and infs
    fill_dict_nans(ydata, wgts=wgts, nan_fill=0.0, inf_fill=0.0)

    # setup linsolve equations
    eqns = odict([(k, "eta_{}_{} + eta_{}_{}".format(k[0], utils.split_pol(k[-1])[0],
                                                     k[1], utils.split_pol(k[-1])[1])) for i, k in enumerate(keys)])
    ls_design_matrix = odict()

    # setup linsolve dictionaries
    ls_data = odict([(eqns[k], ydata[k]) for i, k in enumerate(keys)])
    ls_wgts = odict([(eqns[k], wgts[k]) for i, k in enumerate(keys)])

    # setup linsolve and run
    sol = linsolve.LinearSolver(ls_data, wgts=ls_wgts, **ls_design_matrix)
    echo("...running linsolve", verbose=verbose)
    fit = sol.solve()
    echo("...finished linsolve", verbose=verbose)

    return fit


def phs_logcal(model, data, wgts=None, refant=None, verbose=True):
    """
    calculate per-antenna gain phase via the
    logarithmically linearized equation

    angle(V_ij,xy^data / V_ij,xy^model) = angle(g_i_x) - angle(g_j_y)
                                        = phi_i_x - phi_j_y

    where {x,y} represent the pol of the i-th and j-th antenna respectively.

    Parameters:
    -----------
    model : visibility data of refence model, type=DataContainer
            keys are antenna-pair + polarization tuples, Ex. (1, 2, 'xx').
            values are complex ndarray visibilities.
            these must 2D arrays, with [0] axis indexing time
            and [1] axis indexing frequency.

    data : visibility data of measurements, type=DataContainer
           keys are antenna pair + pol tuples (must match model), values are
           complex ndarray visibilities matching shape of model

    wgts : weights of data, type=DataContainer, [default=None]
           keys are antenna pair + pol tuples (must match model), values are real floats
           matching shape of model and data

    refant : integer antenna number of reference antenna, defult=None
        The refant phase will be set to identically zero in the linear equations.
        By default this takes the first antenna in data.

    Output:
    -------
    fit : dictionary containing phi_i = angle(g_i) for each antenna
    """
    echo("...configuring linsolve data for phs_logcal", verbose=verbose)

    # get keys from match between data and model dictionary
    keys = sorted(set(model.keys()) & set(data.keys()))

    # angle of visibility ratio is ydata independent variable
    ydata = odict([(k, np.angle(data[k] / model[k])) for k in keys])

    # make weights if None
    if wgts is None:
        wgts = odict()
        for i, k in enumerate(keys):
            wgts[k] = np.ones_like(ydata[k], dtype=np.float)

    # fill nans and infs
    fill_dict_nans(ydata, wgts=wgts, nan_fill=0.0, inf_fill=0.0)

    # setup linsolve equations
    eqns = odict([(k, "phi_{}_{} - phi_{}_{}".format(k[0], utils.split_pol(k[2])[0],
                                                     k[1], utils.split_pol(k[2])[1])) for i, k in enumerate(keys)])
    ls_design_matrix = odict()

    # setup linsolve dictionaries
    ls_data = odict([(eqns[k], ydata[k]) for i, k in enumerate(keys)])
    ls_wgts = odict([(eqns[k], wgts[k]) for i, k in enumerate(keys)])

    # get unique gain polarizations
    gain_pols = np.unique(list(map(lambda k: list(utils.split_pol(k[2])), keys)))

    # set reference antenna phase to zero
    if refant is None:
        refant = keys[0][0]
    assert np.array(list(map(lambda k: refant in k, keys))).any(), "refant {} not found in data and model".format(refant)

    for p in gain_pols:
        ls_data['phi_{}_{}'.format(refant, p)] = np.zeros_like(list(ydata.values())[0])
        ls_wgts['phi_{}_{}'.format(refant, p)] = np.ones_like(list(wgts.values())[0])

    # setup linsolve and run
    sol = linsolve.LinearSolver(ls_data, wgts=ls_wgts, **ls_design_matrix)
    echo("...running linsolve", verbose=verbose)
    fit = sol.solve()
    echo("...finished linsolve", verbose=verbose)

    return fit


def delay_lincal(model, data, wgts=None, refant=None, df=9.765625e4, solve_offsets=True, medfilt=True,
                 kernel=(1, 5), verbose=True, antpos=None, four_pol=False, edge_cut=0):
    """
    Solve for per-antenna delays according to the equation

    delay(V_ij,xy^data / V_ij,xy^model) = delay(g_i_x) - delay(g_j_y)

    Can also solve for per-antenna phase offsets with the solve_offsets kwarg.

    Parameters:
    -----------
    model : visibility data of refence model, type=DataContainer
            keys are antenna-pair + polarization tuples, Ex. (1, 2, 'xx').
            values are complex ndarray visibilities.
            these must 2D arrays, with [0] axis indexing time
            and [1] axis indexing frequency.

    data : visibility data of measurements, type=DataContainer
           keys are antenna pair + pol tuples (must match model), values are
           complex ndarray visibilities matching shape of model

    wgts : weights of data, type=DataContainer, [default=None]
           keys are antenna pair + pol tuples (must match model), values are real floats
           matching shape of model and data. These are only used to find delays from
           itegrations that are unflagged for at least two frequency bins. In this case,
           the delays are assumed to have equal weight, otherwise the delays take zero weight.

    refant : antenna number integer to use as reference
        Set the reference antenna to have zero delay, such that its phase is set to identically
        zero across all freqs. By default use the first key in data.

    df : type=float, frequency spacing between channels in Hz

    medfilt : type=boolean, median filter visiblity ratio before taking fft

    kernel : type=tuple, dtype=int, kernel for multi-dimensional median filter

    antpos : type=dictionary, antpos dictionary. antenna num as key, position vector as value.

    four_pol : type=boolean, if True, fit multiple polarizations together

    edge_cut : int, number of channels to exclude at each band edge in FFT window

    Output:
    -------
    fit : dictionary containing delay (tau_i_x) for each antenna and optionally
            offset (phi_i_x) for each antenna.
    """
    echo("...configuring linsolve data for delay_lincal", verbose=verbose)

    # get shared keys
    keys = sorted(set(model.keys()) & set(data.keys()))

    # make wgts
    if wgts is None:
        wgts = odict()
        for i, k in enumerate(keys):
            wgts[k] = np.ones_like(data[k], dtype=np.float)

    # median filter and FFT to get delays
    ratio_delays = []
    ratio_offsets = []
    ratio_wgts = []
    for i, k in enumerate(keys):
        ratio = data[k] / model[k]

        # replace nans
        nan_select = np.isnan(ratio)
        ratio[nan_select] = 0.0
        wgts[k][nan_select] = 0.0

        # replace infs
        inf_select = np.isinf(ratio)
        ratio[inf_select] = 0.0
        wgts[k][inf_select] = 0.0

        # get delays
        dly, offset = fft_dly(ratio, df, wgts=wgts[k], medfilt=medfilt, kernel=kernel, edge_cut=edge_cut)

        # set nans to zero
        rwgts = np.nanmean(wgts[k], axis=1, keepdims=True)
        isnan = np.isnan(dly)
        dly[isnan] = 0.0
        rwgts[isnan] = 0.0
        offset[isnan] = 0.0

        ratio_delays.append(dly)
        ratio_offsets.append(offset)
        ratio_wgts.append(rwgts)

    ratio_delays = np.array(ratio_delays)
    ratio_offsets = np.array(ratio_offsets)
    ratio_wgts = np.array(ratio_wgts)

    # form ydata
    ydata = odict(zip(keys, ratio_delays))

    # form wgts
    ywgts = odict(zip(keys, ratio_wgts))

    # setup linsolve equation dictionary
    eqns = odict([(k, 'tau_{}_{} - tau_{}_{}'.format(k[0], utils.split_pol(k[2])[0],
                                                     k[1], utils.split_pol(k[2])[1])) for i, k in enumerate(keys)])

    # setup design matrix dictionary
    ls_design_matrix = odict()

    # setup linsolve data dictionary
    ls_data = odict([(eqns[k], ydata[k]) for i, k in enumerate(keys)])
    ls_wgts = odict([(eqns[k], ywgts[k]) for i, k in enumerate(keys)])

    # get unique gain polarizations
    gain_pols = np.unique(list(map(lambda k: [utils.split_pol(k[2])[0], utils.split_pol(k[2])[1]], keys)))

    # set reference antenna phase to zero
    if refant is None:
        refant = keys[0][0]
    assert np.array(list(map(lambda k: refant in k, keys))).any(), "refant {} not found in data and model".format(refant)

    for p in gain_pols:
        ls_data['tau_{}_{}'.format(refant, p)] = np.zeros_like(list(ydata.values())[0])
        ls_wgts['tau_{}_{}'.format(refant, p)] = np.ones_like(list(ywgts.values())[0])

    # setup linsolve and run
    sol = linsolve.LinearSolver(ls_data, wgts=ls_wgts, **ls_design_matrix)
    echo("...running linsolve", verbose=verbose)
    fit = sol.solve()
    echo("...finished linsolve", verbose=verbose)

    # setup linsolve parameters
    ydata = odict(zip(keys, ratio_offsets))
    eqns = odict([(k, 'phi_{}_{} - phi_{}_{}'.format(k[0], utils.split_pol(k[2])[0],
                                                     k[1], utils.split_pol(k[2])[1])) for i, k in enumerate(keys)])
    ls_data = odict([(eqns[k], ydata[k]) for i, k in enumerate(keys)])
    ls_wgts = odict([(eqns[k], ywgts[k]) for i, k in enumerate(keys)])
    ls_design_matrix = odict()
    for p in gain_pols:
        ls_data['phi_{}_{}'.format(refant, p)] = np.zeros_like(list(ydata.values())[0])
        ls_wgts['phi_{}_{}'.format(refant, p)] = np.ones_like(list(ywgts.values())[0])
    sol = linsolve.LinearSolver(ls_data, wgts=ls_wgts, **ls_design_matrix)
    echo("...running linsolve", verbose=verbose)
    offset_fit = sol.solve()
    echo("...finished linsolve", verbose=verbose)
    fit.update(offset_fit)

    return fit


def delay_slope_lincal(model, data, antpos, wgts=None, refant=None, df=9.765625e4, medfilt=True,
                       kernel=(1, 5), verbose=True, four_pol=False, edge_cut=0):
    """
    Solve for an array-wide delay slope according to the equation

    delay(V_ij,xy^data / V_ij,xy^model) = dot(T_x, r_i) - dot(T_y, r_j)

    This does not solve for per-antenna delays, but rather a delay slope across the array.

    Parameters:
    -----------
    model : visibility data of refence model, type=DataContainer
            keys are antenna-pair + polarization tuples, Ex. (1, 2, 'xx').
            values are complex ndarray visibilities.
            these must 2D arrays, with [0] axis indexing time
            and [1] axis indexing frequency.

    data : visibility data of measurements, type=DataContainer
           keys are antenna pair + pol tuples (must match model), values are
           complex ndarray visibilities matching shape of model

    antpos : type=dictionary, antpos dictionary. antenna num as key, position vector as value.

    wgts : weights of data, type=DataContainer, [default=None]
           keys are antenna pair + pol tuples (must match model), values are real floats
           matching shape of model and data. These are only used to find delays from
           itegrations that are unflagged for at least two frequency bins. In this case,
           the delays are assumed to have equal weight, otherwise the delays take zero weight.

    refant : antenna number integer to use as a reference,
        The antenna position coordaintes are centered at the reference, such that its phase
        is identically zero across all frequencies. If None, use the first key in data as refant.

    df : type=float, frequency spacing between channels in Hz

    medfilt : type=boolean, median filter visiblity ratio before taking fft

    kernel : type=tuple, dtype=int, kernel for multi-dimensional median filter

    four_pol : type=boolean, if True, fit multiple polarizations together

    edge_cut : int, number of channels to exclude at each band edge of vis in FFT window

    Output:
    -------
    fit : dictionary containing delay slope (T_x) for each pol [seconds / meter].
    """
    echo("...configuring linsolve data for delay_slope_lincal", verbose=verbose)

    # get shared keys
    keys = sorted(set(model.keys()) & set(data.keys()))
    ants = np.unique(list(antpos.keys()))

    # make wgts
    if wgts is None:
        wgts = odict()
        for i, k in enumerate(keys):
            wgts[k] = np.ones_like(data[k], dtype=np.float)

    # center antenna positions about the reference antenna
    if refant is None:
        refant = keys[0][0]
    assert refant in ants, "reference antenna {} not found in antenna list".format(refant)
    antpos = odict(list(map(lambda k: (k, antpos[k] - antpos[refant]), antpos.keys())))

    # median filter and FFT to get delays
    ratio_delays = []
    ratio_offsets = []
    ratio_wgts = []
    for i, k in enumerate(keys):
        ratio = data[k] / model[k]

        # replace nans
        nan_select = np.isnan(ratio)
        ratio[nan_select] = 0.0
        wgts[k][nan_select] = 0.0

        # replace infs
        inf_select = np.isinf(ratio)
        ratio[inf_select] = 0.0
        wgts[k][inf_select] = 0.0

        # get delays
        dly, _ = fft_dly(ratio, df, wgts=wgts[k], medfilt=medfilt, kernel=kernel, edge_cut=edge_cut)

        # set nans to zero
        rwgts = np.nanmean(wgts[k], axis=1, keepdims=True)
        isnan = np.isnan(dly)
        dly[isnan] = 0.0
        rwgts[isnan] = 0.0

        ratio_delays.append(dly)
        ratio_wgts.append(rwgts)

    ratio_delays = np.array(ratio_delays)
    ratio_wgts = np.array(ratio_wgts)

    # form ydata
    ydata = odict(zip(keys, ratio_delays))

    # form wgts
    ywgts = odict(zip(keys, ratio_wgts))

    # setup antenna position terms
    r_ew = odict(list(map(lambda a: (a, "r_ew_{}".format(a)), ants)))
    r_ns = odict(list(map(lambda a: (a, "r_ns_{}".format(a)), ants)))

    # setup linsolve equations
    if four_pol:
        eqns = odict([(k, "T_ew*{} + T_ns*{} - T_ew*{} - T_ns*{}"
                       "".format(r_ew[k[0]], r_ns[k[0]], r_ew[k[1]], r_ns[k[1]])) for i, k in enumerate(keys)])
    else:
        eqns = odict([(k, "T_ew_{}*{} + T_ns_{}*{} - T_ew_{}*{} - T_ns_{}*{}"
                       "".format(utils.split_pol(k[2])[0], r_ew[k[0]], utils.split_pol(k[2])[0], r_ns[k[0]],
                                 utils.split_pol(k[2])[1], r_ew[k[1]], utils.split_pol(k[2])[1], r_ns[k[1]]))
                      for i, k in enumerate(keys)])

    # set design matrix entries
    ls_design_matrix = odict(list(map(lambda a: ("r_ew_{}".format(a), antpos[a][0]), ants)))
    ls_design_matrix.update(odict(list(map(lambda a: ("r_ns_{}".format(a), antpos[a][1]), ants))))

    # setup linsolve data dictionary
    ls_data = odict([(eqns[k], ydata[k]) for i, k in enumerate(keys)])
    ls_wgts = odict([(eqns[k], ywgts[k]) for i, k in enumerate(keys)])

    # setup linsolve and run
    sol = linsolve.LinearSolver(ls_data, wgts=ls_wgts, **ls_design_matrix)
    echo("...running linsolve", verbose=verbose)
    fit = sol.solve()
    echo("...finished linsolve", verbose=verbose)

    return fit


def global_phase_slope_logcal(model, data, antpos, wgts=None, refant=None, verbose=True, tol=1.0, edge_cut=0):
    """
    Solve for a frequency-independent spatial phase slope using the equation

    median_over_freq(angle(V_ij,xy^data / V_ij,xy^model)) = dot(Phi_x, r_i) - dot(Phi_y, r_j)

    Parameters:
    -----------
    model : visibility data of refence model, type=DataContainer
            keys are antenna-pair + polarization tuples, Ex. (1, 2, 'xx').
            values are complex ndarray visibilities.
            these must 2D arrays, with [0] axis indexing time
            and [1] axis indexing frequency.

    data : visibility data of measurements, type=DataContainer
           keys are antenna pair + pol tuples (must match model), values are
           complex ndarray visibilities matching shape of model

    antpos : type=dictionary, antpos dictionary. antenna num as key, position vector as value.

    wgts : weights of data, type=DataContainer, [default=None]
           keys are antenna pair + pol tuples (must match model), values are real floats
           matching shape of model and data. These are only used to find delays from
           itegrations that are unflagged for at least two frequency bins. In this case,
           the delays are assumed to have equal weight, otherwise the delays take zero weight.

    refant : antenna number integer to use as a reference,
        The antenna position coordaintes are centered at the reference, such that its phase
        is identically zero across all frequencies. If None, use the first key in data as refant.

    verbose : print output, type=boolean, [default=False]

    tol : type=float, baseline match tolerance in units of baseline vectors (e.g. meters)

    edge_cut : int, number of channels to exclude at each band edge in phase slope solver

    Output:
    -------
    fit : dictionary containing frequency-indpendent phase slope, e.g. Phi_ns_x
          for each position component and polarization [radians / meter].
    """
    echo("...configuring linsolve data for global_phase_slope_logcal", verbose=verbose)
    assert 2 * edge_cut < list(data.values())[0].shape[1] - 1, "edge_cut cannot be >= Nfreqs/2 - 1"

    # get keys from model and data dictionaries
    keys = sorted(set(model.keys()) & set(data.keys()))
    ants = np.unique(list(antpos.keys()))

    # make weights if None and make flags
    if wgts is None:
        wgts = odict()
        for i, k in enumerate(keys):
            wgts[k] = np.ones_like(data[k], dtype=np.float)

    # center antenna positions about the reference antenna
    if refant is None:
        refant = keys[0][0]
    assert refant in ants, "reference antenna {} not found in antenna list".format(refant)
    antpos = odict(list(map(lambda k: (k, antpos[k] - antpos[refant]), antpos.keys())))

    # average data over baselines
    _reds = redcal.get_reds(antpos, bl_error_tol=tol, pols=data.pols())
    reds = []
    for _red in _reds:
        red = [bl for bl in _red if bl in keys]
        if len(red) > 0:
            reds.append(red)

    avg_data, avg_wgts, red_keys = avg_data_across_red_bls(DataContainer({k: data[k] for k in keys}),
                                                           antpos, wgts=wgts, broadcast_wgts=False, tol=tol, reds=reds)
    avg_model, _, _ = avg_data_across_red_bls(DataContainer({k: model[k] for k in keys}),
                                              antpos, wgts=wgts, broadcast_wgts=False, tol=tol, reds=reds)

    # build linear system
    ls_data, ls_wgts = {}, {}
    for rk in red_keys:
        # build equation string
        eqn_str = '{}*Phi_ew_{} + {}*Phi_ns_{} - {}*Phi_ew_{} - {}*Phi_ns_{}'
        eqn_str = eqn_str.format(antpos[rk[0]][0], utils.split_pol(rk[2])[0], antpos[rk[0]][1], utils.split_pol(rk[2])[0],
                                 antpos[rk[1]][0], utils.split_pol(rk[2])[1], antpos[rk[1]][1], utils.split_pol(rk[2])[1])

        # calculate median of unflagged angle(data/model)
        # ls_weights are sum of non-binary weights
        delta_phi = np.angle(avg_data[rk] / avg_model[rk])
        binary_flgs = np.isclose(avg_wgts[rk], 0.0)
        delta_phi[binary_flgs] *= np.nan
        avg_wgts[rk][np.isinf(delta_phi) + np.isnan(delta_phi)] = 0.0
        delta_phi[np.isinf(delta_phi) + np.isnan(delta_phi)] *= np.nan
        ls_data[eqn_str] = np.nanmedian(delta_phi[:, edge_cut:(delta_phi.shape[1] - edge_cut)], axis=1, keepdims=True)
        ls_wgts[eqn_str] = np.sum(avg_wgts[rk][:, edge_cut:(delta_phi.shape[1] - edge_cut)], axis=1, keepdims=True)

        # set unobserved data to 0 with 0 weight
        ls_wgts[eqn_str][np.isnan(ls_data[eqn_str])] = 0
        ls_data[eqn_str][np.isnan(ls_data[eqn_str])] = 0

    # setup linsolve and run
    solver = linsolve.LinearSolver(ls_data, wgts=ls_wgts)
    echo("...running linsolve", verbose=verbose)
    fit = solver.solve()
    echo("...finished linsolve", verbose=verbose)
    return fit


def merge_gains(gains):
    """
    merge multiple gain dictionaries. will merge only shared keys.

    Parameters:
    -----------
    gains : type=list or tuple, series of gain dictionaries with (ant, pol) keys
            and complex ndarrays as values.

    Output:
    -------
    merged_gains : type=dictionary, merged gain dictionary with same key-value
                   structure as input gain dictionaries.
    """
    # get shared keys
    keys = sorted(reduce(operator.and_, list(map(lambda g: set(g.keys()), gains))))

    # form merged_gains dict
    merged_gains = odict()

    # iterate over keys
    for i, k in enumerate(keys):
        merged_gains[k] = reduce(operator.mul, list(map(lambda g: g.get(k, 1.0), gains)))

    return merged_gains


def apply_gains(data, gains, gain_convention='divide'):
    """
    Apply gain solutions to data. If a requested antenna doesn't
    exist in gains, eliminate associated visibilities from new_data.

    Parameters:
    -----------
    data : type=DataContainer, holds complex visibility data.
        keys are antenna-pair tuples + pol tuples.
        values are ndarray complex visibility data.

    gains : type=dictionary, holds complex, per-antenna gain data.
            keys are antenna integer + gain pol tuples, Ex. (1, 'x').
            values are complex ndarrays
            with shape matching data's visibility ndarrays

            Optionally, can be a tuple holding multiple gains dictionaries
            that will all be multiplied together.

    gain_convention : type=str, options=['multiply', 'divide']
                      option to multiply in or divide in gain solutions to data.

    Output:
    -------
    new_data : type=DataContainer, data with gains applied

    Notes:
    ------
    gain convention == 'divide' means that the gains need to be divided out of the observation
    to get the model/truth, i.e. that V_obs = gi gj* V_true. 'multiply' means that the gains need
    to be multiplied into the observation to get the model/truth, i.e. that V_true = gi gj* V_obs.
    In Abscal (as on omnical and redcal), the standard gain convention is 'divide'.
    """
    # form new dictionary
    new_data = odict()

    # get keys
    keys = list(data.keys())

    # merge gains if multiple gain dictionaries fed
    if isinstance(gains, list) or isinstance(gains, tuple) or isinstance(gains, np.ndarray):
        gains = merge_gains(gains)

    # iterate over keys:
    for i, k in enumerate(keys):
        # get gain keys
        g1 = (k[0], utils.split_pol(k[-1])[0])
        g2 = (k[1], utils.split_pol(k[-1])[1])

        # ensure keys are in gains
        if g1 not in gains or g2 not in gains:
            continue

        # form visbility gain product
        vis_gain = gains[g1] * np.conj(gains[g2])

        # apply to data
        if gain_convention == "multiply":
            new_data[k] = data[k] * vis_gain

        elif gain_convention == "divide":
            new_data[k] = data[k] / vis_gain

    return DataContainer(new_data)


def data_key_to_array_axis(data, key_index, array_index=-1, avg_dict=None):
    """
    move an index of data.keys() into the data axes

    Parameters:
    -----------
    data : type=DataContainer, complex visibility data with
        antenna-pair + pol tuples for keys, in DataContainer dictionary format.

    key_index : integer, index of keys to consolidate into data arrays

    array_index : integer, which axes of data arrays to append to

    avg_dict : DataContainer, a dictionary with same keys as data
        that will have its data arrays averaged along key_index

    Result:
    -------
    new_data : DataContainer, complex visibility data
        with key_index of keys moved into the data arrays

    new_avg_dict : copy of avg_dict. Only returned if avg_dict is not None.

    popped_keys : unique list of keys moved into data array axis
    """
    # instantiate new data object
    new_data = odict()
    new_avg = odict()

    # get keys
    keys = list(data.keys())

    # sort keys across key_index
    key_sort = np.argsort(np.array(keys, dtype=np.object)[:, key_index])
    keys = list(map(lambda i: keys[i], key_sort))
    popped_keys = np.unique(np.array(keys, dtype=np.object)[:, key_index])

    # get new keys
    new_keys = list(map(lambda k: k[:key_index] + k[key_index + 1:], keys))
    new_unique_keys = []

    # iterate over new_keys
    for i, nk in enumerate(new_keys):
        # check for unique keys
        if nk in new_unique_keys:
            continue
        new_unique_keys.append(nk)

        # get all instances of redundant keys
        ravel = list(map(lambda k: k == nk, new_keys))

        # iterate over redundant keys and consolidate into new arrays
        arr = []
        avg_arr = []
        for j, b in enumerate(ravel):
            if b:
                arr.append(data[keys[j]])
                if avg_dict is not None:
                    avg_arr.append(avg_dict[keys[j]])

        # assign to new_data
        new_data[nk] = np.moveaxis(arr, 0, array_index)
        if avg_dict is not None:
            new_avg[nk] = np.nanmean(avg_arr, axis=0)

    if avg_dict is not None:
        return new_data, new_avg, popped_keys
    else:
        return new_data, popped_keys


def array_axis_to_data_key(data, array_index, array_keys, key_index=-1, copy_dict=None):
    """
    move an axes of data arrays in data out of arrays
    and into a unique key index in data.keys()

    Parameters:
    -----------
    data : DataContainer, complex visibility data with
        antenna-pair (+ pol + other) tuples for keys

    array_index : integer, which axes of data arrays
        to extract from arrays and move into keys

    array_keys : list, list of new key from array elements. must have length
        equal to length of data_array along axis array_index

    key_index : integer, index within the new set of keys to insert array_keys

    copy_dict : DataContainer, a dictionary with same keys as data
        that will have its data arrays copied along array_keys

    Output:
    -------
    new_data : DataContainer, complex visibility data
        with array_index of data arrays extracted and moved
        into a unique set of keys

    new_copy : DataContainer, copy of copy_dict
        with array_index of data arrays copied to unique keys
    """
    # instantiate new object
    new_data = odict()
    new_copy = odict()

    # get keys
    keys = sorted(data.keys())
    new_keys = []

    # iterate over keys
    for i, k in enumerate(keys):
        # iterate overy new array keys
        for j, ak in enumerate(array_keys):
            new_key = list(k)
            if key_index == -1:
                new_key.insert(len(new_key), ak)
            else:
                new_key.insert(key_index, ak)
            new_key = tuple(new_key)
            new_data[new_key] = np.take(data[k], j, axis=array_index)
            if copy_dict is not None:
                new_copy[new_key] = copy.copy(copy_dict[k])

    if copy_dict is not None:
        return new_data, new_copy
    else:
        return new_data


def wiener(data, window=(5, 11), noise=None, medfilt=True, medfilt_kernel=(3, 9), array=False):
    """
    wiener filter complex visibility data. this might be used in constructing
    model reference. See scipy.signal.wiener for details on method.

    Parameters:
    -----------
    data : type=DataContainer, ADataContainer dictionary holding complex visibility data
           unelss array is True

    window : type=tuple, wiener-filter window along each axis of data

    noise : type=float, estimate of noise. if None will estimate itself

    medfilt : type=bool, if True, median filter data before wiener filtering

    medfilt_kernel : type=tuple, median filter kernel along each axis of data

    array : type=boolean, if True, feeding a single ndarray, rather than a dictionary

    Output: (new_data)
    -------
    new_data type=DataContainer, DataContainer dictionary holding new visibility data
    """
    # check if data is an array
    if array:
        data = {'arr': data}

    new_data = odict()
    for i, k in enumerate(list(data.keys())):
        real = np.real(data[k])
        imag = np.imag(data[k])
        if medfilt:
            real = signal.medfilt(real, kernel_size=medfilt_kernel)
            imag = signal.medfilt(imag, kernel_size=medfilt_kernel)

        new_data[k] = signal.wiener(real, mysize=window, noise=noise) + \
            1j * signal.wiener(imag, mysize=window, noise=noise)

    if array:
        return new_data['arr']
    else:
        return new_data


def interp2d_vis(model, model_lsts, model_freqs, data_lsts, data_freqs, flags=None,
                 kind='cubic', flag_extrapolate=True, medfilt_flagged=True, medfilt_window=(3, 7),
                 fill_value=None):
    """
    Interpolate complex visibility model onto the time & frequency basis of
    a data visibility. See below for notes on flag propagation if flags is provided.

    Parameters:
    -----------
    model : type=DataContainer, holds complex visibility for model
        keys are antenna-pair + pol tuples, values are 2d complex visibility
        with shape (Ntimes, Nfreqs)

    model_lsts : 1D array of the model time axis, dtype=float, shape=(Ntimes,)

    model_freqs : 1D array of the model freq axis, dtype=float, shape=(Nfreqs,)

    data_lsts : 1D array of the data time axis, dtype=float, shape=(Ntimes,)

    data_freqs : 1D array of the data freq axis, dtype=float, shape=(Nfreqs,)

    flags : type=DataContainer, dictionary containing model flags. Can also contain model wgts
            as floats and will convert to booleans appropriately.

    kind : type=str, kind of interpolation, options=['linear', 'cubic', 'quintic']

    medfilt_flagged : type=bool, if True, before interpolation, replace flagged pixels with output from
                      a median filter centered on each flagged pixel.

    medfilt_window : type=tuple, extent of window for median filter across the (time, freq) axes.
                     Even numbers are rounded down to odd number.

    flag_extrapolate : type=bool, flag extrapolated data_lsts if True.

    fill_value : type=float, if fill_value is None, extrapolated points are extrapolated
                 else they are filled with fill_value.

    Output: (new_model, new_flags)
    -------
    new_model : interpolated model, type=DataContainer
    new_flags : flags associated with interpolated model, type=DataContainer

    Notes:
    ------
    If the data has flagged pixels, it is recommended to turn medfilt_flagged to True. This runs a median
    filter on the flagged pixels and replaces their values with the results, but they remain flagged.
    This happens *before* interpolation. This means that interpolation near flagged pixels
    aren't significantly biased by their presence.

    In general, if flags are fed, flags are propagated if a flagged pixel is a nearest neighbor
    of an interpolated pixel.
    """
    # make flags
    new_model = odict()
    new_flags = odict()

    # get nearest neighbor points
    freq_nn = np.array(list(map(lambda x: np.argmin(np.abs(model_freqs - x)), data_freqs)))
    time_nn = np.array(list(map(lambda x: np.argmin(np.abs(model_lsts - x)), data_lsts)))
    freq_nn, time_nn = np.meshgrid(freq_nn, time_nn)

    # get model indices meshgrid
    mod_F, mod_L = np.meshgrid(np.arange(len(model_freqs)), np.arange(len(model_lsts)))

    # raise warning on flags
    if flags is not None and medfilt_flagged is False:
        print("Warning: flags are fed, but medfilt_flagged=False. \n"
              "This may cause weird behavior of interpolated points near flagged data.")

    # ensure flags are booleans
    if flags is not None:
        if np.issubdtype(flags[list(flags.keys())[0]].dtype, np.float):
            flags = DataContainer(odict(list(map(lambda k: (k, ~flags[k].astype(np.bool)), flags.keys()))))

    # loop over keys
    for i, k in enumerate(list(model.keys())):
        # get model array
        m = model[k]

        # get real and imag separately
        real = np.real(m)
        imag = np.imag(m)

        # median filter flagged data if desired
        if medfilt_flagged and flags is not None:
            # get extent of window along freq and time
            f_ext = int((medfilt_window[1] - 1) / 2.)
            t_ext = int((medfilt_window[0] - 1) / 2.)

            # set flagged data to nan
            real[flags[k]] *= np.nan
            imag[flags[k]] *= np.nan

            # get flagged indices
            f_indices = mod_F[flags[k]]
            l_indices = mod_L[flags[k]]

            # construct fill arrays
            real_fill = np.empty(len(f_indices), np.float)
            imag_fill = np.empty(len(f_indices), np.float)

            # iterate over flagged data and replace w/ medfilt
            for j, (find, tind) in enumerate(zip(f_indices, l_indices)):
                tlow, thi = tind - t_ext, tind + t_ext + 1
                flow, fhi = find - f_ext, find + f_ext + 1
                ll = 0
                while True:
                    # iterate until window has non-flagged data in it
                    # with a max of 10 iterations
                    if tlow < 0:
                        tlow = 0
                    if flow < 0:
                        flow = 0
                    r_med = np.nanmedian(real[tlow:thi, flow:fhi])
                    i_med = np.nanmedian(imag[tlow:thi, flow:fhi])
                    tlow -= 2
                    thi += 2
                    flow -= 2
                    fhi += 2
                    ll += 1
                    if not (np.isnan(r_med) or np.isnan(i_med)):
                        break
                    if ll > 10:
                        break
                real_fill[j] = r_med
                imag_fill[j] = i_med

            # fill real and imag
            real[l_indices, f_indices] = real_fill
            imag[l_indices, f_indices] = imag_fill

            # flag residual nans
            resid_nans = np.isnan(real) + np.isnan(imag)
            flags[k] += resid_nans

            # replace residual nans
            real[resid_nans] = 0.0
            imag[resid_nans] = 0.0

        # propagate flags to nearest neighbor
        if flags is not None:
            f = flags[k][time_nn, freq_nn]
            # check f is boolean type
            if np.issubdtype(f.dtype, np.float):
                f = ~(f.astype(np.bool))
        else:
            f = np.zeros_like(real, bool)

        # interpolate
        interp_real = interpolate.interp2d(model_freqs, model_lsts, real, kind=kind, copy=False, bounds_error=False, fill_value=fill_value)(data_freqs, data_lsts)
        interp_imag = interpolate.interp2d(model_freqs, model_lsts, imag, kind=kind, copy=False, bounds_error=False, fill_value=fill_value)(data_freqs, data_lsts)

        # flag extrapolation if desired
        if flag_extrapolate:
            time_extrap = np.where((data_lsts > model_lsts.max() + 1e-6) | (data_lsts < model_lsts.min() - 1e-6))
            freq_extrap = np.where((data_freqs > model_freqs.max() + 1e-6) | (data_freqs < model_freqs.min() - 1e-6))
            f[time_extrap, :] = True
            f[:, freq_extrap] = True

        # rejoin
        new_model[k] = interp_real + 1j * interp_imag
        new_flags[k] = f

    return DataContainer(new_model), DataContainer(new_flags)


def rephase_vis(model, model_lsts, data_lsts, bls, freqs, inplace=False, flags=None, max_dlst=0.005, latitude=-30.72152):
    """
    Rephase model visibility data onto LST grid of data_lsts.

    Parameters:
    -----------
    model : type=DataContainer, holds complex visibility for model
        keys are antenna-pair + pol tuples, values are 2d complex visibility
        with shape (Ntimes, Nfreqs)

    model_lsts : 1D array of the LST grid in model [radians], dtype=float, shape=(Ntimes,)

    data_lsts : 1D array of the LST grid in data [radians], dtype=float, shape=(Ntimes,)

    bls : type=dictionary, ant-pair keys that holds baseline position vector in ENU frame in meters

    freqs : type=float ndarray, holds frequency channels of model in Hz.

    inplace : type=bool, if True edit data in memory, else make a copy and return

    flags : type=DataContainer, holds model flags

    max_dlst : type=bool, maximum dlst [radians] to allow for rephasing, otherwise flag data.

    latitude : type=float, latitude of array in degrees North

    Return: (new_model, new_flags)
    -------
    new_model : DataContainer with rephased model
    new_flags : DataContainer with new flags
    """
    # unravel LST array if necessary
    if data_lsts.max() < data_lsts.min():
        data_lsts[data_lsts]

    # get nearest neighbor model points
    lst_nn = np.array(list(map(lambda x: np.argmin(np.abs(model_lsts - x)), data_lsts)))

    # get dlst array
    dlst = data_lsts - model_lsts[lst_nn]

    # flag dlst above threshold
    flag_lst = np.zeros_like(dlst, np.bool)
    flag_lst[np.abs(dlst) > max_dlst] = True

    # make new_model and new_flags
    if inplace:
        new_model = model
    else:
        new_model = odict()
    if inplace and flags is not None:
        new_flags = flags
    else:
        new_flags = odict()

    for k in model.keys():
        m = model[k][lst_nn, :]
        new_model[k] = m
        if flags is None:
            new_flags[k] = np.zeros_like(m, np.bool)
        else:
            new_flags[k] = flags[k][lst_nn, :]
        new_flags[k][flag_lst, :] = True

    # rephase
    if inplace:
        utils.lst_rephase(new_model, bls, freqs, dlst, lat=latitude, inplace=True)
        return new_model, new_flags
    else:
        new_model = utils.lst_rephase(new_model, bls, freqs, dlst, lat=latitude, inplace=False)
        return DataContainer(new_model), DataContainer(new_flags)


def fill_dict_nans(data, wgts=None, nan_fill=None, inf_fill=None, array=False):
    """
    take a dictionary and re-fill nan and inf ndarray values.

    Parameters:
    -----------
    data : type=DataContainer, visibility dictionary in AbsCal dictionary format

    wgts : type=DataContainer, weights dictionary matching shape of data to also fill

    nan_fill : if not None, fill nans with nan_fill

    inf_fill : if not None, fill infs with inf_fill

    array : type=boolean, if True, data is a single ndarray to perform operation on
    """
    if array:
        if nan_fill is not None:
            nan_select = np.isnan(data)
            data[nan_select] = nan_fill
            if wgts is not None:
                wgts[nan_select] = 0.0
        if inf_fill is not None:
            inf_select = np.isinf(data)
            data[inf_select] = inf_fill
            if wgts is not None:
                wgts[inf_select] = 0.0

    else:
        for i, k in enumerate(data.keys()):
            if nan_fill is not None:
                # replace nan
                nan_select = np.isnan(data[k])
                data[k][nan_select] = nan_fill
                if wgts is not None:
                    wgts[k][nan_select] = 0.0

            if inf_fill is not None:
                # replace infs
                inf_select = np.isinf(data[k])
                data[k][inf_select] = inf_fill
                if wgts is not None:
                    wgts[k][inf_select] = 0.0


def flatten(l):
    """ flatten a nested list """
    return [item for sublist in l for item in sublist]


class Baseline(object):
    """
    Baseline object for making antenna-independent, unique baseline labels
    for baselines up to 1km in length to an absolute precison of 10 cm.
    Only __eq__ operator is overloaded.
    """

    def __init__(self, bl, tol=2.0):
        """
        bl : list containing [dx, dy, dz] float separation in meters
        tol : tolerance for baseline length comparison in meters
        """
        self.label = "{:06.1f}:{:06.1f}:{:06.1f}".format(float(bl[0]), float(bl[1]), float(bl[2]))
        self.bl = np.array(bl, dtype=np.float)
        self.tol = tol

    def __repr__(self):
        return self.label

    @property
    def unit(self):
        return self.bl / np.linalg.norm(self.bl)

    @property
    def len(self):
        return np.linalg.norm(self.bl)

    def __eq__(self, B2):
        tol = np.max([self.tol, B2.tol])
        # check same length
        if np.isclose(self.len, B2.len, atol=tol):
            # check x, y, z
            equiv = bool(reduce(operator.mul, list(map(lambda x: np.isclose(*x, atol=tol), zip(self.bl, B2.bl)))))
            dot = np.dot(self.unit, B2.unit)
            if equiv:
                return True
            # check conjugation
            elif np.isclose(np.arccos(dot), np.pi, atol=tol / self.len) or (dot < -1.0):
                return 'conjugated'
            # else return False
            else:
                return False
        else:
            return False


def match_red_baselines(model, model_antpos, data, data_antpos, tol=1.0, verbose=True):
    """
    Match unique model baseline keys to unique data baseline keys based on positional redundancy.

    Ideally, both model and data contain only unique baselines, in which case there is a
    one-to-one mapping. If model contains extra redundant baselines, these are not propagated
    to new_model. If data contains extra redundant baselines, the lowest ant1-ant2 pair is chosen
    as the baseline key to insert into model.

    Parameters:
    -----------
    model : type=DataContainer, model dictionary holding complex visibilities
            must conform to DataContainer dictionary format.

    model_antpos : type=dictionary, dictionary holding antennas positions for model dictionary
            keys are antenna integers, values are ndarrays of position vectors in meters

    data : type=DataContainer, data dictionary holding complex visibilities.
            must conform to DataContainer dictionary format.

    data_antpos : type=dictionary, dictionary holding antennas positions for data dictionary
                same format as model_antpos

    tol : type=float, baseline match tolerance in units of baseline vectors (e.g. meters)

    Output: (data)
    -------
    new_model : type=DataContainer, dictionary holding complex visibilities from model that
        had matching baselines to data
    """
    # create baseline keys for model
    model_keys = list(model.keys())
    model_bls = np.array(list(map(lambda k: Baseline(model_antpos[k[1]] - model_antpos[k[0]], tol=tol), model_keys)))

    # create baseline keys for data
    data_keys = list(data.keys())
    data_bls = np.array(list(map(lambda k: Baseline(data_antpos[k[1]] - data_antpos[k[0]], tol=tol), data_keys)))

    # iterate over data baselines
    new_model = odict()
    for i, bl in enumerate(model_bls):
        # compre bl to all model_bls
        comparison = np.array(list(map(lambda mbl: bl == mbl, data_bls)), np.str)

        # get matches
        matches = np.where((comparison == 'True') | (comparison == 'conjugated'))[0]

        # check for matches
        if len(matches) == 0:
            echo("found zero matches in data for model {}".format(model_keys[i]), verbose=verbose)
            continue
        else:
            if len(matches) > 1:
                echo("found more than 1 match in data to model {}: {}".format(model_keys[i], list(map(lambda j: data_keys[j], matches))), verbose=verbose)
            # assign to new_data
            if comparison[matches[0]] == 'True':
                new_model[data_keys[matches[0]]] = model[model_keys[i]]
            elif comparison[matches[0]] == 'conjugated':
                new_model[data_keys[matches[0]]] = np.conj(model[model_keys[i]])

    return DataContainer(new_model)


def avg_data_across_red_bls(data, antpos, wgts=None, broadcast_wgts=True, tol=1.0,
                            mirror_red_data=False, reds=None):
    """
    Given complex visibility data spanning one or more redundant
    baseline groups, average redundant visibilities and return

    Parameters:
    -----------
    data : type=DataContainer, data dictionary holding complex visibilities.
        must conform to AbsCal dictionary format.

    antpos : type=dictionary, antenna position dictionary

    wgts : type=DataContainer, data weights as float

    broadcast_wgts : type=boolean, if True, take geometric mean of input weights as output weights,
        else use mean. If True, this has the effect of broadcasting a single flag from any particular
        baseline to all baselines in a baseline group.

    tol : type=float, redundant baseline tolerance threshold

    mirror_red_data : type=boolean, if True, mirror average visibility across red bls

    reds : list of list of redundant baselines with polarization strings.
           If None, reds is produced from antpos.

    Output: (red_data, red_wgts, red_keys)
    -------
    """
    # get data keys
    keys = list(data.keys())

    # get data, wgts and ants
    pols = np.unique(list(map(lambda k: k[2], data.keys())))
    ants = np.unique(np.concatenate(keys))
    if wgts is None:
        wgts = DataContainer(odict(list(map(lambda k: (k, np.ones_like(data[k]).astype(np.float)), data.keys()))))

    # get redundant baselines if not provided
    if reds is None:
        reds = redcal.get_reds(antpos, bl_error_tol=tol, pols=pols)

    # strip reds of keys not in data
    stripped_reds = []
    for i, bl_group in enumerate(reds):
        group = []
        for k in bl_group:
            if k in data:
                group.append(k)
        if len(group) > 0:
            stripped_reds.append(group)

    # make red_data dictionary
    red_data = odict()
    red_wgts = odict()

    # iterate over reds
    for i, bl_group in enumerate(stripped_reds):
        # average redundant baseline group
        d = np.nansum(list(map(lambda k: data[k] * wgts[k], bl_group)), axis=0)
        d /= np.nansum(list(map(lambda k: wgts[k], bl_group)), axis=0)

        # get wgts
        if broadcast_wgts:
            w = np.array(reduce(operator.mul, list(map(lambda k: wgts[k], bl_group))), np.float) ** (1. / len(bl_group))
        else:
            w = np.array(reduce(operator.add, list(map(lambda k: wgts[k], bl_group))), np.float) / len(bl_group)

        # iterate over bl_group
        for j, key in enumerate(sorted(bl_group)):
            # assign to red_data and wgts
            red_data[key] = d
            red_wgts[key] = w

            # break if no mirror
            if mirror_red_data is False:
                break

    # get red_data keys
    red_keys = list(red_data.keys())

    return DataContainer(red_data), DataContainer(red_wgts), red_keys


def mirror_data_to_red_bls(data, antpos, tol=2.0, weights=False):
    """
    Given unique baseline data (like omnical model visibilities),
    copy the data over to all other baselines in the same redundant group.
    If weights==True, treat data as a wgts dictionary and multiply values
    by their redundant baseline weighting.

    Parameters:
    -----------
    data : data DataContainer in hera_cal.DataContainer form

    antpos : type=dictionary, antenna positions dictionary
                keys are antenna integers, values are ndarray baseline vectors.

    tol : type=float, redundant baseline distance tolerance in units of baseline vectors

    weights : type=bool, if True, treat data as a wgts dictionary and multiply by redundant weighting.

    Output: (red_data)
    -------
    red_data : type=DataContainer, data dictionary in AbsCal form, with unique baseline data
                distributed to redundant baseline groups.
    if weights == True:
        red_data is a real-valued wgts dictionary with redundant baseline weighting muliplied in.
    """
    # get data keys
    keys = list(data.keys())

    # get polarizations in data
    pols = data.pols()

    # get redundant baselines
    reds = redcal.get_reds(antpos, bl_error_tol=tol, pols=pols)

    # make red_data dictionary
    red_data = odict()

    # iterate over data keys
    for i, k in enumerate(keys):

        # find which bl_group this key belongs to
        match = np.array(list(map(lambda r: k in r, reds)))
        conj_match = np.array(list(map(lambda r: reverse_bl(k) in r, reds)))

        # if no match, just copy data over to red_data
        if True not in match and True not in conj_match:
            red_data[k] = copy.copy(data[k])

        else:
            # iterate over matches
            for j, (m, cm) in enumerate(zip(match, conj_match)):
                if weights:
                    # if weight dictionary, add repeated baselines
                    if m:
                        if k not in red_data:
                            red_data[k] = copy.copy(data[k])
                            red_data[k][red_data[k].astype(np.bool)] = red_data[k][red_data[k].astype(np.bool)] + len(reds[j]) - 1
                        else:
                            red_data[k][red_data[k].astype(np.bool)] = red_data[k][red_data[k].astype(np.bool)] + len(reds[j])
                    elif cm:
                        if k not in red_data:
                            red_data[k] = copy.copy(data[k])
                            red_data[k][red_data[k].astype(np.bool)] = red_data[k][red_data[k].astype(np.bool)] + len(reds[j]) - 1
                        else:
                            red_data[k][red_data[k].astype(np.bool)] = red_data[k][red_data[k].astype(np.bool)] + len(reds[j])
                else:
                    # if match, insert all bls in bl_group into red_data
                    if m:
                        for bl in reds[j]:
                            red_data[bl] = copy.copy(data[k])
                    elif cm:
                        for bl in reds[j]:
                            red_data[bl] = np.conj(data[k])

    # re-sort, square if weights to match linsolve
    if weights:
        for i, k in enumerate(red_data):
            red_data[k][red_data[k].astype(np.bool)] = red_data[k][red_data[k].astype(np.bool)]**(2.0)
    else:
        red_data = odict([(k, red_data[k]) for k in sorted(red_data)])

    return DataContainer(red_data)


def match_times(datafile, modelfiles, filetype='uvh5', atol=1e-5):
    """
    Match start and end LST of datafile to modelfiles. Each file in modelfiles needs
    to have the same integration time.

    Parameters:
    -----------
    datafile : type=str, path to miriad data file
    modelfiles : type=str, list of paths to miriad model files ordered according to file start time
    filetype : str, options=['miriad', 'uvh5']

    Return: (matched_modelfiles)
    -------
    matched_modelfiles : type=list, list of modelfiles that overlap w/ datafile in LST
    """
    # get times
    data_time = np.array(io.get_file_lst_range(datafile, filetype=filetype))[:2]
    model_times = np.array(io.get_file_lst_range(modelfiles, filetype=filetype))
    model_inttime = model_times[2][0]
    model_times = model_times[:2]
    model_times[1] += model_inttime

    # unwrap LST
    if data_time[1] < data_time[0]:
        data_time[1] += 2 * np.pi
    model_start = model_times[0][0]
    model_times[model_times < model_start] += 2 * np.pi
    if data_time[0] < model_start:
        data_time += 2 * np.pi

    # select model files
    matched_modelfiles = np.array(modelfiles)[(model_times[0] < data_time[1] + atol)
                                              & (model_times[1] > data_time[0] - atol)]

    return matched_modelfiles
