# -*- coding: utf-8 -*-
# Copyright 2018 the HERA Project
# Licensed under the MIT License
"""
abscal.py
---------

Calibrate measured visibility
data to a visibility model using
linearizations of the (complex)
antenna-based calibration equation:

V_ij,xy^data = g_i_x * conj(g_j_y) * V_ij,xy^model.

Complex-valued parameters are broken into amplitudes and phases as:

V_ij,xy^model = exp(eta_ij,xy^model + i * phi_ij,xy^model)
g_i_x = exp(eta_i_x + i * phi_i_x)
g_j_y = exp(eta_j_y + i * phi_j_y)
V_ij,xy^data = exp(eta_ij,xy^data + i * phi_ij,xy^data)

where {i,j} index antennas and {x,y} are the polarization of
the i-th and j-th antenna respectively.
"""
from __future__ import print_function, division, absolute_import

import os
import sys
from collections import OrderedDict as odict
import copy
import argparse
import numpy as np
import operator
from functools import reduce
from six.moves import map, range, zip
from scipy import signal, interpolate, spatial
from scipy.optimize import brute, minimize
from pyuvdata import UVCal, UVData
import linsolve

from . import version
from .apply_cal import calibrate_in_place
from .smooth_cal import pick_reference_antenna, rephase_to_refant
from .flag_utils import synthesize_ant_flags
from .noise import predict_noise_variance_from_autos
from . import utils
from . import redcal
from . import io
from . import apply_cal
from .datacontainer import DataContainer
from .utils import echo, polnum2str, polstr2num, reverse_bl, split_pol, split_bl


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
    eqns = odict([(k, "a{}*eta_{}+a{}*eta_{}".format(i, split_pol(k[-1])[0],
                                                     i, split_pol(k[-1])[1])) for i, k in enumerate(keys)])
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
                       "".format(split_pol(k[2])[0], split_pol(k[2])[1], r_ew[k[0]],
                                 r_ns[k[0]], r_ew[k[1]], r_ns[k[1]])) for i, k in enumerate(keys)])
    else:
        eqns = odict([(k, "psi_{}*a1 - psi_{}*a2 + Phi_ew_{}*{} + Phi_ns_{}*{} - Phi_ew_{}*{} - Phi_ns_{}*{}"
                       "".format(split_pol(k[2])[0], split_pol(k[2])[1], split_pol(k[2])[0],
                                 r_ew[k[0]], split_pol(k[2])[0], r_ns[k[0]], split_pol(k[2])[1],
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
    eqns = odict([(k, "eta_{}_{} + eta_{}_{}".format(k[0], split_pol(k[-1])[0],
                                                     k[1], split_pol(k[-1])[1])) for i, k in enumerate(keys)])
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
    eqns = odict([(k, "phi_{}_{} - phi_{}_{}".format(k[0], split_pol(k[2])[0],
                                                     k[1], split_pol(k[2])[1])) for i, k in enumerate(keys)])
    ls_design_matrix = odict()

    # setup linsolve dictionaries
    ls_data = odict([(eqns[k], ydata[k]) for i, k in enumerate(keys)])
    ls_wgts = odict([(eqns[k], wgts[k]) for i, k in enumerate(keys)])

    # get unique gain polarizations
    gain_pols = np.unique(list(map(lambda k: list(split_pol(k[2])), keys)))

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


def delay_lincal(model, data, wgts=None, refant=None, df=9.765625e4, f0=0., solve_offsets=True, medfilt=True,
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

    f0 : type=float, frequency of the first channel in the data (used for offsets)

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
        dly, offset = utils.fft_dly(ratio, df, f0=f0, wgts=wgts[k], medfilt=medfilt, kernel=kernel, edge_cut=edge_cut)

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
    eqns = odict([(k, 'tau_{}_{} - tau_{}_{}'.format(k[0], split_pol(k[2])[0],
                                                     k[1], split_pol(k[2])[1])) for i, k in enumerate(keys)])

    # setup design matrix dictionary
    ls_design_matrix = odict()

    # setup linsolve data dictionary
    ls_data = odict([(eqns[k], ydata[k]) for i, k in enumerate(keys)])
    ls_wgts = odict([(eqns[k], ywgts[k]) for i, k in enumerate(keys)])

    # get unique gain polarizations
    gain_pols = np.unique(list(map(lambda k: [split_pol(k[2])[0], split_pol(k[2])[1]], keys)))

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
    eqns = odict([(k, 'phi_{}_{} - phi_{}_{}'.format(k[0], split_pol(k[2])[0],
                                                     k[1], split_pol(k[2])[1])) for i, k in enumerate(keys)])
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
        dly, _ = utils.fft_dly(ratio, df, wgts=wgts[k], medfilt=medfilt, kernel=kernel, edge_cut=edge_cut)

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
                       "".format(split_pol(k[2])[0], r_ew[k[0]], split_pol(k[2])[0], r_ns[k[0]],
                                 split_pol(k[2])[1], r_ew[k[1]], split_pol(k[2])[1], r_ns[k[1]]))
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


def dft_phase_slope_solver(xs, ys, data):
    '''Solve for sptial phase slopes across an array by looking for the peak in the DFT.
    This is analogous to the method in utils.fft_dly(), except its in 2D and does not 
    assume a regular grid for xs and ys.

    Arguments:
        xs: 1D array of x positions (e.g. of antennas or baselines)
        ys: 1D array of y positions (must be same length as xs)
        data: ndarray of complex numbers to fit with a phase slope. The first dimension must match 
            xs and ys, but subsequent dimensions will be preserved and solved independently. 
            Any np.nan in data is interpreted as a flag.

    Returns:
        slope_x, slope_y: phase slopes in units of 1/[xs] where the best fit phase slope plane
            is np.exp(2.0j * np.pi * (xs * slope_x + ys * slope_y)). Both have the same shape 
            the data after collapsing along the first dimension.
    '''

    # find the range of k values in DFT space to explore and the appropriate sampling
    min_len = np.min(np.sqrt(np.array(xs)**2 + np.array(ys)**2))
    nsamples = 2 * np.max(np.sqrt(np.array(xs)**2 + np.array(ys)**2)) / min_len
    search_slice = slice(-1.0 / (2 * min_len), 1.0 / (2 * min_len), 1.0 / (nsamples * min_len))

    # define cost function
    def dft_abs(k, x, y, z):
        return -np.abs(np.dot(z, np.exp(-2.0j * np.pi * (x * k[0] + y * k[1]))))

    # loop over data, minimizing the cost function
    dflat = data.reshape((len(xs), -1))
    slope_x = np.zeros_like(dflat[0, :].real)
    slope_y = np.zeros_like(dflat[0, :].real)
    for i in range(dflat.shape[1]):
        if not np.all(np.isnan(dflat[:, i])):
            dft_peak = brute(dft_abs, (search_slice, search_slice), 
                             (xs[~np.isnan(dflat[:, i])], ys[~np.isnan(dflat[:, i])], 
                              dflat[:, i][~np.isnan(dflat[:, i])]), finish=minimize)
            slope_x[i] = dft_peak[0]
            slope_y[i] = dft_peak[1]
    return slope_x.reshape(data.shape[1:]), slope_y.reshape(data.shape[1:])


def global_phase_slope_logcal(model, data, antpos, solver='linfit', wgts=None, 
                              refant=None, verbose=True, tol=1.0, edge_cut=0):
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

    solver : 'linfit' uses linsolve to fit phase slope across the array,
             'dft' uses a spatial Fourier transform to find a phase slope 

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
    # check solver and edgecut
    if solver == 'linfit':
        echo("...configuring linsolve data for global_phase_slope_logcal", verbose=verbose)
    elif solver == 'dft':
        echo("...finding global phase slopes using the DFT method", verbose=verbose)
    else:
        raise ValueError("Unrecognized solver {}. Must be either 'linfit' or 'dft'.".format(solver))
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

    ls_data, ls_wgts, bls, pols = {}, {}, {}, {}
    for rk in red_keys:
        # build equation string
        eqn_str = '{}*Phi_ew_{} + {}*Phi_ns_{} - {}*Phi_ew_{} - {}*Phi_ns_{}'
        eqn_str = eqn_str.format(antpos[rk[0]][0], split_pol(rk[2])[0], antpos[rk[0]][1], split_pol(rk[2])[0],
                                 antpos[rk[1]][0], split_pol(rk[2])[1], antpos[rk[1]][1], split_pol(rk[2])[1])
        bls[eqn_str] = antpos[rk[0]] - antpos[rk[1]]
        pols[eqn_str] = rk[2]

        # calculate median of unflagged angle(data/model)
        # ls_weights are sum of non-binary weights
        dm_ratio = avg_data[rk] / avg_model[rk]
        dm_ratio /= np.abs(dm_ratio)  # This gives all channels roughly equal weight, moderating the effect of RFI (as in firstcal)
        binary_flgs = np.isclose(avg_wgts[rk], 0.0)
        dm_ratio[binary_flgs] *= np.nan
        avg_wgts[rk][np.isinf(dm_ratio) + np.isnan(dm_ratio)] = 0.0
        dm_ratio[np.isinf(dm_ratio) + np.isnan(dm_ratio)] *= np.nan
        if solver == 'linfit':  # we want to fit the angles
            ls_data[eqn_str] = np.nanmedian(np.angle(dm_ratio[:, edge_cut:(dm_ratio.shape[1] - edge_cut)]), axis=1, keepdims=True)
        elif solver == 'dft':  # we want the full complex number
            ls_data[eqn_str] = np.nanmedian(dm_ratio[:, edge_cut:(dm_ratio.shape[1] - edge_cut)], axis=1, keepdims=True)
        ls_wgts[eqn_str] = np.sum(avg_wgts[rk][:, edge_cut:(dm_ratio.shape[1] - edge_cut)], axis=1, keepdims=True)

        # set unobserved data to 0 with 0 weight
        ls_wgts[eqn_str][np.isnan(ls_data[eqn_str])] = 0
        ls_data[eqn_str][np.isnan(ls_data[eqn_str])] = 0

    if solver == 'linfit':  # build linear system for phase slopes and solve with linsolve
        # setup linsolve and run
        solver = linsolve.LinearSolver(ls_data, wgts=ls_wgts)
        echo("...running linsolve", verbose=verbose)
        fit = solver.solve()
        echo("...finished linsolve", verbose=verbose)

    elif solver == 'dft':  # look for a peak angle space by 2D DFTing across baselines
        if not np.all([split_pol(pol)[0] == split_pol(pol)[1] for pol in data.pols()]):
            raise NotImplementedError('DFT solving of global phase not implemented for abscal with cross-polarizations.')
        fit = {}
        for pol in data.pols():
            keys = [k for k in bls.keys() if pols[k] == pol]
            blx = np.array([bls[k][0] for k in keys])
            bly = np.array([bls[k][1] for k in keys])
            data_array = np.array([ls_data[k] / (ls_wgts[k] > 0) for k in keys])  # is np.nan if all flagged
            with np.errstate(divide='ignore'):  # is np.nan if all flagged
                data_array = np.array([ls_data[k] / (ls_wgts[k] > 0) for k in keys])
            slope_x, slope_y = dft_phase_slope_solver(blx, bly, data_array)
            fit['Phi_ew_{}'.format(split_pol(pol)[0])] = slope_x * 2.0 * np.pi  # 2pi matches custom_phs_slope_gain
            fit['Phi_ns_{}'.format(split_pol(pol)[0])] = slope_y * 2.0 * np.pi

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
    data_lsts[data_lsts < data_lsts[0]] += 2 * np.pi

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


def cut_bls(datacontainer, bls=None, min_bl_cut=None, max_bl_cut=None, inplace=False):
    """
    Cut visibility data based on min and max baseline length.

    Parameters
    ----------
    datacontainer : DataContainer object to perform baseline cut on

    bls : dictionary, holding baseline position vectors.
        keys are antenna-pair tuples and values are baseline vectors in meters.
        If bls is None, will look for antpos attr in datacontainer.

    min_bl_cut : float, minimum baseline separation [meters] to keep in data

    max_bl_cut : float, maximum baseline separation [meters] to keep in data

    inplace : bool, if True edit data in input object, else make a copy.

    Output
    ------
    datacontainer : DataContainer object with bl cut enacted
    """
    if not inplace:
        datacontainer = copy.deepcopy(datacontainer)
    if min_bl_cut is None:
        min_bl_cut = 0.0
    if max_bl_cut is None:
        max_bl_cut = 1e10
    if bls is None:
        # look for antpos in dc
        if not hasattr(datacontainer, 'antpos'):
            raise ValueError("If bls is not fed, datacontainer must have antpos attribute.")
        bls = odict()
        ap = datacontainer.antpos
        for bl in datacontainer.keys():
            if bl[0] not in ap or bl[1] not in ap:
                continue
            bls[bl] = ap[bl[1]] - ap[bl[0]]
    for k in list(datacontainer.keys()):
        bl_len = np.linalg.norm(bls[k])
        if k not in bls:
            continue
        if bl_len > max_bl_cut or bl_len < min_bl_cut:
            del datacontainer[k]

    assert len(datacontainer) > 0, "no baselines were kept after baseline cut..."

    return datacontainer


class AbsCal(object):
    """
    AbsCal object used to for phasing and scaling visibility data to an absolute reference model.
    A few different calibration methods exist. These include:

    1) per-antenna amplitude logarithmic calibration solves the equation:
            ln[abs(V_ij^data / V_ij^model)] = eta_i + eta_j

    2) per-antenna phase logarithmic calibration solves the equation:
           angle(V_ij^data / V_ij^model) = phi_i - phi_j

    3) delay linear calibration solves the equation:
           delay(V_ij^data / V_ij^model) = delay(g_i) - delay(g_j)
                                         = tau_i - tau_j
       where tau is the delay that can be turned
       into a complex gain via: g = exp(i * 2pi * tau * freqs).

    4) delay slope linear calibration solves the equation:
            delay(V_ij^data / V_ij^model) = dot(T_dly, B_ij)
        where T_dly is a delay slope in [ns / meter]
        and B_ij is the baseline vector between ant i and j.

    5) frequency-independent phase slope calibration
        median_over_freq(angle(V_ij^data / V_ij^model)) = dot(Phi, B_ji)
        where Phi is a phase slope in [radians / meter]
        and B_ij is the baseline vector between ant i and j.

    6) Average amplitude linear calibration solves the equation:
            log|V_ij^data / V_ij^model| = log|g_avg_i| + log|g_avg_j|

    7) Tip-Tilt phase logarithmic calibration solves the equation
            angle(V_ij^data /  V_ij^model) = psi + dot(TT_Phi, B_ij)
        where psi is an overall gain phase scalar,
        TT_Phi is the gain phase slope vector [radians / meter]
        and B_ij is the baseline vector between antenna i and j.

    Methods (1), (2) and (3) can be thought of as general bandpass solvers, whereas
    methods (4), (5), (6), and (7) are methods that would be used for data that has already
    been redundantly calibrated.

    Be warned that the linearizations of the phase solvers suffer from phase wrapping
    pathologies, meaning that a delay calibration should generally precede a
    phs_logcal or a TT_phs_logcal bandpass routine.
    """
    def __init__(self, model, data, refant=None, wgts=None, antpos=None, freqs=None,
                 min_bl_cut=None, max_bl_cut=None, bl_taper_fwhm=None, verbose=True,
                 filetype='miriad', input_cal=None):
        """
        AbsCal object used to for phasing and scaling visibility data to an absolute reference model.

        The format of model, data and wgts is in a dictionary format, with the convention that
        keys contain antennas-pairs + polarization, Ex. (1, 2, 'xx'), and values contain 2D complex
        ndarrays with [0] axis indexing time and [1] axis frequency.

        Parameters:
        -----------
        model : Visibility data of refence model, type=dictionary or DataContainer
                keys are antenna-pair + polarization tuples, Ex. (1, 2, 'xx').
                values are complex ndarray visibilities.
                these must be 2D arrays, with [0] axis indexing time
                and [1] axis indexing frequency.

                Optionally, model can be a path to a pyuvdata-supported file, a
                pyuvdata.UVData object or hera_cal.HERAData object,
                or a list of either.

        data :  Visibility data, type=dictionary or DataContainer
                keys are antenna-pair + polarization tuples, Ex. (1, 2, 'xx').
                values are complex ndarray visibilities.
                these must be 2D arrays, with [0] axis indexing time
                and [1] axis indexing frequency.

                Optionally, data can be a path to a pyuvdata-supported file, a
                pyuvdata.UVData object or hera_cal.HERAData object,
                or a list of either. In this case, antpos, freqs
                and wgts are overwritten from arrays in data.

        refant : antenna number integer for reference antenna
            The refence antenna is used in the phase solvers, where an absolute phase is applied to all
            antennas such that the refant's phase is set to identically zero.

        wgts : weights of the data, type=dictionary or DataContainer, [default=None]
               keys are antenna pair + pol tuples (must match model), values are real floats
               matching shape of model and data

        antpos : type=dictionary, dict of antenna position vectors in ENU (topo) frame in meters.
                 origin of coordinates does not matter, but preferably are centered in the array.
                 keys are antenna integers and values are ndarray position vectors,
                 containing [East, North, Up] coordinates.
                 Can be generated from a pyuvdata.UVData instance via
                 ----
                 #!/usr/bin/env python
                 uvd = pyuvdata.UVData()
                 uvd.read_miriad(<filename>)
                 antenna_pos, ants = uvd.get_ENU_antpos()
                 antpos = dict(zip(ants, antenna_pos))
                 ----
                 This is needed only for Tip Tilt, phase slope, and delay slope calibration.

        freqs : ndarray of frequency array, type=ndarray
                1d array containing visibility frequencies in Hz.
                Needed for delay calibration.

        min_bl_cut : float, eliminate all visibilities with baseline separation lengths
            smaller than min_bl_cut. This is assumed to be in ENU coordinates with units of meters.

        max_bl_cut : float, eliminate all visibilities with baseline separation lengths
            larger than max_bl_cut. This is assumed to be in ENU coordinates with units of meters.

        bl_taper_fwhm : float, impose a gaussian taper on the data weights as a function of
            bl separation length, with a specified fwhm [meters]

        filetype : str, if data and/or model are fed as strings, this is their filetype

        input_cal : filepath to calfits, UVCal or HERACal object with gain solutions to
            apply to data on-the-fly via hera_cal.apply_cal.calibrate_in_place
        """
        # set pols to None
        pols = None

        # load model if necessary
        if isinstance(model, list) or isinstance(model, np.ndarray) or isinstance(model, str) or issubclass(model.__class__, UVData):
            (model, model_flags, model_antpos, model_ants, model_freqs, model_lsts,
             model_times, model_pols) = io.load_vis(model, pop_autos=True, return_meta=True, filetype=filetype)

        # load data if necessary
        if isinstance(data, list) or isinstance(data, np.ndarray) or isinstance(data, str) or issubclass(data.__class__, UVData):
            (data, flags, data_antpos, data_ants, data_freqs, data_lsts,
             data_times, data_pols) = io.load_vis(data, pop_autos=True, return_meta=True, filetype=filetype)
            pols = data_pols
            freqs = data_freqs
            antpos = data_antpos

        # apply calibration
        if input_cal is not None:
            if 'flags' not in locals():
                flags = None
            uvc = io.to_HERACal(input_cal)
            gains, cal_flags, quals, totquals = uvc.read()
            apply_cal.calibrate_in_place(data, gains, data_flags=flags, cal_flags=cal_flags, gain_convention=uvc.gain_convention)

        # get shared keys
        self.keys = sorted(set(model.keys()) & set(data.keys()))
        assert len(self.keys) > 0, "no shared keys exist between model and data"

        # append attributes
        self.model = DataContainer(dict([(k, model[k]) for k in self.keys]))
        self.data = DataContainer(dict([(k, data[k]) for k in self.keys]))

        # setup frequencies
        self.freqs = freqs
        if self.freqs is None:
            self.Nfreqs = None
        else:
            self.Nfreqs = len(self.freqs)

        # get pols is not defined, if so, make sure they are string format
        if pols is None:
            pols = np.unique(list(map(lambda k: k[2], self.keys)))
        elif isinstance(pols, np.ndarray) or isinstance(pols, list):
            if np.issubdtype(type(pols[0]), int):
                pols = list(map(lambda p: polnum2str(p), pols))

        # convert to integer format
        self.pols = pols
        self.pols = list(map(lambda p: polstr2num(p), self.pols))
        self.Npols = len(self.pols)

        # save pols in string format and get gain_pols
        self.polstrings = np.array(list(map(lambda p: polnum2str(p), self.pols)))
        self.gain_pols = np.unique(list(map(lambda p: list(split_pol(p)), self.polstrings)))
        self.Ngain_pols = len(self.gain_pols)

        # setup weights
        if wgts is None:
            # use data flags if present
            if 'flags' in locals() and flags is not None:
                wgts = DataContainer(dict([(k, (~flags[k]).astype(np.float)) for k in self.keys]))
            else:
                wgts = DataContainer(dict([(k, np.ones_like(data[k], dtype=np.float)) for k in self.keys]))
            if 'model_flags' in locals():
                for k in self.keys:
                    wgts[k] *= (~model_flags[k]).astype(np.float)
        self.wgts = wgts

        # setup ants
        self.ants = np.unique(np.concatenate(list(map(lambda k: k[:2], self.keys))))
        self.Nants = len(self.ants)
        if refant is None:
            refant = self.keys[0][0]
            print("using {} for reference antenna".format(refant))
        else:
            assert refant in self.ants, "refant {} not found in self.ants".format(refant)
        self.refant = refant

        # setup antenna positions
        self._overwrite_antpos(antpos)

        # setup gain solution keys
        self._gain_keys = list(map(lambda p: list(map(lambda a: (a, p), self.ants)), self.gain_pols))

        # perform baseline cut
        if min_bl_cut is not None or max_bl_cut is not None:
            assert self.antpos is not None, "can't request a bl_cut if antpos is not fed"

            _model = cut_bls(self.model, self.bls, min_bl_cut, max_bl_cut)
            _data = cut_bls(self.data, self.bls, min_bl_cut, max_bl_cut)
            _wgts = cut_bls(self.wgts, self.bls, min_bl_cut, max_bl_cut)

            # re-init
            self.__init__(_model, _data, refant=self.refant, wgts=_wgts, antpos=self.antpos, freqs=self.freqs, verbose=verbose)

        # enact a baseline weighting taper
        if bl_taper_fwhm is not None:
            assert self.antpos is not None, "can't request a baseline taper if antpos is not fed"

            # make gaussian taper func
            def taper(ratio):
                return np.exp(-0.5 * ratio**2)

            # iterate over baselines
            for k in self.wgts.keys():
                self.wgts[k] *= taper(np.linalg.norm(self.bls[k]) / bl_taper_fwhm)

    def _overwrite_antpos(self, antpos):
        '''Helper function for replacing self.antpos, self.bls, and self.antpos_arr without affecting tapering or baseline cuts.
        Useful for replacing true antenna positions with idealized ones derived from the redundancies.'''
        self.antpos = antpos
        self.antpos_arr = None
        self.bls = None
        if self.antpos is not None:
            # center antpos about reference antenna
            self.antpos = odict(list(map(lambda k: (k, antpos[k] - antpos[self.refant]), self.ants)))
            self.bls = odict([(x, self.antpos[x[0]] - self.antpos[x[1]]) for x in self.keys])
            self.antpos_arr = np.array(list(map(lambda x: self.antpos[x], self.ants)))
            self.antpos_arr -= np.median(self.antpos_arr, axis=0)

    def amp_logcal(self, verbose=True):
        """
        Call abscal_funcs.amp_logcal() method. see its docstring for more details.

        Parameters:
        -----------
        verbose : type=boolean, if True print feedback to stdout

        Result:
        -------
        per-antenna amplitude and per-antenna amp gains
        can be accessed via the getter functions
            self.ant_eta
            self.ant_eta_arr
            self.ant_eta_gain
            self.ant_eta_gain_arr
        """
        # set data quantities
        model = self.model
        data = self.data
        wgts = copy.copy(self.wgts)

        # run linsolve
        fit = amp_logcal(model, data, wgts=wgts, verbose=verbose)

        # form result array
        self._ant_eta = odict(list(map(lambda k: (k, copy.copy(fit["eta_{}_{}".format(k[0], k[1])])), flatten(self._gain_keys))))
        self._ant_eta_arr = np.moveaxis(list(map(lambda pk: list(map(lambda k: self._ant_eta[k], pk)), self._gain_keys)), 0, -1)

    def phs_logcal(self, avg=False, verbose=True):
        """
        call abscal_funcs.phs_logcal() method. see its docstring for more details.

        Parameters:
        -----------
        avg : type=boolean, if True, average solution across time and frequency

        verbose : type=boolean, if True print feedback to stdout

        Result:
        -------
        per-antenna phase and per-antenna phase gains
        can be accessed via the methods
            self.ant_phi
            self.ant_phi_arr
            self.ant_phi_gain
            self.ant_phi_gain_arr
        """
        # assign data
        model = self.model
        data = self.data
        wgts = copy.deepcopy(self.wgts)

        # run linsolve
        fit = phs_logcal(model, data, wgts=wgts, refant=self.refant, verbose=verbose)

        # form result array
        self._ant_phi = odict(list(map(lambda k: (k, copy.copy(fit["phi_{}_{}".format(k[0], k[1])])), flatten(self._gain_keys))))
        self._ant_phi_arr = np.moveaxis(list(map(lambda pk: list(map(lambda k: self._ant_phi[k], pk)), self._gain_keys)), 0, -1)

        # take time and freq average
        if avg:
            self._ant_phi = odict(list(map(lambda k: (k, np.ones_like(self._ant_phi[k])
                                                      * np.angle(np.median(np.real(np.exp(1j * self._ant_phi[k])))
                                                                 + 1j * np.median(np.imag(np.exp(1j * self._ant_phi[k]))))), flatten(self._gain_keys))))
            self._ant_phi_arr = np.moveaxis(list(map(lambda pk: list(map(lambda k: self._ant_phi[k], pk)), self._gain_keys)), 0, -1)

    def delay_lincal(self, medfilt=True, kernel=(1, 11), verbose=True, time_avg=False, edge_cut=0):
        """
        Solve for per-antenna delay according to the equation
        by calling abscal_funcs.delay_lincal method.
        See abscal_funcs.delay_lincal for details.

        Parameters:
        -----------
        medfilt : boolean, if True median filter data before fft

        kernel : size of median filter across (time, freq) axes, type=(int, int)

        time_avg : boolean, if True, average resultant antenna delays across time

        edge_cut : int, number of channels to exclude at each band edge in FFT window

        Result:
        -------
        per-antenna delays, per-antenna delay gains, per-antenna phase + phase gains
        can be accessed via the methods
            self.ant_dly
            self.ant_dly_gain
            self.ant_dly_arr
            self.ant_dly_gain_arr
            self.ant_dly_phi
            self.ant_dly_phi_gain
            self.ant_dly_phi_arr
            self.ant_dly_phi_gain_arr
        """
        # check for freq data
        if self.freqs is None:
            raise AttributeError("cannot delay_lincal without self.freqs array")

        # assign data
        model = self.model
        data = self.data
        wgts = self.wgts

        # get freq channel width
        df = np.median(np.diff(self.freqs))

        # run delay_lincal
        fit = delay_lincal(model, data, wgts=wgts, refant=self.refant, medfilt=medfilt, df=df, 
                           f0=self.freqs[0], kernel=kernel, verbose=verbose, edge_cut=edge_cut)

        # time average
        if time_avg:
            k = flatten(self._gain_keys)[0]
            Ntimes = fit["tau_{}_{}".format(k[0], k[1])].shape[0]
            for i, k in enumerate(flatten(self._gain_keys)):
                tau_key = "tau_{}_{}".format(k[0], k[1])
                tau_avg = np.moveaxis(np.median(fit[tau_key], axis=0)[np.newaxis], 0, 0)
                fit[tau_key] = np.repeat(tau_avg, Ntimes, axis=0)
                phi_key = "phi_{}_{}".format(k[0], k[1])
                gain = np.exp(1j * fit[phi_key])
                real_avg = np.median(np.real(gain), axis=0)
                imag_avg = np.median(np.imag(gain), axis=0)
                phi_avg = np.moveaxis(np.angle(real_avg + 1j * imag_avg)[np.newaxis], 0, 0)
                fit[phi_key] = np.repeat(phi_avg, Ntimes, axis=0)

        # form result
        self._ant_dly = odict(list(map(lambda k: (k, copy.copy(fit["tau_{}_{}".format(k[0], k[1])])), flatten(self._gain_keys))))
        self._ant_dly_arr = np.moveaxis(list(map(lambda pk: list(map(lambda k: self._ant_dly[k], pk)), self._gain_keys)), 0, -1)

        self._ant_dly_phi = odict(list(map(lambda k: (k, copy.copy(fit["phi_{}_{}".format(k[0], k[1])])), flatten(self._gain_keys))))
        self._ant_dly_phi_arr = np.moveaxis(list(map(lambda pk: list(map(lambda k: self._ant_dly_phi[k], pk)), self._gain_keys)), 0, -1)

    def delay_slope_lincal(self, medfilt=True, kernel=(1, 15), verbose=True, time_avg=False,
                           four_pol=False, edge_cut=0):
        """
        Solve for an array-wide delay slope (a subset of the omnical degeneracies) by calling
        abscal_funcs.delay_slope_lincal method. See abscal_funcs.delay_slope_lincal for details.

        Parameters:
        -----------
        medfilt : boolean, if True median filter data before fft

        kernel : size of median filter across (time, freq) axes, type=(int, int)

        verbose : type=boolean, if True print feedback to stdout

        time_avg : boolean, if True, average resultant delay slope across time

        four_pol : boolean, if True, form a joint polarization solution

        edge_cut : int, number of channels to exclude at each band edge in FFT window

        Result:
        -------
        delays slopes, per-antenna delay gains, per-antenna phase + phase gains
        can be accessed via the methods
            self.dly_slope
            self.dly_slope_gain
            self.dly_slope_arr
            self.dly_slope_gain_arr
        """
        # check for freq data
        if self.freqs is None:
            raise AttributeError("cannot delay_slope_lincal without self.freqs array")

        # assign data
        model = self.model
        data = self.data
        wgts = self.wgts
        antpos = self.antpos

        # get freq channel width
        df = np.median(np.diff(self.freqs))

        # run delay_slope_lincal
        fit = delay_slope_lincal(model, data, antpos, wgts=wgts, refant=self.refant, medfilt=medfilt, df=df,
                                 kernel=kernel, verbose=verbose, four_pol=four_pol, edge_cut=edge_cut)

        # separate pols if four_pol
        if four_pol:
            for i, gp in enumerate(self.gain_pols):
                fit['T_ew_{}'.format(gp)] = fit["T_ew"]
                fit['T_ns_{}'.format(gp)] = fit["T_ns"]
                fit.pop('T_ew')
                fit.pop('T_ns')

        # time average
        if time_avg:
            k = flatten(self._gain_keys)[0]
            Ntimes = fit["T_ew_{}".format(k[1])].shape[0]
            for i, k in enumerate(flatten(self._gain_keys)):
                ew_key = "T_ew_{}".format(k[1])
                ns_key = "T_ns_{}".format(k[1])
                ew_avg = np.moveaxis(np.median(fit[ew_key], axis=0)[np.newaxis], 0, 0)
                ns_avg = np.moveaxis(np.median(fit[ns_key], axis=0)[np.newaxis], 0, 0)
                fit[ew_key] = np.repeat(ew_avg, Ntimes, axis=0)
                fit[ns_key] = np.repeat(ns_avg, Ntimes, axis=0)

        # form result
        self._dly_slope = odict(list(map(lambda k: (k, copy.copy(np.array([fit["T_ew_{}".format(k[1])], fit["T_ns_{}".format(k[1])]]))), flatten(self._gain_keys))))
        self._dly_slope_arr = np.moveaxis(list(map(lambda pk: list(map(lambda k: np.array([self._dly_slope[k][0], self._dly_slope[k][1]]), pk)), self._gain_keys)), 0, -1)

    def global_phase_slope_logcal(self, solver='linfit', tol=1.0, edge_cut=0, verbose=True):
        """
        Solve for a frequency-independent spatial phase slope (a subset of the omnical degeneracies) by calling
        abscal_funcs.global_phase_slope_logcal method. See abscal_funcs.global_phase_slope_logcal for details.

        Parameters:
        -----------
        solver : 'linfit' uses linsolve to fit phase slope across the array,
                 'dft' uses a spatial Fourier transform to find a phase slope 

        tol : type=float, baseline match tolerance in units of baseline vectors (e.g. meters)

        edge_cut : int, number of channels to exclude at each band edge in phase slope solver

        verbose : type=boolean, if True print feedback to stdout

        Result:
        -------
        per-antenna delays, per-antenna delay gains, per-antenna phase + phase gains
        can be accessed via the methods
            self.phs_slope
            self.phs_slope_gain
            self.phs_slope_arr
            self.phs_slope_gain_arr
        """

        # assign data
        model = self.model
        data = self.data
        wgts = self.wgts
        antpos = self.antpos

        # run global_phase_slope_logcal
        fit = global_phase_slope_logcal(model, data, antpos, solver=solver, wgts=wgts,
                                        refant=self.refant, verbose=verbose, tol=tol, edge_cut=edge_cut)

        # form result
        self._phs_slope = odict(list(map(lambda k: (k, copy.copy(np.array([fit["Phi_ew_{}".format(k[1])], fit["Phi_ns_{}".format(k[1])]]))), flatten(self._gain_keys))))
        self._phs_slope_arr = np.moveaxis(list(map(lambda pk: list(map(lambda k: np.array([self._phs_slope[k][0], self._phs_slope[k][1]]), pk)), self._gain_keys)), 0, -1)

    def abs_amp_logcal(self, verbose=True):
        """
        call abscal_funcs.abs_amp_logcal() method. see its docstring for more details.

        Parameters:
        -----------
        verbose : type=boolean, if True print feedback to stdout

        Result:
        -------
        Absolute amplitude scalar can be accessed via methods
            self.abs_eta
            self.abs_eta_gain
            self.abs_eta_arr
            self.abs_eta_gain_arr
        """
        # set data quantities
        model = self.model
        data = self.data
        wgts = self.wgts

        # run abs_amp_logcal
        fit = abs_amp_logcal(model, data, wgts=wgts, verbose=verbose)

        # form result
        self._abs_eta = odict(list(map(lambda k: (k, copy.copy(fit["eta_{}".format(k[1])])), flatten(self._gain_keys))))
        self._abs_eta_arr = np.moveaxis(list(map(lambda pk: list(map(lambda k: self._abs_eta[k], pk)), self._gain_keys)), 0, -1)

    def TT_phs_logcal(self, verbose=True, zero_psi=True, four_pol=False):
        """
        call abscal_funcs.TT_phs_logcal() method. see its docstring for more details.

        Parameters:
        -----------
        zero_psi : type=boolean, set overall gain phase (psi) to identically zero in linsolve equations.
            This is separate than the reference antenna's absolute phase being set to zero, as it can account
            for absolute phase offsets between polarizations.

        four_pol : type=boolean, even if multiple polarizations are present in data, make free
                    variables polarization un-aware: i.e. one solution across all polarizations.
                    This is the same assumption as 4-polarization calibration in omnical.

        verbose : type=boolean, if True print feedback to stdout

        Result:
        -------
        Tip-Tilt phase slope and overall phase fit can be accessed via methods
            self.abs_psi
            self.abs_psi_gain
            self.TT_Phi
            self.TT_Phi_gain
            self.abs_psi_arr
            self.abs_psi_gain_arr
            self.TT_Phi_arr
            self.TT_Phi_gain_arr
        """
        # set data quantities
        model = self.model
        data = self.data
        wgts = self.wgts
        antpos = self.antpos

        # run TT_phs_logcal
        fit = TT_phs_logcal(model, data, antpos, wgts=wgts, refant=self.refant, verbose=verbose, zero_psi=zero_psi, four_pol=four_pol)

        # manipulate if four_pol
        if four_pol:
            for i, gp in enumerate(self.gain_pols):
                fit['Phi_ew_{}'.format(gp)] = fit["Phi_ew"]
                fit['Phi_ns_{}'.format(gp)] = fit["Phi_ns"]
                fit.pop('Phi_ew')
                fit.pop('Phi_ns')

        # form result
        self._abs_psi = odict(list(map(lambda k: (k, copy.copy(fit["psi_{}".format(k[1])])), flatten(self._gain_keys))))
        self._abs_psi_arr = np.moveaxis(list(map(lambda pk: list(map(lambda k: self._abs_psi[k], pk)), self._gain_keys)), 0, -1)

        self._TT_Phi = odict(list(map(lambda k: (k, copy.copy(np.array([fit["Phi_ew_{}".format(k[1])], fit["Phi_ns_{}".format(k[1])]]))), flatten(self._gain_keys))))
        self._TT_Phi_arr = np.moveaxis(list(map(lambda pk: list(map(lambda k: np.array([self._TT_Phi[k][0], self._TT_Phi[k][1]]), pk)), self._gain_keys)), 0, -1)

    # amp_logcal results
    @property
    def ant_eta(self):
        """ return _ant_eta dict, containing per-antenna amplitude solution """
        if hasattr(self, '_ant_eta'):
            return copy.deepcopy(self._ant_eta)
        else:
            return None

    @property
    def ant_eta_gain(self):
        """ form complex gain from _ant_eta dict """
        if hasattr(self, '_ant_eta'):
            ant_eta = self.ant_eta
            return odict(list(map(lambda k: (k, np.exp(ant_eta[k]).astype(np.complex)), flatten(self._gain_keys))))
        else:
            return None

    @property
    def ant_eta_arr(self):
        """ return _ant_eta in ndarray format """
        if hasattr(self, '_ant_eta_arr'):
            return copy.copy(self._ant_eta_arr)
        else:
            return None

    @property
    def ant_eta_gain_arr(self):
        """ return _ant_eta_gain in ndarray format """
        if hasattr(self, '_ant_eta_arr'):
            return np.exp(self.ant_eta_arr).astype(np.complex)
        else:
            return None

    # phs_logcal results
    @property
    def ant_phi(self):
        """ return _ant_phi dict, containing per-antenna phase solution """
        if hasattr(self, '_ant_phi'):
            return copy.deepcopy(self._ant_phi)
        else:
            return None

    @property
    def ant_phi_gain(self):
        """ form complex gain from _ant_phi dict """
        if hasattr(self, '_ant_phi'):
            ant_phi = self.ant_phi
            return odict(list(map(lambda k: (k, np.exp(1j * ant_phi[k])), flatten(self._gain_keys))))
        else:
            return None

    @property
    def ant_phi_arr(self):
        """ return _ant_phi in ndarray format """
        if hasattr(self, '_ant_phi_arr'):
            return copy.copy(self._ant_phi_arr)
        else:
            return None

    @property
    def ant_phi_gain_arr(self):
        """ return _ant_phi_gain in ndarray format """
        if hasattr(self, '_ant_phi_arr'):
            return np.exp(1j * self.ant_phi_arr)
        else:
            return None

    # delay_lincal results
    @property
    def ant_dly(self):
        """ return _ant_dly dict, containing per-antenna delay solution """
        if hasattr(self, '_ant_dly'):
            return copy.deepcopy(self._ant_dly)
        else:
            return None

    @property
    def ant_dly_gain(self):
        """ form complex gain from _ant_dly dict """
        if hasattr(self, '_ant_dly'):
            ant_dly = self.ant_dly
            return odict(list(map(lambda k: (k, np.exp(2j * np.pi * self.freqs.reshape(1, -1) * ant_dly[k])), flatten(self._gain_keys))))
        else:
            return None

    @property
    def ant_dly_arr(self):
        """ return _ant_dly in ndarray format """
        if hasattr(self, '_ant_dly_arr'):
            return copy.copy(self._ant_dly_arr)
        else:
            return None

    @property
    def ant_dly_gain_arr(self):
        """ return ant_dly_gain in ndarray format """
        if hasattr(self, '_ant_dly_arr'):
            return np.exp(2j * np.pi * self.freqs.reshape(-1, 1) * self.ant_dly_arr)
        else:
            return None

    @property
    def ant_dly_phi(self):
        """ return _ant_dly_phi dict, containing a single phase solution per antenna """
        if hasattr(self, '_ant_dly_phi'):
            return copy.deepcopy(self._ant_dly_phi)
        else:
            return None

    @property
    def ant_dly_phi_gain(self):
        """ form complex gain from _ant_dly_phi dict """
        if hasattr(self, '_ant_dly_phi'):
            ant_dly_phi = self.ant_dly_phi
            return odict(list(map(lambda k: (k, np.exp(1j * np.repeat(ant_dly_phi[k], self.Nfreqs, 1))), flatten(self._gain_keys))))
        else:
            return None

    @property
    def ant_dly_phi_arr(self):
        """ return _ant_dly_phi in ndarray format """
        if hasattr(self, '_ant_dly_phi_arr'):
            return copy.copy(self._ant_dly_phi_arr)
        else:
            return None

    @property
    def ant_dly_phi_gain_arr(self):
        """ return _ant_dly_phi_gain in ndarray format """
        if hasattr(self, '_ant_dly_phi_arr'):
            return np.exp(1j * np.repeat(self.ant_dly_phi_arr, self.Nfreqs, 2))
        else:
            return None

    # delay_slope_lincal results
    @property
    def dly_slope(self):
        """ return _dly_slope dict, containing the delay slope across the array """
        if hasattr(self, '_dly_slope'):
            return copy.deepcopy(self._dly_slope)
        else:
            return None

    @property
    def dly_slope_gain(self):
        """ form a per-antenna complex gain from _dly_slope dict and the antpos dictionary attached to the class"""
        if hasattr(self, '_dly_slope'):
            # get dly_slope dictionary
            dly_slope = self.dly_slope
            # turn delay slope into per-antenna complex gains, while iterating over self._gain_keys
            return odict(list(map(lambda k: (k, np.exp(2j * np.pi * self.freqs.reshape(1, -1) * np.einsum("i...,i->...", dly_slope[k], self.antpos[k[0]][:2]))),
                                  flatten(self._gain_keys))))
        else:
            return None

    def custom_dly_slope_gain(self, gain_keys, antpos):
        """
        return dly_slope_gain with custom gain keys and antenna positions

        gain_keys : type=list, list of unique (ant, pol). Ex. [(0, 'x'), (1, 'x'), (0, 'y'), (1, 'y')]
        antpos : type=dictionary, contains antenna position vectors. keys are ant integer, values are ant position vectors
        """
        if hasattr(self, '_dly_slope'):
            # get dly slope dictionary
            dly_slope = self.dly_slope[self._gain_keys[0][0]]
            # turn delay slope into per-antenna complex gains, while iterating over gain_keys
            return odict(list(map(lambda k: (k, np.exp(2j * np.pi * self.freqs.reshape(1, -1) * np.einsum("i...,i->...", dly_slope, antpos[k[0]][:2]))),
                                  gain_keys)))
        else:
            return None

    @property
    def dly_slope_arr(self):
        """ return _dly_slope_arr array """
        if hasattr(self, '_dly_slope_arr'):
            return copy.copy(self._dly_slope_arr)
        else:
            return None

    @property
    def dly_slope_gain_arr(self):
        """ form complex gain from _dly_slope_arr array """
        if hasattr(self, '_dly_slope_arr'):
            return np.exp(2j * np.pi * self.freqs.reshape(-1, 1) * np.einsum("hi...,hi->h...", self._dly_slope_arr, self.antpos_arr[:, :2]))
        else:
            return None

    @property
    def dly_slope_ant_dly_arr(self):
        """ form antenna delays from _dly_slope_arr array """
        if hasattr(self, '_dly_slope_arr'):
            return np.einsum("hi...,hi->h...", self._dly_slope_arr, self.antpos_arr[:, :2])
        else:
            return None

    # global_phase_slope_logcal results
    @property
    def phs_slope(self):
        """ return _phs_slope dict, containing the frequency-indpendent phase slope across the array """
        if hasattr(self, '_phs_slope'):
            return copy.deepcopy(self._phs_slope)
        else:
            return None

    @property
    def phs_slope_gain(self):
        """ form a per-antenna complex gain from _phs_slope dict and the antpos dictionary attached to the class"""
        if hasattr(self, '_phs_slope'):
            # get phs_slope dictionary
            phs_slope = self.phs_slope
            # turn phs slope into per-antenna complex gains, while iterating over self._gain_keys
            return odict(list(map(lambda k: (k, np.exp(1.0j * np.ones_like(self.freqs).reshape(1, -1) * np.einsum("i...,i->...", phs_slope[k], self.antpos[k[0]][:2]))),
                                  flatten(self._gain_keys))))
        else:
            return None

    def custom_phs_slope_gain(self, gain_keys, antpos):
        """
        return phs_slope_gain with custom gain keys and antenna positions

        gain_keys : type=list, list of unique (ant, pol). Ex. [(0, 'x'), (1, 'x'), (0, 'y'), (1, 'y')]
        antpos : type=dictionary, contains antenna position vectors. keys are ant integer, values are ant position vectors
        """
        if hasattr(self, '_phs_slope'):
            # get phs slope dictionary
            phs_slope = self.phs_slope[self._gain_keys[0][0]]
            # turn phs slope into per-antenna complex gains, while iterating over gain_keys
            return odict(list(map(lambda k: (k, np.exp(1.0j * np.ones_like(self.freqs).reshape(1, -1) * np.einsum("i...,i->...", phs_slope, antpos[k[0]][:2]))),
                                  gain_keys)))
        else:
            return None

    @property
    def phs_slope_arr(self):
        """ return _phs_slope_arr array """
        if hasattr(self, '_phs_slope_arr'):
            return copy.copy(self._phs_slope_arr)
        else:
            return None

    @property
    def phs_slope_gain_arr(self):
        """ form complex gain from _phs_slope_arr array """
        if hasattr(self, '_phs_slope_arr'):
            return np.exp(1.0j * np.ones_like(self.freqs).reshape(-1, 1) * np.einsum("hi...,hi->h...", self._phs_slope_arr, self.antpos_arr[:, :2]))
        else:
            return None

    @property
    def phs_slope_ant_phs_arr(self):
        """ form antenna delays from _phs_slope_arr array """
        if hasattr(self, '_phs_slope_arr'):
            return np.einsum("hi...,hi->h...", self._phs_slope_arr, self.antpos_arr[:, :2])
        else:
            return None

    # abs_amp_logcal results
    @property
    def abs_eta(self):
        """return _abs_eta dict"""
        if hasattr(self, '_abs_eta'):
            return copy.deepcopy(self._abs_eta)
        else:
            return None

    @property
    def abs_eta_gain(self):
        """form complex gain from _abs_eta dict"""
        if hasattr(self, '_abs_eta'):
            abs_eta = self.abs_eta
            return odict(list(map(lambda k: (k, np.exp(abs_eta[k]).astype(np.complex)), flatten(self._gain_keys))))
        else:
            return None

    def custom_abs_eta_gain(self, gain_keys):
        """
        return abs_eta_gain with custom gain keys

        gain_keys : type=list, list of unique (ant, pol). Ex. [(0, 'x'), (1, 'x'), (0, 'y'), (1, 'y')]
        """
        if hasattr(self, '_abs_eta'):
            abs_eta = self.abs_eta[self._gain_keys[0][0]]
            return odict(list(map(lambda k: (k, np.exp(abs_eta).astype(np.complex)), gain_keys)))
        else:
            return None

    @property
    def abs_eta_arr(self):
        """return _abs_eta_arr array"""
        if hasattr(self, '_abs_eta_arr'):
            return copy.copy(self._abs_eta_arr)
        else:
            return None

    @property
    def abs_eta_gain_arr(self):
        """form complex gain from _abs_eta_arr array"""
        if hasattr(self, '_abs_eta_arr'):
            return np.exp(self._abs_eta_arr).astype(np.complex)
        else:
            return None

    # TT_phs_logcal results
    @property
    def abs_psi(self):
        """return _abs_psi dict"""
        if hasattr(self, '_abs_psi'):
            return copy.deepcopy(self._abs_psi)
        else:
            return None

    @property
    def abs_psi_gain(self):
        """ form complex gain from _abs_psi array """
        if hasattr(self, '_abs_psi'):
            abs_psi = self.abs_psi
            return odict(list(map(lambda k: (k, np.exp(1j * abs_psi[k])), flatten(self._gain_keys))))
        else:
            return None

    def custom_abs_psi_gain(self, gain_keys):
        """
        return abs_psi_gain with custom gain keys

        gain_keys : type=list, list of unique (ant, pol). Ex. [(0, 'x'), (1, 'x'), (0, 'y'), (1, 'y')]
        """
        if hasattr(self, '_abs_psi'):
            abs_psi = self.abs_psi[self._gain_keys[0][0]]
            return odict(list(map(lambda k: (k, np.exp(1j * abs_psi)), gain_keys)))
        else:
            return None

    @property
    def abs_psi_arr(self):
        """return _abs_psi_arr array"""
        if hasattr(self, '_abs_psi_arr'):
            return copy.copy(self._abs_psi_arr)
        else:
            return None

    @property
    def abs_psi_gain_arr(self):
        """ form complex gain from _abs_psi_arr array """
        if hasattr(self, '_abs_psi_arr'):
            return np.exp(1j * self._abs_psi_arr)
        else:
            return None

    @property
    def TT_Phi(self):
        """return _TT_Phi array"""
        if hasattr(self, '_TT_Phi'):
            return copy.deepcopy(self._TT_Phi)
        else:
            return None

    @property
    def TT_Phi_gain(self):
        """ form complex gain from _TT_Phi array """
        if hasattr(self, '_TT_Phi'):
            TT_Phi = self.TT_Phi
            return odict(list(map(lambda k: (k, np.exp(1j * np.einsum("i...,i->...", TT_Phi[k], self.antpos[k[0]][:2]))), flatten(self._gain_keys))))
        else:
            return None

    def custom_TT_Phi_gain(self, gain_keys, antpos):
        """
        return TT_Phi_gain with custom gain keys and antenna positions

        gain_keys : type=list, list of unique (ant, pol). Ex. [(0, 'x'), (1, 'x'), (0, 'y'), (1, 'y')]
        antpos : type=dictionary, contains antenna position vectors. keys are ant integer, values are ant positions
        """
        if hasattr(self, '_TT_Phi'):
            TT_Phi = self.TT_Phi[self._gain_keys[0][0]]
            return odict(list(map(lambda k: (k, np.exp(1j * np.einsum("i...,i->...", TT_Phi, antpos[k[0]][:2]))), gain_keys)))
        else:
            return None

    @property
    def TT_Phi_arr(self):
        """return _TT_Phi_arr array"""
        if hasattr(self, '_TT_Phi_arr'):
            return copy.copy(self._TT_Phi_arr)
        else:
            return None

    @property
    def TT_Phi_gain_arr(self):
        """ form complex gain from _TT_Phi_arr array """
        if hasattr(self, '_TT_Phi_arr'):
            return np.exp(1j * np.einsum("hi...,hi->h...", self._TT_Phi_arr, self.antpos_arr[:, :2]))
        else:
            return None


def get_all_times_and_lsts(hd, solar_horizon=90.0, unwrap=True):
    '''Extract all times and lsts from a HERAData object

    Arguments:
        hd: HERAData object intialized with one ore more uvh5 file's metadata
        solar_horizon: Solar altitude threshold [degrees]. Times are not returned when the Sun is above this altitude.
        unwrap: increase all LSTs smaller than the first one by 2pi to avoid phase wrapping

    Returns:
        all_times: list of times in JD in the file or files
        all_lsts: LSTs (in radians) corresponding to all_times
    '''
    all_times = hd.times
    all_lsts = hd.lsts
    if len(hd.filepaths) > 1:  # in this case, it's a dictionary
        all_times = np.ravel([all_times[f] for f in hd.filepaths])
        all_lsts = np.ravel([all_lsts[f] for f in hd.filepaths])[np.argsort(all_times)]
    if unwrap:  # avoid phase wraps 
        all_lsts[all_lsts < all_lsts[0]] += 2 * np.pi
        
    # remove times when sun was too high
    if solar_horizon < 90.0:
        lat, lon, alt = hd.telescope_location_lat_lon_alt_degrees
        solar_alts = utils.get_sun_alt(all_times, latitude=lat, longitude=lon)
        solar_flagged = solar_alts > solar_horizon
        return all_times[~solar_flagged], all_lsts[~solar_flagged]
    else:  # skip this step for speed
        return all_times, all_lsts


def get_d2m_time_map(data_times, data_lsts, model_times, model_lsts, unwrap=True):
    '''Generate a dictionary that maps data times to model times via shared LSTs.

    Arguments:
        data_times: list of times in the data (in JD)
        data_lsts: list of corresponding LSTs (in radians)
        model_times: list of times in the mdoel (in JD)
        model_lsts: list of corresponing LSTs (in radians)
        unwrap: increase all LSTs smaller than the first one by 2pi to avoid phase wrapping

    Returns:
        d2m_time_map: dictionary uniqely mapping times in the data to times in the model 
            that are closest in LST. Each model time maps to at most one data time and 
            each model time maps to at most one data time. Data times without corresponding
            model times map to None.
    '''
    if unwrap:  # avoid phase wraps
        data_lsts[data_lsts < data_lsts[0]] += 2 * np.pi
        model_lsts[model_lsts < model_lsts[0]] += 2 * np.pi

    # first produce a map of indices using the LSTs
    m2d_ind_map = {}  
    for dind, dlst in enumerate(data_lsts):
        nearest_mind = np.argmin(np.abs(model_lsts - dlst))
        if nearest_mind in m2d_ind_map:
            if np.abs(model_lsts[nearest_mind] < data_lsts[m2d_ind_map[nearest_mind]]):
                m2d_ind_map[nearest_mind] = dind
        else:
            m2d_ind_map[nearest_mind] = dind

    # now use those indicies to produce a map of times
    d2m_time_map = {time: None for time in data_times}
    for mind, dind in m2d_ind_map.items():
        d2m_time_map[data_times[dind]] = model_times[mind]
    return d2m_time_map


def abscal_step(gains_to_update, AC, AC_func, AC_kwargs, gain_funcs, gain_args_list, gain_flags, 
                gain_convention='divide', max_iter=1, phs_conv_crit=1e-6, verbose=True):
    '''Generalized function for performing an abscal step (e.g. abs_amp_logcal or TT_phs_logcal).

    Arguments:
        gains_to_update: the gains produced by abscal up until this step. Updated in place.
        AC: AbsCal object containing data, model, and other metadata. AC.data is recalibrated 
            in place using the gains solved for during this step
        AC_func: function (usually a class method of AC) to call to instantiate the new gains 
            which are then accessible as class properties of AC
        AC_kwargs: dictionary of kwargs to pass into AC_func
        gain_funcs: list of functions to call to return gains after AC_func has been called
        gain_args_list: list of tuples of arguments to pass to the corresponding gain_funcs
        gain_flags: per-antenna flags to apply to AC.Data when performing recalibration
        gain_convention: either 'divide' if raw data is calibrated by dividing it by the gains
            otherwise, 'multiply'.
        max_iter: maximum number of times to run phase solvers iteratively to avoid the effect
            of phase wraps in, e.g. phase_slope_cal or TT_phs_logcal
        phs_conv_crit: convergence criterion for updates to iterative phase calibration that compares
            the updates to all 1.0s. 
        verbose: If True, will print the progress of iterative convergence
    '''
    for i in range(max_iter):
        AC_func(**AC_kwargs)
        gains_here = merge_gains([gf(*gargs) for gf, gargs in zip(gain_funcs, gain_args_list)])
        apply_cal.calibrate_in_place(AC.data, gains_here, AC.wgts, gain_flags, 
                                     gain_convention=gain_convention, flags_are_wgts=True)
        for k in gains_to_update.keys():
            gains_to_update[k] *= gains_here[k]
        if max_iter > 1:
            crit = np.max(np.linalg.norm([gains_here[k] - 1.0 for 
                                          k in gains_here.keys()], axis=(0, 1)))
            echo(AC_func.__name__ + " convergence criterion: " + str(crit), verbose=verbose)
            if crit < phs_conv_crit:
                break


def post_redcal_abscal(model, data, flags, rc_flags, min_bl_cut=None, max_bl_cut=None, edge_cut=0, tol=1.0, 
                       gain_convention='divide', phs_max_iter=100, phs_conv_crit=1e-6, refant_num=None, verbose=True):
    '''Performs Abscal for data that has already been redundantly calibrated.

    Arguments:
        model: DataContainer containing externally calibrated visibilities, LST-matched to the data
        data: DataContainer containing redundantly but not absolutely calibrated visibilities. This gets modified.
        flags: DataContainer containing combined data and model flags
        rc_flags: dictionary mapping keys like (1, 'Jxx') to flag waterfalls from redundant calibration
        min_bl_cut : float, eliminate all visibilities with baseline separation lengths
            smaller than min_bl_cut. This is assumed to be in ENU coordinates with units of meters.
        max_bl_cut : float, eliminate all visibilities with baseline separation lengths
            larger than max_bl_cut. This is assumed to be in ENU coordinates with units of meters.
        edge_cut : integer number of channels to exclude at each band edge in delay and global phase solvers
        tol: float distance for baseline match tolerance in units of baseline vectors (e.g. meters)
        gain_convention: either 'divide' if raw data is calibrated by dividing it by the gains
            otherwise, 'multiply'.
        phs_max_iter: maximum number of iterations of phase_slope_cal or TT_phs_cal allowed
        phs_conv_crit: convergence criterion for updates to iterative phase calibration that compares
            the updates to all 1.0s.
        refant_num: integer antenna number defined to have 0 phase. If None, refant will be automatically chosen.

    Returns:
        abscal_delta_gains: gain dictionary mapping keys like (1, 'Jxx') to waterfalls containing 
            the updates to the gains between redcal and abscal
        AC: AbsCal object containing absolutely calibrated data, model, and other useful metadata
    '''
    abscal_delta_gains = {ant: np.ones_like(g, dtype=complex) for ant, g in rc_flags.items()}

    # instantiate Abscal object
    if refant_num is None:
        refant_num = pick_reference_antenna(abscal_delta_gains, synthesize_ant_flags(flags), data.freqs, per_pol=False)[0]
    wgts = DataContainer({k: (~flags[k]).astype(np.float) for k in flags.keys()})
    AC = AbsCal(model, data, wgts=wgts, antpos=data.antpos, freqs=data.freqs,
                refant=refant_num, min_bl_cut=min_bl_cut, max_bl_cut=max_bl_cut)
    
    # use idealized antpos derived from the reds that results in perfect redundancy, then use tol ~ 0 subsequently
    idealized_antpos = redcal.reds_to_antpos(redcal.get_reds(data.antpos, bl_error_tol=tol))
    AC._overwrite_antpos(idealized_antpos)

    # Per-Channel Absolute Amplitude Calibration
    abscal_step(abscal_delta_gains, AC, AC.abs_amp_logcal, {'verbose': verbose}, [AC.custom_abs_eta_gain], 
                [(rc_flags.keys(),)], rc_flags, gain_convention=gain_convention, verbose=verbose)

    # Global Delay Slope Calibration
    for time_avg in [True, False]:
        abscal_step(abscal_delta_gains, AC, AC.delay_slope_lincal, {'time_avg': time_avg, 'edge_cut': edge_cut, 'verbose': verbose},
                    [AC.custom_dly_slope_gain], [(rc_flags.keys(), idealized_antpos)], rc_flags,
                    gain_convention=gain_convention, verbose=verbose)

    # Global Phase Slope Calibration (first using dft, then using linfit)
    abscal_step(abscal_delta_gains, AC, AC.global_phase_slope_logcal, {'solver': 'dft', 'tol': 1e-8,
                'edge_cut': edge_cut, 'verbose': verbose}, [AC.custom_phs_slope_gain], [(rc_flags.keys(), idealized_antpos)], 
                rc_flags, gain_convention=gain_convention, verbose=verbose)
    abscal_step(abscal_delta_gains, AC, AC.global_phase_slope_logcal, {'tol': 1e-8, 'edge_cut': edge_cut, 'verbose': verbose},
                [AC.custom_phs_slope_gain], [(rc_flags.keys(), idealized_antpos)], rc_flags,
                gain_convention=gain_convention, max_iter=phs_max_iter, phs_conv_crit=phs_conv_crit, verbose=verbose)

    # Per-Channel Tip-Tilt Phase Calibration
    abscal_step(abscal_delta_gains, AC, AC.TT_phs_logcal, {'verbose': verbose}, [AC.custom_TT_Phi_gain, AC.custom_abs_psi_gain], 
                [(rc_flags.keys(), idealized_antpos), (rc_flags.keys(),)], rc_flags,
                gain_convention=gain_convention, max_iter=phs_max_iter, phs_conv_crit=phs_conv_crit, verbose=verbose)

    return abscal_delta_gains, AC


def post_redcal_abscal_run(data_file, redcal_file, model_files, output_file=None, nInt_to_load=None,
                           data_solar_horizon=90, model_solar_horizon=90, min_bl_cut=1.0, max_bl_cut=None, edge_cut=0, 
                           tol=1.0, phs_max_iter=100, phs_conv_crit=1e-6, refant=None, clobber=True, add_to_history='', verbose=True):
    '''Perform abscal on entire data files, picking relevant model_files from a list and doing partial data loading.
    Does not work on data (or models) with baseline-dependant averaging.
    
    Arguments:
        data_file: string path to raw uvh5 visibility file
        redcal_file: string path to calfits file that redundantly calibrates the data_file
        model_files: list of string paths to externally calibrated data. Strings must be sortable 
            to produce a chronological list in LST (wrapping over 2*pi is OK)
        output_file: string path to output abscal calfits file. If None, will be redcal_file.replace('.omni.', '.abs.')
        nInt_to_load: number of integrations to load and calibrate simultaneously. Default None loads all integrations.
        data_solar_horizon: Solar altitude threshold [degrees]. When the sun is too high in the data, flag the integration.
        model_solar_horizon: Solar altitude threshold [degrees]. When the sun is too high in the model, flag the integration.
        min_bl_cut: minimum baseline separation [meters] to keep in data when calibrating. None or 0 means no mininum,
            which will include autocorrelations in the absolute calibration. Usually this is not desired, so the default is 1.0.
        max_bl_cut: maximum baseline separation [meters] to keep in data when calibrating. None (default) means no maximum.
        edge_cut: integer number of channels to exclude at each band edge in delay and global phase solvers
        tol: baseline match tolerance in units of baseline vectors (e.g. meters)
        phs_max_iter: integer maximum number of iterations of phase_slope_cal or TT_phs_cal allowed
        phs_conv_crit: convergence criterion for updates to iterative phase calibration that compares them to all 1.0s.
        refant: tuple of the form (0, 'Jxx') indicating the antenna defined to have 0 phase. If None, refant will be automatically chosen.
        clobber: if True, overwrites existing abscal calfits file at the output path
        add_to_history: string to add to history of output abscal file

    Returns:
        hc: HERACal object which was written to disk. Matches the input redcal_file with an updated history.
            This HERACal object has been updated with the following properties accessible on hc.build_calcontainers():
                * gains: abscal gains for times that could be calibrated, redcal gains otherwise (but flagged)
                * flags: redcal flags, with additional flagging if the data or model are flagged (see flag_utils.synthesize_ant_flags)
                * quals: abscal chi^2 per antenna based on calibrated data minus model (Normalized by noise/nObs, but not with proper DoF)
                * total_qual: abscal chi^2 based on calibrated data minus model (Normalized by noise/nObs, but not with proper DoF)
    '''
    # Raise error if output calfile already exists and clobber is False
    if output_file is None:
        output_file = redcal_file.replace('.omni.', '.abs.')
    if os.path.exists(output_file) and not clobber:
        raise IOError("{} exists, not overwriting.".format(output_file))

    # Load redcal calibration
    hc = io.HERACal(redcal_file)
    rc_gains, rc_flags, rc_quals, rc_tot_qual = hc.read()

    # Initialize full-size, totally-flagged abscal gain/flag/etc. dictionaries
    abscal_gains = copy.deepcopy(rc_gains)
    abscal_flags = {ant: np.ones_like(rf) for ant, rf in rc_flags.items()}
    abscal_chisq_per_ant = {ant: np.zeros_like(rq) for ant, rq in rc_quals.items()}
    abscal_chisq = {pol: np.zeros_like(rtq) for pol, rtq in rc_tot_qual.items()}

    # match times to narrow down model_files
    matched_model_files = sorted(set(match_times(data_file, model_files, filetype='uvh5')))
    if len(matched_model_files) > 0:
        hd = io.HERAData(data_file)
        hdm = io.HERAData(matched_model_files)
        pol_load_list = [pol for pol in hd.pols if split_pol(pol)[0] == split_pol(pol)[1]]
        
        # match integrations in model to integrations in data
        all_data_times, all_data_lsts = get_all_times_and_lsts(hd, solar_horizon=data_solar_horizon, unwrap=True)
        all_model_times, all_model_lsts = get_all_times_and_lsts(hdm, solar_horizon=model_solar_horizon, unwrap=True)
        d2m_time_map = get_d2m_time_map(all_data_times, all_data_lsts, all_model_times, all_model_lsts)
        
        # group matched time indices for partial I/O
        matched_tinds = [tind for tind, time in enumerate(hd.times) if time in d2m_time_map and d2m_time_map[time] is not None]
        if len(matched_tinds) > 0:
            tind_groups = np.array([matched_tinds])  # just load a single group
            if nInt_to_load is not None:  # split up the integrations to load nInt_to_load at a time
                tind_groups = np.split(matched_tinds, np.arange(nInt_to_load, len(matched_tinds), nInt_to_load))

            # loop over polarizations
            for pol in pol_load_list:
                echo('\n\nNow calibrating ' + pol + '-polarization...', verbose=verbose)
                # loop over groups of time indices
                for tinds in tind_groups:
                    echo('\n    Now calibrating times ' + str(hd.times[tinds[0]])
                         + ' through ' + str(hd.times[tinds[-1]]) + '...', verbose=verbose)
                    
                    # load data and apply calibration
                    data, flags, nsamples = hd.read(times=hd.times[tinds], polarizations=[pol])
                    data_ants = set([ant for bl in data.keys() for ant in split_bl(bl)])
                    rc_gains_subset = {k: rc_gains[k][tinds, :] for k in data_ants}
                    rc_flags_subset = {k: rc_flags[k][tinds, :] for k in data_ants}
                    calibrate_in_place(data, rc_gains_subset, data_flags=flags, 
                                       cal_flags=rc_flags_subset, gain_convention=hc.gain_convention)
                    auto_bls = [bl for bl in hd.bls if (bl[0] == bl[1]) and bl[2] == pol]
                    autocorrs = DataContainer({bl: copy.deepcopy(data[bl]) for bl in auto_bls})

                    if not np.all(flags.values()):
                        # load model and rephase
                        model_times_to_load = [d2m_time_map[time] for time in hd.times[tinds]]
                        model, model_flags, _ = io.partial_time_io(hdm, model_times_to_load, polarizations=[pol])
                        model_bls = {bl: model.antpos[bl[0]] - model.antpos[bl[1]] for bl in model.keys()}
                        utils.lst_rephase(model, model_bls, model.freqs, data.lsts - model.lsts,
                                          lat=hdm.telescope_location_lat_lon_alt_degrees[0], inplace=True)
                        
                        # update data flags w/ model flags
                        for k in flags.keys():
                            if k in model_flags:
                                flags[k] += model_flags[k]

                        # run absolute calibration, copying data because it gets modified internally
                        delta_gains, AC = post_redcal_abscal(model, data, flags, rc_flags_subset, edge_cut=edge_cut, 
                                                             tol=tol, min_bl_cut=min_bl_cut, max_bl_cut=max_bl_cut, 
                                                             gain_convention=hc.gain_convention, phs_max_iter=phs_max_iter, 
                                                             phs_conv_crit=phs_conv_crit, verbose=verbose,
                                                             refant_num=(None if refant is None else refant[0]))

                        # calibrate autos, abscal them, and generate abscal Chi^2
                        calibrate_in_place(autocorrs, delta_gains, data_flags=flags, 
                                           cal_flags=rc_flags_subset, gain_convention=hc.gain_convention)
                        chisq_wgts = {}
                        for bl in AC.data.keys():
                            dt = (np.median(np.ediff1d(hd.times_by_bl[bl[:2]])) * 86400.)
                            noise_var = predict_noise_variance_from_autos(bl, autocorrs, dt=dt, df=np.median(np.ediff1d(data.freqs)))
                            chisq_wgts[bl] = noise_var**-1 * (~flags[bl]).astype(np.float)
                        total_qual, nObs, quals, nObs_per_ant = utils.chisq(AC.data, AC.model, chisq_wgts,
                                                                            gain_flags=rc_flags_subset, split_by_antpol=True)
                    
                        # update results
                        delta_flags = synthesize_ant_flags(flags)
                        for ant in data_ants:
                            abscal_gains[ant][tinds, :] = rc_gains_subset[ant] * delta_gains[ant]
                            abscal_flags[ant][tinds, :] = rc_flags_subset[ant] + delta_flags[ant]
                            abscal_chisq_per_ant[ant][tinds, :] = quals[ant] / nObs_per_ant[ant]  # Note, not normalized for DoF
                        for antpol in total_qual.keys():
                            abscal_chisq[antpol][tinds, :] = total_qual[antpol] / nObs[antpol]  # Note, not normalized for DoF
                            
        # impose a single reference antenna on the final antenna solution
        if refant is None:
            refant = pick_reference_antenna(abscal_gains, abscal_flags, hc.freqs, per_pol=True)
        rephase_to_refant(abscal_gains, refant, flags=abscal_flags)
    else:
        echo("No model files overlap with data files in LST. Result will be fully flagged.", verbose=verbose)

    # Save results to disk
    hc.update(gains=abscal_gains, flags=abscal_flags, quals=abscal_chisq_per_ant, total_qual=abscal_chisq)
    hc.quality_array[np.isnan(hc.quality_array)] = 0
    hc.total_quality_array[np.isnan(hc.total_quality_array)] = 0
    hc.history += version.history_string(add_to_history)
    hc.write_calfits(output_file, clobber=clobber)
    return hc


def post_redcal_abscal_argparser():
    ''' Argparser for commandline operation of hera_cal.abscal.post_redcal_abscal_run() '''
    a = argparse.ArgumentParser(description="Command-line drive script for post-redcal absolute calibration using hera_cal.abscal module")
    a.add_argument("data_file", type=str, help="string path to raw uvh5 visibility file")
    a.add_argument("redcal_file", type=str, help="string path to calfits file that redundantly calibrates the data_file")
    a.add_argument("model_files", type=str, nargs='+', help="list of string paths to externally calibrated data. Strings must be sortable to produce a chronological list in LST \
                                                             (wrapping over 2*pi is OK)")
    a.add_argument("--output_file", default=None, type=str, help="string path to output abscal calfits file. If None, will be redcal_file.replace('.omni.', '.abs.'")
    a.add_argument("--nInt_to_load", default=None, type=int, help="number of integrations to load and calibrate simultaneously. Default None loads all integrations.")
    a.add_argument("--data_solar_horizon", default=90.0, type=float, help="Solar altitude threshold [degrees]. When the sun is too high in the data, flag the integration.")
    a.add_argument("--model_solar_horizon", default=90.0, type=float, help="Solar altitude threshold [degrees]. When the sun is too high in the model, flag the integration.")
    a.add_argument("--min_bl_cut", default=1.0, type=float, help="minimum baseline separation [meters] to keep in data when calibrating. None or 0 means no mininum, which will \
                                                                  include autocorrelations in the absolute calibration. Usually this is not desired, so the default is 1.0.")
    a.add_argument("--max_bl_cut", default=None, type=float, help="maximum baseline separation [meters] to keep in data when calibrating. None (default) means no maximum.")
    a.add_argument("--edge_cut", default=0, type=int, help="integer number of channels to exclude at each band edge in delay and global phase solvers")
    a.add_argument("--tol", default=1.0, type=float, help="baseline match tolerance in units of baseline vectors (e.g. meters)")
    a.add_argument("--phs_max_iter", default=100, type=int, help="integer maximum number of iterations of phase_slope_cal or TT_phs_cal allowed")
    a.add_argument("--phs_conv_crit", default=1e-6, type=float, help="convergence criterion for updates to iterative phase calibration that compares them to all 1.0s.")
    a.add_argument("--clobber", default=False, action="store_true", help="overwrites existing abscal calfits file at the output path")
    a.add_argument("--verbose", default=False, action="store_true", help="print calibration progress updates")
    args = a.parse_args()
    return args
