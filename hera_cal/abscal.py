# -*- coding: utf-8 -*-
# Copyright 2020 the HERA Project
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
import os
from collections import OrderedDict as odict
import copy
import argparse
import numpy as np
import operator
from functools import reduce
from scipy import signal, interpolate, spatial, constants
from scipy.optimize import brute, minimize
from pyuvdata import UVCal, UVData
import linsolve
import warnings

from .apply_cal import calibrate_in_place
from .smooth_cal import pick_reference_antenna, rephase_to_refant
from .flag_utils import synthesize_ant_flags
from .noise import predict_noise_variance_from_autos
from . import utils
from . import redcal
from . import io
from . import apply_cal
from .datacontainer import DataContainer
from .utils import echo, polnum2str, polstr2num, reverse_bl, split_pol, split_bl, join_bl, join_pol

PHASE_SLOPE_SOLVERS = ['linfit', 'dft', 'ndim_fft']  # list of valid solvers for global_phase_slope_logcal


def abs_amp_logcal(model, data, wgts=None, verbose=True, return_gains=False, gain_ants=[]):
    """
    calculate absolute (array-wide) gain amplitude scalar
    with a linear solver using the logarithmically linearized equation:

    ln|V_ij,xy^data / V_ij,xy^model| = eta_x + eta_y

    where {i,j} index antenna numbers and {x,y} index polarizations
    of the i-th and j-th antennas respectively.

    Parameters:
    -----------
    model : visibility data of refence model, type=DataContainer
            keys are antenna-pair + polarization tuples, Ex. (1, 2, 'nn').
            values are complex ndarray visibilities.
            these must be 2D arrays, with [0] axis indexing time
            and [1] axis indexing frequency.

    data : visibility data of measurements, type=DataContainer
           keys are antenna pair + pol tuples (must match model), values are
           complex ndarray visibilities matching shape of model

    wgts : weights of data, type=DataContainer, [default=None]
           keys are antenna pair + pol tuples (must match model), values are real floats
           matching shape of model and data

    return_gains : boolean. If True, convert result into a dictionary of gain waterfalls.

    gain_ants : list of ant-pol tuples for return_gains dictionary

    verbose : print output, type=boolean, [default=False]

    Output:
    -------
    if not return_gains:
        fit : dictionary with 'eta_{}' key for amplitude scalar for {} polarization,
                which has the same shape as the ndarrays in the model
    else:
        gains: dictionary with gain_ants as keys and gain waterfall arrays as values
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
            wgts[k] = np.ones_like(ydata[k], dtype=float)

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

    if not return_gains:
        return fit
    else:
        return {ant: np.exp(fit['eta_{}'.format(ant[1])]).astype(complex) for ant in gain_ants}


def abs_amp_lincal(model, data, wgts=None, verbose=True, return_gains=False, gain_ants=[],
                   conv_crit=None, maxiter=100):
    """
    calculate absolute (array-wide) gain amplitude scalar
    with a linear (or linearized) solver using the equation:

    V_ij,xy^data = A_x A_y * V_ij,xy^model

    where {i,j} index antenna numbers and {x,y} index polarizations
    of the i-th and j-th antennas respectively. When no cross-polarized
    visibilities are involved, A^2 is solved for linearly for both real
    and imaginary parts simultaneously as separate equations. Otherwise,
    we have to use a linear-product solving algorithm, using abs_amp_logcal
    as a starting point.

    Parameters:
    -----------
    model : visibility data of refence model, type=DataContainer
            keys are antenna-pair + polarization tuples, Ex. (1, 2, 'nn').
            values are complex ndarray visibilities.
            these must be 2D arrays, with [0] axis indexing time
            and [1] axis indexing frequency.

    data : visibility data of measurements, type=DataContainer
           keys are antenna pair + pol tuples (must match model), values are
           complex ndarray visibilities matching shape of model

    wgts : weights of data, type=DataContainer, [default=None]
           keys are antenna pair + pol tuples (must match model), values are real floats
           matching shape of model and data

    return_gains : boolean. If True, convert result into a dictionary of gain waterfalls.

    gain_ants : list of ant-pol tuples for return_gains dictionary

    conv_crit : A convergence criterion below which to stop iterating LinProductSolver.
                Converegence is measured L2-norm of the change in the solution of the
                variables divided by the L2-norm of the solution itself.
                Default: None (resolves to machine precision for inferred dtype).
                Note: only used when data and model include cross-polarized visibilities.

    maxiter : Integer maximum number of iterations to perform LinProductSolver.
              Note: only used when data and model include cross-polarized visibilities.

    verbose : print output, type=boolean, [default=False]

    Output:
    -------
    if not return_gains:
        fit : dictionary with 'A_{}' key for amplitude scalar for {} polarization,
              which has the same shape as the ndarrays in the model
    else:
        gains: dictionary with gain_ants as keys and gain waterfall arrays as values
    """
    echo("...configuring linsolve data for abs_amp_lincal", verbose=verbose)

    # get keys from model and data dictionary
    keys = sorted(set(model.keys()) & set(data.keys()))

    # check to see whether any cross-polarizations are being used (this will require a different solver)
    cross_pols_used = False
    for k in keys:
        ant0, ant1 = split_bl(k)
        if ant0[1] != ant1[1]:
            cross_pols_used = True
            break

    # make weights if None
    if wgts is None:
        wgts = odict()
        for i, k in enumerate(keys):
            wgts[k] = np.ones_like(data[k], dtype=float)

    # fill nans and infs, minimally duplicating data to save memory
    data_here = {}
    model_here = {}
    for k in keys:
        if np.any(~np.isfinite(data[k])):
            data_here[k] = copy.deepcopy(data[k])
            fill_dict_nans(data_here[k], wgts=wgts[k], nan_fill=0.0, inf_fill=0.0, array=True)
        else:
            data_here[k] = data[k]
        if np.any(~np.isfinite(model[k])):
            model_here[k] = copy.deepcopy(model[k])
            fill_dict_nans(model_here[k], wgts=wgts[k], nan_fill=0.0, inf_fill=0.0, array=True)
        else:
            model_here[k] = model[k]

    # setup linsolve equations, either for A (if cross_pols_used) or A^2
    ls_data = {}
    ls_wgts = {}
    ls_consts = {}
    for i, k in enumerate(keys):
        pol0, pol1 = split_pol(k[-1])
        if cross_pols_used:
            re_eq_str = f'model_re_{i}*A_{pol0}*A_{pol1}'
            im_eq_str = f'model_im_{i}*A_{pol0}*A_{pol1}'
        else:
            re_eq_str = f'model_re_{i}*Asq_{pol0}'
            im_eq_str = f'model_im_{i}*Asq_{pol0}'

        ls_data[re_eq_str] = np.real(data_here[k])
        ls_wgts[re_eq_str] = wgts[k]
        ls_consts[f'model_re_{i}'] = np.real(model_here[k])

        ls_data[im_eq_str] = np.imag(data_here[k])
        ls_wgts[im_eq_str] = wgts[k]
        ls_consts[f'model_im_{i}'] = np.imag(model_here[k])

    # setup linsolve and run
    echo("...running linsolve", verbose=verbose)
    if cross_pols_used:
        # use abs_amp_logcal to get a starting point solution
        sol0 = abs_amp_logcal(model, data, wgts=wgts)
        sol0 = {k.replace('eta_', 'A_'): np.exp(sol) for k, sol in sol0.items()}
        # now solve by linearizing
        solver = linsolve.LinProductSolver(ls_data, sol0, wgts=ls_wgts, constants=ls_consts)
        meta, fit = solver.solve_iteratively(conv_crit=conv_crit, maxiter=maxiter)
    else:
        # in this case, the equations are already linear in A^2
        solver = linsolve.LinearSolver(ls_data, wgts=ls_wgts, constants=ls_consts)
        fit = solver.solve()
        fit = {k.replace('Asq', 'A'): np.sqrt(np.abs(sol)) for k, sol in fit.items()}
    echo("...finished linsolve", verbose=verbose)

    if not return_gains:
        return fit
    else:
        return {ant: np.abs(fit[f'A_{ant[1]}']).astype(complex) for ant in gain_ants}


def _count_nDims(antpos, assume_2D=True):
    '''Antenna position dimension counter helper function used in solvers that support higher-dim abscal.'''
    nDims = len(list(antpos.values())[0])
    for k in antpos.keys():
        assert len(antpos[k]) == nDims, 'Not all antenna positions have the same dimensionality.'
        if assume_2D:
            assert len(antpos[k]) >= 2, 'Since assume_2D is True, all antenna positions must 2D or higher.'
    return nDims


def TT_phs_logcal(model, data, antpos, wgts=None, refant=None, assume_2D=True,
                  zero_psi=True, four_pol=False, verbose=True, return_gains=False, gain_ants=[]):
    """
    calculate overall gain phase and gain phase Tip-Tilt slopes (East-West and North-South)
    with a linear solver applied to the logarithmically linearized equation:

    angle(V_ij,xy^data / V_ij,xy^model) = angle(g_i_x * conj(g_j_y))
                                        = psi_x - psi_y + Phi^ew_x*r_i^ew + Phi^ns_x*r_i^ns
                                          - Phi^ew_y*r_j^ew - Phi^ns_y*r_j^ns

    where psi is the overall gain phase across the array [radians] for x and y polarizations,
    and PHI^ew, PHI^ns are the gain phase slopes across the east-west and north-south axes
    of the array in units of [radians / meter], where x and y denote the pol of the i-th and j-th
    antenna respectively. The phase slopes are polarization independent by default (1pol & 2pol cal),
    but can be merged with the four_pol parameter (4pol cal). r_i is the antenna position vector
    of the i^th antenna.

    If assume_2D is not true, this solves for the tip-tilt degeneracies of antenna positions in an
    arbitary number of dimensions, the output of redcal.reds_to_antpos() for an array with extra
    tip-tilt degeneracies. In that case, the fit parameters are  Phi_0, Phi_1, Phi_2, etc.,
    generalizing the equation above to use the n-dimensional dot product Phi . r.

    Parameters:
    -----------
    model : visibility data of refence model, type=DataContainer
            keys are antenna-pair + polarization tuples, Ex. (1, 2, 'nn').
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
             keys are antenna integers, values are antenna positions vectors
             (preferably centered at center of array). If assume_2D is True, it is assumed that the
             [0] index contains the east-west separation and [1] index the north-south separation

    assume_2D : type=boolean, [default=False]
                If this is true, all dimensions of antpos beyond the first two will be ignored.
                If return_gains is False and assume_2D is False, then the returned variables will
                look like Phi_0, Phi_1, Phi_2, etc. corresponding to the dimensions in antpos.

    zero_psi : set psi to be identically zero in linsolve eqns, type=boolean, [default=False]

    four_pol : type=boolean, even if multiple polarizations are present in data, make free
               variables polarization un-aware: i.e. one solution across all polarizations.
               This is the same assumption as 4-polarization calibration in omnical.

    verbose : print output, type=boolean, [default=False]

    return_gains : boolean. If True, convert result into a dictionary of gain waterfalls.

    gain_ants : list of ant-pol tuples for return_gains dictionary

    Output:
    -------

    if not return_gains:
        fit : dictionary with psi key for overall gain phase and Phi_ew and Phi_ns array containing
                phase slopes across the EW and NS directions of the array. There is a set of each
                of these variables per polarization. If assume_2D is False, then these will be the
                more general Phi_0, Phi_1, Phi_2, etc. corresponding to the dimensions in antpos.
    else:
        gains: dictionary with gain_ants as keys and gain waterfall arrays as values

    """
    echo("...configuring linsolve data for TT_phs_logcal", verbose=verbose)

    # get keys from model dictionary
    keys = sorted(set(model.keys()) & set(data.keys()))
    antnums = np.unique(list(antpos.keys()))

    # angle of phs ratio is ydata independent variable
    # angle after divide
    ydata = {k: np.angle(data[k] / model[k]) for k in keys}

    # make unit weights if None
    if wgts is None:
        wgts = {k: np.ones_like(ydata[k], dtype=float) for k in keys}

    # fill nans and infs
    fill_dict_nans(ydata, wgts=wgts, nan_fill=0.0, inf_fill=0.0)

    # center antenna positions about the reference antenna
    if refant is None:
        refant = keys[0][0]
    assert refant in antnums, "reference antenna {} not found in antenna list".format(refant)
    antpos = {k: antpos[k] - antpos[refant] for k in antpos.keys()}

    # count dimensions of antenna positions, figure out how many to solve for
    nDims = _count_nDims(antpos, assume_2D=assume_2D)

    # setup linsolve equations
    eqns = {}
    for k in keys:
        ap0, ap1 = split_pol(k[2])
        eqns[k] = f'psi_{ap0}*a1 - psi_{ap1}*a2'
        for d in range((nDims, 2)[assume_2D]):
            if four_pol:
                eqns[k] += f' + Phi_{d}*r_{d}_{k[0]} - Phi_{d}*r_{d}_{k[1]}'
            else:
                eqns[k] += f' + Phi_{d}_{ap0}*r_{d}_{k[0]} - Phi_{d}_{ap1}*r_{d}_{k[1]}'

    # set design matrix entries
    ls_design_matrix = {}
    for a in antnums:
        for d in range((nDims, 2)[assume_2D]):
            ls_design_matrix[f'r_{d}_{a}'] = antpos[a][d]
    if zero_psi:
        ls_design_matrix.update({"a1": 0.0, "a2": 0.0})
    else:
        ls_design_matrix.update({"a1": 1.0, "a2": 1.0})

    # setup linsolve dictionaries
    ls_data = {eqns[k]: ydata[k] for k in keys}
    ls_wgts = {eqns[k]: wgts[k] for k in keys}

    # setup linsolve and run
    sol = linsolve.LinearSolver(ls_data, wgts=ls_wgts, **ls_design_matrix)
    echo("...running linsolve", verbose=verbose)
    fit = sol.solve()
    echo("...finished linsolve", verbose=verbose)

    if not return_gains:
        # rename variables ew/ns instead of 0/1 to maintain backwards compatability
        if assume_2D:
            params = list(fit.keys())
            for p in params:
                if 'Phi_0' in p:
                    fit[p.replace('Phi_0', 'Phi_ew')] = fit[p]
                    del fit[p]
                if 'Phi_1' in p:
                    fit[p.replace('Phi_1', 'Phi_ns')] = fit[p]
                    del fit[p]
        return fit
    else:
        # compute gains, dotting each parameter into the corresponding coordinate in that dimension
        gains = {}
        for ant in gain_ants:
            gains[ant] = np.exp(1.0j * fit['psi_{}'.format(ant[1])])
            if four_pol:
                Phis = [fit[f'Phi_{d}'] for d in range((nDims, 2)[assume_2D])]
            else:
                Phis = [fit[f'Phi_{d}_{ant[1]}'] for d in range((nDims, 2)[assume_2D])]
            gains[ant] *= np.exp(1.0j * (np.einsum('i,ijk->jk', antpos[ant[0]][0:len(Phis)], Phis)))
        return gains


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
            keys are antenna-pair + polarization tuples, Ex. (1, 2, 'nn').
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
            wgts[k] = np.ones_like(ydata[k], dtype=float)

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
            keys are antenna-pair + polarization tuples, Ex. (1, 2, 'nn').
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
            wgts[k] = np.ones_like(ydata[k], dtype=float)

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
    gain_pols = np.unique([split_pol(k[2]) for k in keys])

    # set reference antenna phase to zero
    if refant is None:
        refant = keys[0][0]
    assert any(refant in k for k in keys), f"refant {refant} not found in data and model"

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
            keys are antenna-pair + polarization tuples, Ex. (1, 2, 'nn').
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
            wgts[k] = np.ones_like(data[k], dtype=float)

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
    gain_pols = np.unique([list(split_pol(k[2])[:2]) for k in keys])

    # set reference antenna phase to zero
    if refant is None:
        refant = keys[0][0]
    assert any(refant in k for k in keys), f"refant {refant} not found in data and model"

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


def delay_slope_lincal(model, data, antpos, wgts=None, refant=None, df=9.765625e4, f0=0.0, medfilt=True,
                       kernel=(1, 5), assume_2D=True, four_pol=False, edge_cut=0, time_avg=False,
                       return_gains=False, gain_ants=[], verbose=True):
    """
    Solve for an array-wide delay slope according to the equation

    delay(V_ij,xy^data / V_ij,xy^model) = dot(T_x, r_i) - dot(T_y, r_j)

    This does not solve for per-antenna delays, but rather a delay slope across the array.

    Parameters:
    -----------
    model : visibility data of refence model, type=DataContainer
            keys are antenna-pair + polarization tuples, Ex. (1, 2, 'nn').
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

    f0 : type=float, frequency of 0th channel in Hz.
         Optional, but used to get gains without a delay offset.

    medfilt : type=boolean, median filter visiblity ratio before taking fft

    kernel : type=tuple, dtype=int, kernel for multi-dimensional median filter

    assume_2D : type=boolean, [default=False]
                If this is true, all dimensions of antpos beyond the first two will be ignored.
                If return_gains is False and assume_2D is False, then the returned variables will
                look like T_0, T_1, T_2, etc. corresponding to the dimensions in antpos.

    four_pol : type=boolean, if True, fit multiple polarizations together

    edge_cut : int, number of channels to exclude at each band edge of vis in FFT window

    time_avg : boolean, if True, replace resultant antenna delay slope with the median across time

    return_gains : boolean. If True, convert result into a dictionary of gain waterfalls.

    gain_ants : list of ant-pol tuples for return_gains dictionary

    Output:
    -------
    if not return_gains:
        fit : dictionary containing delay slope (T_x) for each pol [seconds / meter].
              If assume_2D is False, then these will be the more general T_0, T_1, T_2, etc.
              corresponding to the dimensions in antpos, instead of T_ew or T_ns.
    else:
        gains: dictionary with gain_ants as keys and gain waterfall arrays as values
    """
    echo("...configuring linsolve data for delay_slope_lincal", verbose=verbose)

    # get shared keys
    keys = sorted(set(model.keys()) & set(data.keys()))
    antnums = np.unique(list(antpos.keys()))

    # make unit wgts if None
    if wgts is None:
        wgts = {k: np.ones_like(data[k], dtype=float) for k in keys}

    # center antenna positions about the reference antenna
    if refant is None:
        refant = keys[0][0]
    assert refant in antnums, "reference antenna {} not found in antenna list".format(refant)
    antpos = {k: antpos[k] - antpos[refant] for k in antpos.keys()}

    # count dimensions of antenna positions, figure out how many to solve for
    nDims = _count_nDims(antpos, assume_2D=assume_2D)

    # median filter and FFT to get delays
    ydata = {}
    ywgts = {}
    for i, k in enumerate(keys):
        ratio = data[k] / model[k]
        ratio /= np.abs(ratio)

        # replace nans and infs
        wgts[k][~np.isfinite(ratio)] = 0.0
        ratio[~np.isfinite(ratio)] = 0.0

        # get delays
        ydata[k], _ = utils.fft_dly(ratio, df, wgts=wgts[k], f0=f0, medfilt=medfilt, kernel=kernel, edge_cut=edge_cut)

        # set nans to zero
        ywgts[k] = np.nanmean(wgts[k], axis=1, keepdims=True)
        isnan = np.isnan(ydata[k])
        ydata[k][isnan] = 0.0
        ywgts[k][isnan] = 0.0

    # setup antenna position terms
    r_ew = {a: f"r_ew_{a}" for a in antnums}
    r_ns = {a: f"r_ns_{a}" for a in antnums}

    # setup linsolve equations
    eqns = {k: '' for k in keys}
    for k in keys:
        ap0, ap1 = split_pol(k[2])
        for d in range((nDims, 2)[assume_2D]):
            if len(eqns[k]) > 0:
                eqns[k] += ' + '
            if four_pol:
                eqns[k] += f'T_{d}*r_{d}_{k[0]} - T_{d}*r_{d}_{k[1]}'
            else:
                eqns[k] += f'T_{d}_{ap0}*r_{d}_{k[0]} - T_{d}_{ap1}*r_{d}_{k[1]}'

    # set design matrix entries
    ls_design_matrix = {}
    for a in antnums:
        for d in range((nDims, 2)[assume_2D]):
            ls_design_matrix[f'r_{d}_{a}'] = antpos[a][d]

    # setup linsolve data dictionary
    ls_data = {eqns[k]: ydata[k] for k in keys}
    ls_wgts = {eqns[k]: ywgts[k] for k in keys}

    # setup linsolve and run
    sol = linsolve.LinearSolver(ls_data, wgts=ls_wgts, **ls_design_matrix)
    echo("...running linsolve", verbose=verbose)
    fit = sol.solve()
    echo("...finished linsolve", verbose=verbose)

    # time average
    if time_avg:
        Ntimes = list(fit.values())[0].shape[0]
        for k in fit:
            fit[k] = np.repeat(np.moveaxis(np.median(fit[k], axis=0)[np.newaxis], 0, 0), Ntimes, axis=0)

    if not return_gains:
        # rename variables ew/ns instead of 0/1 to maintain backwards compatability
        if assume_2D:
            params = list(fit.keys())
            for p in params:
                if 'T_0' in p:
                    fit[p.replace('T_0', 'T_ew')] = fit[p]
                    del fit[p]
                if 'T_1' in p:
                    fit[p.replace('T_1', 'T_ns')] = fit[p]
                    del fit[p]
        return fit
    else:
        gains = {}
        for ant in gain_ants:
            # construct delays from delay slopes
            if four_pol:
                Taus = [fit[f'T_{d}'] for d in range((nDims, 2)[assume_2D])]
            else:
                Taus = [fit[f'T_{d}_{ant[1]}'] for d in range((nDims, 2)[assume_2D])]
            delays = np.einsum('ijk,i->j', Taus, antpos[ant[0]][0:len(Taus)])

            # construct gains from freqs and delays
            freqs = f0 + np.arange(list(data.values())[0].shape[1]) * df
            gains[ant] = np.exp(2.0j * np.pi * np.outer(delays, freqs))
        return gains


def RFI_delay_slope_cal(reds, antpos, red_data, freqs, rfi_chans, rfi_headings, rfi_wgts=None,
                        min_tau=-500e-9, max_tau=500e-9, delta_tau=0.1e-9, return_gains=False, gain_ants=None):
    '''Finds a per-unique baseline delay relative to a set of RFI transmitters with known frequency and heading,
    and then fits that slope across the array for each degeneracy dimension. Namely, we fit a set of T_{pol}_{dim}
    such that:

    V_ij * e^(-2i π b_ij.rhat(ν) ν / c) = Σ_dims[T_{pol}_{dim}]

    where b is the baseline vector and rhat is transmitter unit vector.

    Arguments:
        reds: list of list of baseline-pol tuples, e.g. (0, 1, 'ee'), considered redundant
        antpos: dictionary mapping antenna index to length-3 vector of antenna position in meters in ENU coordinates
        red_data: DataContainer with redundantly averaged visibility solutions.
        freqs: array of frequencies in Hz with length equal to that of the second dimension of the data
        rfi_chans: length Nrfi list of channel indices with RFI with known heading
        rfi_headings: (3, Nrfi) numpy array of direciton unit vectors pointed toward stable transmitters.
        rfi_wgts: length Nrfi list of linear multiplicative weights representating the relative confidence
            in the delay expected in a particular channel
        min_tau: Smallest delay for brute-force search in s (default -500 ns)
        max_tau: Largest delay for brute-force search in s (default 500 ns)
        delta_tau: Brute force delay search resolution (default 0.1 ns)
        return_gains: Bool. If True, convert delay slope into gains. Otherwise, return delay slopes.
        gain_ants: If return_gains is True, these are the keys. Ignored otherwise.

    Returns:
        if return_gains:
            Returns a dictionary with gain_ants as keys mapping to complex gains, each the shape of the data.
        else:
            Returns a dictionary of delay slopes for each integration in the data, keyed by 'T_{pol}_{dim_index}'
            where dimensions are computed using abstracted antenna positions with redcal.reds_to_antpos(reds).
    '''
    # check that reds are 1pol or 2pol
    if redcal.parse_pol_mode(reds) not in ['1pol', '2pol']:
        raise NotImplementedError('RFI_delay_slope_cal cannot currently handle 4pol calibration.')

    # compute unique baseline vectors and idealized baseline vectors if desired
    unique_blvecs = {red[0]: np.mean([antpos[bl[1]] - antpos[bl[0]] for bl in red], axis=0) for red in reds}
    idealized_antpos = redcal.reds_to_antpos(reds)
    idealized_blvecs = {red[0]: idealized_antpos[red[0][1]] - idealized_antpos[red[0][0]] for red in reds}

    # brute-force find per-unique baseline delays that make the per-UBL visibilities, dotted into the rfi_phase, closest to real
    vis = np.array([red_data[red[0]][:, rfi_chans] for red in reds])
    vis /= np.abs(vis)
    rfi_phs = np.array([np.exp(-2j * np.pi * np.dot(unique_blvecs[red[0]], rfi_headings) * freqs[rfi_chans] / constants.c) for red in reds])
    dlys_to_check = np.arange(min_tau, max_tau, delta_tau)
    dly_terms = np.exp(2j * np.pi * np.outer(freqs[rfi_chans], dlys_to_check))
    # dimensions: i = Nubls, j = Ntimes, k = Ndlys
    to_minimize = np.zeros(vis.shape[0:2] + (len(dlys_to_check),))
    for ci in range(len(rfi_chans)):
        wgt_here = (rfi_wgts[ci] if rfi_wgts is not None else 1.0)
        to_minimize += np.abs(np.einsum('ij,i,k->ijk', vis[:, :, ci], rfi_phs[:, ci], dly_terms[ci, :]) - 1) * wgt_here
    dly_sol_args = np.argmin(to_minimize, axis=-1)
    delay_sols = {red[0]: dlys_to_check[dly_sol_args[i]] for i, red in enumerate(reds)}

    # Use per-baseline delays to solve for the DoF, which is one phase slope per polarization per dimension of the idealized antpos
    ls_data, ls_wgts = {}, {}
    for red in reds:
        eq_str = ' + '.join([f'T_{red[0][2]}_{dim} * {ibl_comp}' for dim, ibl_comp in enumerate(idealized_blvecs[red[0]])])
        ls_data[eq_str] = delay_sols[red[0]]
        ls_wgts[eq_str] = len(red) * np.ones_like(delay_sols[red[0]])  # weight by number of baselines in group
    dly_slope_sol = linsolve.LinearSolver(ls_data, wgts=ls_wgts).solve()

    if not return_gains:
        return dly_slope_sol

    # compute gains from DoF for antennas provided
    gains = {}
    for ant in gain_ants:
        pol = join_pol(ant[1], ant[1])
        ipos = idealized_antpos[ant[0]]
        dlys = np.dot(ipos, [dly_slope_sol[f'T_{pol}_{dim}'] for dim in range(len(ipos))])
        gains[ant] = np.exp(2j * np.pi * np.outer(dlys, freqs))
    return gains


def dft_phase_slope_solver(xs, ys, data, flags=None):
    '''Solve for spatial phase slopes across an array by looking for the peak in the DFT.
    This is analogous to the method in utils.fft_dly(), except its in 2D and does not
    assume a regular grid for xs and ys.

    Arguments:
        xs: 1D array of x positions (e.g. of antennas or baselines)
        ys: 1D array of y positions (must be same length as xs)
        data: ndarray of complex numbers to fit with a phase slope. The first dimension must match
            xs and ys, but subsequent dimensions will be preserved and solved independently.
            Any np.nan in data is interpreted as a flag.
        flags: optional array of flags of data not to include in the phase slope solver.

    Returns:
        slope_x, slope_y: phase slopes in units of radians/[xs] where the best fit phase slope plane
            is np.exp(2.0j * np.pi * (xs * slope_x + ys * slope_y)). Both have the same shape
            the data after collapsing along the first dimension.
    '''

    # use the minimum and maximum difference between positions to define the search range and sampling in Fourier space
    deltas = [((xi - xj)**2 + (yi - yj)**2)**.5 for i, (xi, yi) in enumerate(zip(xs, ys))
              for (xj, yj) in zip(xs[i + 1:], ys[i + 1:])]
    search_slice = slice(-1.0 / np.min(deltas), 1.0 / np.min(deltas), 1.0 / np.max(deltas))

    # define cost function
    def dft_abs(k, x, y, z):
        return -np.abs(np.dot(z, np.exp(-2.0j * np.pi * (x * k[0] + y * k[1]))))

    # set up flags, treating nans as flags
    if flags is None:
        flags = np.zeros_like(data, dtype=bool)
    flags = flags | np.isnan(data)

    # loop over data, minimizing the cost function
    dflat = data.reshape((len(xs), -1))
    fflat = flags.reshape((len(xs), -1))
    slope_x = np.zeros_like(dflat[0, :].real)
    slope_y = np.zeros_like(dflat[0, :].real)
    for i in range(dflat.shape[1]):
        if not np.all(np.isnan(dflat[:, i])):
            dft_peak = brute(dft_abs, (search_slice, search_slice),
                             (xs[~fflat[:, i]], ys[~fflat[:, i]],
                              dflat[:, i][~fflat[:, i]]), finish=minimize)
            slope_x[i] = dft_peak[0]
            slope_y[i] = dft_peak[1]
    return 2 * np.pi * slope_x.reshape(data.shape[1:]), 2 * np.pi * slope_y.reshape(data.shape[1:])


def ndim_fft_phase_slope_solver(data, bl_vecs, assume_2D=True, zero_pad=2, bl_error_tol=1.0):
    '''Find phase slopes across the array in the data. Similar to utils.fft_dly,
    but can grid arbitarary bl_vecs in N dimensions (for example, when using
    generealized antenna positions from redcal.reds_to_antpos in arrays with
    extra degeneracies).

    Parameters:
    -----------
    data : dictionary or DataContainer mapping keys to (complex) ndarrays.
           All polarizations are treated equally and solved for together.

    bl_vecs : dictionary mapping keys in data to vectors in N dimensions

    assume_2D : if True, assume N == 2 and only use the first two dimensions of bl_vecs.

    zero_pad : float factor by which to expand the grid onto which the data is binned.
               Increases resolution in Fourier space at the cost of runtime/memory.
               Must be >= 1.

    bl_error_tol : float used to define non-zero elements of baseline vectors.
                   This helps set the fundamental resolution of the grid.

    Output:
    -------
    phase_slopes : list of length N dimensions. Each element is the same shape
                   as each entry in data. Contains the phase gradients in units
                   of 1 / [bl_vecs].
    '''
    nDim = _count_nDims(bl_vecs, assume_2D=assume_2D)
    if assume_2D:
        nDim = 2
    keys = sorted(list(bl_vecs.keys()))

    # Figure out a grid for baselines and
    coords = []
    all_bins = []
    bl_vecs_array = np.array([bl_vecs[k] for k in keys])
    assert zero_pad >= 1, f'zero_pad={zero_pad}, but it must be greater than or equal to 1.'
    for d in range(nDim):
        min_comp = np.min(bl_vecs_array[:, d])
        max_comp = np.max(bl_vecs_array[:, d])
        # pick minimum delta in this dimension inconsistent with 0 using bl_error_tol
        dbl = np.min(np.abs(bl_vecs_array[:, d])[np.abs(bl_vecs_array[:, d]) >= bl_error_tol])
        comp_range = max_comp - min_comp
        bins = np.arange(min_comp - dbl - comp_range * (zero_pad - 1) / 2,
                         max_comp + 2 * dbl + comp_range * (zero_pad - 1) / 2, dbl)
        all_bins.append(bins)
        coords.append(np.digitize(bl_vecs_array[:, d], bins))
    coords = np.array(coords).T

    # create and fill grid with complex data
    digitized = np.zeros(tuple([len(b) for b in all_bins]) + data[keys[0]].shape, dtype=complex)
    for i, k in enumerate(keys):
        digitized[tuple(coords[i])] = data[k]
    digitized[~np.isfinite(digitized)] = 0

    # FFT along first nDim dimensions
    digitized_fft = np.fft.fftn(digitized, axes=tuple(range(nDim)))
    # Condense the FFTed dimensions and find the max along them
    new_shape = (np.prod(digitized_fft.shape[0:nDim]),) + data[keys[0]].shape
    arg_maxes = digitized_fft.reshape(new_shape).argmax(0)
    # Find the coordinates of the peaks in the FFT dimensions
    peak_coords = np.unravel_index(arg_maxes, digitized_fft.shape[0:nDim])

    # Convert coordinates to phase slopes using fft_freq
    phase_slopes = []
    for d in range(nDim):
        fourier_modes = np.fft.fftfreq(len(all_bins[d]), np.median(np.diff(all_bins[d])))
        phase_slopes.append(fourier_modes[peak_coords[d]] * 2 * np.pi)
    return phase_slopes


def global_phase_slope_logcal(model, data, antpos, reds=None, solver='linfit', wgts=None, refant=None,
                              assume_2D=True, verbose=True, tol=1.0, edge_cut=0, time_avg=False,
                              zero_pad=2, return_gains=False, gain_ants=[]):
    """
    Solve for a frequency-independent spatial phase slope using the equation

    median_over_freq(angle(V_ij,xy^data / V_ij,xy^model)) = dot(Phi_x, r_i) - dot(Phi_y, r_j)

    Parameters:
    -----------
    model : visibility data of refence model, type=DataContainer
            keys are antenna-pair + polarization tuples, Ex. (1, 2, 'nn').
            values are complex ndarray visibilities.
            these must 2D arrays, with [0] axis indexing time
            and [1] axis indexing frequency.

    data : visibility data of measurements, type=DataContainer
           keys are antenna pair + pol tuples (must match model), values are
           complex ndarray visibilities matching shape of model

    antpos : type=dictionary, antpos dictionary. antenna num as key, position vector as value.

    reds : list of list of redundant baselines. If left as None (default), will try to infer
           reds from antpos, though if the antenna position dimensionaility is > 3, this will fail.

    solver : 'linfit' uses linsolve to fit phase slope across the array.
             'dft' uses a spatial Fourier transform to find a phase slope, only works in 2D.
             'ndim_fft' uses a gridded spatial Fourier transform instead, but works in ND.

    wgts : weights of data, type=DataContainer, [default=None]
           keys are antenna pair + pol tuples (must match model), values are real floats
           matching shape of model and data. These are only used to find delays from
           itegrations that are unflagged for at least two frequency bins. In this case,
           the delays are assumed to have equal weight, otherwise the delays take zero weight.

    refant : antenna number integer to use as a reference,
             The antenna position coordaintes are centered at the reference, such that its phase
             is identically zero across all frequencies. If None, use the first key in data as refant.

    assume_2D : type=boolean, [default=False]
                If this is true, all dimensions of antpos beyond the first two will be ignored.
                If return_gains is False and assume_2D is False, then the returned variables will
                look like Phi_0, Phi_1, Phi_2, etc. corresponding to the dimensions in antpos.

    verbose : print output, type=boolean, [default=False]

    tol : type=float, baseline match tolerance in units of baseline vectors (e.g. meters)

    edge_cut : int, number of channels to exclude at each band edge in phase slope solver

    time_avg : boolean, if True, replace resultant antenna phase slopes with the median across time

    zero_pad : float factor by which to expand the grid onto which the data is binned. Only used
               for ndim_fft mode. Must be >= 1.

    return_gains : boolean. If True, convert result into a dictionary of gain waterfalls.

    gain_ants : list of ant-pol tuples for return_gains dictionary

    Output:
    -------
    if not return_gains:
        fit : dictionary containing frequency-indpendent phase slope, e.g. Phi_ns_Jxx
              for each position component and polarization in units of radians / [antpos].
              If assume_2D is False, then these will be the more general Phi_0, Phi_1,
              Phi_2, etc. corresponding to the dimensions in antpos.
    else:
        gains : dictionary with gain_ants as keys and gain waterfall arrays as values
    """
    # check solver and edgecut
    assert solver in PHASE_SLOPE_SOLVERS, f"Unrecognized solver {solver}"
    echo(f"...configuring global_phase_slope_logcal for the {solver} algorithm", verbose=verbose)
    assert 2 * edge_cut < list(data.values())[0].shape[1] - 1, "edge_cut cannot be >= Nfreqs/2 - 1"

    # get keys from model and data dictionaries
    keys = sorted(set(model.keys()) & set(data.keys()))
    antnums = np.unique(list(antpos.keys()))

    # make weights if None and make flags
    if wgts is None:
        wgts = odict()
        for i, k in enumerate(keys):
            wgts[k] = np.ones_like(data[k], dtype=float)
    flags = DataContainer({k: ~wgts[k].astype(bool) for k in wgts})

    # center antenna positions about the reference antenna
    if refant is None:
        refant = keys[0][0]
    assert refant in antnums, "reference antenna {} not found in antenna list".format(refant)
    antpos = odict([(k, antpos[k] - antpos[refant]) for k in antpos.keys()])

    # count dimensions of antenna positions, figure out how many to solve for
    nDims = _count_nDims(antpos, assume_2D=assume_2D)

    # average data over baselines
    if reds is None:
        reds = redcal.get_pos_reds(antpos, bl_error_tol=tol)
    ap = data.antpairs()
    reds_here = []
    for red in reds:
        red_here = [bl[0:2] for bl in red if bl[0:2] in ap or bl[0:2][::-1] in ap]  # if the reds have polarizations, ignore them
        if len(red_here) > 0:
            reds_here.append(red_here)
    avg_data, avg_flags, _ = utils.red_average(data, reds=reds_here, flags=flags, inplace=False)
    red_keys = list(avg_data.keys())
    avg_wgts = DataContainer({k: (~avg_flags[k]).astype(float) for k in avg_flags})
    avg_model, _, _ = utils.red_average(model, reds=reds_here, flags=flags, inplace=False)

    ls_data, ls_wgts, bl_vecs, pols = {}, {}, {}, {}
    for rk in red_keys:
        # build equation string
        eqn_str = ''
        ap0, ap1 = split_pol(rk[2])
        for d in range(nDims):
            if len(eqn_str) > 0:
                eqn_str += ' + '
            eqn_str += f'{antpos[rk[0]][d]}*Phi_{d}_{ap0} - {antpos[rk[1]][d]}*Phi_{d}_{ap1}'
        bl_vecs[eqn_str] = antpos[rk[0]] - antpos[rk[1]]
        pols[eqn_str] = rk[2]

        # calculate median of unflagged angle(data/model)
        # ls_weights are sum of non-binary weights
        dm_ratio = avg_data[rk] / avg_model[rk]
        dm_ratio /= np.abs(dm_ratio)  # This gives all channels roughly equal weight, moderating the effect of RFI (as in firstcal)
        binary_flgs = np.isclose(avg_wgts[rk], 0.0) | np.isinf(dm_ratio) | np.isnan(dm_ratio)
        avg_wgts[rk][binary_flgs] = 0.0
        dm_ratio[binary_flgs] *= np.nan
        if solver == 'linfit':  # we want to fit the angles
            ls_data[eqn_str] = np.nanmedian(np.angle(dm_ratio[:, edge_cut:(dm_ratio.shape[1] - edge_cut)]), axis=1, keepdims=True)
        elif solver in ['dft', 'ndim_fft']:  # we want the full complex number
            ls_data[eqn_str] = np.nanmedian(dm_ratio[:, edge_cut:(dm_ratio.shape[1] - edge_cut)], axis=1, keepdims=True)
        ls_wgts[eqn_str] = np.sum(avg_wgts[rk][:, edge_cut:(dm_ratio.shape[1] - edge_cut)], axis=1, keepdims=True)

        # set unobserved data to 0 with 0 weight
        ls_wgts[eqn_str][~np.isfinite(ls_data[eqn_str])] = 0
        ls_data[eqn_str][~np.isfinite(ls_data[eqn_str])] = 0

    if solver == 'linfit':  # build linear system for phase slopes and solve with linsolve
        # setup linsolve and run
        solver = linsolve.LinearSolver(ls_data, wgts=ls_wgts)
        echo("...running linsolve", verbose=verbose)
        fit = solver.solve()
        echo("...finished linsolve", verbose=verbose)

    elif solver in ['dft', 'ndim_fft']:  # look for a peak angle slope by FTing across the array

        if not np.all([split_pol(pol)[0] == split_pol(pol)[1] for pol in data.pols()]):
            raise NotImplementedError('DFT/FFT solving of global phase not implemented for abscal with cross-polarizations.')
        for k in ls_data:
            ls_data[k][ls_wgts[k] == 0] = np.nan

        # solve one polarization at a time
        fit = {}
        for pol in data.pols():
            eqkeys = [k for k in bl_vecs.keys() if pols[k] == pol]
            # reformat data into arrays for dft_phase_slope_solver
            if solver == 'dft':
                assert assume_2D, 'dft solver only works when the array is 2D. Try using ndim_fft instead.'
                blx = np.array([bl_vecs[k][0] for k in eqkeys])
                bly = np.array([bl_vecs[k][1] for k in eqkeys])
                data_array = np.array([ls_data[k] for k in eqkeys])
                slope_x, slope_y = dft_phase_slope_solver(blx, bly, data_array)
                fit['Phi_0_{}'.format(split_pol(pol)[0])] = slope_x
                fit['Phi_1_{}'.format(split_pol(pol)[0])] = slope_y
            # Perform ndim_fft solver
            elif solver == 'ndim_fft':
                slopes = ndim_fft_phase_slope_solver({k: ls_data[k] for k in eqkeys}, {k: bl_vecs[k] for k in eqkeys},
                                                     assume_2D=assume_2D, zero_pad=zero_pad, bl_error_tol=tol)
                for d, slope in enumerate(slopes):
                    fit[f'Phi_{d}_{split_pol(pol)[0]}'] = slope

    # time average
    if time_avg:
        Ntimes = list(fit.values())[0].shape[0]
        for k in fit:
            fit[k] = np.repeat(np.moveaxis(np.median(fit[k], axis=0)[np.newaxis], 0, 0), Ntimes, axis=0)

    if not return_gains:
        # rename variables ew/ns instead of 0/1 to maintain backwards compatability
        if assume_2D:
            params = list(fit.keys())
            for p in params:
                if 'Phi_0' in p:
                    fit[p.replace('Phi_0', 'Phi_ew')] = fit[p]
                    del fit[p]
                if 'Phi_1' in p:
                    fit[p.replace('Phi_1', 'Phi_ns')] = fit[p]
                    del fit[p]
        return fit
    else:
        # compute gains, dotting each slope into the corresponding coordinate in that dimension
        gains = {}
        for ant in gain_ants:
            Phis = [fit[f'Phi_{d}_{ant[1]}'] for d in range((nDims, 2)[assume_2D])]
            gains[ant] = np.exp(1.0j * np.einsum('i,ijk,k->jk', antpos[ant[0]][0:len(Phis)],
                                                 Phis, np.ones(data[keys[0]].shape[1])))
        return gains


def merge_gains(gains, merge_shared=True):
    """
    Merge a list of gain (or flag) dictionaries.

    If gains has boolean ndarray keys, interpret as flags
    and merge with a logical OR.

    Parameters:
    -----------
    gains : type=list or tuple, series of gain dictionaries with (ant, pol) keys
            and complex ndarrays as values (or boolean ndarrays if flags)
    merge_shared : type=bool, If True merge only shared keys, eliminating the others.
        Otherwise, merge all keys.

    Output:
    -------
    merged_gains : type=dictionary, merged gain (or flag) dictionary with same key-value
                   structure as input dict.
    """
    # get shared keys
    if merge_shared:
        keys = sorted(set(reduce(operator.and_, [set(g.keys()) for g in gains])))
    else:
        keys = sorted(set(reduce(operator.add, [list(g.keys()) for g in gains])))

    # form merged_gains dict
    merged_gains = odict()

    # determine if gains or flags from first entry in gains
    fedflags = False
    if gains[0][list(gains[0].keys())[0]].dtype == np.bool_:
        fedflags = True

    # iterate over keys
    for i, k in enumerate(keys):
        if fedflags:
            merged_gains[k] = reduce(operator.add, [g.get(k, True) for g in gains])
        else:
            merged_gains[k] = reduce(operator.mul, [g.get(k, 1.0) for g in gains])

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
    key_sort = np.argsort(np.array(keys, dtype=object)[:, key_index])
    keys = [keys[i] for i in key_sort]
    popped_keys = np.unique(np.array(keys, dtype=object)[:, key_index])

    # get new keys
    new_keys = [k[:key_index] + k[key_index + 1:] for k in keys]
    new_unique_keys = []

    # iterate over new_keys
    for i, nk in enumerate(new_keys):
        # check for unique keys
        if nk in new_unique_keys:
            continue
        new_unique_keys.append(nk)

        # get all instances of redundant keys
        ravel = [k == nk for k in new_keys]

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
        return DataContainer(new_data)


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
        with shape (Ntimes, Nfreqs).

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
    freq_nn = np.array([np.argmin(np.abs(model_freqs - x)) for x in data_freqs])
    time_nn = np.array([np.argmin(np.abs(model_lsts - x)) for x in data_lsts])
    freq_nn, time_nn = np.meshgrid(freq_nn, time_nn)

    # get model indices meshgrid
    mod_F, mod_L = np.meshgrid(np.arange(len(model_freqs)), np.arange(len(model_lsts)))

    # raise warning on flags
    if flags is not None and medfilt_flagged is False:
        print("Warning: flags are fed, but medfilt_flagged=False. \n"
              "This may cause weird behavior of interpolated points near flagged data.")

    # ensure flags are booleans
    if flags is not None and np.issubdtype(
        flags[list(flags.keys())[0]].dtype, np.floating
    ):
        flags = DataContainer(
            odict([(k, ~flags[k].astype(bool)) for k in flags.keys()])
        )

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
            real_fill = np.empty(len(f_indices), float)
            imag_fill = np.empty(len(f_indices), float)

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
            if np.issubdtype(f.dtype, np.floating):
                f = ~(f.astype(bool))
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
    lst_nn = np.array([np.argmin(np.abs(model_lsts - x)) for x in data_lsts])

    # get dlst array
    dlst = data_lsts - model_lsts[lst_nn]

    # flag dlst above threshold
    flag_lst = np.zeros_like(dlst, bool)
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
            new_flags[k] = np.zeros_like(m, bool)
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


def flatten(nested_list):
    """ flatten a nested list """
    return [item for sublist in nested_list for item in sublist]


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
        self.bl = np.array(bl, dtype=float)
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
            equiv = np.all([np.isclose(*x, atol=tol) for x in zip(self.bl, B2.bl)])
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
    model_bls = np.array([Baseline(model_antpos[k[1]] - model_antpos[k[0]], tol=tol) for k in model_keys])

    # create baseline keys for data
    data_keys = list(data.keys())
    data_bls = np.array([Baseline(data_antpos[k[1]] - data_antpos[k[0]], tol=tol) for k in data_keys])

    # iterate over data baselines
    new_model = odict()
    for i, bl in enumerate(model_bls):
        # compre bl to all model_bls
        comparison = np.array(list(map(lambda mbl: bl == mbl, data_bls)), str)

        # get matches
        matches = np.where((comparison == 'True') | (comparison == 'conjugated'))[0]

        # check for matches
        if len(matches) == 0:
            echo("found zero matches in data for model {}".format(model_keys[i]), verbose=verbose)
            continue
        else:
            if len(matches) > 1:
                echo(
                    f"found more than 1 match in data to model {model_keys[i]}: {[data_keys[j] for j in matches]}",
                    verbose=verbose
                )
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
    warnings.warn("Warning: This function will be deprecated in the next hera_cal release.")

    # get data keys
    keys = list(data.keys())

    # get data, wgts and ants
    pols = np.unique([k[2] for k in data.keys()])
    ants = np.unique(np.concatenate(keys))
    if wgts is None:
        wgts = DataContainer(
            odict([(k, np.ones_like(data[k]).astype(float)) for k in data.keys()])
        )

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
        d = np.nansum([data[k] * wgts[k] for k in bl_group], axis=0)
        d /= np.nansum([wgts[k] for k in bl_group], axis=0)

        # get wgts
        if broadcast_wgts:
            w = np.array(reduce(operator.mul, [wgts[k] for k in bl_group]), float) ** (1. / len(bl_group))
        else:
            w = np.array(reduce(operator.add, [wgts[k] for k in bl_group]), float) / len(bl_group)

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
        match = np.array([k in r for r in reds])
        conj_match = np.array([reverse_bl(k) in r for r in reds])

        # if no match, just copy data over to red_data
        if not np.any(match) and not np.any(conj_match):
            red_data[k] = copy.copy(data[k])

        else:
            # iterate over matches
            for j, (m, cm) in enumerate(zip(match, conj_match)):
                if weights:
                    # if weight dictionary, add repeated baselines
                    if m:
                        if k not in red_data:
                            red_data[k] = copy.copy(data[k])
                            red_data[k][red_data[k].astype(bool)] = red_data[k][red_data[k].astype(bool)] + len(reds[j]) - 1
                        else:
                            red_data[k][red_data[k].astype(bool)] = red_data[k][red_data[k].astype(bool)] + len(reds[j])
                    elif cm:
                        if k not in red_data:
                            red_data[k] = copy.copy(data[k])
                            red_data[k][red_data[k].astype(bool)] = red_data[k][red_data[k].astype(bool)] + len(reds[j]) - 1
                        else:
                            red_data[k][red_data[k].astype(bool)] = red_data[k][red_data[k].astype(bool)] + len(reds[j])
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
            red_data[k][red_data[k].astype(bool)] = red_data[k][red_data[k].astype(bool)]**(2.0)
    else:
        red_data = odict([(k, red_data[k]) for k in sorted(red_data)])

    return DataContainer(red_data)


def match_times(datafile, modelfiles, filetype='uvh5', atol=1e-5):
    """
    Match start and end LST of datafile to modelfiles. Each file in modelfiles needs
    to have the same integration time.

    Args:
        datafile : type=str, path to data file
        modelfiles : type=list of str, list of filepaths to model files ordered according to file start time
        filetype : str, options=['uvh5', 'miriad']

    Returns:
        matched_modelfiles : type=list, list of modelfiles that overlap w/ datafile in LST
    """
    # get lst arrays
    data_dlst, data_dtime, data_lsts, data_times = io.get_file_times(datafile, filetype=filetype)
    model_dlsts, model_dtimes, model_lsts, model_times = io.get_file_times(modelfiles, filetype=filetype)

    # shift model files relative to first file & first index if needed
    for ml in model_lsts:
        ml[ml < model_lsts[0][0]] += 2 * np.pi
        # also ensure that ml is increasing
        ml[ml < ml[0]] += 2 * np.pi
    # get model start and stop, buffering by dlst / 2
    model_starts = np.asarray([ml[0] - md / 2.0 for ml, md in zip(model_lsts, model_dlsts)])
    model_ends = np.asarray([ml[-1] + md / 2.0 for ml, md in zip(model_lsts, model_dlsts)])

    # shift data relative to model if needed
    data_lsts[data_lsts < model_starts[0]] += 2 * np.pi
    # make sure monotonically increasing.
    data_lsts = np.unwrap(data_lsts)
    # select model files
    match = np.asarray(modelfiles)[(model_starts < data_lsts[-1] + atol)
                                   & (model_ends > data_lsts[0] - atol)]

    return match


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
        keys contain antennas-pairs + polarization, Ex. (1, 2, 'nn'), and values contain 2D complex
        ndarrays with [0] axis indexing time and [1] axis frequency.

        Parameters:
        -----------
        model : Visibility data of refence model, type=dictionary or DataContainer
                keys are antenna-pair + polarization tuples, Ex. (1, 2, 'nn').
                values are complex ndarray visibilities.
                these must be 2D arrays, with [0] axis indexing time
                and [1] axis indexing frequency.

                Optionally, model can be a path to a pyuvdata-supported file, a
                pyuvdata.UVData object or hera_cal.HERAData object,
                or a list of either.

        data :  Visibility data, type=dictionary or DataContainer
                keys are antenna-pair + polarization tuples, Ex. (1, 2, 'nn').
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

        # get shared keys and pols
        self.keys = sorted(set(model.keys()) & set(data.keys()))
        assert len(self.keys) > 0, "no shared keys exist between model and data"
        if pols is None:
            pols = np.unique([k[2] for k in self.keys])
        self.pols = pols
        self.Npols = len(self.pols)
        self.gain_pols = np.unique([list(split_pol(p)) for p in self.pols])
        self.Ngain_pols = len(self.gain_pols)

        # append attributes
        self.model = DataContainer(dict([(k, model[k]) for k in self.keys]))
        self.data = DataContainer(dict([(k, data[k]) for k in self.keys]))

        # setup frequencies
        self.freqs = freqs
        if self.freqs is None:
            self.Nfreqs = None
        else:
            self.Nfreqs = len(self.freqs)

        # setup weights
        if wgts is None:
            # use data flags if present
            if 'flags' in locals() and flags is not None:
                wgts = DataContainer(dict([(k, (~flags[k]).astype(float)) for k in self.keys]))
            else:
                wgts = DataContainer(dict([(k, np.ones_like(data[k], dtype=float)) for k in self.keys]))
            if 'model_flags' in locals():
                for k in self.keys:
                    wgts[k] *= (~model_flags[k]).astype(float)
        self.wgts = wgts

        # setup ants
        self.ants = np.unique(np.concatenate([k[:2] for k in self.keys]))
        self.Nants = len(self.ants)
        if refant is None:
            refant = self.keys[0][0]
            print("using {} for reference antenna".format(refant))
        else:
            assert refant in self.ants, "refant {} not found in self.ants".format(refant)
        self.refant = refant

        # setup antenna positions
        self._set_antpos(antpos)

        # setup gain solution keys
        self._gain_keys = [[(a, p) for a in self.ants] for p in self.gain_pols]

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

    def _set_antpos(self, antpos):
        '''Helper function for replacing self.antpos, self.bls, and self.antpos_arr without affecting tapering or baseline cuts.
        Useful for replacing true antenna positions with idealized ones derived from the redundancies.'''
        self.antpos = antpos
        self.antpos_arr = None
        self.bls = None
        if self.antpos is not None:
            # center antpos about reference antenna
            self.antpos = odict([(k, antpos[k] - antpos[self.refant]) for k in self.ants])
            self.bls = odict([(x, self.antpos[x[0]] - self.antpos[x[1]]) for x in self.keys])
            self.antpos_arr = np.array([self.antpos[x] for x in self.ants])
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
        self._ant_eta = odict([(k, copy.copy(fit[f"eta_{k[0]}_{k[1]}"])) for k in flatten(self._gain_keys)])
        self._ant_eta_arr = np.moveaxis([[self._ant_eta[k] for k in pk] for pk in self._gain_keys], 0, -1)

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
        self._ant_phi = odict([(k, copy.copy(fit[f"phi_{k[0]}_{k[1]}"])) for k in flatten(self._gain_keys)])
        self._ant_phi_arr = np.moveaxis([[self._ant_phi[k] for k in pk] for pk in self._gain_keys], 0, -1)

        # take time and freq average
        if avg:
            self._ant_phi = odict([
                (
                    k,
                    np.ones_like(self._ant_phi[k]) * np.angle(
                        np.median(np.real(np.exp(1j * self._ant_phi[k])))
                        + 1j * np.median(np.imag(np.exp(1j * self._ant_phi[k])))
                    )
                ) for k in flatten(self._gain_keys)
            ])
            self._ant_phi_arr = np.moveaxis([[self._ant_phi[k] for k in pk] for pk in self._gain_keys], 0, -1)

    def delay_lincal(self, medfilt=True, kernel=(1, 11), verbose=True, time_avg=False, edge_cut=0):
        """
        Solve for per-antenna delay according to the equation
        by calling abscal_funcs.delay_lincal method.
        See abscal_funcs.delay_lincal for details.

        Parameters:
        -----------
        medfilt : boolean, if True median filter data before fft

        kernel : size of median filter across (time, freq) axes, type=(int, int)

        time_avg : boolean, if True, replace resultant antenna delays with the median across time

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
        self._ant_dly = odict([(k, copy.copy(fit[f"tau_{k[0]}_{k[1]}"])) for k in flatten(self._gain_keys)])
        self._ant_dly_arr = np.moveaxis([[self._ant_dly[k] for k in pk] for pk in self._gain_keys], 0, -1)

        self._ant_dly_phi = odict([(k, copy.copy(fit[f"phi_{k[0]}_{k[1]}"])) for k in flatten(self._gain_keys)])
        self._ant_dly_phi_arr = np.moveaxis([[self._ant_dly_phi[k] for k in pk] for pk in self._gain_keys], 0, -1)

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

        time_avg : boolean, if True, replace the resultant delay slope with the median across time

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
                                 time_avg=time_avg, kernel=kernel, verbose=verbose, four_pol=four_pol, edge_cut=edge_cut)

        # separate pols if four_pol
        if four_pol:
            for i, gp in enumerate(self.gain_pols):
                fit['T_ew_{}'.format(gp)] = fit["T_ew"]
                fit['T_ns_{}'.format(gp)] = fit["T_ns"]
                fit.pop('T_ew')
                fit.pop('T_ns')

        # form result
        self._dly_slope = odict([(k, copy.copy(np.array([fit[f"T_ew_{k[1]}"], fit[f"T_ns_{k[1]}"]]))) for k in flatten(self._gain_keys)])
        self._dly_slope_arr = np.moveaxis([[np.array([self._dly_slope[k][0], self._dly_slope[k][1]]) for k in pk] for pk in self._gain_keys], 0, -1)

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
        self._phs_slope = odict([(k, copy.copy(np.array([fit[f"Phi_ew_{k[1]}"], fit[f"Phi_ns_{k[1]}"]]))) for k in flatten(self._gain_keys)])
        self._phs_slope_arr = np.moveaxis([[np.array([self._phs_slope[k][0], self._phs_slope[k][1]]) for k in pk] for pk in self._gain_keys], 0, -1)

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
        self._abs_eta = odict([(k, copy.copy(fit[f"eta_{k[1]}"])) for k in flatten(self._gain_keys)])
        self._abs_eta_arr = np.moveaxis([[self._abs_eta[k] for k in pk] for pk in self._gain_keys], 0, -1)

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
        self._abs_psi = odict([(k, copy.copy(fit[f"psi_{k[1]}"])) for k in flatten(self._gain_keys)])
        self._abs_psi_arr = np.moveaxis([[self._abs_psi[k] for k in pk] for pk in self._gain_keys], 0, -1)

        self._TT_Phi = odict([(k, copy.copy(np.array([fit[f"Phi_ew_{k[1]}"], fit[f"Phi_ns_{k[1]}"]]))) for k in flatten(self._gain_keys)])
        self._TT_Phi_arr = np.moveaxis([[np.array([self._TT_Phi[k][0], self._TT_Phi[k][1]]) for k in pk] for pk in self._gain_keys], 0, -1)

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
            return odict([(k, np.exp(ant_eta[k]).astype(complex)) for k in flatten(self._gain_keys)])
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
            return np.exp(self.ant_eta_arr).astype(complex)
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

    def _make_new_odict(self, fnc):
        """Create new odict with keys from gain_keys, applying fnc to every key."""
        return odict([(k, fnc(k)) for k in flatten(self._gain_keys)])

    @property
    def ant_phi_gain(self):
        """ form complex gain from _ant_phi dict """
        if hasattr(self, '_ant_phi'):
            ant_phi = self.ant_phi
            return self._make_new_odict(lambda k: np.exp(1j * ant_phi[k]))
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
            factor = 2j * np.pi * self.freqs.reshape(1, -1)
            return self._make_new_odict(
                lambda k: np.exp(factor * ant_dly[k])
            )
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
            return self._make_new_odict(
                lambda k: np.exp(1j * np.repeat(ant_dly_phi[k], self.Nfreqs, 1))
            )
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
            # einsum sums over antenna position
            factor = 2j * np.pi * self.freqs.reshape(1, -1)
            return self._make_new_odict(
                lambda k: np.exp(
                    factor * np.einsum("i...,i->...", dly_slope[k], self.antpos[k[0]][:2])
                )
            )
        else:
            return None

    def custom_dly_slope_gain(self, gain_keys, antpos):
        """
        return dly_slope_gain with custom gain keys and antenna positions

        gain_keys : type=list, list of unique (ant, pol). Ex. [(0, 'Jee'), (1, 'Jee'), (0, 'Jnn'), (1, 'Jnn')]
        antpos : type=dictionary, contains antenna position vectors. keys are ant integer, values are ant position vectors
        """
        if hasattr(self, '_dly_slope'):
            # form dict of delay slopes for each polarization in self._gain_keys
            # b/c they are identical for all antennas of the same polarization
            dly_slope_dict = {ants[0][1]: self.dly_slope[ants[0]] for ants in self._gain_keys}

            # turn delay slope into per-antenna complex gains, while iterating over input gain_keys
            dly_slope_gain = odict()
            for gk in gain_keys:
                # einsum sums over antenna position
                dly_slope_gain[gk] = np.exp(2j * np.pi * self.freqs.reshape(1, -1) * np.einsum("i...,i->...", dly_slope_dict[gk[1]], antpos[gk[0]][:2]))
            return dly_slope_gain
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
            # einsum sums over antenna position
            return np.exp(2j * np.pi * self.freqs.reshape(-1, 1) * np.einsum("hi...,hi->h...", self._dly_slope_arr, self.antpos_arr[:, :2]))
        else:
            return None

    @property
    def dly_slope_ant_dly_arr(self):
        """ form antenna delays from _dly_slope_arr array """
        if hasattr(self, '_dly_slope_arr'):
            # einsum sums over antenna position
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
            # einsum sums over antenna position
            fac = 1.0j * np.ones_like(self.freqs).reshape(1, -1)
            return self._make_new_odict(
                lambda k: np.exp(
                    fac * np.einsum("i...,i->...", phs_slope[k], self.antpos[k[0]][:2])
                )
            )
        else:
            return None

    def custom_phs_slope_gain(self, gain_keys, antpos):
        """
        return phs_slope_gain with custom gain keys and antenna positions

        gain_keys : type=list, list of unique (ant, pol). Ex. [(0, 'Jee'), (1, 'Jee'), (0, 'Jnn'), (1, 'Jnn')]
        antpos : type=dictionary, contains antenna position vectors. keys are ant integer, values are ant position vectors
        """
        if hasattr(self, '_phs_slope'):
            # form dict of phs slopes for each polarization in self._gain_keys
            # b/c they are identical for all antennas of the same polarization
            phs_slope_dict = {ants[0][1]: self.phs_slope[ants[0]] for ants in self._gain_keys}

            # turn phs slope into per-antenna complex gains, while iterating over input gain_keys
            phs_slope_gain = odict()
            for gk in gain_keys:
                # einsum sums over antenna position
                phs_slope_gain[gk] = np.exp(1.0j * np.ones_like(self.freqs).reshape(1, -1) * np.einsum("i...,i->...", phs_slope_dict[gk[1]], antpos[gk[0]][:2]))
            return phs_slope_gain

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
            # einsum sums over antenna position
            return np.exp(1.0j * np.ones_like(self.freqs).reshape(-1, 1) * np.einsum("hi...,hi->h...", self._phs_slope_arr, self.antpos_arr[:, :2]))
        else:
            return None

    @property
    def phs_slope_ant_phs_arr(self):
        """ form antenna delays from _phs_slope_arr array """
        if hasattr(self, '_phs_slope_arr'):
            # einsum sums over antenna position
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
            return self._make_new_odict(lambda k: np.exp(abs_eta[k]).astype(complex))
        else:
            return None

    def custom_abs_eta_gain(self, gain_keys):
        """
        return abs_eta_gain with custom gain keys

        gain_keys : type=list, list of unique (ant, pol). Ex. [(0, 'Jee'), (1, 'Jee'), (0, 'Jnn'), (1, 'Jnn')]
        """
        if hasattr(self, '_abs_eta'):
            # form dict of abs eta for each polarization in self._gain_keys
            # b/c they are identical for all antennas of the same polarization
            abs_eta_dict = {ants[0][1]: self.abs_eta[ants[0]] for ants in self._gain_keys}

            # turn abs eta into per-antenna complex gains, while iterating over input gain_keys
            abs_eta_gain = odict()
            for gk in gain_keys:
                abs_eta_gain[gk] = np.exp(abs_eta_dict[gk[1]]).astype(complex)
            return abs_eta_gain

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
            return np.exp(self._abs_eta_arr).astype(complex)
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
            return self._make_new_odict(lambda k: np.exp(1j * abs_psi[k]))
        else:
            return None

    def custom_abs_psi_gain(self, gain_keys):
        """
        return abs_psi_gain with custom gain keys

        gain_keys : type=list, list of unique (ant, pol). Ex. [(0, 'Jee'), (1, 'Jee'), (0, 'Jnn'), (1, 'Jnn')]
        """
        if hasattr(self, '_abs_psi'):
            # form dict of abs psi for each polarization in self._gain_keys
            # b/c they are identical for all antennas of the same polarization
            abs_psi_dict = {ants[0][1]: self.abs_psi[ants[0]] for ants in self._gain_keys}

            # turn abs psi into per-antenna complex gains, while iterating over input gain_keys
            abs_psi_gain = odict()
            for gk in gain_keys:
                abs_psi_gain[gk] = np.exp(1j * abs_psi_dict[gk[1]])
            return abs_psi_gain
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
            # einsum sums over antenna position
            return self._make_new_odict(
                lambda k: np.exp(
                    1j * np.einsum("i...,i->...", TT_Phi[k], self.antpos[k[0]][:2])
                )
            )
        else:
            return None

    def custom_TT_Phi_gain(self, gain_keys, antpos):
        """
        return TT_Phi_gain with custom gain keys and antenna positions

        gain_keys : type=list, list of unique (ant, pol). Ex. [(0, 'Jee'), (1, 'Jee'), (0, 'Jnn'), (1, 'Jnn')]
        antpos : type=dictionary, contains antenna position vectors. keys are ant integer, values are ant positions
        """
        if hasattr(self, '_TT_Phi'):
            # form dict of TT_Phi for each polarization in self._gain_keys
            # b/c they are identical for all antennas of the same polarization
            TT_Phi_dict = {ants[0][1]: self.TT_Phi[ants[0]] for ants in self._gain_keys}

            # turn TT_Phi into per-antenna complex gains, while iterating over input gain_keys
            TT_Phi_gain = odict()
            for gk in gain_keys:
                # einsum sums over antenna position
                TT_Phi_gain[gk] = np.exp(1j * np.einsum("i...,i->...", TT_Phi_dict[gk[1]], antpos[gk[0]][:2]))
            return TT_Phi_gain
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
            # einsum sums over antenna position
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
        all_times = np.array([time for f in hd.filepaths for time in all_times[f]])
        all_lsts = np.array([lst for f in hd.filepaths for lst in all_lsts[f]])[np.argsort(all_times)]
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


def get_d2m_time_map(data_times, data_lsts, model_times, model_lsts, extrap_limit=.5):
    '''Generate a dictionary that maps data times to model times via shared LSTs.

    Arguments:
        data_times: list of times in the data (in JD)
        data_lsts: list of corresponding LSTs (in radians)
        model_times: list of times in the mdoel (in JD)
        model_lsts: list of corresponing LSTs (in radians)
        extrap_limit: float that sets the maximum distance away in LST, in unit of the median Delta
            in model_lsts, that a data time can be mapped to model time. If no model_lst is within
            this distance, the data_time is mapped to None. If there is only one model lst, this
            is ignored and the nearest time is always returned.

    Returns:
        d2m_time_map: dictionary uniqely mapping times in the data to times in the model
            that are closest in LST. Data times map to None when the nearest model LST is too far,
            as defined by the extrap_limit.
    '''
    # check that the input is sensible
    if len(data_times) != len(data_lsts):
        raise ValueError('data_times and data_lsts must have the same length.')
    if len(model_times) != len(model_lsts):
        raise ValueError('model_times and model_lsts must have the same length.')

    # compute maximum acceptable distance on the unit circle
    max_complex_dist = 2.0
    if len(model_lsts) > 1:
        max_complex_dist = np.median(np.abs(np.diff(np.exp(1j * model_lsts)))) * extrap_limit

    # find indices of nearest model lst for a given data lst
    d2m_ind_map = {}
    for dind, dlst in enumerate(data_lsts):
        lst_complex_distances = np.abs(np.exp(1j * model_lsts) - np.exp(1j * dlst))
        # check to see that the nearst model_lst is close enough
        if np.min(lst_complex_distances) <= max_complex_dist:
            d2m_ind_map[dind] = np.argmin(lst_complex_distances)
        else:
            d2m_ind_map[dind] = None

    # return map of data times to model times using those indices
    return {data_times[dind]: model_times[mind] if mind is not None else None
            for dind, mind in d2m_ind_map.items()}


def abscal_step(gains_to_update, AC, AC_func, AC_kwargs, gain_funcs, gain_args_list, gain_flags,
                gain_convention='divide', max_iter=1, phs_conv_crit=1e-6, verbose=True):
    '''Generalized function for performing an abscal step (e.g. abs_amp_logcal or TT_phs_logcal).

    NOTE: This function is no longer used and will likely be removed in a future version.

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
    warnings.warn('abscal_step is no longer used by post_redcal_abscal and thus subject to future removal.', DeprecationWarning)

    for i in range(max_iter):
        AC_func(**AC_kwargs)
        gains_here = merge_gains([gf(*gargs) for gf, gargs in zip(gain_funcs, gain_args_list)])
        apply_cal.calibrate_in_place(AC.data, gains_here, AC.wgts, gain_flags,
                                     gain_convention=gain_convention, flags_are_wgts=True)
        for k in gains_to_update.keys():
            gains_to_update[k] *= gains_here[k]
        if max_iter > 1:
            crit = np.median(np.linalg.norm([gains_here[k] - 1.0 for
                                             k in gains_here.keys()], axis=(0, 1)))
            echo(AC_func.__name__ + " convergence criterion: " + str(crit), verbose=verbose)
            if crit < phs_conv_crit:
                break


def match_baselines(data_bls, model_bls, data_antpos, model_antpos=None, pols=[], data_is_redsol=False,
                    model_is_redundant=False, tol=1.0, min_bl_cut=None, max_bl_cut=None, max_dims=2, verbose=False):
    '''Figure out which baselines to use in the data and the model for abscal and their correspondence.

    Arguments:
        data_bls: list of baselines in data file in the form (0, 1, 'ee')
        model_bls: list of baselines in model files in the form (0, 1, 'ee')
        data_antpos: dictionary mapping antenna number to ENU position in meters for antennas in the data
        model_antpos: same as data_antpos, but for the model. If None, assumed to match data_antpos
        pols: list of polarizations to use. If empty, will use all polarizations in the data or model.
        data_is_redsol: if True, the data file only contains one visibility per unique baseline
        model_is_redundant: if True, the model file only contains one visibility per unique baseline
        tol: float distance for baseline match tolerance in units of baseline vectors (e.g. meters)
        min_bl_cut : float, eliminate all visibilities with baseline separation lengths
            smaller than min_bl_cut. This is assumed to be in ENU coordinates with units of meters.
        max_bl_cut : float, eliminate all visibilities with baseline separation lengths
            larger than max_bl_cut. This is assumed to be in ENU coordinates with units of meters.

    Returns:
        data_bl_to_load: list of baseline tuples in the form (0, 1, 'ee') to load from the data file
        model_bl_to_load: list of baseline tuples in the form (0, 1, 'ee') to load from the model file(s)
        data_to_model_bl_map: dictionary mapping data baselines to the corresponding model baseline
    '''
    if data_is_redsol and not model_is_redundant:
        raise NotImplementedError('If the data is just unique baselines, the model must also be just unique baselines.')
    if model_antpos is None:
        model_antpos = copy.deepcopy(data_antpos)

    # Perform cut on baseline length and polarization
    if len(pols) == 0:
        pols = list(set([bl[2] for bl_list in [data_bls, model_bls] for bl in bl_list]))
    data_bl_to_load = set(utils.filter_bls(data_bls, pols=pols, antpos=data_antpos, min_bl_cut=min_bl_cut, max_bl_cut=max_bl_cut))
    model_bl_to_load = set(utils.filter_bls(model_bls, pols=pols, antpos=model_antpos, min_bl_cut=min_bl_cut, max_bl_cut=max_bl_cut))

    # If we're working with full data sets, only pick out matching keys (or ones that work reversably)
    if not data_is_redsol and not model_is_redundant:
        data_bl_to_load = [bl for bl in data_bl_to_load if (bl in model_bl_to_load) or (reverse_bl(bl) in model_bl_to_load)]
        model_bl_to_load = [bl for bl in model_bl_to_load if (bl in data_bl_to_load) or (reverse_bl(bl) in data_bl_to_load)]
        data_to_model_bl_map = {bl: bl for bl in data_bl_to_load if bl in model_bl_to_load}
        data_to_model_bl_map.update({bl: reverse_bl(bl) for bl in data_bl_to_load if reverse_bl(bl) in model_bl_to_load})

    # Either the model is just unique baselines, or both the data and the model are just unique baselines
    else:
        # build reds using both sets of antpos to find matching baselines
        # increase all antenna indices in the model by model_offset to distinguish them from data antennas
        model_offset = np.max(list(data_antpos.keys())) + 1
        joint_antpos = {**data_antpos, **{ant + model_offset: pos for ant, pos in model_antpos.items()}}
        joint_reds = redcal.get_reds(joint_antpos, pols=pols, bl_error_tol=tol)

        # filter out baselines not in data or model or between data and model
        joint_reds = [[bl for bl in red if not ((bl[0] < model_offset) ^ (bl[1] < model_offset))] for red in joint_reds]
        joint_reds = [[bl for bl in red if (bl in data_bl_to_load) or (reverse_bl(bl) in data_bl_to_load)
                       or ((bl[0] - model_offset, bl[1] - model_offset, bl[2]) in model_bl_to_load)
                       or reverse_bl((bl[0] - model_offset, bl[1] - model_offset, bl[2])) in model_bl_to_load] for red in joint_reds]
        joint_reds = [red for red in joint_reds if len(red) > 0]

        # map baselines in data to unique baselines in model
        data_to_model_bl_map = {}
        for red in joint_reds:
            data_bl_candidates = [bl for bl in red if bl[0] < model_offset]
            model_bl_candidates = [(bl[0] - model_offset, bl[1] - model_offset, bl[2]) for bl in red if bl[0] >= model_offset]
            assert len(model_bl_candidates) <= 1, ('model_is_redundant is True, but the following model baselines are '
                                                   'redundant and in the model file: {}'.format(model_bl_candidates))
            if len(model_bl_candidates) == 1:
                for bl in red:
                    if bl[0] < model_offset:
                        if bl in data_bl_to_load:
                            data_to_model_bl_map[bl] = model_bl_candidates[0]
                        elif reverse_bl(bl) in data_bl_to_load:
                            data_to_model_bl_map[reverse_bl(bl)] = reverse_bl(model_bl_candidates[0])
                        else:
                            raise ValueError("Baseline {} looks like a data baseline, but isn't in data_bl_to_load.".format(bl))
            assert ((len(data_bl_candidates) <= 1)
                    or (not data_is_redsol)), ('data_is_redsol is True, but the following data baselines are redundant in the ',
                                               'data file: {}'.format(data_bl_candidates))
        # only load baselines in map
        data_bl_to_load = [bl for bl in data_bl_to_load if bl in data_to_model_bl_map.keys()]
        model_bl_to_load = [bl for bl in model_bl_to_load if (bl in data_to_model_bl_map.values())
                            or (reverse_bl(bl) in data_to_model_bl_map.values())]

    echo("Selected {} data baselines and {} model baselines to load.".format(len(data_bl_to_load), len(model_bl_to_load)), verbose=verbose)
    return list(data_bl_to_load), list(model_bl_to_load), data_to_model_bl_map


def build_data_wgts(data_flags, data_nsamples, model_flags, autocorrs, auto_flags, times_by_bl=None,
                    df=None, data_is_redsol=False, gain_flags=None, tol=1.0, antpos=None):
    '''Build linear weights for data in abscal (or calculating chisq) defined as
    wgts = (noise variance * nsamples)^-1 * (0 if data or model is flagged).
    Note: if there are discontinunities into the autocorrelations, the nsamples, etc., this may
    introduce spectral strucutre into the calibration soltuion.

    Arguments:
        data_flags: DataContainer containing flags on data to be abscaled
        data_nsamples: DataContainer containing the number of samples in each data point
        model_flags: DataContainer with model flags. Assumed to have all the same keys as the data_flags.
        autocorrs: DataContainer with autocorrelation visibilities
        auto_flags: DataContainer containing flags for autocorrelation visibilities
        times_by_bl: dictionary mapping antenna pairs like (0,1) to float Julian Date. Optional if
            inferable from data_flags and all times have length > 1.
        df: If None, inferred from data_flags.freqs
        data_is_redsol: If True, data_file only contains unique visibilities for each baseline group.
            In this case, gain_flags and tol are required and antpos is required if not derivable
            from data_flags. In this case, the noise variance is inferred from autocorrelations from
            all baselines in the represented unique baseline group.
        gain_flags: Used to exclude ants from the noise variance calculation from the autocorrelations
            Ignored if data_is_redsol is False.
        tol: float distance for baseline match tolerance in units of baseline vectors (e.g. meters).
            Ignored if data_is_redsol is False.
        antpos: dictionary mapping antenna number to ENU position in meters for antennas in the data.
            Ignored if data_is_redsol is False. If left as None, can be inferred from data_flags.data_antpos.
    Returns:
        wgts: Datacontainer mapping data_flags baseline to weights
    '''
    # infer times and df if necessary
    if times_by_bl is None:
        times_by_bl = data_flags.times_by_bl
    if df is None:
        df = np.median(np.ediff1d(data_flags.freqs))

    # if data_is_redsol, get reds, using data_flags.antpos if antpos is unspecified
    if data_is_redsol:
        if antpos is None:
            antpos = data_flags.data_antpos
        reds = redcal.get_reds(antpos, bl_error_tol=tol, pols=data_flags.pols())
        reds = redcal.filter_reds(reds, ants=[split_bl(bl)[0] for bl in autocorrs])

    # build weights dict using (noise variance * nsamples)^-1 * (0 if data or model is flagged)
    wgts = {}
    for bl in data_flags:
        dt = (np.median(np.ediff1d(times_by_bl[bl[:2]])) * 86400.)
        wgts[bl] = (data_nsamples[bl] * (~data_flags[bl]) * (~model_flags[bl])).astype(float)

        if not np.all(wgts[bl] == 0.0):
            # use autocorrelations to produce weights
            if not data_is_redsol:
                noise_var = predict_noise_variance_from_autos(bl, autocorrs, dt=dt, df=df)
            # use autocorrelations from all unflagged antennas in unique baseline to produce weights
            else:
                try:  # get redundant group that includes this baseline
                    red_here = [red for red in reds if (bl in red) or (reverse_bl(bl) in red)][0]
                except IndexError:  # this baseline has no unflagged redundancies
                    noise_var = np.inf
                else:
                    noise_vars = []
                    for rbl in red_here:
                        noise_var_here = predict_noise_variance_from_autos(rbl, autocorrs, dt=dt, df=df)
                        for ant in split_bl(rbl):
                            noise_var_here[auto_flags[join_bl(ant, ant)]] = np.nan
                        noise_vars.append(noise_var_here)
                    # estimate noise variance per baseline, assuming inverse variance weighting, but excluding flagged autos
                    noise_var = np.nansum(np.array(noise_vars)**-1, axis=0)**-1 * np.sum(~np.isnan(noise_vars), axis=0)
            wgts[bl] *= noise_var**-1
        wgts[bl][~np.isfinite(wgts[bl])] = 0.0

    return DataContainer(wgts)


def _get_idealized_antpos(cal_flags, antpos, pols, tol=1.0, keep_flagged_ants=True, data_wgts={}):
    '''Figure out a set of idealized antenna positions that doesn't introduce additional
    redcal degeneracies.

    Arguments:
        cal_flags: dictionary mapping keys like (1, 'Jnn') to flag waterfalls
        antpos: dictionary mapping antenna numbers to numpy array positions
        pols: list of polarizations like ['ee', 'nn']
        tol: float distance for baseline match tolerance in units of baseline vectors (e.g. meters)
        keep_flagged_ants: If True, flagged antennas that are off-grid (i.e. would introduce an
            additional degeneracy) are placed at the origin. Otherwise, flagged antennas in cal_flags
            are excluded from idealized_antpos.
        data_wgts: DataContainer mapping baselines like (0, 1, 'ee') to weights. Used to check if
            flagged antennas off the calibratable grid have no weight. Ignored if keep_flagged_ants
            is False.

    Returns:
        idealized_antpos: dictionary mapping antenna numbers to antenna positions on an N-dimensional
            grid where redundant real-world baselines (up to the tol) are perfectly redundant (up to
            numerical precision). These baselines will be arbitrarily linearly transformed (stretched,
            skewed, etc.) and antennas that introduce extra degeneracies will introduce extra dimensions.
            See redcal.reds_to_antpos() for more detail.
    '''
    # build list of reds without flagged untennas
    all_ants = list(cal_flags.keys())
    unflagged_ants = [ant for ant in cal_flags if not np.all(cal_flags[ant])]
    all_reds = redcal.get_reds(antpos, bl_error_tol=tol, pols=pols)
    unflagged_reds = redcal.filter_reds(all_reds, ants=unflagged_ants)

    # count the number of dimensions describing the redundancies of unflagged antennas
    unflagged_idealized_antpos = redcal.reds_to_antpos(unflagged_reds, tol=redcal.IDEALIZED_BL_TOL)
    unflagged_nDims = _count_nDims(unflagged_idealized_antpos, assume_2D=False)

    # get the potentially calibratable ants, reds, and idealized_antpos. These are antennas that may
    # be flagged, but they they are still on the grid of unflagged antennas and can thus be updated
    # without introducing additional degeneracies.
    if keep_flagged_ants:
        reds = redcal.filter_reds(all_reds, max_dims=unflagged_nDims)
    else:
        reds = unflagged_reds
    calibratable_ants = set([ant for red in reds for bl in red for ant in split_bl(bl)])
    idealized_antpos = redcal.reds_to_antpos(reds, tol=redcal.IDEALIZED_BL_TOL)
    for ant in unflagged_ants:
        if ant not in calibratable_ants:
            raise ValueError(f'{ant}, which is not flagged in cal_flags, but is not in the on-grid ants '
                             f'which are {sorted(list(calibratable_ants))}.')

    if keep_flagged_ants:
        # figure out which atennas have non-zero weight
        ants_with_wgts = set([])
        for bl in data_wgts:
            if not np.all(data_wgts[bl] == 0.0):
                for ant in split_bl(bl):
                    if ant not in all_ants:
                        raise ValueError(f'Antenna {ant} has non-zero weight in data_wgts but is not in cal_flags, '
                                         f'which has keys {sorted(list(cal_flags.keys()))}.')
                    ants_with_wgts.add(ant)

        # add off-grid antennas that have no weight at idealized position = 0
        for ant in all_ants:
            if ant not in calibratable_ants:
                if ant in ants_with_wgts:
                    raise ValueError(f'Antenna {ant} appears in data with non-zero weight, but is not in the on-grid ants '
                                     f'which are {sorted(list(calibratable_ants))}.')
                idealized_antpos[ant[0]] = np.zeros(unflagged_nDims)

    return idealized_antpos


def post_redcal_abscal(model, data, data_wgts, rc_flags, edge_cut=0, tol=1.0, kernel=(1, 15),
                       phs_max_iter=100, phs_conv_crit=1e-6, verbose=True,
                       use_abs_amp_logcal=True, use_abs_amp_lincal=True):
    '''Performs Abscal for data that has already been redundantly calibrated.

    Arguments:
        model: DataContainer containing externally calibrated visibilities, LST-matched to the data.
            The model keys must match the data keys.
        data: DataContainer containing redundantly but not absolutely calibrated visibilities. This gets modified.
        data_wgts: DataContainer containing same keys as data, determines their relative weight in the abscal
            linear equation solvers.
        rc_flags: dictionary mapping keys like (1, 'Jnn') to flag waterfalls from redundant calibration.
        edge_cut : integer number of channels to exclude at each band edge in delay and global phase solvers
        tol: float distance for baseline match tolerance in units of baseline vectors (e.g. meters)
        kernel: tuple of integers, size of medfilt kernel used in the first step of delay slope calibration.
            otherwise, 'multiply'.
        phs_max_iter: maximum number of iterations of phase_slope_cal or TT_phs_cal allowed
        phs_conv_crit: convergence criterion for updates to iterative phase calibration that compares
            the updates to all 1.0s.
        use_abs_amp_logcal: start absolute amplitude calibration with a biased but robust first step. Default True.
        use_abs_amp_lincal: finish calibration with an unbiased amplitude lincal step. Default True.

    Returns:
        abscal_delta_gains: gain dictionary mapping keys like (1, 'Jnn') to waterfalls containing
            the updates to the gains between redcal and abscal. Uses keys from rc_flags. Will try to
            update flagged antennas if they fall on the grid and don't introduce additional degeneracies.
    '''

    # get ants, idealized_antpos, and reds
    ants = sorted(list(rc_flags.keys()))
    idealized_antpos = _get_idealized_antpos(rc_flags, data.antpos, data.pols(),
                                             data_wgts=data_wgts, tol=tol, keep_flagged_ants=True)
    reds = redcal.get_reds(idealized_antpos, pols=data.pols(), bl_error_tol=redcal.IDEALIZED_BL_TOL)

    # Abscal Step 1: Per-Channel Logarithmic Absolute Amplitude Calibration
    if use_abs_amp_logcal:
        gains_here = abs_amp_logcal(model, data, wgts=data_wgts, verbose=verbose, return_gains=True, gain_ants=ants)
        abscal_delta_gains = {ant: gains_here[ant] for ant in ants}
        apply_cal.calibrate_in_place(data, gains_here)
    else:
        abscal_delta_gains = {ant: np.ones_like(rc_flags[ant], dtype=np.complex64) for ant in ants}

    # Abscal Step 2: Global Delay Slope Calibration
    binary_wgts = DataContainer({bl: (data_wgts[bl] > 0).astype(float) for bl in data_wgts})
    df = np.median(np.diff(data.freqs))
    for time_avg in [True, False]:  # first use the time-averaged solution to try to avoid false minima
        gains_here = delay_slope_lincal(model, data, idealized_antpos, wgts=binary_wgts, df=df, f0=data.freqs[0], medfilt=True, kernel=kernel,
                                        assume_2D=False, time_avg=time_avg, verbose=verbose, edge_cut=edge_cut, return_gains=True, gain_ants=ants)
        abscal_delta_gains = {ant: abscal_delta_gains[ant] * gains_here[ant] for ant in ants}
        apply_cal.calibrate_in_place(data, gains_here)

    # Abscal Step 3: Global Phase Slope Calibration (first using ndim_fft, then using linfit)
    for time_avg in [True, False]:
        gains_here = global_phase_slope_logcal(model, data, idealized_antpos, reds=reds, solver='ndim_fft', wgts=binary_wgts, verbose=verbose, assume_2D=False,
                                               tol=redcal.IDEALIZED_BL_TOL, edge_cut=edge_cut, time_avg=time_avg, return_gains=True, gain_ants=ants)
        abscal_delta_gains = {ant: abscal_delta_gains[ant] * gains_here[ant] for ant in ants}
        apply_cal.calibrate_in_place(data, gains_here)
    for time_avg in [True, False]:
        for i in range(phs_max_iter):
            gains_here = global_phase_slope_logcal(model, data, idealized_antpos, reds=reds, solver='linfit', wgts=binary_wgts, verbose=verbose, assume_2D=False,
                                                   tol=redcal.IDEALIZED_BL_TOL, edge_cut=edge_cut, time_avg=time_avg, return_gains=True, gain_ants=ants)
            abscal_delta_gains = {ant: abscal_delta_gains[ant] * gains_here[ant] for ant in ants}
            apply_cal.calibrate_in_place(data, gains_here)
            crit = np.median(np.linalg.norm([gains_here[k] - 1.0 for k in gains_here.keys()], axis=(0, 1)))
            echo("global_phase_slope_logcal convergence criterion: " + str(crit), verbose=verbose)
            if crit < phs_conv_crit:
                break

    # Abscal Step 4: Per-Channel Tip-Tilt Phase Calibration
    angle_wgts = DataContainer({bl: 2 * np.abs(model[bl])**2 * data_wgts[bl] for bl in model})
    # This is because, in the high SNR limit, if Var(model) = 0 and Var(data) = Var(noise),
    # then Var(angle(data / model)) = Var(noise) / (2 |model|^2). Here data_wgts = Var(noise)^-1.
    for i in range(phs_max_iter):
        gains_here = TT_phs_logcal(model, data, idealized_antpos, wgts=angle_wgts, verbose=verbose, assume_2D=False, return_gains=True, gain_ants=ants)
        abscal_delta_gains = {ant: abscal_delta_gains[ant] * gains_here[ant] for ant in ants}
        apply_cal.calibrate_in_place(data, gains_here)
        crit = np.median(np.linalg.norm([gains_here[k] - 1.0 for k in gains_here.keys()], axis=(0, 1)))
        echo("TT_phs_logcal convergence criterion: " + str(crit), verbose=verbose)
        if crit < phs_conv_crit:
            break

    # Abscal Step 5: Per-Channel Linear Absolute Amplitude Calibration
    if use_abs_amp_lincal:
        gains_here = abs_amp_lincal(model, data, wgts=data_wgts, verbose=verbose, return_gains=True, gain_ants=ants)
        abscal_delta_gains = {ant: abscal_delta_gains[ant] * gains_here[ant] for ant in ants}

    return abscal_delta_gains


def post_redcal_abscal_run(data_file, redcal_file, model_files, raw_auto_file=None, data_is_redsol=False, model_is_redundant=False, output_file=None,
                           nInt_to_load=None, data_solar_horizon=90, model_solar_horizon=90, extrap_limit=.5, min_bl_cut=1.0, max_bl_cut=None,
                           edge_cut=0, tol=1.0, phs_max_iter=100, phs_conv_crit=1e-6, refant=None, clobber=True, add_to_history='', verbose=True, skip_abs_amp_lincal=False,
                           write_delta_gains=False, output_file_delta=None):
    '''Perform abscal on entire data files, picking relevant model_files from a list and doing partial data loading.
    Does not work on data (or models) with baseline-dependant averaging.

    Arguments:
        data_file: string path to raw uvh5 visibility file or omnical_visibility solution
            (in the later case, one must also set data_is_redsol to True).
        redcal_file: string path to redcal calfits file. This forms the basis of the resultant abscal calfits file.
            If data_is_redsol is False, this will also be used to calibrate the data_file and raw_auto_file
        model_files: list of string paths to externally calibrated data or a reference simulation.
            Strings must be sortable to produce a chronological list in LST (wrapping over 2*pi is OK).
        raw_auto_file: path to data file that contains raw autocorrelations for all antennas in redcal_file.
            These are used for weighting and calculating chi^2. If data_is_redsol, this must be provided.
            If this is None and data_file will be used.
        data_is_redsol: If True, data_file only contains unique visibilities for each baseline group. This means it has been
            redundantly calibrated by the gains in redcal_file already. If this is True, model_is_redundant must also be True
            and raw_auto_file must be provided. If both this and model_is_redundant are False, then only exact baseline
            matches are used in absolute calibration.
        model_is_redundant: If True, then model_files only containe unique visibilities. In this case, data and model
            antenna numbering do not need to agree, as redundant baselines will be found automatically.
        output_file: string path to output abscal calfits file. If None, will be redcal_file.replace('.omni.', '.abs.')
        nInt_to_load: number of integrations to load and calibrate simultaneously. Default None loads all integrations.
        data_solar_horizon: Solar altitude threshold [degrees]. When the sun is too high in the data, flag the integration.
        model_solar_horizon: Solar altitude threshold [degrees]. When the sun is too high in the model, flag the integration.
        extrap_limit: float maximum LST difference (in units of delta LST of the model) allowed between matching data and model times
        min_bl_cut: minimum baseline separation [meters] to keep in data when calibrating. None or 0 means no mininum,
            which will include autocorrelations in the absolute calibration. Usually this is not desired, so the default is 1.0.
        max_bl_cut: maximum baseline separation [meters] to keep in data when calibrating. None (default) means no maximum.
        edge_cut: integer number of channels to exclude at each band edge in delay and global phase solvers
        tol: baseline match tolerance in units of baseline vectors (e.g. meters)
        phs_max_iter: integer maximum number of iterations of phase_slope_cal or TT_phs_cal allowed
        phs_conv_crit: convergence criterion for updates to iterative phase calibration that compares them to all 1.0s.
        refant: tuple of the form (0, 'Jnn') indicating the antenna defined to have 0 phase. If None, refant will be automatically chosen.
        clobber: if True, overwrites existing abscal calfits file at the output path
        add_to_history: string to add to history of output abscal file
        skip_abs_amp_lincal: if False, finish calibration with an unbiased amplitude lincal step. Default False.
        write_delta_gains: write the degenerate gain component solved by abscal so a separate file specified by output_file_delta. Quality and flag arrays are equal to abscal file.
        output_file_delta: path to file to write delta gains if write_delta_gains=True

    Returns:
        hc: HERACal object which was written to disk. Matches the input redcal_file with an updated history.
            This HERACal object has been updated with the following properties accessible on hc.build_calcontainers():
                * gains: abscal gains for times that could be calibrated, redcal gains otherwise (but flagged)
                * flags: redcal flags, with additional flagging if the data is flagged (see flag_utils.synthesize_ant_flags) or if
                    if the model is completely flagged for a given freq/channel when reduced to a single flagging waterfall
                * quals: abscal chi^2 per antenna based on calibrated data minus model (Normalized by noise/nObs, but not with proper DoF)
                * total_qual: abscal chi^2 based on calibrated data minus model (Normalized by noise/nObs, but not with proper DoF)
    '''
    # Raise error if output calfile already exists and clobber is False
    if output_file is None:
        output_file = redcal_file.replace('.omni.', '.abs.')
    if os.path.exists(output_file) and not clobber:
        raise IOError("{} exists, not overwriting.".format(output_file))

    # Make raw_auto_file the data_file if None when appropriate, otherwise raise an error
    if raw_auto_file is None:
        if not data_is_redsol:
            raw_auto_file = data_file
        else:
            raise ValueError('If the data is a redundant visibility solution, raw_auto_file must be specified.')

    # Load redcal calibration
    hc = io.HERACal(redcal_file)
    rc_gains, rc_flags, rc_quals, rc_tot_qual = hc.read()
    assert hc.gain_convention == 'divide', "The calibration gain convention in {} is not the HERA standard 'divide'.".format(redcal_file)

    # Initialize full-size, totally-flagged abscal gain/flag/etc. dictionaries
    abscal_gains = copy.deepcopy(rc_gains)
    abscal_flags = {ant: np.ones_like(rf) for ant, rf in rc_flags.items()}
    abscal_chisq_per_ant = {ant: np.zeros_like(rq) for ant, rq in rc_quals.items()}  # this stays zero, as it's not particularly meaningful
    abscal_chisq = {pol: np.zeros_like(rtq) for pol, rtq in rc_tot_qual.items()}

    # match times to narrow down model_files
    matched_model_files = sorted(set(match_times(data_file, model_files, filetype='uvh5')))
    if len(matched_model_files) == 0:
        echo("No model files overlap with data files in LST. Result will be fully flagged.", verbose=verbose)
    else:
        echo("The following model files overlap with data files in LST:\n" + "\n".join(matched_model_files), verbose=verbose)
        hd = io.HERAData(data_file)
        hdm = io.HERAData(matched_model_files)
        if hc.gain_scale is not None and hc.gain_scale.lower() != "uncalib":
            warnings.warn(f"Warning: Overwriting redcal gain_scale of {hc.gain_scale} with model gain_scale of {hdm.vis_units}", RuntimeWarning)
        hc.gain_scale = hdm.vis_units  # set vis_units of hera_cal based on model files.
        hd_autos = io.HERAData(raw_auto_file)
        assert hdm.x_orientation.lower() == hd.x_orientation.lower(), 'Data x_orientation, {}, does not match model x_orientation, {}'.format(hd.x_orientation.lower(), hdm.x_orientation.lower())
        assert hc.x_orientation.lower() == hd.x_orientation.lower(), 'Data x_orientation, {}, does not match redcal x_orientation, {}'.format(hd.x_orientation.lower(), hc.x_orientation.lower())
        pol_load_list = [pol for pol in hd.pols if split_pol(pol)[0] == split_pol(pol)[1]]

        # get model bls and antpos to use later in baseline matching
        model_bls = hdm.bls
        model_antpos = hdm.data_antpos
        if len(matched_model_files) > 1:  # in this case, it's a dictionary
            model_bls = list(set([bl for bls in list(hdm.bls.values()) for bl in bls]))
            model_antpos = {ant: pos for antpos in hdm.data_antpos.values() for ant, pos in antpos.items()}

        # match integrations in model to integrations in data
        all_data_times, all_data_lsts = get_all_times_and_lsts(hd, solar_horizon=data_solar_horizon, unwrap=True)
        all_model_times, all_model_lsts = get_all_times_and_lsts(hdm, solar_horizon=model_solar_horizon, unwrap=True)
        d2m_time_map = get_d2m_time_map(all_data_times, all_data_lsts, all_model_times, all_model_lsts, extrap_limit=extrap_limit)

        # group matched time indices for partial I/O
        matched_tinds = [tind for tind, time in enumerate(hd.times) if time in d2m_time_map and d2m_time_map[time] is not None]
        if len(matched_tinds) > 0:
            tind_groups = np.array([matched_tinds])  # just load a single group
            if nInt_to_load is not None:  # split up the integrations to load nInt_to_load at a time
                tind_groups = np.split(matched_tinds, np.arange(nInt_to_load, len(matched_tinds), nInt_to_load))

            # loop over polarizations
            for pol in pol_load_list:
                echo('\n\nNow calibrating ' + pol + '-polarization...', verbose=verbose)
                ants = [ant for ant in abscal_gains if join_pol(ant[1], ant[1]) == pol]

                # figure out which baselines to load from the data and the model and their correspondence (if one or both is redundantly averaged)
                (data_bl_to_load,
                 model_bl_to_load,
                 data_to_model_bl_map) = match_baselines(hd.bls, model_bls, hd.data_antpos, model_antpos=model_antpos, pols=[pol],
                                                         data_is_redsol=data_is_redsol, model_is_redundant=model_is_redundant,
                                                         tol=tol, min_bl_cut=min_bl_cut, max_bl_cut=max_bl_cut, verbose=verbose)
                if (len(data_bl_to_load) == 0) or (len(model_bl_to_load) == 0):
                    echo("No baselines in the data match baselines in the model. Results for this polarization will be fully flagged.", verbose=verbose)
                else:
                    # loop over groups of time indices
                    for tinds in tind_groups:
                        echo('\n    Now calibrating times ' + str(hd.times[tinds[0]])
                             + ' through ' + str(hd.times[tinds[-1]]) + '...', verbose=verbose)

                        # load data and apply calibration (unless data_is_redsol, so it's already redcal'ed)
                        data, flags, nsamples = hd.read(times=hd.times[tinds], bls=data_bl_to_load)
                        rc_gains_subset = {k: rc_gains[k][tinds, :] for k in ants}
                        rc_flags_subset = {k: rc_flags[k][tinds, :] for k in ants}
                        if not data_is_redsol:  # data is raw, so redundantly calibrate it
                            calibrate_in_place(data, rc_gains_subset, data_flags=flags, cal_flags=rc_flags_subset)

                        if not np.all(list(flags.values())):
                            # load model and rephase
                            model_times_to_load = [d2m_time_map[time] for time in hd.times[tinds]]
                            model, model_flags, _ = io.partial_time_io(hdm, np.unique(model_times_to_load), bls=model_bl_to_load)
                            if not np.array_equal(model_times_to_load, model.times):
                                # if multiple data times map to a single model time, this expands the model to match the data in time
                                model.select_or_expand_times(model_times_to_load)
                                model_flags.select_or_expand_times(model_times_to_load)
                            model_blvecs = {bl: model.antpos[bl[0]] - model.antpos[bl[1]] for bl in model.keys()}
                            utils.lst_rephase(model, model_blvecs, model.freqs, data.lsts - model.lsts,
                                              lat=hdm.telescope_location_lat_lon_alt_degrees[0], inplace=True)

                            # Flag frequencies and times in the data that are entirely flagged in the model
                            model_flag_waterfall = np.all([f for f in model_flags.values()], axis=0)
                            for k in flags.keys():
                                flags[k] += model_flag_waterfall

                            # get the relative wgts for each piece of data
                            auto_bls = [join_bl(ant, ant) for ant in rc_gains if join_bl(ant, ant)[2] == pol]
                            autocorrs, auto_flags, _ = hd_autos.read(times=hd.times[tinds], bls=auto_bls)
                            calibrate_in_place(autocorrs, rc_gains_subset, data_flags=auto_flags, cal_flags=rc_flags_subset)

                            # use data_to_model_bl_map to rekey model. Does not copy to save memory.
                            model = DataContainer({bl: model[data_to_model_bl_map[bl]] for bl in data})
                            model_flags = DataContainer({bl: model_flags[data_to_model_bl_map[bl]] for bl in data})

                            # build data weights based on inverse noise variance and nsamples and flags
                            data_wgts = build_data_wgts(flags, nsamples, model_flags, autocorrs, auto_flags,
                                                        times_by_bl=hd.times_by_bl, df=np.median(np.ediff1d(data.freqs)),
                                                        data_is_redsol=data_is_redsol, gain_flags=rc_flags_subset, antpos=hd.data_antpos)

                            # run absolute calibration to get the gain updates
                            delta_gains = post_redcal_abscal(model, data, data_wgts, rc_flags_subset, edge_cut=edge_cut, tol=tol,
                                                             phs_max_iter=phs_max_iter, phs_conv_crit=phs_conv_crit, verbose=verbose, use_abs_amp_lincal=not(skip_abs_amp_lincal))

                            # abscal autos, rebuild weights, and generate abscal Chi^2
                            calibrate_in_place(autocorrs, delta_gains)
                            chisq_wgts = build_data_wgts(flags, nsamples, model_flags, autocorrs, auto_flags,
                                                         times_by_bl=hd.times_by_bl, df=np.median(np.ediff1d(data.freqs)),
                                                         data_is_redsol=data_is_redsol, gain_flags=rc_flags_subset, antpos=hd.data_antpos)
                            total_qual, nObs, quals, nObs_per_ant = utils.chisq(data, model, chisq_wgts,
                                                                                gain_flags=rc_flags_subset, split_by_antpol=True)

                            # update results
                            for ant in ants:
                                # new gains are the product of redcal gains and delta gains from abscal
                                abscal_gains[ant][tinds, :] = rc_gains_subset[ant] * delta_gains[ant]
                                # new flags are the OR of redcal flags and times/freqs totally flagged in the model
                                abscal_flags[ant][tinds, :] = rc_flags_subset[ant] + model_flag_waterfall
                            for antpol in total_qual.keys():
                                abscal_chisq[antpol][tinds, :] = total_qual[antpol] / nObs[antpol]  # Note, not normalized for DoF
                                abscal_chisq[antpol][tinds, :][~np.isfinite(abscal_chisq[antpol][tinds, :])] = 0.

        # impose a single reference antenna on the final antenna solution
        if refant is None:
            refant = pick_reference_antenna(abscal_gains, abscal_flags, hc.freqs, per_pol=True)
        rephase_to_refant(abscal_gains, refant, flags=abscal_flags, propagate_refant_flags=True)

    # flag any nans, infs, etc.
    for ant in abscal_gains:
        abscal_flags[ant][~np.isfinite(abscal_gains[ant])] = True
        abscal_gains[ant][~np.isfinite(abscal_gains[ant])] = 1.0 + 0.0j

    # Save results to disk
    hc.update(gains=abscal_gains, flags=abscal_flags, quals=abscal_chisq_per_ant, total_qual=abscal_chisq)
    hc.quality_array[np.isnan(hc.quality_array)] = 0
    hc.total_quality_array[np.isnan(hc.total_quality_array)] = 0
    hc.history += utils.history_string(add_to_history)
    hc.write_calfits(output_file, clobber=clobber)
    if write_delta_gains:
        hcdelta = copy.deepcopy(hc)
        # set delta gains to be the ratio between abscal gains (phased to refant) and redcal gains
        delta_gains = {ant: abscal_gains[ant] / rc_gains[ant] for ant in abscal_gains}
        hcdelta.update(gains=delta_gains)
        assert output_file_delta is not None, "output_file_delta must be specified if write_delta_gains=True"
        hcdelta.write_calfits(output_file_delta, clobber=True)
    return hc


def multiply_gains(gain_file_1, gain_file_2, output_file, clobber=False, divide_gains=False):
    """Multiply gains from two files.

    Flags are always Ord
    total quality array is set to None
    quality array is set to Nan since it's a required Parameter.

    Parameters
    ----------
    gain_file_1: str
        path to first gain file to multiply
    gain_file_2: str
        path to second gain file to multiply
    output_file: str
        path to output file.
    clobber: bool, optional
        overwrite existing output_file
        default is False.
    divide_gains: bool, optional
        divide gain 1 by gain 2. Note -- This is not related to the gain convention.
        Just whether we want to take the ratio of two gains we input or their product.
        default is False.
    """
    hc1 = io.HERACal(gain_file_1)
    hc1.read()
    hc2 = io.HERACal(gain_file_2)
    hc2.read()
    if divide_gains:
        hc1.gain_array /= hc2.gain_array
    else:
        hc1.gain_array *= hc2.gain_array
    hc1.flag_array = hc1.flag_array | hc2.flag_array
    hc1.total_quality_array = None
    hc1.quality_array[:] = np.nan
    hc1.write_calfits(output_file, clobber=clobber)


def multiply_gains_argparser():
    ap = argparse.ArgumentParser(description="Command-line drive script to multiply two gains together.")
    ap.add_argument("gain_file_1", type=str, help="Path to first gain to multiply.")
    ap.add_argument("gain_file_2", type=str, help="Path to second gain to multiply.")
    ap.add_argument("output_file", type=str, help="Path to write out multiplied gains.")
    ap.add_argument("--divide_gains", default=False, action="store_true", help="divide gain 1 by gain 2 instead of multiplying.")
    ap.add_argument("--clobber", default=False, action="store_true", help="overwrite any existing output files.")
    return ap


def run_model_based_calibration(data_file, model_file, output_filename, auto_file=None, precalibration_gain_file=None,
                                inflate_model_by_redundancy=False, constrain_model_to_data_ants=False,
                                clobber=False, tol=1e-6, max_iter=10,
                                refant=None, ant_threshold=0.0, no_ampcal=False, no_phscal=False,
                                dly_lincal=False, spoof_missing_channels=False,
                                verbose=False, **dlycal_kwargs):
    """Driver function for model based calibration including i/o

    Solve for gain parameters based on a foreground model.

    Parameters
    ----------
    data_file: str
        path to pyuvdata visibility file.
        Or UVData object containing data to be calibrated.
        flags from this datafile are used for gains.
    model_file: str
        path to pyuvdata model file.
        Or UVData object containing model to calibrate against.
    output_filename: str
        path to output calibration file.
    auto_file: str, optional
        path to file containing autocorrelations and nsamples for inverse
        Or UVData object containing autocorrelations to use as calibration weights.
        variance weights. Default None -> use binary flag weights in calibration.
    precalibration_gain_file: str, optional
        Path to a gain file to apply to data before running calibration
        default is None.
    inflate_model_by_redundancy: bool, optional
        expand model to match the redundant groups present in data file.
        default is False.
        Should be set to True if a redundant model is supplied or bad things
        will happen!
    constrain_model_to_data_ants: bool, optional
        before inflating by redundancy, downselect array antennas in model to only
        include antennas in the data. This avoids inflating model to full HERA array
        which is memory inefficient for analyses using a small fraction of the array
        but will break if the redundant baselines are keyed to antennas that are not
        present in the data so only use this if you are confident that this is the case.
        default is False.
    clobber: bool, optional
        overwrite outputs if True.
    tol: float, optional
        perform calibration loop until differences in gain solutions are at this level.
    max_iter: int, optional
        maximum number of iterations to perform
        default is 10
    refant: tuple, optional
        referance antenna in form of (antnum, polstr).
        Default is None -> automatically select a refant.
    ant_threshold: float, optional
        threshold of flags in frequency and time for a given antenna to completely
        flag this antenna rom calibration.
    no_ampcal: bool, optional
        skip amplitude calibration (only calibrate phases)
        default is False.
    no_phscal: bool, optional
        skip phase calibration (only calibrate amplitudes)
        default is False.
    dly_lincal: bool, optional
        perform initial delay calibration before abscal
        default is False.
    spoof_missing_channels: bool, optional
        insert flagged gains in any frequency discontinunities.
        default is False.
    verbose: bool, optional
        lots of outputs.
    **dlycal_kwargs
        keyword args for abscal.delay_lincal
    """
    hdd = io.to_HERAData(data_file, filetype="uvh5")
    if hasattr(hdd, "data_array") and hdd.data_array is not None:
        data, data_flags, data_nsamples = hdd.build_datacontainers()
    else:
        data, data_flags, data_nsamples = hdd.read()

    hdm = io.to_HERAData(model_file, filetype="uvh5")
    if not hasattr(hdm, "data_array") or hdm.data_array is None:
        hdm.read()

    if precalibration_gain_file is not None:
        uvc_precal = io.HERACal(precalibration_gain_file)
        gains_precal, flags_precal, _, _ = uvc_precal.read()
        # apply precal gains to data
        calibrate_in_place(data=data, data_flags=data_flags,
                           new_gains=gains_precal, cal_flags=flags_precal)

    # expand hdm by redundancy
    if inflate_model_by_redundancy:
        if constrain_model_to_data_ants:
            # In order to avoid inflating to full interferometer dataset
            # which will be unecessary for many earlier analyses
            # prune model antennas to only include data antennas in hdd
            # this will only work if the antennas in the baseline keys for each redundant group
            # in the model are also present in the dataset so only use this if you are confident
            # that this is the case.
            all_data_ants = np.unique(np.hstack([hdd.ant_1_array, hdd.ant_2_array]))
            all_model_ants = np.unique(np.hstack([hdm.ant_1_array, hdm.ant_2_array]))
            assert np.all([ant in all_data_ants for ant in all_model_ants]), "All model antennas must be present in data if constrain_model_to_data_ants is True!"
            hdm.select(antenna_nums=all_data_ants,
                       keep_all_metadata=False)
        hdm.inflate_by_redundancy()
        # also make sure to only include baselines present in hdd.
        hdm.select(bls=hdd.bls)
        hdm._determine_blt_slicing()
        hdm._determine_pol_indexing()

    model, model_flags, model_nsamples = hdm.build_datacontainers()

    data_ant_flags = synthesize_ant_flags(data_flags, threshold=ant_threshold)
    model_ant_flags = synthesize_ant_flags(model_flags, threshold=ant_threshold)

    lst_center = hdd.lsts[hdd.Ntimes // 2] * 12 / np.pi
    field_str = 'LST={lst_center:%.2f} hrs'

    if refant is None:
        # dummy refant. Could be completely flagged for all we know.
        refant_init = str(hdd.ant_1_array[0])
    else:
        refant_init = str(refant)
    # initialize HERACal to store new cal solutions
    hc = UVCal()
    hc = hc.initialize_from_uvdata(uvdata=hdd, gain_convention='divide', cal_style='sky',
                                   ref_antenna_name=refant_init, sky_catalog=f'{model_file}',
                                   metadata_only=False, sky_field=field_str, cal_type='gain',
                                   future_array_shapes=True)
    hc = io.to_HERACal(hc)
    hc.update(flags=data_ant_flags)
    # generate cal object from model to hold model flags.
    hcm = UVCal()
    hcm = hcm.initialize_from_uvdata(uvdata=hdm, gain_convention='divide', cal_style='sky',
                                     ref_antenna_name=refant_init, sky_catalog=f'{model_file}',
                                     metadata_only=False, sky_field=field_str, cal_type='gain',
                                     future_array_shapes=True)
    hcm = io.to_HERACal(hcm)
    hcm.update(flags=model_ant_flags)
    # init all gains to unity.
    hcm.gain_array[:] = 1. + 0.j
    hc.gain_array[:] = 1. + 0.j
    # set calibration flags to be or of data and model flags
    hc.flag_array = hc.flag_array | hcm.flag_array

    abscal_gains, abscal_flags, _, _ = hc.build_calcontainers()

    # load in weights if supplied.
    if auto_file is not None:

        hda = io.to_HERAData(auto_file, filetype="uvh5")
        bls = [ap for ap in hda.get_antpairs() if ap[0] == ap[1]]
        auto_data, auto_flags, auto_nsamples = hda.read(bls=bls)
        # generate wgts from autocorrelations
        wgts = build_data_wgts(data_flags=data_flags, data_nsamples=data_nsamples, model_flags=model_flags,
                               autocorrs=auto_data, auto_flags=auto_flags)
    else:
        # if no autocorrelation file supplied, use nsamples weights times
        # or of data and model flags
        wgts = DataContainer({k: (~data_flags[k]).astype(np.float64)
                             * (~model_flags[k]).astype(np.float64)
                             * data_nsamples[k] for k in data_flags})

    # select refant if not specified.
    if refant is None:
        refant = pick_reference_antenna(abscal_gains, abscal_flags, hc.freqs, per_pol=True)
    hc.ref_antenna_name = str(refant)
    hcm.ref_antenna_name = str(refant)

    my_abscal = AbsCal(data=data, model=model, wgts=wgts, freqs=data.freqs)

    # initialize convergence metric to be very large. This will always be greater then
    # all non-infinite tols provided so calibration loop will run at least once.
    delta = np.inf

    abscal_gains_iteration = {k: np.ones((hc.Ntimes, hc.Nfreqs), dtype=complex) for k in abscal_gains}

    # find initial starting point with lincal before performing abscal.
    if dly_lincal:
        my_abscal.delay_lincal(**dlycal_kwargs)
        abscal_gains_iteration = merge_gains([my_abscal.ant_dly_phi_gain, my_abscal.ant_dly_gain])

        apply_cal.calibrate_in_place(data=my_abscal.data, new_gains=abscal_gains_iteration)
        abscal_gains = merge_gains([abscal_gains, abscal_gains_iteration])

    niter = 0
    while delta > tol and niter < max_iter:
        for k in abscal_gains_iteration:
            abscal_gains_iteration[k][:] = 1. + 0j
        if not no_ampcal:
            my_abscal.amp_logcal(verbose=verbose)
            abscal_gains_iteration = merge_gains([abscal_gains_iteration, my_abscal.ant_eta_gain])

        if not no_phscal:
            my_abscal.phs_logcal(verbose=verbose)
            abscal_gains_iteration = merge_gains([abscal_gains_iteration, my_abscal.ant_phi_gain])

        # phase to refant.
        rephase_to_refant(abscal_gains_iteration, refant, flags=abscal_flags, propagate_refant_flags=True)
        # update abscal gains with iteration.
        abscal_gains_new = merge_gains([abscal_gains, abscal_gains_iteration])
        maxvals = [np.max(np.abs(abscal_gains_new[k][np.invert(abscal_flags[k])]
                   - abscal_gains[k][np.invert(abscal_flags[k])])) for k in abscal_gains if np.any(~abscal_flags[k])]
        if len(maxvals) > 0:
            delta = np.max(maxvals)
        else:
            echo(f"All gains are flagged! Exiting...", verbose=verbose)
            break

        for k in abscal_gains:
            abscal_gains[k] = abscal_gains_new[k]

        # use hcm to apply iteration gains.
        hcm.update(gains=abscal_gains_iteration)

        # apply iteration gains to my_abscal.data
        apply_cal.calibrate_in_place(data=my_abscal.data, new_gains=abscal_gains_iteration)
        niter += 1
        if delta <= tol:
            echo(f"Convergence achieved after {niter} iterations with max delta of {delta}", verbose=verbose)
        if niter == max_iter - 1 and delta > tol:
            echo(f"Convergence not achieved but max_iter of {max_iter} reached. delta={delta}.", verbose=verbose)

    # multiply gains by precal gains
    if precalibration_gain_file is not None:
        abscal_gains = merge_gains([abscal_gains, gains_precal])

    # update the calibration array.
    hc.update(gains=abscal_gains)

    hc.write(output_filename, clobber=clobber, spoof_missing_channels=spoof_missing_channels)


def post_redcal_abscal_argparser():
    ''' Argparser for commandline operation of hera_cal.abscal.post_redcal_abscal_run() '''
    a = argparse.ArgumentParser(description="Command-line drive script for post-redcal absolute calibration using hera_cal.abscal module")
    a.add_argument("data_file", type=str, help="string path to raw uvh5 visibility file or omnical_visibility solution")
    a.add_argument("redcal_file", type=str, help="string path to calfits file that serves as the starting point of abscal")
    a.add_argument("model_files", type=str, nargs='+', help="list of string paths to externally calibrated data or reference solution. Strings \
                                                             must be sortable to produce a chronological list in LST (wrapping over 2*pi is OK)")
    a.add_argument("--raw_auto_file", default=None, type=str, help="path to data file that contains raw autocorrelations for all antennas in redcal_file. \
                                                                  If not provided, data_file is used instead. Required if data_is_redsol is True.")
    a.add_argument("--data_is_redsol", default=False, action="store_true", help="If True, data_file only contains unique, redcal'ed visibilities.")
    a.add_argument("--model_is_redundant", default=False, action="store_true", help="If True, then model_files only containe unique visibilities.")
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
    a.add_argument("--skip_abs_amp_lincal", default=False, action="store_true", help="finish calibration with an unbiased amplitude lincal step")
    a.add_argument("--write_delta_gains", default=False, action="store_true", help="Write degenerate abscal component of gains separately.")
    a.add_argument("--output_file_delta", type=str, default=None, help="Filename to write delta gains too.")
    args = a.parse_args()
    return args


def model_calibration_argparser():
    '''Argparser for commandline operation of run_model_based_calibration'''
    ap = argparse.ArgumentParser(description="Command-line drive script for model based calibration")
    ap.add_argument("data_file", type=str, help="string path to data file to calibrate.")
    ap.add_argument("model_file", type=str, help="string path to model file to calibrate.")
    ap.add_argument("output_filename", type=str, help="string path to output calfits file to store gains.")
    ap.add_argument("--auto_file", type=str, default=None, help="string path to file containing autocorrelations to use as inverse variants weights. If not specified, use uniform weights with flags.")
    ap.add_argument("--clobber", default=False, action="store_true", help="overwrite output calfits if it already exists.")
    ap.add_argument("--tol", default=1e-6, type=float, help="number of calibration rounds to run.")
    ap.add_argument("--inflate_model_by_redundancy", default=False, action="store_true", help="If redundant model file is provided, inflate it!")
    ap.add_argument("--constrain_model_to_data_ants", default=False, action="store_true", help="before inflating by redundancy, downselect array antennas in model to only \
                                                                                                include antennas in the data. This avoids inflating model to full HERA array \
                                                                                                which is memory inefficient for analyses using a small fraction of the array \
                                                                                                but will break if the redundant baselines are keyed to antennas that are not \
                                                                                                present in the data so only use this if you are confident that this is the case. \
                                                                                                Default is False.")
    ap.add_argument("--precalibration_gain_file", default=None, type=str, help="Path to a gain file to apply to data before running calibration \
                                                                                default is None.")
    ap.add_argument("--verbose", default=False, action="store_true", help="lots of outputs.")
    ap.add_argument("--no_ampcal", default=False, action="store_true", help="disable amp_cal")
    ap.add_argument("--no_phscal", default=False, action="store_true", help="disable phs_cal")
    ap.add_argument("--dly_lincal", default=False, action="store_true", help="dly lincal to find starting point.")
    ap.add_argument("--spoof_missing_channels", default=False, action="store_true", help="Fill in missing channels with flagged gains. This ensures compatibility with calfits which only supports unifrom spaced channels.")
    return ap
