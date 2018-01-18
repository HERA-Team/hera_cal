"""
abscal_funcs.py
---------------

Self contained routines
used by abscal.py
"""
import os
import sys
from collections import OrderedDict as odict
import copy
import argparse
import functools
import numpy as np
from pyuvdata import UVCal, UVData
from pyuvdata import utils as uvutils
from hera_cal import omni, utils, firstcal, cal_formats, redcal
from hera_cal.datacontainer import DataContainer
from scipy import signal
from scipy import interpolate
import linsolve
import itertools
import operator
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import gaussian_process

def abs_amp_logcal(model, data, wgts=None, verbose=True):
    """
    calculate absolute (array-wide) gain amplitude scalar
    with a linear solver using the logarithmically linearized equation:

    ln|V_ij,xy^model / V_ij,xy^data| = eta_x + eta_y

    where {i,j} index antenna numbers and {x,y} index polarizations
    of the i-th and j-th antennas respectively.

    Parameters:
    -----------
    model : visibility data of refence model, type=dictionary
            keys are antenna-pair + polarization tuples, Ex. (1, 2, 'xx').
            values are complex ndarray visibilities.
            these must be 2D arrays, with [0] axis indexing time
            and [1] axis indexing frequency.

    data : visibility data of measurements, type=dictionary
           keys are antenna pair + pol tuples (must match model), values are
           complex ndarray visibilities matching shape of model

    wgts : weights of data, type=dictionry, [default=None]
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
    ydata = odict([(k, np.log(np.abs(model[k]/data[k]))) for k in keys])

    # make weights if None
    if wgts is None:
        wgts = odict()
        for i, k in enumerate(keys):
            wgts[k] = np.ones_like(ydata[k], dtype=np.float)

    # fill nans and infs
    fill_dict_nans(ydata, wgts=wgts, nan_fill=0.0, inf_fill=0.0)

    # setup linsolve equations
    # a{} is a dummy variable to prevent linsolve from overwriting repeated measurements 
    eqns = odict([(k, "a{}*eta_{}+a{}*eta_{}".format(i, k[-1][0], i, k[-1][1])) for i, k in enumerate(keys)])
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


def TT_phs_logcal(model, data, antpos, wgts=None, verbose=True, zero_psi=False,
                  merge_pols=False):
    """
    calculate overall gain phase and gain phase Tip-Tilt slopes (East-West and North-South)
    with a linear solver applied to the logarithmically linearized equation:

    angle(V_ij,xy^model / V_ij,xy^data) = angle(g_i_x) * angle(conj(g_j_y))
                                        = psi_x - psi_y + PHI^ew_x*r_i^ew + PHI^ns_x*r_i^ns
                                          - PHI^ew_y*r_j^ew - PHI^ns_y*r_j^ns

    where psi is the overall gain phase across the array [radians] for x and y polarizations,
    and PHI^ew, PHI^ns are the gain phase slopes across the east-west and north-south axes
    of the array in units of [radians / meter], where x and y denote the pol of the i-th and j-th
    antenna respectively. The phase slopes are polarization independent by default (1pol & 2pol cal),
    but can be merged with the merge_pols parameter (4pol cal). r_i is the antenna position vector
    of the i^th antenna.

    Parameters:
    -----------
    model : visibility data of refence model, type=dictionary
            keys are antenna-pair + polarization tuples, Ex. (1, 2, 'xx').
            values are complex ndarray visibilities.
            these must 2D arrays, with [0] axis indexing time
            and [1] axis indexing frequency.

    data : visibility data of measurements, type=dictionary
           keys are antenna pair + pol tuples (must match model), values are
           complex ndarray visibilities matching shape of model

    wgts : weights of data, type=dictionry, [default=None]
           keys are antenna pair + pol tuples (must match model), values are real floats
           matching shape of model and data

    antpos : antenna position vectors, type=dictionary
          keys are antenna integers, values are 2D
          antenna vectors in meters (preferably centered at center of array),
          with [0] index containing east-west separation and [1] index north-south separation

    zero_psi : set psi to be identically zero in linsolve eqns, type=boolean, [default=False]

    merge_pols : type=boolean, even if multiple polarizations are present in data, make free
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
    ants = np.unique(antpos.keys())

    # angle of phs ratio is ydata independent variable
    # angle after divide
    ydata = odict([(k, np.angle(model[k]/data[k])) for k in keys])

    # make weights if None
    if wgts is None:
        wgts = odict()
        for i, k in enumerate(keys):
            wgts[k] = np.ones_like(ydata[k], dtype=np.float)

    # fill nans and infs
    fill_dict_nans(ydata, wgts=wgts, nan_fill=0.0, inf_fill=0.0)

    # setup antenna position terms
    r_ew = odict(map(lambda a: (a, "r_ew_{}".format(a)), ants))
    r_ns = odict(map(lambda a: (a, "r_ns_{}".format(a)), ants))

    # setup linsolve equations
    if merge_pols:
        eqns = odict([(k, "psi_{}*a1 - psi_{}*a2 + Phi_ew*{} + Phi_ns*{} - Phi_ew*{} - Phi_ns*{}"
                    "".format(k[2][0], k[2][1], r_ew[k[0]], r_ns[k[0]], r_ew[k[1]], r_ns[k[1]])) for i, k in enumerate(keys)])
    else:
        eqns = odict([(k, "psi_{}*a1 - psi_{}*a2 + Phi_ew_{}*{} + Phi_ns_{}*{} - Phi_ew_{}*{} - Phi_ns_{}*{}"
                    "".format(k[2][0], k[2][1], k[2][0], r_ew[k[0]], k[2][0], r_ns[k[0]], k[2][1],
                              r_ew[k[1]], k[2][1], r_ns[k[1]])) for i, k in enumerate(keys)])

    # set design matrix entries
    ls_design_matrix = odict(map(lambda a: ("r_ew_{}".format(a), antpos[a][0]), ants))
    ls_design_matrix.update(odict(map(lambda a: ("r_ns_{}".format(a), antpos[a][1]), ants)))

    if zero_psi:
        ls_design_matrix.update({"a1":0.0, "a2":0.0})
    else:
        ls_design_matrix.update({"a1":1.0, "a2":1.0})

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

    ln|V_ij,xy^model / V_ij,xy^data| = ln|g_i_x| + ln|g_j_y|
                                     = eta_i_x + eta_j_y

    where {x,y} represent the polarization of the i-th and j-th antenna
    respectively.

    Parameters:
    -----------
    model : visibility data of refence model, type=dictionary
            keys are antenna-pair + polarization tuples, Ex. (1, 2, 'xx').
            values are complex ndarray visibilities.
            these must 2D arrays, with [0] axis indexing time
            and [1] axis indexing frequency.

    data : visibility data of measurements, type=dictionary
           keys are antenna pair + pol tuples (must match model), values are
           complex ndarray visibilities matching shape of model

    wgts : weights of data, type=dictionry, [default=None]
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
    ydata = odict([(k, np.log(np.abs(model[k]/data[k]))) for k in keys])

    # make weights if None
    if wgts is None:
        wgts = odict()
        for i, k in enumerate(keys):
            wgts[k] = np.ones_like(ydata[k], dtype=np.float)

    # fill nans and infs
    fill_dict_nans(ydata, wgts=wgts, nan_fill=0.0, inf_fill=0.0)

    # setup linsolve equations
    eqns = odict([(k, "eta_{}_{} + eta_{}_{}".format(k[0], k[-1][0], k[1], k[-1][1])) for i, k in enumerate(keys)])
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


def phs_logcal(model, data, wgts=None, verbose=True):
    """
    calculate per-antenna gain phase via the 
    logarithmically linearized equation

    angle(V_ij,xy^model / V_ij,xy^data) = angle(g_i_x) - angle(g_j_y)
                                        = phi_i_x - phi_j_y

    where {x,y} represent the pol of the i-th and j-th antenna respectively.

    Parameters:
    -----------
    model : visibility data of refence model, type=dictionary
            keys are antenna-pair + polarization tuples, Ex. (1, 2, 'xx').
            values are complex ndarray visibilities.
            these must 2D arrays, with [0] axis indexing time
            and [1] axis indexing frequency.

    data : visibility data of measurements, type=dictionary
           keys are antenna pair + pol tuples (must match model), values are
           complex ndarray visibilities matching shape of model

    wgts : weights of data, type=dictionry, [default=None]
           keys are antenna pair + pol tuples (must match model), values are real floats
           matching shape of model and data

    Output:
    -------
    fit : dictionary containing phi_i = angle(g_i) for each antenna
    """
    echo("...configuring linsolve data for phs_logcal", verbose=verbose)

    # get keys from match between data and model dictionary
    keys = sorted(set(model.keys()) & set(data.keys()))

    # angle of visibility ratio is ydata independent variable
    ydata = odict([(k, np.angle(model[k]/data[k])) for k in keys])

    # make weights if None
    if wgts is None:
        wgts = odict()
        for i, k in enumerate(keys):
            wgts[k] = np.ones_like(ydata[k], dtype=np.float)

    # fill nans and infs
    fill_dict_nans(ydata, wgts=wgts, nan_fill=0.0, inf_fill=0.0)

    # setup linsolve equations
    eqns = odict([(k, "phi_{}_{} - phi_{}_{}".format(k[0], k[-1][0], k[1], k[-1][1])) for i, k in enumerate(keys)])
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


def delay_lincal(model, data, wgts=None, df=9.765625e4, solve_offsets=True, medfilt=True, kernel=(1, 5),
                 verbose=True, time_ax=0, freq_ax=1):
    """
    Solve for per-antenna delay according to the equation

    delay(V_ij,xy^model / V_ij,xy^data) = delay(g_i_x) - delay(g_j_y)

    Can also solve for the phase offset per antenna per polarization.

    Parameters:
    -----------
    model : visibility data of refence model, type=dictionary
            keys are antenna-pair + polarization tuples, Ex. (1, 2, 'xx').
            values are complex ndarray visibilities.
            these must be at least 2D arrays, with [0] axis indexing time
            and [1] axis indexing frequency. If the arrays are 3D arrays, the [2] axis
            should index polarization, in which case the key loses its pol entry, Ex. (1, 2).

    data : visibility data of measurements, type=dictionary
           keys are antenna pair + pol tuples (must match model), values are
           complex ndarray visibilities matching shape of model

    wgts : weights of data, type=dictionry, [default=None]
           keys are antenna pair + pol tuples (must match model), values are real floats
           matching shape of model and data

    df : type=float, frequency spacing between channels in Hz

    solve_offsets : type=boolean, if True, setup a system of linear equations for per-antenna phase offset
                    and solve.

    medfilt : type=boolean, median filter visiblity ratio before taking fft

    kernel : type=tuple, dtype=int, kernel for multi-dimensional median filter

    time_ax : type=int, time axis of model and data

    freq_ax : type=int, freq axis of model and data

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
        for i, k in enumerate(keys): wgts[k] = np.ones_like(data[k], dtype=np.float)

    # median filter and FFT to get delays
    ratio_delays = []
    ratio_offsets = []
    for i, k in enumerate(keys):
        ratio = model[k]/data[k]

        # replace nans
        nan_select = np.isnan(ratio)
        ratio[nan_select] = 0.0
        wgts[k][nan_select] = 0.0

        # replace infs
        inf_select = np.isinf(ratio)
        ratio[inf_select] = 0.0
        wgts[k][inf_select] = 0.0

        # get delays
        dly, offset = fft_dly(ratio, wgts=wgts[k], df=df, medfilt=medfilt, kernel=kernel, time_ax=time_ax, freq_ax=freq_ax, solve_phase=solve_offsets)
        ratio_delays.append(dly)
        ratio_offsets.append(offset)
       
    ratio_delays = np.array(ratio_delays)
    ratio_offsets = np.array(ratio_offsets)

    # form ydata
    ydata = odict(zip(keys, ratio_delays))

    # setup linsolve equation dictionary
    eqns = odict([(k, 'tau_{}_{} - tau_{}_{}'.format(k[0], k[-1][0], k[1], k[-1][1])) for i, k in enumerate(keys)])

    # setup design matrix dictionary
    ls_design_matrix = odict()

    # setup linsolve data dictionary
    ls_data = odict([(eqns[k], ydata[k]) for i, k in enumerate(keys)])

    # setup linsolve and run
    sol = linsolve.LinearSolver(ls_data, **ls_design_matrix)
    echo("...running linsolve", verbose=verbose)
    fit = sol.solve()
    echo("...finished linsolve", verbose=verbose)

    # solve for offsets
    if solve_offsets:
        # setup linsolve parameters
        ydata = odict(zip(keys, ratio_offsets))
        eqns = odict([(k, 'phi_{}_{} - phi_{}_{}'.format(k[0], k[-1][0], k[1], k[-1][1])) for i, k in enumerate(keys)])
        ls_design_matrix = odict()
        ls_data = odict([(eqns[k], ydata[k]) for i, k in enumerate(keys)])
        sol = linsolve.LinearSolver(ls_data, **ls_design_matrix)
        echo("...running linsolve", verbose=verbose)
        offset_fit = sol.solve()
        echo("...finished linsolve", verbose=verbose)
        fit.update(offset_fit)

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
    keys = sorted(reduce(operator.and_, map(lambda g: set(g.keys()), gains)))

    # form merged_gains dict
    merged_gains = odict()

    # iterate over keys
    for i, k in enumerate(keys):
        merged_gains[k] = reduce(operator.mul, map(lambda g: g.get(k, 1.0), gains))

    return merged_gains

def apply_gains(data, gains, gain_convention='multiply'):
    """
    apply gain solutions to data

    Parameters:
    -----------
    data : type=dictionary, holds complex visibility data.
        keys are antenna-pair tuples + pol tuples.
        values are ndarray complex visibility data.

    gains : type=dictionary, holds complex, per-antenna gain data.
            keys are antenna integer + gain pol tuples, Ex. (1, 'x').
            values are complex ndarrays
            with shape matching data's visibility ndarrays

        Optionally, can be a tuple holding multiple gains dictionaries
        that will all be multiplied together.

    Output:
    -------
    new_data : type=dictionary, data with gains applied
    """
    # form new dictionary
    new_data = odict()

    # get keys
    keys = data.keys()

    # merge gains if multiple gain dictionaries fed
    if type(gains) == list or type(gains) == tuple or type(gains) == np.ndarray:
        gains = merge_gains(gains)

    # iterate over keys:
    for i, k in enumerate(keys):
        # get gain keys
        g1 = (k[0], k[-1][0])
        g2 = (k[1], k[-1][1])

        # form visbility gain product
        vis_gain = gains[g1] * np.conj(gains[g2])

        # apply to data
        if gain_convention == "multiply":
            new_data[k] = data[k] * vis_gain

        elif gain_convention == "divide":
            new_data[k] = data[k] / vis_gain

    return new_data


def data_key_to_array_axis(data, key_index, array_index=-1, avg_dict=None):
    """
    move an index of data.keys() into the data axes

    Parameters:
    -----------
    data : type=dictionary, complex visibility data with
        antenna-pair + pol tuples for keys, in AbsCal dictionary format.
    
    key_index : integer, index of keys to consolidate into data arrays

    array_index : integer, which axes of data arrays to append to

    avg_dict : dictionary, a dictionary with same keys as data
        that will have its data arrays averaged along key_index

    Result:
    -------
    new_data : dictionary, complex visibility data
        with key_index of keys moved into the data arrays

    new_avg_dict : copy of avg_dict. Only returned if avg_dict is not None.

    popped_keys : unique list of keys moved into data array axis
    """
    # instantiate new data object
    new_data = odict()
    new_avg = odict()

    # get keys
    keys = data.keys()

    # sort keys across key_index
    key_sort = np.argsort(np.array(keys, dtype=np.object)[:, key_index])
    keys = map(lambda i: keys[i], key_sort)
    popped_keys = np.unique(np.array(keys, dtype=np.object)[:, key_index])

    # get new keys
    new_keys = map(lambda k: k[:key_index] + k[key_index+1:], keys)
    new_unique_keys = []

    # iterate over new_keys
    for i, nk in enumerate(new_keys):
        # check for unique keys
        if nk in new_unique_keys:
            continue
        new_unique_keys.append(nk)

        # get all instances of redundant keys
        ravel = map(lambda k: k == nk, new_keys)

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
    data : dictionary, complex visibility data with
        antenna-pair (+ pol + other) tuples for keys
    
    array_index : integer, which axes of data arrays
        to extract from arrays and move into keys

    array_keys : list, list of new key from array elements. must have length
        equal to length of data_array along axis array_index

    key_index : integer, index within the new set of keys to insert array_keys

    copy_dict : dictionary, a dictionary with same keys as data
        that will have its data arrays copied along array_keys

    Output:
    -------
    new_data : dictionary, complex visibility data
        with array_index of data arrays extracted and moved
        into a unique set of keys

    new_copy : dictionary, copy of copy_dict
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


def UVData2AbsCalDict(filenames, pol_select=None, pop_autos=True, return_meta=False, filetype='miriad'):
    """
    turn pyuvdata.UVData objects or miriad filenames 
    into the datacontainer dictionary form that AbsCal requires. This format is
    keys as antennas-pair + polarization format, Ex. (1, 2, 'xx')
    and values as 2D complex ndarrays with [0] axis indexing time and [1] axis frequency.

    Parameters:
    -----------
    filenames : list of either strings to miriad filenames or list of UVData instances
                to concatenate into a single dictionary

    pol_select : list of polarization strings to keep

    pop_autos : boolean, if True: remove autocorrelations

    return_meta : boolean, if True: also return antenna and unique frequency and LST arrays

    filetype : string, filetype of data if filenames is a string

    Output:
    -------
    if return_meta is True:
        (data, flags, antpos, ants, freqs, times, pols)
    else:
        (data, flags)

    data : dictionary containing baseline-pol complex visibility data
    flags : dictionary containing data flags
    antpos : dictionary containing antennas numbers as keys and position vectors
    ants : ndarray containing unique antennas
    freqs : ndarray containing frequency channels (Hz)
    times : ndarray containing LST bins of data (radians)
    """
    # check filenames is a list
    if type(filenames) is not list and type(filenames) is not np.ndarray:
        if type(filenames) is str:
            uvd = UVData()
            suffix = os.path.splitext(filenames)[1]
            if filetype == 'uvfits' or suffix == '.uvfits':
                uvd.read_uvfits(filenames)
                uvd.unphase_to_drift()
            elif filetype == 'miriad':
                uvd.read_miriad(filenames)

        else:
            uvd = filenames
    else:
        if type(filenames[0]) is str:
            uvd = UVData()
            suffix = os.path.splitext(filenames[0])[1]
            if filetype == 'uvfits' or suffix == '.uvfits':
                uvd.read_uvfits(filenames)
                uvd.unphase_to_drift()
            elif filetype == 'miriad':
                uvd.read_miriad(filenames)
        else:
            uvd = reduce(operator.add, filenames)

    # load data
    d, f = firstcal.UVData_to_dict([uvd])

    # pop autos
    if pop_autos:
        for i, k in enumerate(d.keys()):
            if k[0] == k[1]:
                d.pop(k)
                f.pop(k)

    # turn into datacontainer
    data, flags = DataContainer(d), DataContainer(f)

    # get meta
    if return_meta:
        freqs = np.unique(uvd.freq_array)
        times = np.unique(uvd.lst_array)
        antpos, ants = uvd.get_ENU_antpos(center=True, pick_data_ants=True)
        antpos = odict(zip(ants, antpos))
        pols = uvd.polarization_array
        return data, flags, antpos, ants, freqs, times, pols
    else:
        return data, flags


def fft_dly(vis, wgts=None, df=9.765625e4, medfilt=True, kernel=(1, 11), time_ax=0, freq_ax=1,
            solve_phase=True):
    """
    get delay of visibility across band using FFT w/ tukey window
    and quadratic fit to delay peak.

    Parameters:
    -----------
    vis : ndarray of visibility data, dtype=complex, shape=(Ntimes, Nfreqs, +)

    df : frequency channel width in Hz

    medfilt : boolean, median filter data before fft

    kernel : size of median filter kernel along (time, freq) axes

    time_ax : time axis of data

    freq_ax : frequency axis of data

    Output: (dlys, phi)
    -------
    dlys : ndarray containing delay for each integration

    phi : ndarray containing phase of delay mode for each integration
    """
    # get array params
    Nfreqs = vis.shape[freq_ax]
    Ntimes = vis.shape[time_ax]

    # get wgt
    if wgts is None:
        wgts = np.ones_like(vis, dtype=np.float)

    # smooth via median filter
    kernel += tuple(np.ones((vis.ndim - len(kernel)), np.int))
    if medfilt:
        vis_smooth = signal.medfilt(np.real(vis), kernel_size=kernel) + 1j*signal.medfilt(np.imag(vis), kernel_size=kernel)
    else:
        vis_smooth = vis

    # fft
    window = np.repeat(signal.windows.tukey(Nfreqs)[np.newaxis], Ntimes, axis=time_ax)
    window = np.moveaxis(window, 0, time_ax)
    window *= wgts
    vfft = np.fft.fft(vis_smooth * window, axis=freq_ax)

    # get argmax of abs
    amp = np.abs(vfft)
    argmax = np.moveaxis(np.argmax(amp, axis=freq_ax)[np.newaxis], 0, freq_ax)

    # get delays
    fftfreqs = np.fft.fftfreq(Nfreqs, 1)
    dfreq = np.median(np.diff(fftfreqs))
    dlys = fftfreqs[argmax]

    # get peak shifts, and add to dlys
    def get_peak(amp, max_ind):
        Nchan = len(amp)
        y = np.concatenate([amp,amp,amp])
        max_ind += Nchan
        y = y[max_ind-1:max_ind+2]
        r = np.abs(np.diff(y))
        r = r[0] / r[1]
        peak = 0.5 * (r-1) / (r+1)
        return peak

    peak_shifts = np.array([get_peak(np.take(amp, i, axis=time_ax), np.take(argmax, i, axis=time_ax)[0]) for i in range(Ntimes)])
    dlys += np.moveaxis(peak_shifts.reshape(-1, 1) * dfreq, 0, time_ax)

    phi = None
    if solve_phase:
        # get phase offsets by interpolating real and imag component of FFT
        vfft_real = []
        vfft_imag = []
        for i, a in enumerate(argmax):
            # get real and imag of each argmax
            real = np.take(vfft.real, i, axis=time_ax)
            imag = np.take(vfft.imag, i, axis=time_ax)

            # wrap around
            real = np.concatenate([real, real, real])
            imag = np.concatenate([imag, imag, imag])
            a += Nfreqs

            # add interpolation component
            rl = interpolate.interp1d(np.arange(Nfreqs*3), real)(a + peak_shifts[i])
            im = interpolate.interp1d(np.arange(Nfreqs*3), imag)(a + peak_shifts[i])

            # insert into arrays
            vfft_real.append(rl)
            vfft_imag.append(im)

        vfft_real = np.moveaxis(np.array(vfft_real), 0, time_ax)
        vfft_imag = np.moveaxis(np.array(vfft_imag), 0, time_ax)
        vfft_interp = vfft_real + 1j*vfft_imag
        phi = np.angle(vfft_interp)
        dlys /= df

    return dlys, phi


def wiener(data, window=(5, 15), noise=None, medfilt=True, medfilt_kernel=(1,13), array=False):
    """
    wiener filter complex visibility data. this might be used in constructing
    model reference. See scipy.signal.wiener for details on method.

    Parameters:
    -----------
    data : type=dictionary, AbsCal-format dictionary holding complex visibility data
        unelss array is True

    window : type=tuple, wiener-filter window along each axis of data

    noise : type=float, estimate of noise. if None will estimate itself

    medfilt : type=bool, if True, median filter data before wiener filtering

    medfilt_kernel : type=tuple, median filter kernel along each axis of data

    array : type=boolean, if True, feeding a single ndarray, rather than a dictionary

    Output: (new_data)
    -------
    new_data type=dictionary, AbsCal-format dictionary holding new visibility data
    """
    # check if data is an array
    if array:
        data = {'arr': data}

    new_data = odict()
    for i, k in enumerate(data.keys()):
        real = np.real(data[k])
        imag = np.imag(data[k])
        if medfilt:
            real = signal.medfilt(real, kernel_size=medfilt_kernel)
            imag = signal.medfilt(imag, kernel_size=medfilt_kernel)

        new_data[k] = signal.wiener(real, mysize=window, noise=noise) + \
                      1j*signal.wiener(imag, mysize=window, noise=noise)

    if array:
        return new_data['arr']
    else:
        return new_data


def interp2d_vis(data, data_times, data_freqs, model_times, model_freqs,
                 kind='cubic', fill_value=0, zero_tol=1e-10, flag_extrapolate=True,
                 presmooth=False, bounds_error=True, **wiener_kwargs):
    """
    interpolate complex visibility data onto the time & frequency basis of
    a model visibility.

    Parameters:
    -----------
    data : type=dictionary, holds complex visibility data
        keys are antenna-pair + pol tuples, values are 2d complex visibility data
        with shape (Ntimes, Nfreqs)

    data_times : 1D array of the data time axis, dtype=float, shape=(Ntimes,)

    data_freqs : 1D array of the data freq axis, dtype=float, shape=(Nfreqs,)

    model_times : 1D array of the model time axis, dtype=float, shape=(Ntimes,)

    model_freqs : 1D array of the model freq axis, dtype=float, shape=(Nfreqs,)

    kind : kind of interpolation method, type=str, options=['linear', 'cubic', ...]
        see scipy.interpolate.interp2d for details

    fill_value : values to put for interpolation points outside training set
        if None, values are extrapolated

    zero_tol : for amplitudes lower than this tolerance, set real and imag components to zero

    bounds_error : type=boolean, if True, raise ValueError when extrapolating. If False, extrapolate.

    flag_extrapolate : flag extrapolated data if True

    presmooth : type=boolean, if True perform the following steps on real and imag separately:
        smooth data, perform interpolation, find difference between original and interpolation
        and add difference to original data.

    Output: (data, flags)
    -------
    data : interpolated data, type=dictionary
    flags : flags associated with data, type=dictionary
    """
    # make flags
    new_data = odict()
    flags = odict()

    # loop over keys
    for i, k in enumerate(data.keys()):
        # interpolate real and imag separately
        d = data[k]
        real = np.real(d)
        imag = np.imag(d)

        if presmooth:
            # wiener smooth
            _real = copy.copy(real)
            _imag = copy.copy(imag)

            d = wiener(d, array=True, **wiener_kwargs)
            real = np.real(d)
            imag = np.imag(d)

        # interpolate
        interp_real = interpolate.interp2d(data_freqs, data_times, real,
                                           kind=kind, fill_value=np.nan, bounds_error=bounds_error)(model_freqs, model_times)
        interp_imag = interpolate.interp2d(data_freqs, data_times, imag,
                                           kind=kind, fill_value=np.nan, bounds_error=bounds_error)(model_freqs, model_times)
        if presmooth:
            # get differences and add
            real_diff = interp_real - real
            imag_diff = interp_imag - imag
            interp_real = _real + real_diff
            interp_imag = _imag + imag_diff

        # set flags
        f = np.zeros_like(interp_real, dtype=float)
        if flag_extrapolate:
            f[np.isnan(interp_real) + np.isnan(interp_imag)] = 1.0
        interp_real[np.isnan(interp_real)] = fill_value
        interp_imag[np.isnan(interp_imag)] = fill_value

        # force things near amplitude of zero to (positive) zero
        zero_select = np.isclose(np.sqrt(interp_real**2 + interp_imag**2), 0.0, atol=zero_tol)
        interp_real[zero_select] *= 0.0 * interp_real[zero_select]
        interp_imag[zero_select] *= 0.0 * interp_imag[zero_select]

        # rejoin
        new_data[k] = interp_real + 1j*interp_imag
        flags[k] = f

    return DataContainer(new_data), DataContainer(flags)


def gains2calfits(calfits_fname, abscal_gains, freq_array, time_array, pol_array,
                  gain_convention='multiply', overwrite=False, **kwargs):
    """
    write out gain_array in calfits file format.

    Parameters:
    -----------
    calfits_fname : string, path and filename to output calfits file

    abscal_gains : dictionary, antenna integer as key, ndarray complex gain
        as values with shape (Ntimes, Nfreqs, Npols)

    freq_array : ndarray, frequency array of data in Hz

    time_array : ndarray, time array of data in Julian Date

    pol_array : ndarray, polarization array of data, in 'x' or 'y' form. 

    kwargs : additional kwargs for meta in cal_formats.HERACal(meta, gains)
    """
    # ensure pol is string
    int2pol = {-5: 'x', -6: 'y'}
    if pol_array.dtype == np.int:
        pol_array = list(pol_array)
        for i, p in enumerate(pol_array):
            pol_array[i] = int2pol[p]

    # reconfigure gain dictionary into HERACal gain dictionary
    heracal_gains = {}
    for i, p in enumerate(pol_array):
        pol_dict = {}
        for j, k in enumerate(abscal_gains.keys()):
            pol_dict[k] = abscal_gains[k][:, :, i]
        heracal_gains[p] = pol_dict

    # configure meta
    inttime = np.median(np.diff(time_array)) * 24. * 3600.
    meta = {'times':time_array, 'freqs':freq_array, 'inttime':inttime, 'gain_convention': gain_convention}
    meta.update(**kwargs)

    # convert to UVCal
    uvc = cal_formats.HERACal(meta, heracal_gains)

    # write to file
    if os.path.exists(calfits_fname) is True and overwrite is False:
        echo("{} already exists, not overwriting...".format(calfits_fname))
    else:
        echo("saving {}".format(calfits_fname))
        uvc.write_calfits(calfits_fname, clobber=overwrite)

def fill_dict_nans(data, wgts=None, nan_fill=None, inf_fill=None, array=False):
    """
    take a dictionary and re-fill nan and inf ndarray values.

    Parameters:
    -----------
    data : type=dictionary, visibility dictionary in AbsCal dictionary format

    wgts : type=dictionary, weights dictionary matching shape of data to also fill

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


def echo(message, type=0, verbose=True):
    if verbose:
        if type == 0:
            print(message)
        elif type == 1:
            print('')
            print(message)
            print("-"*40)


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
        self.tol = tol

    def __repr__(self):
        return self.label

    @property
    def bl(self):
        return np.array(map(float, self.label.split(':')))

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
            equiv = bool(reduce(operator.mul, map(lambda x: np.isclose(*x, atol=tol), zip(self.bl, B2.bl))))
            dot = np.dot(self.unit, B2.unit)
            if equiv:
                return True
            # check conjugation
            elif np.isclose(np.arccos(dot), np.pi, atol=tol/self.len) or (dot < -1.0):
                return 'conjugated'
            # else return False
            else:
                return False
        else:
            return False


def match_red_baselines(data, data_antpos, model, model_antpos, tol=1.0, verbose=True):
    """
    match data baseline keys to model baseline keys based on positional redundancy

    Parameters:
    -----------
    data : type=dictionary, data dictionary holding complex visibilities.
        must conform to AbsCal dictionary format.

    data_antpos : type=dictionary, dictionary holding antennas positions for data dictionary
        keys are antenna integers, values are ndarrays of position vectors in meters

    model : type=dictionary, model dictionary holding complex visibilities
        must conform to AbsCal dictionary format.

    model_antpos : type=dictionary, dictionary holding antennas positions for model dictionary
        same format as data_antpos

    tol : type=float, baseline match tolerance in units of baseline vectors (e.g. meters)

    Output: (data)
    -------
    data : type=dictionary, dictionary holding complex visibilities from data that
        had matching baselines to model
    """
    # create baseline keys for model
    model_keys = model.keys()
    model_bls = np.array(map(lambda k: Baseline(model_antpos[k[1]] - model_antpos[k[0]], tol=tol), model_keys))

    # create baseline keys for data
    data_keys = data.keys()
    data_bls = np.array(map(lambda k: Baseline(data_antpos[k[1]] - data_antpos[k[0]], tol=tol), data_keys))

    # iterate over data baselines
    new_data = odict()
    for i, bl in enumerate(data_bls):
        # compre bl to all model_bls
        comparison = np.array(map(lambda mbl: bl == mbl, model_bls), np.str)

        # get matches
        matches = np.where((comparison=='True')|(comparison=='conjugated'))[0]

        # check for matches
        if len(matches) == 0:
            echo("found zero matches in model for data {}".format(data_keys[i]), verbose=verbose)
            continue
        else:
            if len(matches) > 1:
                echo("found more than 1 match in model to data {}: {}".format(data_keys[i], map(lambda j: model_keys[j], matches)), verbose=verbose)
            # assign to new_data
            if comparison[matches[0]] == 'True':
                new_data[model_keys[matches[0]]] = data[data_keys[i]]
            elif comparison[matches[0]] == 'conjugated':
                new_data[model_keys[matches[0]]] = np.conj(data[data_keys[i]])

    return DataContainer(new_data)


def smooth_solutions(Xdata, Ydata, Xpred=None, gains=True, kind='gp', n_restart=3, degree=1, ls=10.0, return_model=False):
    """
    Smooth gain (or calibration) solutions across time and/or frequency.

    Parameters:
    -----------
    Xdata : ndarray containing flattened x-values for regression, type=ndarray, dtype=float
            shape=(Ndata, Ndimensions)
  
    Ydata : type=dictionary, dict holding gain or calibration solutions with values as
            ndarray containing complex gains or real calibration solution across
            time and/or frequency (single pol) with shape=(Ndata, 1)
  
    Xpred : ndarray containing flattened x-values for prediction. if None will use Xdata.
            type=ndarray, dtype=float, shape=(Ndata, Ndimensions)

    gains : type=boolean, if True input Ydata is complex gains,
            if False input Ydata is real calibration solutions


    Output:
    -------

    """
    # get number of dimensions
    ndim = Xtrain.shape[1]

    if kind == 'poly':
        # robust linear regression via sklearn
        model = make_pipeline(PolynomialFeatures(degree), linear_model.RANSACRegressor())
        model.fit(Xtrain, ytrain)

    elif kind == 'gp':
        # construct GP kernel
        ls = np.array([1e-1 for i in range(ndim)])
        kernel = 1.0**2 * gaussian_process.kernels.RBF(ls, np.array([1e-2, 1e1])) + \
                 gaussian_process.kernels.WhiteKernel(1e-4, (1e-8, 1e1))
        model = gaussian_process.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restart)
        # fit GP
        model.fit(Xtrain, ytrain)

    ypred = model.Predict(X)


def abscal_run(data_files, model_files, unravel_pol=False, unravel_freq=False, unravel_time=False, verbose=True,
               save=False, calfits_fname=None, output_gains=False, overwrite=False, zero_psi=False,
               smooth=False, **kwargs):
    """
    run AbsCal on a set of contiguous data files

    Parameters:
    -----------
    data_files : path(s) to data miriad files, type=list
        a list of one or more miriad file(s) containing complex
        visibility data

    model_files : path(s) to miriad files(s), type=list
        a list of one or more miriad files containing complex visibility data
        that *overlaps* the time and frequency range of data_files

    output_gains : boolean, if True: return AbsCal gains
    """
    # load data
    echo("loading data files", type=1, verbose=verbose)
    for i, df in enumerate(data_files):
        echo("loading {}".format(df), type=0, verbose=verbose)
        uv = UVData()
        try:
            uv.read_miriad(df)
        except:
            uv.read_uvfits(df)
            uv.unphase_to_drift()
        if i == 0:
            uvd = uv
        else:
            uvd += uv

    data, flags, pols = UVData2AbsCalDict([uvd])
    for i, k in enumerate(data.keys()):
        if k[0] == k[1]:
            data.pop(k)
    for i, k in enumerate(flags.keys()):
        if k[0] == k[1]:
            flags.pop(k)

    # get data params
    data_times = np.unique(uvd.lst_array)
    data_freqs = uvd.freq_array.squeeze()
    data_pols = uvd.polarization_array

    # load weights
    wgts = copy.deepcopy(flags)
    for k in wgts.keys():
        wgts[k] = (~wgts[k]).astype(np.float)

    # load antenna positions and make antpos and baseline dictionary
    antpos, ants = uvd.get_ENU_antpos(center=True, pick_data_ants=True)
    antpos = odict(map(lambda x: (x, antpos[ants.tolist().index(x)]), ants))
    bls = odict([((x[0], x[1]), antpos[x[1]] - antpos[x[0]]) for x in data.keys()])

    # load models
    for i, mf in enumerate(model_files):
        echo("loading {}".format(mf), type=0, verbose=verbose)
        uv = UVData()
        try:
            uv.read_miriad(mf)
        except:
            uv.read_uvfits(mf)
            uv.unphase_to_drift()
        if i == 0:
            uvm = uv
        else:
            uvm += uv

    model, mflags, mpols = UVData2AbsCalDict([uvm], pol_select=pols)
    for i, k in enumerate(model.keys()):
        if k[0] == k[1]:
            model.pop(k)

    # get model params
    model_times = np.unique(uvm.lst_array)
    model_freqs = uvm.freq_array.squeeze()

    # align model freq-time axes to data axes
    model = interp_model(model, model_times, model_freqs, data_times, data_freqs,
                        kind='cubic', fill_value=0, zero_tol=1e-6)

    # check if model has only unique baseline data
    # this is the case if, for example, the model Nbls in less than the data Nbls
    if uvm.Nbls < uvd.Nbls:
        # try to expand model data into redundant baseline groups
        model = mirror_data_to_red_bls(model, bls, antpos, tol=2.0)

        # ensure data keys match model keys
        for i, k in enumerate(data):
            if k not in model:
                data.pop(k)

    # run abscal
    AC = AbsCal(model, data, wgts=wgts, antpos=antpos, freqs=data_freqs, times=data_times, pols=data_pols)
    AC.run(unravel_pol=unravel_pol, unravel_freq=unravel_freq, unravel_time=unravel_time,
           verbose=verbose, gains2dict=True, zero_psi=zero_psi, **kwargs)

    # smooth gains
    if smooth:
        AC.smooth_params()

    # make gains
    AC.make_gains()

    # write to file
    if save:
        if calfits_fname is None:
            calfits_fname = os.path.basename(data_file) + '.abscal.calfits'
        AC.write_calfits(calfits_fname, overwrite=overwrite, verbose=verbose)

    if output_gains:
        return AC.gain_array

def abscal_arg_parser():
    a = argparse.ArgumentParser()
    a.add_argument("--data_files", type=str, nargs='*', help="list of miriad files of data to-be-calibrated.", required=True)
    a.add_argument("--model_files", type=str, nargs='*', default=[], help="list of data-overlapping miriad files for visibility model.", required=True)
    a.add_argument("--calfits_fname", type=str, default=None, help="name of output calfits file.")
    a.add_argument("--overwrite", default=False, action='store_true', help="overwrite output calfits file if it exists.")
    a.add_argument("--silence", default=False, action='store_true', help="silence output from abscal while running.")
    a.add_argument("--zero_psi", default=False, action='store_true', help="set overall gain phase 'psi' to zero in linsolve equations.")
    return a


def avg_data_across_red_bls(data, bls, antpos, flags=None, broadcast_flags=True, median=False, tol=0.5,
                            mirror_red_data=False):
    """
    Given complex visibility data spanning one or more redundant
    baseline groups, average redundant visibilities and write to file
    """
    # get data keys
    keys = data.keys()

    # get data, flags and ants
    data = copy.deepcopy(data)
    ants = np.unique(np.concatenate(keys))
    if flags is None:
        flags = copy.deepcopy(data)
        for k in flags.keys(): flags[k] = np.zeros_like(flags[k]).astype(np.bool)

    # get redundant baselines
    reds = compute_reds(bls, antpos, tol=tol)

    # make red_data dictionary
    red_data = odict()
    red_flags = odict()

    # iterate over reds
    for i, bl_group in enumerate(reds):
        # average redundant baseline group
        if median:
            d = np.nanmedian(map(lambda k: data[k], bl_group), axis=0)
        else:
            d = np.nanmean(map(lambda k: data[k], bl_group), axis=0)

        # assign to red_data
        for j, key in enumerate(sorted(bl_group)):
            red_data[key] = copy.copy(d)
            if mirror_red_data is False:
                break

        # assign flags
        if broadcast_flags:
            red_flags[key] = np.max(map(lambda k: flags[k], bl_group), axis=0).astype(np.bool)
        else:
            red_flags[key] = np.min(map(lambda k: flags[k], bl_group), axis=0).astype(np.bool)

    # get red_data keys
    red_keys = red_data.keys()

    return red_data, red_flags, red_keys


def avg_file_across_red_bls(data_fname, outdir=None, output_fname=None,
                            write_miriad=True, output_data=False, overwrite=False, 
                            verbose=True, **kwargs):
    """
    """
    # check output file
    if outdir is None:
        outdir = os.path.dirname(data_fname)
    if output_fname is None:
        output_fname = os.path.basename(data_fname) + 'M'
    output_fname = os.path.join(outdir, output_fname)
    if os.path.exists(output_fname) is True and overwrite is False:
        raise IOError("{} exists, not overwriting".format(output_fname))

    if type(data_fname) == str:
        uvd = UVData()
        uvd.read_miriad(data_fname)

    # get data
    data, flags, pols = UVData2AbsCalDict([uvd])

    # get antpos and baselines
    antpos, ants = uvd.get_ENU_antpos()
    antpos = dict(zip(ants, antpos))
    bls = data.keys()

    # avg data across reds
    red_data, red_flags, red_keys = avg_data_across_red_bls(data, bls, antpos, **kwargs)
    uvd_data = np.array(map(lambda k: red_data[k], red_keys))
    uvd_flags = np.array(map(lambda k: red_flags[k], red_keys))
    uvd_bls = np.array(red_keys)
    blts_select = np.array(map(lambda k: uvd.antpair2ind(*k), uvd_bls)).reshape(-1)
    Nbls = len(uvd_bls)
    Nblts = len(blts_select)
    uvd_bls = np.array(map(lambda k: uvd.baseline_to_antnums(k), uvd.baseline_array[blts_select]))

    # resort data
    uvd_data = uvd_data.reshape(-1, 1, uvd.Nfreqs, uvd.Npols)
    uvd_flags = uvd_flags.reshape(-1, 1, uvd.Nfreqs, uvd.Npols)

    # write to file
    if write_miriad:
        echo("saving {}".format(output_fname), verbose=verbose)
        uvd.data_array = uvd_data
        uvd.flag_array = uvd_flags
        uvd.time_array = uvd.time_array[blts_select]
        uvd.lst_array = uvd.lst_array[blts_select]
        uvd.baseline_array = uvd.baseline_array[blts_select]
        uvd.ant_1_array = uvd_bls[:, 0]
        uvd.ant_2_array = uvd_bls[:, 1]
        uvd.uvw_array = uvd.uvw_array[blts_select, :]
        uvd.nsample_array = np.ones_like(uvd.data_array, dtype=np.float)
        uvd.Nbls = Nbls
        uvd.Nblts = Nblts
        uvd.zenith_dec = uvd.zenith_dec[blts_select]
        uvd.zenith_ra = uvd.zenith_ra[blts_select]
        uvd.write_miriad(output_fname, clobber=True)

    # output data
    if output_data:
        return red_data, red_flags, red_keys


def mirror_data_to_red_bls(data, antpos, tol=2.0, pol=None, weights=False):
    """
    Given unique baseline data (like omnical model visibilities),
    copy the data over to all other baselines in the same redundant group.
    If weights==True, treat data as a wgts dictionary and multiply values
    by their redundant baseline weighting.

    Parameters:
    -----------
    data : data dictionary in hera_cal.DataContainer form

    antpos : type=dictionary, antenna positions dictionary
                keys are antenna integers, values are ndarray baseline vectors.

    tol : type=float, redundant baseline distance tolerance in units of baseline vectors

    pol : type=str, polarization in data.keys()

    weights : type=bool, if True, treat data as a wgts dictionary and multiply by redundant weighting.

    Output: (red_data)
    -------
    red_data : type=dictionary, data dictionary in AbsCal form, with unique baseline data
                distributed to redundant baseline groups.
    if weights == True:
        red_data is a real-valued wgts dictionary with redundant baseline weighting muliplied in.
    """
    # get data keys
    keys = data.keys()

    # get polarizations in data
    pols = data.pols()

    # get redundant baselines
    reds = redcal.get_reds(antpos, bl_error_tol=tol, pols=pols)

    # make red_data dictionary
    red_data = odict()

    # iterate over data keys
    for i, k in enumerate(keys):

        # find which bl_group this key belongs to
        match = np.array(map(lambda r: k in r, reds))
        conj_match = np.array(map(lambda r: data._switch_bl(k) in r, reds))

        # if no match, just copy data over to red_data
        if True not in match and True not in conj_match:
            red_data[k] = copy.copy(data[k])

        else:
            # iterate over matches
            for j, (m, cm) in enumerate(zip(match, conj_match)):
                if weights:
                    # if weight dictionary, add repeated baselines in inverse quadrature
                    if m == True:
                        if (k in red_data) == False:
                            red_data[k] = copy.copy(data[k])
                            red_data[k][red_data[k].astype(np.bool)] = red_data[k][red_data[k].astype(np.bool)] + len(reds[j]) - 1
                        else:
                            red_data[k][red_data[k].astype(np.bool)] = red_data[k][red_data[k].astype(np.bool)] + len(reds[j])
                    elif cm == True:
                        if (k in red_data) == False:
                            red_data[k] = copy.copy(data[k])
                            red_data[k][red_data[k].astype(np.bool)] = red_data[k][red_data[k].astype(np.bool)] + len(reds[j]) - 1
                        else:
                            red_data[k][red_data[k].astype(np.bool)] = red_data[k][red_data[k].astype(np.bool)] + len(reds[j])
                else:
                    # if match, insert all bls in bl_group into red_data
                    if m == True:
                        for bl in reds[j]:
                            red_data[bl] = copy.copy(data[k])
                    elif cm == True:
                        for bl in reds[j]:
                            red_data[bl] = np.conj(data[k])

    # re-sort, inverse quad if weights
    if weights:
        red_data = odict([(k, red_data[k]**(-2)) for k in sorted(red_data)])
    else:
        red_data = odict([(k, red_data[k]) for k in sorted(red_data)])


    return DataContainer(red_data)

'''
def lst_align(data_fname, model_fnames=None, dLST=0.00299078, output_fname=None, outdir=None,
              overwrite=False, verbose=True, write_miriad=True, output_data=False,
              match='nearest', filetype='miriad', **interp2d_kwargs):
    """
    Interpolate complex visibilities to align time integrations with an LST grid.
    If output_fname is not provided, write interpolated data as
    input filename + "L.hour.decimal" miriad file. The LST grid can be created from
    scratch using the dLST parameter, or an LST grid can be imported from a model file.

    Parameters:
    -----------


    match : type=str, LST-bin matching method, options=['nearest','forward','backward']

    """
    # try to load model
    echo("loading models", verbose=verbose)
    if model_fnames is not None:
        uvm = UVData()
        # parse suffix
        if type(model_fnames) == np.str:
            suffix = os.path.splitext(model_fnames)[1]
        elif type(model_fnames) == list or type(model_fnames) == np.ndarray:
            suffix = os.path.splitext(model_fnames[0])[1]
        else:
            raise IOError("couldn't parse type of {}".format(model_fnames))
        if filetype == 'uvfits' or suffix == '.uvfits':
            uvm.read_uvfits(model_fnames)
            uvm.unphase_to_drift()
        elif filetype == 'miriad':
            uvm.read_miriad(model_fnames)
        # get meta data
        model_lsts = np.unique(uvm.lst_array) * 12 / np.pi
        model_freqs = np.unique(uvm.freq_array)
    else:
        # generate LST array
        model_lsts = np.arange(0, 24, dLST)
        model_freqs = None

    # load file
    echo("loading {}".format(data_fname), verbose=verbose)
    uvd = UVData()
    uvd.read_miriad(data_fname)

    # get data and metadata
    (data, flags, antpos, ants, data_freqs, data_lsts) = UVData2AbsCalDict(uvd, pop_autos=False, return_meta=True)
    data_lsts *= 12 / np.pi
    Ntimes = len(data_lsts)

    # get closest lsts
    sort = np.argsort(np.abs(model_lsts - data_lsts[0]))[:2]
    if match == 'nearest':
        start = sort[0]
    elif match == 'forward':
        start = np.max(sort)
    elif match == 'backward':
        start = np.min(sort)

    # create lst grid
    lst_indices = np.arange(start, start+Ntimes)
    model_lsts = model_lsts[lst_indices]

    # specify freqs
    if model_freqs is None:
        model_freqs = data_freqs
    Nfreqs = len(model_freqs)

    # interpolate data
    echo("interpolating data", verbose=verbose)
    interp_data, interp_flags = interp2d_vis(data, data_lsts, data_freqs, model_lsts, model_freqs, **interp2d_kwargs)
    Nbls = len(interp_data)

    # reorder into arrays
    uvd_data = np.array(interp_data.values())
    uvd_data = uvd_data.reshape(-1, 1, Nfreqs, 1)
    uvd_flags = np.array(map(lambda k: interp_flags[k], flags.keys())).astype(np.bool) + \
                np.array(map(lambda k: flags[k], flags.keys())).astype(np.bool) 
    uvd_flags = uvd_flags.reshape(-1, 1, Nfreqs, 1)
    uvd_keys = np.repeat(np.array(interp_data.keys()).reshape(-1, 1, 2), Ntimes, axis=1).reshape(-1, 2)
    uvd_bls = np.array(map(lambda k: uvd.antnums_to_baseline(k[0], k[1]), uvd_keys))
    uvd_times = np.array(map(lambda x: utils.JD2LST.LST2JD(x, np.median(np.floor(uvd.time_array)), uvd.telescope_location_lat_lon_alt_degrees[1]), model_lsts))
    uvd_times = np.repeat(uvd_times[np.newaxis], Nbls, axis=0).reshape(-1)
    uvd_lsts = np.repeat(model_lsts[np.newaxis], Nbls, axis=0).reshape(-1)
    uvd_freqs = model_freqs.reshape(1, -1)

    # assign to uvdata object
    uvd.data_array = uvd_data
    uvd.flag_array = uvd_flags
    uvd.baseline_array = uvd_bls
    uvd.ant_1_array = uvd_keys[:, 0]
    uvd.ant_2_array = uvd_keys[:, 1]
    uvd.time_array = uvd_times
    uvd.lst_array = uvd_lsts * np.pi / 12
    uvd.freq_array = uvd_freqs
    uvd.Nfreqs = Nfreqs

    # write miriad
    if write_miriad:
        # check output
        if outdir is None:
            outdir = os.path.dirname(data_fname)
        if output_fname is None:
            output_fname = os.path.basename(data_fname) + 'L.{:07.4f}'.format(model_lsts[0])
        output_fname = os.path.join(outdir, output_fname)
        if os.path.exists(output_fname) and overwrite is False:
            raise IOError("{} exists, not overwriting".format(output_fname))

        # write to file
        echo("saving {}".format(output_fname), verbose=verbose)
        uvd.write_miriad(output_fname, clobber=True)

    # output data and flags
    if output_data:
        return interp_data, interp_flags, model_lsts, model_freqs


def lstbin_arg_parser():
    a = argparse.ArgumentParser()
    a.add_argument("--data_files", type=str, nargs='*', help="list of miriad files of data to-be-calibrated.", required=True)
    a.add_argument("--model_files", type=str, nargs='*', default=[], help="list of data-overlapping miriad files for visibility model.", required=True)
    a.add_argument("--calfits_fname", type=str, default=None, help="name of output calfits file.")
    a.add_argument("--overwrite", default=False, action='store_true', help="overwrite output calfits file if it exists.")
    a.add_argument("--silence", default=False, action='store_true', help="silence output from abscal while running.")
    a.add_argument("--zero_psi", default=False, action='store_true', help="set overall gain phase 'psi' to zero in linsolve equations.")
    return a




'''
