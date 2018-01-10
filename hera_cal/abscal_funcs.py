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
from hera_cal import omni, utils, firstcal, cal_formats
from scipy import signal
from scipy import interpolate
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import gaussian_process
import linsolve
from astropy import stats as astats
import itertools


def abs_amp_lincal(model, data, wgts=None, verbose=True):
    """
    calculate absolute (array-wide) gain amplitude scalar
    with a linear solver using equation:

    |V_ij^model| = A * |V_ij^data|

    Parameters:
    -----------
    model : visibility data of refence model, type=dictionary
            keys are antenna-pair tuples.
            values are complex ndarray visibilities.
            these visibilities must be at least 2D arrays, with [0] axis indexing time
            and [1] axis indexing frequency. If the arrays are 3D arrays, the [2] axis
            should index polarization.

    data : visibility data of measurements, type=dictionary
           keys are antenna pair tuples (must match model), values are
           complex ndarray visibilities matching shape of model

    wgts : weights of data, type=dictionry, [default=None]
           keys are antenna pair tuples (must match model), values are real floats
           matching shape of model and data

    verbose : print output, type=boolean, [default=False]

    Output:
    -------
    fit : dictionary with 'amp' key for amplitude scalar, which has the same shape as
        the ndarrays in model
    """
    echo("...configuring linsolve data for abs_amp_lincal", verbose=verbose)

    # get keys from model dictionary
    keys = sorted(set(model.keys()) & set(data.keys()))

    # abs of amplitude ratio is ydata independent variable
    ydata = odict([(k, np.abs(model[k]/data[k])) for k in model.keys()])

    # make weights if None
    if wgts is None:
        wgts = copy.deepcopy(ydata)
        for i, k in enumerate(keys):
            wgts[k] = np.ones_like(ydata[k], dtype=np.float)

    # replace nans
    for i, k in enumerate(keys):
        nan_select = np.isnan(ydata[k])
        ydata[k][nan_select] = 0.0
        wgts[k][nan_select] = 0.0
        nan_select = np.isnan(model[k])
        model[k][nan_select] = 0.0
        wgts[k][nan_select] = 0.0

    # replace infs
    for i, k in enumerate(keys):
        inf_select = np.isinf(ydata[k])
        ydata[k][inf_select] = 0.0
        wgts[k][inf_select] = 0.0
        inf_select = np.isinf(model[k])
        model[k][inf_select] = 0.0
        wgts[k][inf_select] = 0.0

    # setup linsolve equations
    eqns = odict([(k, "a{}*amp".format(str(i))) for i, k in enumerate(keys)])
    ls_design_matrix = odict([("a{}".format(str(i)), 1.0) for i, k in enumerate(keys)])

    # setup linsolve dictionaries
    ls_data = odict([(eqns[k], ydata[k]) for i, k in enumerate(keys)])
    ls_wgts = odict([(eqns[k], wgts[k]) for i, k in enumerate(keys)])

    # setup linsolve and run
    sol = linsolve.LinearSolver(ls_data, wgts=ls_wgts, **ls_design_matrix)
    echo("...running linsolve", verbose=verbose)
    fit = sol.solve()
    echo("...finished linsolve", verbose=verbose)

    return fit


def TT_phs_logcal(model, data, bls, wgts=None, verbose=True, zero_psi=False):
    """
    calculate overall gain phase and gain phase Tip-Tilt slopes (EW and NS)
    with a linear solver applied to the logarithmically
    linearized equation:

    phi_ij^model = phi_ij^data + psi + PHI^x * b_ij^x + PHI^y * b_ij^y

    where psi is the overall gain phase across the array [radians],
    and PHI = <PHI^x, PHI^y> is the gain phase slopes across the
    x axis (east-west) and y axis (north-south) of the array respectively
    in units of [radians / meter].

    Parameters:
    -----------
    model : visibility data of refence model, type=dictionary
            keys are antenna-pair tuples, values are complex ndarray visibilities
            these visibilities must be at least 2D arrays, with [0] axis indexing time
            and [1] axis indexing frequency. If the arrays are 3D arrays, the [2] axis
            should index polarization.

    data : visibility data of measurements, type=dictionary
           keys are antenna pair tuples (must match model), values are
           complex ndarray visibilities matching shape of model

    bls : baseline vectors of antenna pairs, type=dictionary
          keys are antenna pair tuples (must match model), values are 2D or 3D ndarray
          baseline vectors in meters, with [0] index containing X (E-W) separation
          and [1] index Y (N-S) separation.

    wgts : weights of data, type=dictionry, [default=None]
           keys are antenna pair tuples (must match model), values are real floats
           matching shape of model and data

    verbose : print output, type=boolean, [default=False]

    zero_psi : set psi to be identically zero in linsolve eqns, type=boolean, [default=False]

    Output:
    -------
    fit : dictionary with psi key for overall gain phase and PHI array containing
        PHIx (x-axis phase slope) and PHIy (y-axis phase slope)
        which all have the same shape as the ndarrays in model
    """
    echo("...configuring linsolve data for TT_phs_logcal", verbose=verbose)

    # get keys from model dictionary
    keys = sorted(set(model.keys()) & set(data.keys()))

    # angle of phs ratio is ydata independent variable
    # angle after divide
    ydata = odict([(k, np.angle(model[k]/data[k])) for k in model.keys()])

    # make weights if None
    if wgts is None:
        wgts = copy.deepcopy(ydata)
        for i, k in enumerate(keys):
            wgts[k] = np.ones_like(ydata[k], dtype=np.float)

    # replace nans
    for i, k in enumerate(keys):
        nan_select = np.isnan(ydata[k])
        ydata[k][nan_select] = 0.0
        wgts[k][nan_select] = 0.0
        nan_select = np.isnan(model[k])
        model[k][nan_select] = 0.0
        wgts[k][nan_select] = 0.0

    # replace infs
    for i, k in enumerate(keys):
        inf_select = np.isinf(ydata[k])
        ydata[k][inf_select] = 0.0
        wgts[k][inf_select] = 0.0
        inf_select = np.isinf(model[k])
        model[k][inf_select] = 0.0
        wgts[k][inf_select] = 0.0

    # setup baseline terms
    bx = odict([(k, ["bx_{}_{}".format(k[0], k[1]), bls[k][0]]) for i, k in enumerate(keys)])
    by = odict([(k, ["by_{}_{}".format(k[0], k[1]), bls[k][1]]) for i, k in enumerate(keys)])

    # setup linsolve equations
    if zero_psi:
        eqns = odict([(k, "psi*0 + PHIx*{} + PHIy*{}".format(bx[k][0], by[k][0])) for i, k in enumerate(keys)])
    else:
        eqns = odict([(k, "psi + PHIx*{} + PHIy*{}".format(bx[k][0], by[k][0])) for i, k in enumerate(keys)])

    ls_design_matrix = odict(bx.values() + by.values())

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

    ln|V_ij^model| - ln|V_ij^data| = ln|g_i| + ln|g_j|

    Parameters:
    -----------
    model : visibility data of refence model, type=dictionary
            keys are antenna-pair tuples, values are complex ndarray visibilities
            these visibilities must be at least 2D arrays, with [0] axis indexing time
            and [1] axis indexing frequency. If the arrays are 3D arrays, the [2] axis
            should index polarization.

    data : visibility data of measurements, type=dictionary
           keys are antenna pair tuples (must match model), values are
           complex ndarray visibilities matching shape of model

    wgts : weights of data, type=dictionry, [default=None]
           keys are antenna pair tuples (must match model), values are real floats
           matching shape of model and data

    Output:
    -------
    fit : dictionary containing eta_i = ln|g_i| for each antenna
    """
    echo("...configuring linsolve data for amp_logcal", verbose=verbose)

    # get keys from model dictionary
    keys = sorted(set(model.keys()) & set(data.keys()))

    # difference of log-amplitudes is ydata independent variable
    ydata = odict([(k, np.log(np.abs(model[k]/data[k]))) for k in model.keys()])

    # make weights if None
    if wgts is None:
        wgts = copy.deepcopy(ydata)
        for i, k in enumerate(keys):
            wgts[k] = np.ones_like(ydata[k], dtype=np.float)

    # replace nans
    for i, k in enumerate(keys):
        nan_select = np.isnan(ydata[k])
        ydata[k][nan_select] = 0.0
        wgts[k][nan_select] = 0.0
        nan_select = np.isnan(model[k])
        model[k][nan_select] = 0.0
        wgts[k][nan_select] = 0.0

    # replace infs
    for i, k in enumerate(keys):
        inf_select = np.isinf(ydata[k])
        ydata[k][inf_select] = 0.0
        wgts[k][inf_select] = 0.0
        inf_select = np.isinf(model[k])
        model[k][inf_select] = 0.0
        wgts[k][inf_select] = 0.0

    # setup linsolve equations
    eqns = odict([(k, "eta{} + eta{}".format(k[0], k[1])) for i, k in enumerate(keys)])
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

    angle(V_ij^model) - angle(V_ij^data) = angle(g_i) - angle(g_j)

    Parameters:
    -----------
    model : visibility data of refence model, type=dictionary
            keys are antenna-pair tuples (ant1, ant2), or antenna-pair-pol tuples (ant1, ant2, pol).
            values are complex ndarray visibilities.
            these visibilities must be at least 2D arrays, with [0] axis indexing time
            and [1] axis indexing frequency. If the arrays are 3D arrays, the [2] axis
            should index polarization.

    data : visibility data of measurements, type=dictionary
           must match model in format and array shape

    wgts : weights of data, type=dictionry, [default=None]
           keys are antenna pair tuples (must match model), values are real floats
           matching shape of model and data

    Output:
    -------
    fit : dictionary containing phi_i = angle(g_i) for each antenna
    """
    echo("...configuring linsolve data for phs_logcal", verbose=verbose)

    # get keys from match between data and model dictionary
    keys = sorted(set(model.keys()) & set(data.keys()))

    # angle of visibility ratio is ydata independent variable
    ydata = odict([(k, np.angle(model[k]/data[k])) for k in model.keys()])

    # make weights if None
    if wgts is None:
        wgts = copy.deepcopy(ydata)
        for i, k in enumerate(keys):
            wgts[k] = np.ones_like(ydata[k], dtype=np.float)

    # replace nans
    for i, k in enumerate(keys):
        nan_select = np.isnan(ydata[k])
        ydata[k][nan_select] = 0.0
        wgts[k][nan_select] = 0.0
        nan_select = np.isnan(model[k])
        model[k][nan_select] = 0.0
        wgts[k][nan_select] = 0.0

    # replace infs
    for i, k in enumerate(keys):
        inf_select = np.isinf(ydata[k])
        ydata[k][inf_select] = 0.0
        wgts[k][inf_select] = 0.0
        inf_select = np.isinf(model[k])
        model[k][inf_select] = 0.0
        wgts[k][inf_select] = 0.0

    # setup linsolve equations
    eqns = odict([(k, "phi{} - phi{}".format(k[0], k[1])) for i, k in enumerate(keys)])
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


def delay_lincal(model, data, df=9.765625e4, kernel=(1, 11), verbose=True, time_ax=0, freq_ax=1):
    """
    Solve for per-antenna delay according to the equation

    tau_ij^model - tau_ij^data = tau_i - tau_j

    Parameters:
    -----------
    model : visibility data of refence model, type=dictionary
            keys are antenna-pair tuples, values are complex ndarray visibilities
            these visibilities must be at least 2D arrays, with [0] axis indexing time
            and [1] axis indexing frequency. If the arrays are 3D arrays, the [2] axis
            should index polarization.

    data : visibility data of measurements, type=dictionary
           keys are antenna pair tuples (must match model), values are
           complex ndarray visibilities matching shape of model

    wgts : weights of data, type=dictionry, [default=None]
           keys are antenna pair tuples (must match model), values are real floats
           matching shape of model and data

    Output:
    -------
    fit : dictionary containing delay (tau_i) for each antenna
    """
    # get shared keys
    keys = sorted(set(model.keys()) & set(data.keys()))

    # median filter and FFT to get delays
    model_dlys, data_dlys = [], []
    for i, k in enumerate(keys):
        m_dly, m_off = fft_dly(model[k], df=df, kernel=kernel, time_ax=time_ax, freq_ax=freq_ax)
        model_dlys.append(m_dly)
        d_dly, d_off = fft_dly(data[k], df=df, kernel=kernel, time_ax=time_ax, freq_ax=freq_ax)
        data_dlys.append(d_dly)
       
    model_dlys = np.array(model_dlys)
    data_dlys = np.array(data_dlys)

    # form ydata
    ydata = odict(zip(keys, model_dlys - data_dlys))

    # setup linsolve equation dictionary
    eqns = odict([(k, 'tau{} - tau{}'.format(k[0], k[1])) for i, k in enumerate(keys)])

    # setup design matrix dictionary
    ls_design_matrix = odict()

    # setup linsolve data dictionary
    ls_data = odict([(eqns[k], ydata[k]) for i, k in enumerate(keys)])

    # setup linsolve and run
    sol = linsolve.LinearSolver(ls_data, **ls_design_matrix)
    echo("...running linsolve", verbose=verbose)
    fit = sol.solve()
    echo("...finished linsolve", verbose=verbose)

    return fit


def data_key_to_array_axis(data, key_index, array_index=-1, avg_dict=None):
    """
    move an index of data.keys() into the data axes

    Parameters:
    -----------
    data : dictionary, complex visibility data with
        antenna-pair (+ pol + other) tuples for keys.
    
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


def UVData2AbsCalDict(filenames, pol_select=None, pop_autos=True):
    """
    turn pyuvdata.UVData objects or miriad filenames 
    into the dictionary form that AbsCal requires

    Parameters:
    -----------
    filenames : list of either strings to miriad filenames or list of UVData instances
                to concatenate into a single dictionary

    pol_select : list of polarization strings to keep

    pop_autos : boolean, if True: remove autocorrelations

    Output:
    -------
    data, flags

    data : dictionary containing baseline-pol complex visibility data
    flags : dictionary containing data flags
    """
    # check filenames is a list
    if type(filenames) is not list and type(filenames) is not np.ndarray:
        if type(filenames) is str:
            uvd = UVData()
            uvd.read_miriad(filenames)
        else:
            uvd = filenames
    else:
        # iterate over extra files and concatenate to original object
        for i, fname in enumerate(filenames):
            if i == 0:
                if type(fname) is str:
                    uvd = UVData()
                    uvd.read_miriad(fname)
                else:
                    uvd = fname
            else:
                if type(fname) is str:
                    uv_d = UVData()
                    uv_d.read_miriad(fname)
                    uvd += uv_d
                else:
                    uvd += fname

    # load data
    d, f = firstcal.UVData_to_dict([uvd])

    # pop autos
    if pop_autos:
        for i, k in enumerate(d.keys()):
            if k[0] == k[1]:
                d.pop(k)
                f.pop(k)

    # reconfigure polarization nesting
    data = odict()
    flags = odict()

    # select pols
    if pol_select is None:
        pol_select = d[d.keys()[0]].keys()

    # loop over data keys
    for i, k in enumerate(sorted(d.keys())):
        for j, p in enumerate(pol_select):
            key = k + (p,)
            data[key] = d[k][p]
            flags[key] = f[k][p]

    return data, flags


def fft_dly(vis, df=9.765625e4, kernel=(1, 11), time_ax=0, freq_ax=1):
    """
    get delay of visibility across band using FFT w/ blackman harris window
    and quadratic fit to delay peak.

    Parameters:
    -----------
    vis : ndarray of visibility data, dtype=complex, shape=(Ntimes, Nfreqs, +)
    
    df : frequency channel width in Hz

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

    # smooth via median filter
    kernel += tuple(np.ones((vis.ndim - len(kernel)), np.int))
    vis_smooth = signal.medfilt(np.real(vis), kernel_size=kernel) + 1j*signal.medfilt(np.imag(vis), kernel_size=kernel)

    # fft
    window = np.moveaxis(signal.windows.blackmanharris(Nfreqs).reshape(-1, 1), 0, freq_ax)
    vfft = np.fft.fft(vis_smooth, axis=freq_ax)

    # get argmax of abs
    amp = np.abs(vfft)
    argmax = np.moveaxis(np.argmax(amp, axis=freq_ax)[np.newaxis], 0, freq_ax)

    # get delays
    fftfreqs = np.fft.fftfreq(Nfreqs, 1)
    dfreq = np.median(np.diff(fftfreqs))
    dlys = fftfreqs[argmax]

    # get peak shifts, and add to dlys
    def get_peak(amp, max_ind):
        y = amp[max_ind-1:max_ind+2]
        if len(y) < 3:
            return 0
        r = np.abs(np.diff(y))
        r = r[0] / r[1]
        peak = 0.5 * (r-1) / (r+1)
        return peak

    peak_shifts = np.array([get_peak(np.take(amp, i, axis=time_ax), np.take(argmax, i, axis=time_ax)[0]) for i in range(Ntimes)])
    dlys += np.moveaxis(peak_shifts.reshape(-1, 1) * dfreq, 0, time_ax)

    # get phase offsets by interpolating real and imag component of FFT
    vfft_real = []
    vfft_imag = []
    for i, a in enumerate(argmax):
        # get real and imag of each argmax
        real = np.take(vfft.real, i, axis=time_ax)
        imag = np.take(vfft.imag, i, axis=time_ax)

        # add interpolation component
        rl = interpolate.interp1d(np.arange(Nfreqs), real)(a + peak_shifts[i])
        im = interpolate.interp1d(np.arange(Nfreqs), imag)(a + peak_shifts[i])

        # insert into arrays
        vfft_real.append(rl)
        vfft_imag.append(im)

    vfft_real = np.moveaxis(np.array(vfft_real), 0, time_ax)
    vfft_imag = np.moveaxis(np.array(vfft_imag), 0, time_ax)
    vfft_interp = vfft_real + 1j*vfft_imag
    phi = np.angle(vfft_interp)
    dlys /= df

    return dlys, phi


def interp2d_vis(data, data_times, data_freqs, model_times, model_freqs,
                 kind='cubic', fill_value=0, zero_tol=1e-10, flag_extrapolate=True):
    """
    interpolate complex visibility data onto the time & frequency basis of
    a model visibility.
    ** Note: this is just a simple wrapper for scipy.interpolate.interp2d **

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

    flag_extrapolate : flag extrapolated data if True

    Output: (data, flags)
    -------
    data : interpolated data, type=dictionary
    flags : flags associated with data, type=dictionary
    """
    # make flags
    flags = odict()

    # loop over keys
    for i, k in enumerate(data.keys()):
        # interpolate real and imag separately
        interp_real = interpolate.interp2d(data_freqs, data_times, np.real(data[k]),
                                           kind=kind, fill_value=np.nan, bounds_error=False)(model_freqs, model_times)
        interp_imag = interpolate.interp2d(data_freqs, data_times, np.imag(data[k]),
                                           kind=kind, fill_value=np.nan, bounds_error=False)(model_freqs, model_times)

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
        new_data = interp_real + 1j*interp_imag
        new_flags = f

        data[k] = new_data
        flags[k] = f

    return data, flags


def compute_reds(antpos, ex_ants=[], tol=1.0):
    """
    compute redundant baselines groups.

    Parameters:
    -----------
    antpos : dictionary, antennas integers as keys, baseline vectors as values

    ex_ants : list of flagged (excluded) antennas

    tol : float, tolerance for redundant baseline determination in units of the baseline vector units

    Output:
    -------
    red_bls : redundant baseline list of input bls list
        ordered by smallest separation to longest separation    
    """
    if type(antpos) is not dict and type(antpos) is not odict:
        raise AttributeError("antpos is not a dictionary type")

    # calculate all permutations
    bls = sorted(itertools.combinations(antpos.keys(), 2))

    red_bl_vecs = []
    red_bl_dists = []
    red_bls = []
    for i, k in enumerate(bls):
        if k[0] in ex_ants or k[1] in ex_ants:
            continue
        try:
            bl_vec = antpos[k[1]] - antpos[k[0]]
        except KeyError:
            continue
        unique = map(lambda x: np.linalg.norm(bl_vec - x) > tol, red_bl_vecs)
        if len(unique) == 0 or functools.reduce(lambda x, y: x*y, unique) == 1:
            red_bl_vecs.append(bl_vec)
            red_bl_dists.append(np.linalg.norm(bl_vec))
            red_bls.append([k])
        else:
            red_id = np.where(np.array(unique) == False)[0][0]
            red_bls[red_id].append(k)

    red_bls = list(np.array(red_bls)[np.argsort(red_bl_dists)])
    return red_bls


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


def echo(message, type=0, verbose=True):
    if verbose:
        if type == 0:
            print(message)
        elif type == 1:
            print('')
            print(message)
            print("-"*40)

''' TO DO

def match_red_baselines(data_bls, data_ants, data_antpos, model_bls, model_ants, model_antpos, tol=1.0):
    """
    match unique data baselines to unique model baselines
    """
    # ensure ants are lists
    if type(data_ants) is not list:
        data_ants = data_ants.tolist()
    if type(model_ants) is not list:
        model_ants = model_ants.tolist()

    # create unique baseline id
    data_bl_id = np.array(map(lambda bl: Baseline(data_antpos[data_ants.index(bl[1])]-data_antpos[data_ants.index(bl[0])], tol=tol), data_bls))
    model_bl_id = np.array(map(lambda bl: Baseline(model_antpos[data_ants.index(bl[0])]-model_antpos[model_ants.index(bl[1])], tol=tol), model_bls))

    # iterate over data bls
    for i, bl in data_bl_id:
        pass


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
            if equiv:
                return True
            # check conjugation
            elif np.isclose(np.arccos(np.dot(self.unit, B2.unit)), np.pi, atol=tol/self.len):
                return 'conjugated'
            # else return False
            else:
                return False
        else:
            return False

def smooth_data(Xtrain, ytrain, Xpred, kind='gp', n_restart=3, degree=1, ls=10.0, return_model=False):
    """


    Parameters:
    -----------
    Xtrain : ndarray containing x-values for regression
        type=ndarray, dtype=float, shape=(Ndata, Ndimensions)
  
    ytrain : 1D np.array containing parameter y-values across time and/or frequency (single pol)
        type=1D-ndarray, dtype=float, shape=(Ndata, Nfeatures)
  
    Xpred : ndarray containing x-values for prediction
        type=ndarray, dtype=float, shape=(Ndata, Ndimensions)

    kind : kind of smoothing to perform, type=str, opts=['linear','const']
        'poly' : fit a Nth order polynomial across frequency, with order set by "degree"
        'gp' : fit a gaussian process across frequency
        'boxcar' : convolve ytrain with a boxcar window
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
    run AbsCal on a single data miriad file

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


def mirror_data_to_red_bls(data, bls, antpos, tol=2.0):
    """
    Given unique baseline data (like omnical model visibilities),
    copy the data over to all other baselines in the same redundant group

    Parameters:
    -----------
    data : data dictionary in AbsCal form, see AbsCal docstring for details

    bls : baseline list
        list of all antenna pair tuples that "data" needs to expand into

    antpos : antenna positions dictionary
        keys are antenna integers, values are ndarray baseline vectors.
        This can be created via
        ```
        antpos, ants = UVData.get_ENU_pos()
        antpos = dict(zip(ants, antpos))
        ```

    tol : redundant baseline distance tolerance, dtype=float
        fed into abscal.compute_reds

    Output:
    -------
    red_data : data dictionary in AbsCal form, with unique baseline data
        distributed to redundant baseline groups.

    """
    # get data keys
    keys = data.keys()

    # get data and ants
    data = copy.deepcopy(data)
    ants = np.unique(np.concatenate([keys]))

    # get redundant baselines
    reds = compute_reds(bls, antpos, tol=tol)

    # make red_data dictionary
    red_data = odict()

    # iterate over red bls
    for i, bl_group in enumerate(reds):
        # find which key in data is in this group
        select = np.array(map(lambda x: x in keys or x[::-1] in keys, reds[i]))

        if True not in select:
            continue
        k = reds[i][np.argmax(select)]

        # iterate over bls and insert data into red_data
        for j, bl in enumerate(bl_group):

            red_data[bl] = copy.copy(data[k])

    # re-sort
    red_data = odict([(k, red_data[k]) for k in sorted(red_data)])

    return red_data


def lst_align(data_fname, model_fnames=None, dLST=0.00299078, output_fname=None, outdir=None, overwrite=False,
              verbose=True, write_miriad=True, output_data=False, kind='linear'):
    """
    """
    # try to load model
    if model_fnames is not None:
        if type(model_fnames) is str:
            uvm = UVData()
            uvm.read_miriad(model_fnames)
            lst_arr = np.unique(uvm.lst_array) * 12 / np.pi
            model_freqs = np.unique(uvm.freq_array)
        elif type(model_fnames) is list:
            uvm = UVData()
            uvm.read_miriad(model_fnames[0])
            for i, f in enumerate(model_fnames[1:]):
                uv = UVData()
                uv.read_miriad(f)
                uvm += uv
            lst_arr = np.unique(uvm.lst_array) * 12 / np.pi
            model_freqs = np.unique(uvm.freq_array)
    else:
        # generate LST array
        lst_arr = np.arange(0, 24, dLST)
        model_freqs = None

    # load file
    echo("loading {}".format(data_fname), verbose=verbose)
    uvd = UVData()
    uvd.read_miriad(data_fname)

    # get data
    data, flags, pols = UVData2AbsCalDict([uvd], pop_autos=False)

    # get data lst and freq arrays
    data_lsts, data_freqs = np.unique(uvd.lst_array) * 12/np.pi, np.unique(uvd.freq_array)
    Ntimes = len(data_lsts)

    # get closest lsts
    start = np.argmin(np.abs(lst_arr - data_lsts[0]))
    lst_indices = np.arange(start, start+Ntimes)
    model_lsts = lst_arr[lst_indices]
    if model_freqs is None:
        model_freqs = data_freqs
    Nfreqs = len(model_freqs)

    # interpolate data
    echo("interpolating data", verbose=verbose)
    interp_data, interp_flags = interp_vis(data, data_lsts, data_freqs, model_lsts, model_freqs, kind=kind)
    Nbls = len(interp_data)

    # reorder into arrays
    uvd_data = np.array(interp_data.values())
    uvd_data = uvd_data.reshape(-1, 1, Nfreqs, 1)
    uvd_flags = np.array(interp_flags.values()).astype(np.bool)
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

'''


