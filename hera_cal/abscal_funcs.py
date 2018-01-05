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
from JD2LST import LST2JD


def amp_lincal(model, data, wgts=None, verbose=False):
    """
    calculate gain amplitude scalar with a linear solver
    using equation:

    |V_ij^model| = A * |V_ij^data|

    Parameters:
    -----------
    model : visibility data of refence model, type=dictionary
        keys are antenna-pair tuples, values are complex ndarray visibilities
        these visibilities must be at least 2D arrays, with [0] axis indexing time
        and [1] axis indexing frequency. If the arrays are 3D arrays, the [2] axis
        should index polarization.

        Example: Ntimes = 2, Nfreqs = 3, Npol = 0
        model = {(0, 1): np.array([[1+0j, 2+1j, 0+2j], [3-1j, -1+2j, 0+2j]]), ...}

        Example: Ntimes = 2, Nfreqs = 3, Npol = 2
        model = {(0, 1): np.array([ [[1+0j, 2+1j, 0+2j],
                                     [3-1j,-1+2j, 0+2j]],
                                    [[3+1j, 4+0j,-1-3j],
                                     [4+2j, 0+0j, 0-1j]] ]), ...}

    data : visibility data of measurements, type=dictionary
        keys are antenna pair tuples (must match model), values are
        complex ndarray visibilities matching shape of model

    wgts : weights of data, type=dictionry, [default=None]
        keys are antenna pair tuples (must match model), values are real floats
        matching shape of model and data

    verbose : print output, type=boolean, [default=False]

    Output:
    -------
    fit : dictionary with 'A' key for amplitude scalar, which has the same shape as
        the ndarrays in model

    """
    echo("...configuring linsolve data", verbose=verbose)

    # get keys from model dictionary
    keys = model.keys()

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


def phs_logcal(model, data, bls, wgts=None, verbose=False, zero_psi=False):
    """
    calculate overall gain phase and gain phase slopes (EW and NS)
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

        Example: Ntimes = 2, Nfreqs = 3, Npol = 0
        model = {(0, 1): np.array([[1+0j, 2+1j, 0+2j], [3-1j, -1+2j, 0+2j]]), ...}

        Example: Ntimes = 2, Nfreqs = 3, Npol = 2
        model = {(0, 1): np.array([ [[1+0j, 2+1j, 0+2j],
                                     [3-1j,-1+2j, 0+2j]],
                                    [[3+1j, 4+0j,-1-3j],
                                     [4+2j, 0+0j, 0-1j]] ]), ...}

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
    fit : dictionary with psi key for overall gain phase
        PHIx for x-axis phase slope and PHIy for y-axis phase slope
        which all have the same shape as the ndarrays in model
    """
    echo("...configuring linsolve data", verbose=verbose)

    # get keys from model dictionary
    keys = model.keys()

    # angle of phs ratio is ydata independent variable
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
    eqns = odict([(k, "psi*a{} + PHIx*{} + PHIy*{}".format(str(i), bx[k][0], by[k][0])) for i, k in enumerate(keys)])

    # fill in other variables in the design matrix
    if zero_psi:
        ls_design_matrix = odict([("a{}".format(str(i)), 0.0) for i, k in enumerate(keys)])
    else:
        ls_design_matrix = odict([("a{}".format(str(i)), 1.0) for i, k in enumerate(keys)])
    ls_design_matrix.update(odict(bx.values()))
    ls_design_matrix.update(odict(by.values()))

    # setup linsolve dictionaries
    ls_data = odict([(eqns[k], ydata[k]) for i, k in enumerate(keys)])
    ls_wgts = odict([(eqns[k], wgts[k]) for i, k in enumerate(keys)])

    # setup linsolve and run
    sol = linsolve.LinearSolver(ls_data, wgts=ls_wgts, **ls_design_matrix)
    echo("...running linsolve", verbose=verbose)
    fit = sol.solve()
    echo("...finished linsolve", verbose=verbose)

    return fit


def delay_lincal(model, data, refant, df=9.765625e4, kernel=(1, 11), verbose=True, time_ax=0, freq_ax=1):
    """
    Solve for per-antenna delay according to the equation

    tau_ij^model = tau_i - tau_j + tau_ij^data

    Parameters:
    -----------

    """
    # verify refant
    if refant not in np.concatenate(model.keys()):
        raise KeyError("refant {} not in model dictionary".format(refant))

    # unpack model and data
    model_values = np.array(model.values())
    model_keys = model.keys()
    data_values = np.array(data.values())
    data_keys = data.keys()

    # median filter and FFT to get delays
    model_dlys, data_dlys = [], []
    for i, mv in enumerate(model_values):
        m_dly, m_off = fft_dly(mv, df=df, kernel=kernel, time_ax=time_ax, freq_ax=freq_ax)
        model_dlys.append(m_dly)
    for i, dv in enumerate(data_values):
        d_dly, d_off = fft_dly(mv, df=df, kernel=kernel, time_ax=time_ax, freq_ax=freq_ax)
        data_dlys.append(d_dly)

    model_dlys = odict([(k, model_dlys[i]) for i, k in enumerate(model_keys)])
    data_dlys = odict([(k, data_dlys[i]) for i, k in enumerate(data_keys)])

    # form ydata
    ydata = odict([(k, model_dlys[k] - data_dlys[k]) for i, k in enumerate(model_keys)])

    # setup linsolve equation dictionary
    eqns = odict([(k, "a_{}_{}*tau_{} + a_{}_{}*tau_{}".format(k[0], k[1], k[0], k[1], k[0], k[1])) for i, k in enumerate(model_keys)])

    # setup design matrix dictionary
    ls_design_matrix = odict([("a_{}_{}".format(k[0], k[1]), 1.0) if k[0] != refant else ("a_{}_{}".format(k[0], k[1]), 0.0) for i, k in enumerate(model_keys)])
    ls_design_matrix.update(odict([("a_{}_{}".format(k[1], k[0]), -1.0) if k[1] != refant else ("a_{}_{}".format(k[1], k[0]), 0.0) for i, k in enumerate(model_keys)]))

    # setup linsolve data dictionary
    ls_data = odict([(eqns[k], ydata[k]) for i, k in enumerate(model_keys)])

    # setup linsolve and run
    sol = linsolve.LinearSolver(ls_data, **ls_design_matrix)
    echo("...running linsolve", verbose=verbose)
    fit = sol.solve()
    echo("...finished linsolve", verbose=verbose)

    return fit


def run_abscal(data_files, model_files, unravel_pol=False, unravel_freq=False, unravel_time=False, verbose=True,
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


def UVData2AbsCalDict(filenames, pol_select=None, pop_autos=True):
    """
    turn pyuvdata.UVData objects or miriad filenames 
    into the dictionary form that AbsCal requires

    Parameters:
    -----------
    filenames : list of either strings to miriad filenames or list of UVData instances

    pol_select : list of polarization strings to keep

    pop_autos : boolean, if True: remove autocorrelations

    Output:
    -------
    DATA_LIST, FLAG_LIST, POL_LIST

    DATA_LIST : list of dictionaries containing data from UVData objects
        if len(filenames) == 1, just outputs the dictionary itself

    FLAG_LIST : list of dictionaries containing flag data
        if len(filenames) == 1, just outputs dictionary

    POL_LIST : list of polarizations in UVData objects
        if len(filenames) == 1, just outputs polarization string itself
    """
    # initialize containers for data dicts and pol keys
    DATA_LIST = []
    FLAG_LIST = []
    POL_LIST = []

    # loop over filenames    
    for i, fname in enumerate(filenames):
        # initialize UVData object
        if type(fname) == str:
            uvd = UVData()
            uvd.read_miriad(fname)
        elif type(fname) == UVData:
            uvd = fname
        # load data into dictionary
        data_temp, flag_temp = firstcal.UVData_to_dict([uvd])

        if pop_autos:
            # eliminate autos
            for i, k in enumerate(data_temp.keys()):
                if k[0] == k[1]:
                    data_temp.pop(k)
                    flag_temp.pop(k)

        ## reconfigure polarization nesting ##
        # setup empty dictionaries
        data = odict()
        flags = odict()

        # configure pol keys
        pol_keys = sorted(data_temp[data_temp.keys()[0]].keys())
        if pol_select is not None:
            # ensure it is a list
            if type(pol_select) == str:
                pol_select = [pol_select]

            # ensure desired pols are in data
            if functools.reduce(lambda x, y: x*y, map(lambda x: x in pol_keys, pol_select)) == 0:
                raise KeyError("desired pols(s) {} not found in data pols {}".format(pol_select, pol_keys))
            pol_keys = pol_select

        # configure data keys
        data_keys = sorted(data_temp.keys())

        # iterate over pols and data keys
        for i, p in enumerate(pol_keys):
            for j, k in enumerate(data_keys):
                if i == 0:
                    data[k] = data_temp[k][p][:, :, np.newaxis]
                    flags[k] = flag_temp[k][p][:, :, np.newaxis]
                elif i > 0:
                    data[k] = np.dstack([data[k], data_temp[k][p][:, :, np.newaxis]])
                    flags[k] = np.dstack([flags[k], flag_temp[k][p][:, :, np.newaxis]])
        # append
        DATA_LIST.append(data)
        FLAG_LIST.append(flags)
        POL_LIST.append(pol_keys)

    if len(DATA_LIST) == 1:
        DATA_LIST = DATA_LIST[0]
        FLAG_LIST = FLAG_LIST[0]
        POL_LIST = POL_LIST[0]

    return DATA_LIST, FLAG_LIST, POL_LIST


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





def unravel(data, prefix, axis, copy_dict=None):
    """
    promote visibility data from within a data key
    to being its own key

    Parameters:
    -----------
    data : "data" dictionary, see class docstring for details on specs

    prefix : prefix of the key we are adding to data, type=string

    axis : which axis of the visibility to "promote" or unravel, type=int

    copy_dict : ancillary dictionary that matches data in keys, but 
        needs to get its values directly copied (not unraveled)
        just to match shape of new data dictionary. This is necessary, 
        for example, for the baselines (bls) dictionary 
    """
    # loop over keys
    for i, k in enumerate(data.keys()):
        # loop over row / columns of data
        for j in range(data[k].shape[axis]):
            if axis == 0:
                data[k+("{}{}".format(prefix, str(j)),)] = copy.copy(data[k][j:j+1])
            elif axis == 1:
                data[k+("{}{}".format(prefix, str(j)),)] = copy.copy(data[k][:, j:j+1])
            elif axis == 2:
                data[k+("{}{}".format(prefix, str(j)),)] = copy.copy(data[k][:, :, j:j+1])
            elif axis == 3:
                data[k+("{}{}".format(prefix, str(j)),)] = copy.copy(data[k][:, :, :, j:j+1])
            else:
                raise TypeError("can't support axis > 3")
            
            if copy_dict is not None:
                copy_dict[k+("{}{}".format(prefix, str(j)),)] = copy_dict[k]
        # remove original key
        data.pop(k)
        if copy_dict is not None:
            copy_dict.pop(k)


def fft_dly(vis, df=9.765625e4, kernel=(1, 11), time_ax=0, freq_ax=1):
    """
    get delay of visibility across band

    vis : 2D ndarray of visibility data, dtype=complex, shape=(Ntimes, Nfreqs)

    df : frequency channel width in Hz

    time_ax : time axis of data

    freq_ax : frequency axis of data

    pol_ax : polarization axis of data (if exists)

    edgecut : fraction of band edges to cut before FFT
    """
    # get array params
    Nfreqs = vis.shape[freq_ax]
    Ntimes = vis.shape[time_ax]

    # smooth via median filter
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


def interp_vis(data, data_times, data_freqs, model_times, model_freqs,
                 kind='cubic', fill_value=0, zero_tol=1e-10, flag_extrapolate=True):
    """
    interpolate complex visibility data onto the time & frequency basis of
    a model visibility.
    ** Note: this is just a simple wrapper for scipy.interpolate.interp2d **

    Parameters:
    -----------
    data : complex visibility data, type=dictionary, see AbsCal for details on format

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
    """
    # copy data and flags
    data = copy.deepcopy(data)
    flags = odict()

    # loop over keys
    for i, k in enumerate(data.keys()):
        # loop over polarizations
        new_data = []
        new_flags = []
        for p in range(data[k].shape[2]):
            # interpolate real and imag separately
            interp_real = interpolate.interp2d(data_freqs, data_times, np.real(data[k][:, :, p]),
                                               kind=kind, fill_value=np.nan, bounds_error=False)(model_freqs, model_times)
            interp_imag = interpolate.interp2d(data_freqs, data_times, np.imag(data[k][:, :, p]),
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
            new_data.append(interp_real + 1j*interp_imag)
            new_flags.append(f)

        data[k] = np.moveaxis(np.array(new_data), 0, 2)
        flags[k] = np.moveaxis(np.array(new_flags), 0, 2)

    return data, flags


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
        keys are antenna integers, values are ndarray baseline vectors

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


def compute_reds(bls, antpos, tol=2.0):
    """
    compute redundant baselines

    Parameters:
    -----------
    bls : baseline list, list of antenna pair tuples

    antpos : dictionary, antennas integers as keys, baseline vectors as values

    tol : float, tolerance for redundant baseline determination in units of the baseline vector units

    Output:
    -------
    red_bls : redundant baseline list of input bls list
        ordered by smallest separation to longest separation    
    """
    if type(antpos) is not dict:
        raise AttributeError("antpos is not a dict")
    red_bl_vecs = []
    red_bl_dists = []
    red_bls = []
    for i, k in enumerate(bls):
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
                  gain_convention='multiply', inttime=10.7, overwrite=False, **kwargs):
    """
    write out gain_array in calfits file format

    Parameters:
    -----------

    calfits_fname : string

    abscal_gains : complex gain in dictionary form from AbsCal.make_gains()

    freq_array : frequency array of data in Hz

    time_array : time array of data in Julian Date

    pol_array : polarization array of data, in 'x' or 'y' form. 

    """
    # ensure pol is string
    int2pol = {-5: 'x', -6: 'y'}
    if pol_array.dtype == np.int:
        pol_array = list(pol_array)
        for i, p in enumerate(pol_array):
            pol_array[i] = int2pol[p]

    # reconfigure AbsCal gain dictionary into HERACal gain dictionary
    heracal_gains = {}
    for i, p in enumerate(pol_array):
        pol_dict = {}
        for j, k in enumerate(abscal_gains.keys()):
            pol_dict[k] = abscal_gains[k][:, :, i]
        heracal_gains[p] = pol_dict

    # configure meta
    meta = {'times':time_array, 'freqs':freq_array, 'inttime':inttime, 'gain_convention': gain_convention}
    meta.update(**kwargs)

    # convert to UVCal
    uvc = cal_formats.HERACal(meta, heracal_gains)

    # write to file
    if os.path.exists(calfits_fname) is True and overwrite is False:
        print("{} already exists, not overwriting...".format(calfits_fname))
    else:
        uvc.write_calfits(calfits_fname, clobber=overwrite)


def param2calfits(calfits_fname, abscal_param, param_name, freq_array, time_array, pol_array, overwrite=False):
    """
    """
    pass
    

def abscal_arg_parser():
    a = argparse.ArgumentParser()
    a.add_argument("--data_files", type=str, nargs='*', help="list of miriad files of data to-be-calibrated.", required=True)
    a.add_argument("--model_files", type=str, nargs='*', default=[], help="list of data-overlapping miriad files for visibility model.", required=True)
    a.add_argument("--calfits_fname", type=str, default=None, help="name of output calfits file.")
    a.add_argument("--overwrite", default=False, action='store_true', help="overwrite output calfits file if it exists.")
    a.add_argument("--silence", default=False, action='store_true', help="silence output from abscal while running.")
    a.add_argument("--zero_psi", default=False, action='store_true', help="set overall gain phase 'psi' to zero in linsolve equations.")
    return a


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
    for baselines up to 1km in length. Only __eq__ operator is overloaded.
    """
    def __init__(self, bl, tol=1.0):
        """
        bl : list containing [dx, dy] float separation in meters
        tol : tolerance for baseline length comparison in meters
        """
        self.label = "{:06.1f}:{:06.1f}".format(float(bl[0]), float(bl[1]))
        self.tol = tol

    def __repr__(self):
        return self.label

    @property
    def bl(self):
        return map(float, self.label.split(':'))

    @property
    def len(self):
        return np.linalg.norm(self.bl)

    def __eq__(self, B2):
        if np.isclose(self.len, B2.len, atol=np.max([self.tol, B2.tol])):
            if np.isclose(self.bl[0], -B2.bl[0], atol=np.max([self.tol, B2.tol])):
                return 'conjugated'
            else:
                return True
        else:
            return False


def lst_align(data_fname, model_fname=None, dLST=0.00299078, output_fname=None, outdir=None, overwrite=False,
              verbose=True):
    """
    """
    # try to load model
    if model_fname is not None:
        uvm = UVData()
        uvm.read_miriad(model_fname)
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
    interp_data, interp_flags = interp_vis(data, data_lsts, data_freqs, model_lsts, model_freqs, kind='cubic')
    Nbls = len(interp_data)

    # reorder into arrays
    uvd_data = np.array(interp_data.values())
    uvd_data = uvd_data.reshape(-1, 1, Nfreqs, 1)
    uvd_flags = np.array(interp_flags.values()).astype(np.bool)
    uvd_flags = uvd_flags.reshape(-1, 1, Nfreqs, 1)
    uvd_keys = np.repeat(np.array(interp_data.keys()).reshape(-1, 1, 2), Ntimes, axis=1).reshape(-1, 2)
    uvd_bls = np.array(map(lambda k: uvd.antnums_to_baseline(k[0], k[1]), uvd_keys))
    uvd_times = np.array(map(lambda x: JD2LST.LST2JD(x, np.median(np.floor(uvd.time_array)), uvd.telescope_location_lat_lon_alt_degrees[1]), model_lsts))
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

    # check output
    if outdir is None:
        outdir = os.path.dirname(data_fname)
    if output_fname is None:
        output_fname = data_fname.split('.')
        output_fname.pop(2)
        output_fname = '.'.join(output_fname) + 'L.{:07.4f}'.format(model_lsts[0])
    output_fname = os.path.join(outdir, output_fname)
    if os.path.exists(output_fname) and overwrite is False:
        raise IOError("{} exists, not overwriting".format(output_fname))

    # write to file
    echo("saving {}".format(output_fname), verbose=verbose)
    uvd.write_miriad(output_fname, clobber=True)





