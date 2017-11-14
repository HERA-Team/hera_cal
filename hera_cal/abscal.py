"""
abscal.py
---------

calibrate measured visibility
data to a visibility model
"""
import os
import sys
import argparse
import numpy as np
from pyuvdata import UVCal, UVData
from hera_cal import omni, utils
import linsolve
from collections import OrderedDict as odict
import copy
from scipy import signal



def amp_lincal(model, data, wgts=None, verbose=False):
    """
    calculate gain amplitude scalar with a linear solver
    using equation:

    |V_ij^model| = A * |V_ij^data|

    Parameters:
    -----------
    model : visibility data of refence model, type=dictionary
        keys are antenna pair tuples, values are complex ndarray visibilities
        these visibilities must be 2D arrays, with [0] axis indexing time
        and [1] axis indexing frequency
        Example: Ntimes = 2, Nfreqs = 3
        model = {(0, 1): np.array([[1+0j, 2+1j, 0+2j], [3-1j, -1+2j, 0+2j]]), ...}

    data : visibility data of measurements, type=dictionary
        keys are antenna pair tuples (must match model), values are complex ndarray visibilities
            these visibilities must be 2D arrays, with [0] axis indexing time
            and [1] axis indexing frequency 

    wgts : least square weights of data, type=dictionry, [default=None]
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
            wgts[k] = np.ones_like(ydata[k]).astype(np.float)

    # replace nans
    for i, k in enumerate(keys):
        nan_select = np.isnan(ydata[k])
        ydata[k][nan_select] = -1.0
        wgts[k][nan_select] = 1e-10

    # setup linsolve equations
    eqn = "1*A"

    # setup linsolve dictionaries
    ls_data = odict([(eqn, ydata[k]) for i, k in enumerate(keys)])
    ls_wgts = odict([(eqn, wgts[k]) for i, k in enumerate(keys)])

    # setup linsolve and run
    sol = linsolve.LinearSolver(ls_data, wgts=ls_wgts)
    echo("...running linsolve", verbose=verbose)
    fit = sol.solve()
    echo("...finished linsolve", verbose=verbose)

    return fit


def phs_logcal(model, data, bls, wgts=None, verbose=False):
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
        keys are antenna pair tuples, values are complex ndarray visibilities
        these visibilities must be 2D arrays, with [0] axis indexing time
        and [1] axis indexing frequency
        Example: Ntimes = 2, Nfreqs = 3
        model = {(0, 1): np.array([[1+0j, 2+1j, 0+2j], [3-1j, -1+2j, 0+2j]]), ...}

    data : visibility data of measurements, type=dictionary
        keys are antenna pair tuples (must match model), values are complex ndarray visibilities
            these visibilities must be 2D arrays, with [0] axis indexing time
            and [1] axis indexing frequency

    bls : baseline vectors of antenna pairs, type=dictionary
        keys are antenna pair tuples (must match model), values are 2D or 3D ndarray
        baseline vectors in meters, with [0] index containing X separation, and [1] index Y separation.

    wgts : least square weights of data, type=dictionry, [default=None]
        keys are antenna pair tuples (must match model), values are real floats
        matching shape of model and data

    verbose : print output, type=boolean, [default=False]

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
            wgts[k] = np.ones_like(ydata[k]).astype(np.float)

    # replace nans
    for i, k in enumerate(keys):
        nan_select = np.isnan(ydata[k])
        ydata[k][nan_select] = 0.0
        wgts[k][nan_select] = 1e-10

    # setup baseline terms
    bx = odict([("bx_{}_{}".format(k[0], k[1]), bls[k][0]) for i, k in enumerate(keys)])
    by = odict([("by_{}_{}".format(k[0], k[1]), bls[k][1]) for i, k in enumerate(keys)])
    ls_bldata = copy.deepcopy(bx)
    ls_bldata.update(by)

    # setup linsolve equations
    eqns = odict([(k, "psi + PHIx*bx_{}_{} + PHIy*by_{}_{}".format(k[0], k[1], k[0], k[1])) for i, k in enumerate(keys)])

    # setup linsolve dictionaries
    ls_data = odict([(eqns[k], ydata[k]) for i, k in enumerate(keys)])
    ls_wgts = odict([(eqns[k], wgts[k]) for i, k in enumerate(keys)])

    # setup linsolve and run
    sol = linsolve.LinearSolver(ls_data, wgts=ls_wgts, **ls_bldata)
    echo("...running linsolve", verbose=verbose)
    fit = sol.solve()
    echo("...finished linsolve", verbose=verbose)

    return fit


class AbsCal(object):
    """
    """
    def __init__(self, model, data, wgts=None, bls=None, antpos=None, ants=None, freqs=None):
        """
        AbsCal object for absolute calibration of flux scale and phasing
        given a visibility model and measured data. model, data and weights
        should be fed as dictionary types,

        Parameters:
        -----------
        model : visibility data of refence model, type=dictionary
            keys are antenna pair tuples, values are complex ndarray visibilities
            these visibilities must be 2D arrays, with [0] axis indexing time
            and [1] axis indexing frequency
            Example: Ntimes = 2, Nfreqs = 3
            model = {(0, 1): np.array([[1+0j, 2+1j, 0+2j], [3-1j, -1+2j, 0+2j]]), ...}

        data : visibility data of measurements, type=dictionary
            keys are antenna pair tuples (must match model), values are complex ndarray visibilities
            these visibilities must be 2D arrays, with [0] axis indexing time
            and [1] axis indexing frequency

        bls : baseline vectors of antenna pairs, type=dictionary
            keys are antenna pair tuples (must match model), values are 2D or 3D ndarray
            baseline vectors in meters (topocentric coordinates),
            with [0] index containing East separation, and [1] index North separation.

        wgts : least square weights of data, type=dictionry, [default=None]
            keys are antenna pair tuples (must match model), values are real floats
            matching shape of model and data

        freqs : frequency array, type=ndarray, dtype=float
            1d array containing visibility frequencies in Hz
    
        verbose : print output, type=boolean, [default=False]

        """
        # append attributes
        self.model = model
        self.data = data

        # setup baselines and antenna positions
        self.bls = bls

        # setup frequencies
        self.freqs = freqs

        # setup weights
        if wgts is None:
            wgts = copy.deepcopy(data)
            for i, k in enumerate(data.keys()):
                wgts[k] = np.ones_like(wgts[k], dtype=np.float)
        self.wgts = wgts

        # get data parameters
        self.Ntimes = model[model.keys()[0]].shape[0]
        self.Nfreq = model[model.keys()[0]].shape[1]
        self.ants = np.unique(sorted(np.array(map(lambda x: [x[0], x[1]], model.values())).ravel()))
        self.Nants = len(self.ants)
        # TODO add polarization (correlation) axis to everything


    def amp_lincal(self, unravel_freq=False, unravel_time=False, verbose=False):
        """
        call abscal.amp_lincal() method. see its docstring for more details.

        Parameters:
        -----------
        unravel_freq : tie all frequencies together, type=boolean, [default=False]
            if True, unravel frequency axis in linsolve call, such that you get
            one result for all frequencies

        unravel_time : tie all times together, type=boolean, [default=False]
            if True, unravel time axis in linsolve call, such that you get
            one result for all times
        """
        # copy data
        model = copy.deepcopy(self.model)
        data = copy.deepcopy(self.data)
        wgts = copy.deepcopy(self.wgts)

        if unravel_time:
            unravel(data, 't', 0)
            unravel(model, 't', 0)
            unravel(wgts, 't', 0)

        if unravel_freq:
            unravel(data, 'f', 1)
            unravel(model, 'f', 1)
            unravel(wgts, 'f', 1)

        # run linsolve
        fit = amp_lincal(model, data, wgts=wgts, verbose=verbose)
        self.gain_amp = fit['A']

        
    def phs_logcal(self, unravel_freq=False, unravel_time=False, verbose=False):
        """
        call abscal.amp_lincal() method. see its docstring for more details.

        Parameters:
        -----------
        unravel_freq : tie all frequencies together, type=boolean, [default=False]
            if True, unravel frequency axis in linsolve call, such that you get
            one result for all frequencies

        unravel_time : tie all times together, type=boolean, [default=False]
            if True, unravel time axis in linsolve call, such that you get
            one result for all times
        """
        # copy data
        model = copy.deepcopy(self.model)
        data = copy.deepcopy(self.data)
        wgts = copy.deepcopy(self.wgts)
        bls = copy.deepcopy(self.bls)

        if unravel_time:
            unravel(data, 't', 0)
            unravel(model, 't', 0, copy_dict=bls)
            unravel(wgts, 't', 0)

        if unravel_freq:
            unravel(data, 'f', 1)
            unravel(model, 'f', 1, copy_dict=bls)
            unravel(wgts, 'f', 1)

        # run linsolve
        fit = phs_logcal(model, data, bls, wgts=wgts, verbose=verbose)
        self.gain_psi = fit['psi']
        self.gain_phi = np.array([fit['PHIx'], fit['PHIy']])

    def run(self, calfits_filename=None, save=False, overwrite=False, unravel_freq=False, unravel_time=False, verbose=False):
        """
        """

        # run amp cal
        self.amp_logcal(unravel_freq=unravel_freq, unravel_time=unravel_time, verbose=verbose)

        # run phs cal
        self.phs_logcal(unravel_freq=unravel_freq, unravel_time=unravel_time, verbose=verbose)

        # form gains
        self.gain_array = np.ones((self.Nants, self.Nfreqs, self.Ntimes, self.Npols), dtype=np.complex)

        # add amplitude to gains
        self.gain_array *= self.gain_amp

        # add phases to gains
        for i, a in enumerate(self.ants):
            self.gain_array[i] *= np.exp(self.gain_psi - 1j*np.dot(self.gain_phi, self.antpos[:, :2].T))

        # TODO : extract select degrees of freedom

        # TODO : write to calfits



def run_abscal(data_files, model_files):
    """
    iterate over miriad visibility files and run AbsCal



    Parameters:
    -----------
    """

    # configure vars



    # loop over data files
    for i, f in enumerate(data_files):
        # load data

        # load models

        # load weights

        # get params

        # lst align

        # run abscal
        AC = AbsCal()
        AC.run()


def smooth_model(model, kernel=(3, 15)):
    """
    smooth model visibility real and imag components separately
    using a median filter, then recombine into complex visibility

    Warning: too aggressive smoothing can lead to signal loss
    and can introduce artifacts
    
    Parameters:
    -----------
    model : dictionary

    """
    model = copy.deepcopy(model)
    for i, k in enumerate(model.keys()):
        mreal = np.real(model[k])
        mimag = np.imag(model[k])
        mreal_smoothed = signal.medfilt(mreal, kernel_size=kernel)
        mimag_smoothed = signal.medfilt(mimag, kernel_size=kernel)
        model[k] = (mreal_smoothed + 1j*mimag_smoothed)

    return model



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
            data[k+("{}{}".format(prefix, str(j)),)] = np.take(data[k], j, axis=axis)
            if copy_dict is not None:
                copy_dict[k+("{}{}".format(prefix, str(j)),)] = copy_dict[k]
        # remove original key
        data.pop(k)
        if copy_dict is not None:
            copy_dict.pop(k)


def echo(message, type=0, verbose=True):
    if verbose:
        if type == 0:
            print(message)
        elif type == 1:
            print('')
            print(message)
            print("-"*40)


def vis_align(model, data, time_array, freq_array):
    """
    interpolate model complex visibility onto the time-frequency basis of data

    Parameters:
    -----------
    model : model (reference) visibility data, type=2D-ndarray, shape=(Ntimes, Nfreqs)


    """
    


def gains2calfits(fname, gain_array, ants, freq_array, time_array, pol_array,
                gain_convention='multiply', clobber=False):
    """
    write out gain_array in calfits file format

    Parameters:
    -----------



    """
    pass




















