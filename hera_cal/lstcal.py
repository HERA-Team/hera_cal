"""
lstcal.py
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



def amp_lincal(model, data, wgts=None, verbose=False):
    """
    calculate gain amplitude scalar with a linear solver
    using equation:

    |V_ij^model| = A * |V_ij^data|

    Parameters:
    -----------
    model : visibility data of refence model, type=dictionary
        keys are antenna pair tuples, values are complex ndarray visibilities

    data : visibility data of measurements, type=dictionary
        keys are antenna pair tuples (must match model), values are complex ndarray visibilities

    wgts : least square weights of data, type=dictionry, [default=None]
        keys are antenna pair tuples (must match model), values are real floats

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

    data : visibility data of measurements, type=dictionary
        keys are antenna pair tuples (must match model), values are complex ndarray visibilities

    bls : baseline vectors of antenna pairs, type=dictionary
        keys are antenna pair tuples (must match model), values are 2D or 3D ndarray
        baseline vectors in meters, with [0] index containing X separation, and [1] index Y separation.

    wgts : least square weights of data, type=dictionry, [default=None]
        keys are antenna pair tuples (must match model), values are real floats

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


class LSTCal(object):
    """
    """
    def __init__(self, model, data, freqs, antpos, wgts=None):
        """
        LSTCal object for absolute calibration of flux scale and phasing
        given a visibility model and measured data. model, data and weights
        should be fed as dictionary types,

        model : 


        data : 


        wgts : 


        freqs : 


        antpos : 

        """
        

        self.model = model
        self.data = data
        self.wgts = wgts
        self.freqs = freqs
        self.antpos = antpos




    def amp_lincal(self, freq_ind=True, time_ind=True):
        """

        """
        amp_lincal(self.model, self.data, self.wgts)

        

    def phs_logcal(self):
        """
        """
        pass

    def extract_amp(self):
        """
        """
        pass

    def extract_phs(self):
        """
        """
        pass


    def run(self):

        # run amp cal
        self.amp_logcal()
        self.extract_amp()

        # run phs cal
        self.phs_logcal()
        self.extract_phs()

        # combine gains





def run_lstcal():
    """
    Run LSTCal on a series of files
    """

    for i, fname in enumerate(files):



        LC = LSTCal(model, data, wgts, freqs, antpos)
        LC.run()



def echo(message, type=0, verbose=True):
    if verbose:
        if type == 0:
            print(message)
        elif type == 1:
            print('')
            print(message)
            print("-"*40)



def lst_align():
    """
    """
    pass

def uvdata2dict():
    """
    """
    pass

def data2calfits():
    """
    """
    pass


def lstcal_option_parser():
    """
    """
    a = argparse.ArgumentParser()



    return a







