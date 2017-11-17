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
from pyuvdata import utils as uvutils
from hera_cal import omni, utils, firstcal, cal_formats
import linsolve
from collections import OrderedDict as odict
import copy
from scipy import signal
from scipy import interpolate

from get_antpos import get_antpos


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
    ls_design_matrix = odict([("a{}".format(str(i)), np.ones(ydata[k].shape, dtype=np.float)) for i, k in enumerate(keys)])

    # setup linsolve dictionaries
    ls_data = odict([(eqns[k], ydata[k]) for i, k in enumerate(keys)])
    ls_wgts = odict([(eqns[k], wgts[k]) for i, k in enumerate(keys)])

    # setup linsolve and run
    sol = linsolve.LinearSolver(ls_data, wgts=ls_wgts, **ls_design_matrix)
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
    ls_design_matrix = odict([("a{}".format(str(i)), np.ones(ydata[k].shape, dtype=np.float)) for i, k in enumerate(keys)])
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


class AbsCal(object):

    def __init__(self, model, data, wgts=None, antpos=None, freqs=None, pols=None):
        """
        AbsCal object for absolute calibration of flux scale and phasing
        given a visibility model and measured data. model, data and weights
        should be fed as dictionary types,

        Parameters:
        -----------
        model : visibility data of refence model, type=dictionary
            keys are antenna pair tuples, values are complex ndarray visibilities
            these visibilities must be 3D arrays, with the [0] axis indexing time,
            the [1] axis indexing frequency and the [2] axis indexing polarization

            Example: Ntimes = 2, Nfreqs = 3, Npol = 2
            model = {(0, 1): np.array([ [[1+0j, 2+1j, 0+2j],
                                         [3-1j,-1+2j, 0+2j]],
                                        [[3+1j, 4+0j,-1-3j],
                                         [4+2j, 0+0j, 0-1j]] ]), ...}

        data : visibility data of measurements, type=dictionary
            keys are antenna pair tuples (must match model), values are
            complex ndarray visibilities, with shape matching model

        antpos : antenna position vectors in TOPO frame in meters, type=dictionary
            keys are antenna integers and values are 2D or 3D ndarray
            position vectors in meters (topocentric coordinates),
            with [0] index containing X (E-W) distance, and [1] index Y (N-S) distance.

        wgts : weights of data, type=dictionry, [default=None]
            keys are antenna pair tuples (must match model), values are real floats
            matching shape of model and data

        freqs : frequency array, type=ndarray, dtype=float
            1d array containing visibility frequencies in Hz
    
        pols : polarization array, type=ndarray, dtype=int
            array containing polarization integers
            in pyuvdata.UVData.polarization_array format

        verbose : print output, type=boolean, [default=False]
        """
        # append attributes
        self.model = model
        self.data = data

        # setup frequencies
        self.Nfreqs = model[model.keys()[0]].shape[1]
        if freqs is None:
            self.freqs = np.zeros(self.Nfreqs)
        else:
            self.freqs = freqs
        if self.Nfreqs != len(self.freqs):
            raise TypeError("shape of 'freqs' does not match shape of arrays in model dictionary: {}".format(model[model.keys()[0]].shape[1]))

        # setup polarization
        self.Npols = model[model.keys()[0]].shape[2]
        if pols is None:
            self.pols = np.zeros((self.Npols))
        else:
            self.pols = pols
        if len(self.pols) != self.Npols:
            raise TypeError("shape of 'pols' does not match shape of arrays in model dictionary: {}".format(model[model.keys()[2]].shape[1]))

        # setup weights
        if wgts is None:
            wgts = copy.deepcopy(data)
            for i, k in enumerate(data.keys()):
                wgts[k] = np.ones_like(wgts[k], dtype=np.float)
        self.wgts = wgts

        # setup times
        self.Ntimes = model[model.keys()[0]].shape[0]

        # setup ants
        self.ants = np.unique(sorted(np.array(map(lambda x: [x[0], x[1]], model.keys())).ravel()))
        self.Nants = len(self.ants)

        # setup baselines and antenna positions
        self.antpos = antpos
        if self.antpos is not None:
            self.bls = odict([((x[0], x[1]), self.antpos[x[1]] - self.antpos[x[0]]) for x in self.model.keys()])
            self.antpos = np.array(map(lambda x: self.antpos[x], self.ants))
            self.antpos -= np.median(self.antpos, axis=0)

    def amp_lincal(self, unravel_freq=False, unravel_time=False, unravel_pol=False, verbose=False):
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

        unravel_pol : tie all polarizations together, type=boolean, [default=False]
            if True, unravel polarization
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

        if unravel_pol:
            unravel(data, 'p', 2)
            unravel(model, 'p', 2)
            unravel(wgts, 'p', 2)

        # run linsolve
        fit = amp_lincal(model, data, wgts=wgts, verbose=verbose)
        self.gain_amp = np.sqrt(fit['amp'])

        
    def phs_logcal(self, unravel_freq=False, unravel_time=False, unravel_pol=False, verbose=False):
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

        unravel_pol : tie all polarizations together, type=boolean, [default=False]
            if True, unravel polarization
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

        if unravel_pol:
            unravel(data, 'p', 2)
            unravel(model, 'p', 2, copy_dict=bls)
            unravel(wgts, 'p', 2)
           
        # run linsolve
        fit = phs_logcal(model, data, bls, wgts=wgts, verbose=verbose)
        self.gain_psi = fit['psi']
        self.gain_phi = np.array([fit['PHIx'], fit['PHIy']])

    def make_gains(self, gains2dict=False):
        """
        use self.gain_amp and self.gain_phi ane self.gain_psi
        to construct a complex gain array per antenna assuming
        a gain convention of multiply.

        Parameters:
        -----------
        gains2dict : boolean, if True convert gains into dictionary form
            with antenna number as key and ndarray as value
        """
        if hasattr(self, "gain_amp") is False:
            raise NameError("self.gain_amp doesn't exist...")
        if hasattr(self, "gain_psi") is False:
            raise NameError("self.gain_psi doesn't exist...")
        if hasattr(self, "gain_phi") is False:
            raise NameError("self.gain_phi doesn't exist...")

        # form gains
        gain_array = np.ones((self.Nants, self.Ntimes, self.Nfreqs, self.Npols), dtype=np.complex)

        # multiply amplitude
        amps = self.gain_amp[np.newaxis]
        gain_array *= amps

        # multiply phase
        phases = np.exp(-1j*self.gain_psi[np.newaxis] - 1j*np.einsum("ijkl, hi -> hjkl", self.gain_phi, self.antpos[:, :2]))
        gain_array *= phases

        self.gain_array = gain_array

        if gains2dict:
            self.gain_array = odict((a, self.gain_array[i]) for i, a in enumerate(self.ants))

    def run(self, unravel_pol=False, unravel_freq=False, unravel_time=False, verbose=False, gains2dict=False):
        """
        run amp_lincal and phs_logcal on self.model and self.data, and optionally write out 
        gains to a calfits file.

        run Parameters:
        -----------
        calfits_filename : string, path to output calfits file, default=None
        save : boolean, if True, save gains to a calfits file
        overwrite : boolean, if True, overwrite if calfits_filename exists

        amp_lincal & phs_logcal Parameters:
        -----------------------------------
        unravel_pol : type=boolean, see amp_lincal or phs_logcal for details
        unravel_freq : type=boolean, see amp_lincal or phs_logcal for details
        unravel_time : type=boolean, see amp_lincal or phs_logcal for details
        verbose : type=boolean, see amp_lincal or phs_logcal for details
        """

        # run amp cal
        echo("running amp_lincal", type=1, verbose=verbose)
        self.amp_lincal(unravel_freq=unravel_freq, unravel_time=unravel_time, unravel_pol=False, verbose=verbose)

        # run phs cal
        echo("running phs_logcal", type=1, verbose=verbose)
        self.phs_logcal(unravel_freq=unravel_freq, unravel_time=unravel_time, unravel_pol=False, verbose=verbose)

        # make gains
        echo("making gains", type=1, verbose=verbose)
        self.make_gains(gains2dict=gains2dict)


def run_abscal(data_file, model_files, unravel_pol=False, unravel_freq=False, unravel_time=False, verbose=True,
               save=False, calfits_fname=None, output_gains=False, overwrite=False, **kwargs):
    """
    run AbsCal on a single data miriad file

    Parameters:
    -----------
    data_file : path to data miriad file, type=string
        a single miriad file containing complex visibility data

    model_files : path(s) to miriad files(s), type=list
        a list of one or more miriad files containing complex visibility data
        that ** overlaps ** the time and frequency range of data_file

    output_gains : boolean, if True: return AbsCal gains
    """
    # load data
    echo("loading data files", type=1, verbose=verbose)
    echo("loading {}".format(data_file), type=0, verbose=verbose)
    uvd = UVData()
    uvd.read_miriad(data_file)
    data, flags, pols = UVData2AbsCalDict([uvd])
    for i, k in enumerate(data.keys()):
        if k[0] == k[1]:
            data.pop(k)
    for i, k in enumerate(flags.keys()):
        if k[0] == k[1]:
            flags.pop(k)

    # get data params
    data_times = uvd.time_array.reshape(uvd.Ntimes, uvd.Nbls)[:, 0]
    data_freqs = uvd.freq_array.squeeze()
    data_pols = uvd.polarization_array

    # load weights
    wgts = copy.deepcopy(flags)
    for k in wgts.keys():
        wgts[k] = (~wgts[k]).astype(np.float)

    # load antenna positions and make baseline dictionary
    antpos, ants = get_antpos(uvd, center=True, pick_data_ants=True)
    antpos = odict(map(lambda x: (x, antpos[ants.tolist().index(x)]), ants))
    bls = odict([((x[0], x[1]), antpos[x[1]] - antpos[x[0]]) for x in data.keys()])

    # load models
    for i, mf in enumerate(model_files):
        echo("loading {}".format(mf), type=0, verbose=verbose)
        uv = UVData()
        uv.read_miriad(mf)
        if i == 0:
            uvm = uv
        else:
            uvm += uv
    model, mflags, mpols = UVData2AbsCalDict([uvm])
    for i, k in enumerate(model.keys()):
        if k[0] == k[1]:
            model.pop(k)

    # get model params
    model_times = uvm.time_array.reshape(uvm.Ntimes, uvm.Nbls)[:, 0]
    model_freqs = uvm.freq_array.squeeze()

    # align model freq-time axes to data axes
    model = interp_model(model, model_times, model_freqs, data_times, data_freqs, kind='cubic', fill_value=0)

    # check if model has only unique baseline data
    # this is the case if, for example, the model Nbls in less than the data Nbls
    if uvm.Nbls < uvd.Nbls:
        # try to expand model data into redundant baseline groups
        red_info = hc.omni.aa_to_info(hc.utils.get_aa_from_uv(uvm))
        model = mirror_data_to_red_bls(model, red_info)

    # run abscal
    AC = AbsCal(model, data, wgts=wgts, antpos=antpos, freqs=data_freqs, pols=data_pols)
    AC.run(unravel_pol=unravel_pol, unravel_freq=unravel_freq, unravel_time=unravel_time, verbose=verbose, gains2dict=True)

    # write to file
    if save:
        if calfits_fname is None:
            calfits_fname = os.path.basename(data_file) + '.abscal.calfits'
        echo("saving {}".format(calfits_fname), type=1, verbose=verbose)
        gains2calfits(calfits_fname, AC.gain_array, data_freqs, data_times, data_pols,
                      gain_convention='multiply', inttime=10.7, overwrite=overwrite, **kwargs)

    if output_gains:
        return AC.gain_array


def UVData2AbsCalDict(filenames):
    """
    turn pyuvdata.UVData objects or miriad filenames 
    into the dictionary form that AbsCal requires

    Parameters:
    -----------
    filenames : list of either strings to miriad filenames or list of UVData instances

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
        # eliminate autos
        for i, k in enumerate(data_temp.keys()):
            if k[0] == k[1]:
                data_temp.pop(k)
                flag_temp.pop(k)
        # reconfigure polarization nesting
        data = odict()
        flags = odict()
        pol_keys = sorted(data_temp[data_temp.keys()[0]].keys())
        data_keys = sorted(data_temp.keys())
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


def interp_model(model, model_times, model_freqs, data_times, data_freqs,
                 kind='cubic', fill_value=0, zero_tol=1e-5):
    """
    interpolate model complex visibility onto the time-frequency basis of data.
    ** Note: ** this is just a simple wrapper for scipy.interpolate.interp2d

    Parameters:
    -----------
    model : visibility data of model, type=dictionary, see AbsCal for details on format

    model_times : 1D array of the model time axis, dtype=float, shape=(Ntimes,)

    model_freqs : 1D array of the model freq axis, dtype=float, shape=(Nfreqs,)

    data_times : 1D array of the data time axis, dtype=float, shape=(Ntimes,)

    data_freqs : 1D array of the data freq axis of, dtype=float, shape=(Nfreqs,)

    kind : kind of interpolation method, type=str, options=['linear', 'cubic', ...]
        see scipy.interpolate.interp2d for details

    fill_value : values to put for interpolation points outside training set
        if None, values are extrapolated

    zero_tol : for amplitudes lower than this tolerance, set real and imag components to zero
    """
    model = copy.deepcopy(model)
    # loop over keys
    for i, k in enumerate(model.keys()):
        # loop over polarizations
        new_data = []
        for p in range(model[k].shape[2]):
            # interpolate real and imag separately
            interp_real = interpolate.interp2d(model_freqs, model_times, np.real(model[k][:, :, p]),
                                               kind=kind, fill_value=fill_value, bounds_error=False)(data_freqs, data_times)
            interp_imag = interpolate.interp2d(model_freqs, model_times, np.imag(model[k][:, :, p]),
                                               kind=kind, fill_value=fill_value, bounds_error=False)(data_freqs, data_times)

            # force things near amplitude of zero to zero
            zero_select = np.isclose(np.sqrt(interp_real**2 + interp_imag**2), 0.0, atol=zero_tol)
            interp_real[zero_select] *= 0.0 * interp_real[zero_select]
            interp_imag[zero_select] *= 0.0 * interp_imag[zero_select]

            # rejoin
            new_data.append(interp_real + 1j*interp_imag)

        model[k] = np.moveaxis(np.array(new_data), 0, 2)

    return model


def mirror_data_to_red_bls(data, red_info):
    """
    Given unique baseline data (like omnical model visibilities),
    copy the data over to all other baselines in the same redundant group

    Parameters:
    -----------
    data : data dictionary in AbsCal form, see AbsCal docstring for details

    red_info : RedundantInfo object of the array.
        See hera_cal.utils.get_aa_from_uv and hera_cal.omni.aa_to_info methods
        to generate a red_info object.

    Output:
    -------
    red_data : data dictionary in AbsCal form, with unique baseline data
        distributed to redundant baseline groups.

    """
    # get data and ants
    data = copy.deepcopy(data)
    ants = np.unique(np.concatenate([data.keys()]))

    # get redundant baselines
    reds = red_info.get_reds()

    # ensure these reds are antennas pairs of antennas in the data
    pop_reds = []
    for i, bls in enumerate(reds):
        pop_bls = []
        for j, bl in enumerate(bls):
            if bl[0] not in ants or bl[1] not in ants:
                pop_bls.append(j)
        for j in pop_bls[::-1]:
            reds[i].pop(j)
        if len(reds[i]) == 0:
            pop_reds.append(i)
    for i in pop_reds[::-1]:
        reds.pop(i)

    # make red_data dictionary
    red_data = odict()

    # iterate over red bls
    for i, bls in enumerate(reds):
        # find which key in data is in this group
        select = np.array(map(lambda x: x in data.keys(), reds[i]))
        if True not in select:
            continue
        k = reds[i][np.argmax(select)]

        # iterate over bls and insert data into red_data
        for j, bl in enumerate(bls):
            red_data[bl] = copy.copy(data[k])

    # re-sort
    red_data = odict([(k, red_data[k]) for k in sorted(red_data)])

    return red_data


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


def abscal_arg_parser():
    a = argparse.ArgumentParser()
    a.add_argument("--data_file", type=str, help="path to miriad data file to-be-calibrated.", required=True)
    a.add_argument("--model_files", type=str, nargs='*', default=[], help="list of miriad files for visibility model.", required=True)
    a.add_argument("--calfits_fname", type=str, default=None, help="name of output calfits file.")
    a.add_argument("--overwrite", default=False, action='store_true', help="overwrite output calfits file if it exists.")
    a.add_argument("--silence", default=False, action='store_true', help="silence output from abscal while running.")
    a.add_argument("--unravel_time", default=False, action='store_true', help="couple all times together into linsolve equations.")
    return a


def echo(message, type=0, verbose=True):
    if verbose:
        if type == 0:
            print(message)
        elif type == 1:
            print('')
            print(message)
            print("-"*40)


