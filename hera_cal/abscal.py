"""
abscal.py
---------

Calibrate measured visibility
data to a visibility model using
linerizations of the
(complex) calibration equation:

V_ij^model = g_i * conj(g_j) * V_ij^data

where

V_ij^model = exp(eta_ij^model + i * phi_ij^model)
g_i = exp(eta_i + i * phi_i)
g_j = exp(eta_j + i * phi_j)
V_ij^data = exp(eta_ij^data + i * phi_ij^data)
"""
from abscal_funcs import *


class ParentAbsCal(object):

    def __init__(self, model, data, wgts=None, freqs=None, times=None, pols=[None]):
        """
        ParentAbsCal object used to setup abscal instance.

        model, data and wgts should be fed as dictionary types.

        Parameters:
        -----------
        model : dict of visibility data of refence model, type=dictionary
            keys are antenna pair tuples, values are complex ndarray visibilities
            these visibilities must be 3D arrays, with the [0] axis indexing time,
            the [1] axis indexing frequency and the [2] axis indexing polarization

        data : dict of visibility data of measurements, type=dictionary
            keys are antenna pair tuples (must match model), values are
            complex ndarray visibilities, with shape matching model

        wgts : dict of weights of data, type=dictionry, [default=None]
            keys are antenna pair tuples (must match model), values are real floats
            matching shape of model and data

        freqs : ndarray of frequency array, type=ndarray, dtype=float
            1d array containing visibility frequencies in Hz. Needed to write gain solutions
            out to calfits.
    
        times : ndarray of time array, type=ndarray, dtype=float
            1d array containing visibility times in Julian Date. Needed to write out to calfits.

        pols : ndarray of polarization array, type=ndarray, dtype=int
            array containing polarization integers in pyuvdata.UVData.polarization_array 
            format. Needed to write out to calfits.
        """
        # append attributes
        self.model = model
        self.data = data

        # setup frequencies
        self.Nfreqs = model[model.keys()[0]].shape[1]
        self.freqs = freqs

        # setup polarization
        self.str2pol = {"xx": -5, "yy": -6, "xy": -7, "yx": -8}
        self.Npols = model[model.keys()[0]].shape[2]
        if pols is not None:
            if type(pols) is list:
                if type(pols[0]) is str:
                    pols = map(lambda x: self.str2pol[x], pols)
            elif type(pols) is str:
                pols = [self.str2pol[pols]]
            elif type(pols) is int:
                pols = [pols]
        self.pols = pols

        # setup weights
        if wgts is None:
            wgts = copy.deepcopy(data)
            for i, k in enumerate(data.keys()):
                wgts[k] = np.ones_like(wgts[k], dtype=np.float)
        self.wgts = wgts

        # setup times
        self.Ntimes = model[model.keys()[0]].shape[0]
        self.times = times

        # setup ants
        self.ants = np.unique(sorted(np.array(map(lambda x: [x[0], x[1]], model.keys())).ravel()))
        self.Nants = len(self.ants)


class OmniAbsCal(ParentAbsCal):
    """
    OmniAbsCal object used specifically for calibrating omnical degeneracies:
    the absolute amplitude scalar, and Tip-Tilt phase slopes.

    1) Average amplitude linear calibration solves the linear equation:
            |V_ij^model / V_ij^data| = |g_avg|
 
    2) Tip-Tilt phase logarithmic calibration 
            angle(V_ij^model /  V_ij^data) = psi + dot(TT_Phi, B_ij)
        where psi is an overall gain phase scalar, 
        TT_Phi is the gain phase slope vector [radians / meter]
        and B_ij is the baseline vector between antenna i and j.
    """

    def __init__(self, antpos, *init_args, **init_kwargs):
        """
        OmniAbsCal object used specifically for calibrating omnical degeneracies:
        the absolute amplitude scalar, and Tip-Tilt phase slopes.
        model, data and wgts should be fed as dictionary types.

        Parameters:
        -----------
        See hera_cal.abscal.ParentAbsCal doc string for details on positional and keyword arguments.

        antpos : type=dictionary, dict of antenna position vectors in TOPO frame in meters
                 keys are antenna integers and values are 2D or 3D ndarray
                 position vectors in meters (topocentric coordinates),
                 with [0] index containing X (E-W) distance, and [1] index Y (N-S) distance.
                 Can be generated from a pyuvdata.UVData instance via
                 ----
                 #!/usr/bin/env python
                 uvd = pyuvdata.UVData()
                 uvd.read_miriad(<filename>)
                 antenna_pos, ants = uvd.get_ENU_antpos()
                 antpos = dict(zip(ants, antenna_pos))
                 ----
                 This is needed for Tip Tilt phase calibration, but not for absolute amplitude
                 calibration.
        """
        super(OmniAbsCal, self).__init__(*init_args, **init_kwargs)

        # setup baselines and antenna positions
        self.antpos = antpos
        if self.antpos is not None:
            self.bls = odict([((x[0], x[1]), self.antpos[x[1]] - self.antpos[x[0]]) for x in self.model.keys()])
            self.antpos = np.array(map(lambda x: self.antpos[x], self.ants))
            self.antpos -= np.median(self.antpos, axis=0)

    def abs_amp_lincal(self, unravel_freq=False, unravel_time=False, unravel_pol=False,
                       apply_cal=False, verbose=True):
        """
        call abscal_funcs.abs_amp_lincal() method. see its docstring for more details.

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

        apply_cal : turn calibration solution into gains and apply to data

        Result:
        -------
        Results can be accessd via the instance methods
            self.get_abs_amp()
            self.get_abs_amp_gain()
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
        fit = abs_amp_lincal(model, data, wgts=wgts, verbose=verbose)
        self._abs_amp = copy.copy(np.sqrt(fit['amp']))
        self._abs_amp_gain = np.exp(self.abs_amp).astype(np.complex)[np.newaxis]

    def TT_phs_logcal(self, unravel_freq=False, unravel_time=False, unravel_pol=False,
                      verbose=True, zero_psi=False):
        """
        call abscal_funcs.TT_phs_logcal() method. see its docstring for more details.

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

        Result:
        -------
        Results can be accessed via instance methods
            self.get_abs_psi()
            self.get_abs_psi_gain()
            self.get_TT_Phi()
            self.get_TT_Phi_gain()
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
        fit = TT_phs_logcal(model, data, bls, wgts=wgts, verbose=verbose, zero_psi=zero_psi)
        self._abs_psi = fit['psi']
        self._TT_Phi = np.array([fit['PHIx'], fit['PHIy']])
        self._abs_psi_gain = np.exp(-1j*self._abs_psi)[np.newaxis]
        self._TT_Phi_gain = np.exp(-1j*np.einsum("ijkl, hi -> hjkl",self._TT_Phi, self.antpos[:, :2]))

    @property
    def get_abs_amp(self):
        """return _abs_amp array"""
        if hasattr(self, '_abs_amp'):
            return copy.copy(self._abs_amp)
        else:
            return None

    @property
    def get_abs_amp_gain(self):
        """return _abs_amp_gain array"""
        if hasattr(self, '_abs_amp_gain'):
            return copy.copy(self._abs_amp_gain)
        else:
            return None

    @property
    def get_abs_psi(self):
        """return _abs_psi array"""
        if hasattr(self, '_abs_psi'):
            return copy.copy(self._abs_psi)
        else:
            return None

    @property
    def get_abs_psi_gain(self):
        """return _abs_psi_gain array"""
        if hasattr(self, '_abs_psi_gain'):
            return copy.copy(self._abs_psi_gain)
        else:
            return None

    @property
    def get_TT_Phi(self):
        """return _TT_Phi array"""
        if hasattr(self, '_TT_Phi'):
            return copy.copy(self._TT_Phi)
        else:
            return None

    @property
    def get_TT_Phi_gain(self):
        """return _TT_Phi_gain array"""
        if hasattr(self, '_TT_Phi_gain'):
            return copy.copy(self._TT_Phi_gain)
        else:
            return None


class AbsCal(ParentAbsCal):
    """
    AbsCal object to perform delay and absolute bandpass calibration.

    Three calibration methods exist.

    1) per-antenna amplitude logarithmic calibration
       solves the equation:
            ln[abs(V_ij^model / V_ij^data)] = eta_i + eta_j

    2) per-antenna phase logarithmic calibration
       solves the equation:
           angle(V_ij^model / V_ij^data) = phi_i - phi_j

    3) delay linear calibration solves the equation:
           tau_ij^model - tau_ij^data = tau_i - tau_j
       where tau is the delay that can be turned
       into a complex gain via: g = exp(i * 2pi * tau * freqs).
    """ 

    def __init__(self, *init_args, **init_kwargs):
        """
        AbsCal object used for absolute bandpass calibration.

        model, data and wgts should be fed as dictionary types.

        Parameters:
        -----------
        See hera_cal.abscal.ParentAbsCal doc string for details on positional and keyword arguments.
        """
        super(AbsCal, self).__init__(*init_args, **init_kwargs)

    def amp_logcal(self, unravel_freq=False, unravel_time=False, unravel_pol=False, verbose=True):
        """
        call abscal_funcs.amp_logcal() method. see its docstring for more details.

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
        fit = amp_logcal(model, data, wgts=wgts, verbose=verbose)
        self._ant_eta = np.array(map(lambda a: fit['eta{}'.format(a)], self.ants))
        self._ant_eta_gain = np.exp(self._ant_eta)

    def phs_logcal(self, unravel_freq=False, unravel_time=False, unravel_pol=False, verbose=True):
        """
        call abscal_funcs.phs_logcal() method. see its docstring for more details.

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
        fit = phs_logcal(model, data, wgts=wgts, verbose=verbose)
        self._ant_phi = np.array(map(lambda a: fit['phi{}'.format(a)], self.ants))
        self._ant_phi_gain = np.exp(-1j*self._ant_phi)

    def delay_lincal(self, kernel=(1, 11), verbose=True):
        """
        Solve for per-antenna delay according to the equation
        by calling abscal_funcs.delay_lincal method.
        See abscal_funcs.delay_lincal for details.

        Parameters:
        -----------
        kernel : size of median filter across (time, freq) axes, type=(int, int)
        """
        # check for freq data
        if hasattr(self, 'freqs') is False:
            raise AttributeError("cannot delay_lincal without self.freqs array")

        # copy data
        model = copy.deepcopy(self.model)
        data = copy.deepcopy(self.data)

        # get freq channel width
        df = np.median(np.diff(self.freqs))

        # iterate over polarizations
        dlys = odict(map(lambda a: ('tau{}'.format(a), []), self.ants))
        for i, p in enumerate(self.pols):
            # run linsolve
            m = odict(zip(model.keys(), map(lambda k: model[k][:, :, i], model.keys())))
            d = odict(zip(data.keys(), map(lambda k: data[k][:, :, i], data.keys())))
            fit = delay_lincal(m, d, df=df, kernel=kernel, verbose=verbose, time_ax=0, freq_ax=1)
            for j, k in enumerate(fit.keys()):
                dlys[k].append(fit[k])

        # turn into array
        self._ant_dly = np.array(map(lambda a: np.moveaxis(dlys['tau{}'.format(a)], 0, 2), self.ants))
        self._ant_dly_gain = np.exp(-2j*np.pi*self.freqs.reshape(-1, 1)*self._delays)

    @property
    def get_ant_eta(self):
        if hasattr(self, '_ant_eta'):
            return copy.copy(self._ant_eta)
        else:
            return None

    @property
    def get_ant_eta_gain(self):
        if hasattr(self, '_ant_eta_gain'):
            return copy.copy(self._ant_eta_gain)
        else:
            return None

    @property
    def get_ant_phi(self):
        if hasattr(self, '_ant_phi'):
            return copy.copy(self._ant_phi)
        else:
            return None

    @property
    def get_ant_phi_gain(self):
        if hasattr(self, '_ant_phi_gain'):
            return copy.copy(self._ant_phi)
        else:
            return None

    @property
    def get_ant_dly(self):
        if hasattr(self, '_ant_dly'):
            return copy.copy(self._ant_dly)
        else:
            return None

    @property
    def get_ant_dly_gain(self):
        if hasattr(self, '_ant_dly_gain'):
            return copy.copy(self._ant_dly_gain)
        else:
            return None



''' TO DO
    def smooth_data(self, data, flags=None, kind='linear'):
        """


        Parameters:
        -----------
        data : 


        flags : 
    
        """
        # configure parameters



        Ntimes
        Nfreqs

        # interpolate flagged data


        # sphere training data x-values


        # ravel training data


        if kind == 'poly':
            # fit polynomial
            data = smooth_data(Xtrain_raveled, ytrain_raveled, Xpred_raveled, kind=kind, degree=degree)

        if kind == 'gp':
            # construct GP mean function from a degree-order polynomial
            MF = make_pipeline(PolynomialFeatures(degree), linear_model.RANSACRegressor())
            MF.fit(Xtrain_raveled, ytrain_raveled)
            y_mean = MF.predict(Xtrain_raveled)

            # make residual and normalize by std
            y_resid = (ytrain_raveled - y_mean).reshape(Ntimes, Nfreqs)
            y_std = np.sqrt(astats.biweight_midvariance(y_resid.ravel()))
            y_resid /= y_std

            # average residual across time
            ytrain = np.mean(y_resid, axis=0)

            # ravel training data
            Xtrain_raveled = Xtrain[0, :].reshape(-1, 1)
            ytrain_raveled = ytrain
            Xpred_raveled = Xpred[0, :]

            # fit GP and predict MF
            y_pred = smooth_data(Xtrain_raveled, ytrain_raveled, Xpred_raveled) * y_std
            y_pred = np.repeat(y_pred, Ntimes)
            data = (y_pred + y_mean).reshape(Ntimes, Nfreqs)
'''