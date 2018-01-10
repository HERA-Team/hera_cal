"""
abscal.py
---------

Calibrate measured visibility
data to a visibility model using
linearizations of the
(complex) antenna-based calibration equation:

V_ij^model = g_i * conj(g_j) * V_ij^data

where

V_ij^model = exp(eta_ij^model + i * phi_ij^model)
g_i = exp(eta_i + i * phi_i)
g_j = exp(eta_j + i * phi_j)
V_ij^data = exp(eta_ij^data + i * phi_ij^data)
"""
from abscal_funcs import *


class ParentAbsCal(object):

    def __init__(self, model, data, wgts=None, freqs=None, times=None, pols=None):
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
        # get shared keys
        self.keys = sorted(set(model.keys()) & set(data.keys()))

        # append attributes
        self.model = model
        self.data = data

        # setup frequencies
        self.freqs = freqs
        if self.freqs is None:
            self.Nfreqs = None
        else:
            self.Nfreqs = len(self.freqs)

        # setup polarization
        self.str2pol = {"xx": -5, "yy": -6, "xy": -7, "yx": -8}
        if pols is not None:
            if type(pols) is list:
                if type(pols[0]) is str:
                    pols = map(lambda x: self.str2pol[x], pols)
                elif type(pols[0]) is int:
                    pols = pols
            elif type(pols) is str:
                pols = [self.str2pol[pols]]
            elif type(pols) is int:
                pols = [pols]
            self.pols = pols
            self.Npols = len(pols)
        else:
            self.pols = None
            self.Npols = None

        # setup weights
        if wgts is None:
            wgts = odict()
            for i, k in enumerate(self.keys):
                wgts[k] = np.ones_like(data[k], dtype=np.float)
        self.wgts = wgts

        # setup times
        self.times = times
        if times is None:
            self.Ntimes = None
        else:
            self.Ntimes = len(self.times)

        # setup ants
        self.ants = np.unique(np.concatenate(map(lambda k: k[:2], self.keys)))
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

    def __init__(self, model, data, antpos, wgts=None, freqs=None, times=None, pols=[None]):
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
        super(OmniAbsCal, self).__init__(model, data, wgts=wgts, freqs=freqs, times=times, pols=pols)

        # setup baselines and antenna positions
        self.antpos = antpos
        if self.antpos is not None:
            self.bls = odict([(x, self.antpos[x[1]] - self.antpos[x[0]]) for x in self.keys])
            self.antpos = np.array(map(lambda x: self.antpos[x], self.ants))
            self.antpos -= np.median(self.antpos, axis=0)

    def abs_amp_lincal(self, separate_pol=False, verbose=True):
        """
        call abscal_funcs.abs_amp_lincal() method. see its docstring for more details.

        Parameters:
        -----------
        separate_pol : bool, separate polarization to have independent solutions
            if False, form a joint solution across both pols

        Result:
        -------
        Absolute amplitude scalar can be accessed via methods
            self.get_abs_amp()
            self.get_abs_amp_gain()
        """
        # set data quantities
        model = self.model
        data = self.data
        wgts = self.wgts

        if separate_pol:
            model, pols = data_key_to_array_axis(model, 2)
            data, pols = data_key_to_array_axis(data, 2)
            wgts, pols = data_key_to_array_axis(wgts, 2)

        # run abs_amp_lincal
        fit = abs_amp_lincal(model, data, wgts=wgts, verbose=verbose)

        # form result
        self._abs_amp = np.sqrt(fit['amp'])
        if separate_pol is False:
            self._abs_amp = np.moveaxis(self._abs_amp[np.newaxis], 0, -1)

        # form gain
        self._abs_amp_gain = np.sqrt(self._abs_amp.astype(np.complex)[np.newaxis])

    def TT_phs_logcal(self, separate_pol=False, verbose=True, zero_psi=False):
        """
        call abscal_funcs.TT_phs_logcal() method. see its docstring for more details.

        Parameters:
        -----------
        separate_pol : bool, separate polarization to have independent solutions
            if False, form a joint solution across both pols

        Result:
        -------
        Tip-Tilt phase slope fit can be accessed via methods
            self.get_abs_psi()
            self.get_abs_psi_gain()
            self.get_TT_Phi()
            self.get_TT_Phi_gain()
        """
        # set data quantities
        model = self.model
        data = self.data
        wgts = self.wgts
        bls = self.bls

        if separate_pol:
            model, bls, pols = data_key_to_array_axis(model, 2, avg_dict=bls)
            data, pols = data_key_to_array_axis(data, 2)
            wgts, pols = data_key_to_array_axis(wgts, 2)

        # run TT_phs_logcal
        fit = TT_phs_logcal(model, data, bls, wgts=wgts, verbose=verbose, zero_psi=zero_psi)

        # form result
        self._abs_psi = fit['psi']
        self._TT_Phi = np.array([fit['PHIx'], fit['PHIy']])
        if separate_pol is False:
            self._abs_psi = np.moveaxis(self._abs_psi[np.newaxis], 0, -1)
            self._TT_Phi = np.moveaxis(self._TT_Phi[np.newaxis], 0, -1)

        # form gains
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

    def __init__(self, model, data, wgts=None, freqs=None, times=None, pols=[None]):
        """
        AbsCal object used for absolute bandpass calibration.

        model, data and wgts should be fed as dictionary types.

        Parameters:
        -----------
        See hera_cal.abscal.ParentAbsCal doc string for details on positional and keyword arguments.
        """
        super(AbsCal, self).__init__(model, data, wgts=wgts, freqs=freqs, times=times, pols=pols)

    def amp_logcal(self, separate_pol=False, verbose=True):
        """
        call abscal_funcs.amp_logcal() method. see its docstring for more details.

        Parameters:
        -----------
        separate_pol : bool, separate polarization to have independent solutions
            if False, form a joint solution across both pols

        Result:
        -------
        per-antenna amplitude and per-antenna amp gains
        can be accessed via the methods
            self.get_ant_eta()
            self.get_ant_eta_gain()
        """
        # set data quantities
        model = self.model
        data = self.data
        wgts = self.wgts

        if separate_pol:
            model, pols = data_key_to_array_axis(model, 2)
            data, pols = data_key_to_array_axis(data, 2)
            wgts, pols = data_key_to_array_axis(wgts, 2)

        # run linsolve
        fit = amp_logcal(model, data, wgts=wgts, verbose=verbose)

        # form result array
        self._ant_eta = np.array(map(lambda a: fit['eta{}'.format(a)], self.ants))

        # add polarization axis if formed a joint pol solution
        if separate_pol is False:
            self._ant_eta = np.moveaxis(self._ant_eta[np.newaxis], 0, -1)

        # form gain array
        self._ant_eta_gain = np.exp(self._ant_eta).astype(np.complex)

    def phs_logcal(self, separate_pol=False, verbose=True):
        """
        call abscal_funcs.phs_logcal() method. see its docstring for more details.

        Parameters:
        -----------
        separate_pol : bool, separate polarization to have independent solutions
            if False, form a joint solution across both pols

        Result:
        -------
        per-antenna phase and per-antenna phase gains
        can be accessed via the methods
            self.get_ant_phi()
            self.get_ant_phi_gain()
        """
        # assign data
        model = self.model
        data = self.data
        wgts = self.wgts

        # separate pol
        if separate_pol:
            model, pols = data_key_to_array_axis(model, 2)
            data, pols = data_key_to_array_axis(data, 2)
            wgts, pols = data_key_to_array_axis(wgts, 2)

        # run linsolve
        fit = phs_logcal(model, data, wgts=wgts, verbose=verbose)

        # form result array
        self._ant_phi = np.array(map(lambda a: fit['phi{}'.format(a)], self.ants))

        # add polarization axis if formed one solutions
        if separate_pol is False:
            self._ant_phi = np.moveaxis(self._ant_phi[np.newaxis], 0, -1)

        # form gain array
        self._ant_phi_gain = np.exp(-1j*self._ant_phi)

    def delay_lincal(self, kernel=(1, 11), time_ax=0, freq_ax=1, verbose=True):
        """
        Solve for per-antenna delay according to the equation
        by calling abscal_funcs.delay_lincal method.
        See abscal_funcs.delay_lincal for details.

        Currently only supports joint polarization solutions.

        Parameters:
        -----------
        kernel : size of median filter across (time, freq) axes, type=(int, int)

        Result:
        -------
        per-antenna delays and per-antenna delay gains
        can be accessed via the methods
            self.get_ant_dly()
            self.get_ant_dly_gain()
        """
        # check for freq data
        if hasattr(self, 'freqs') is False:
            raise AttributeError("cannot delay_lincal without self.freqs array")

        # assign data
        model = self.model
        data = self.data
        wgts = self.wgts

        # get freq channel width
        df = np.median(np.diff(self.freqs))

        # run delay_lincal
        fit = delay_lincal(model, data, df=df, kernel=kernel, verbose=verbose, time_ax=time_ax, freq_ax=freq_ax)

        # turn into array
        self._ant_dly = np.array(map(lambda a: np.moveaxis(dlys['tau{}'.format(a)], 0, 2), self.ants))
        self._ant_dly = np.moveaxis(self._ant_dly[np.newaxis], 0, -1)
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
            return copy.copy(self._ant_phi_gain)
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