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


class AbsCal(object):
    """
    AbsCal object used to for phasing and scaling visibility data to an absolute reference model. 
    A few different calibration methods exist. These include:

    1) per-antenna amplitude logarithmic calibration solves the equation:
            ln[abs(V_ij^model / V_ij^data)] = eta_i + eta_j

    2) per-antenna phase logarithmic calibration solves the equation:
           angle(V_ij^model / V_ij^data) = phi_i - phi_j

    3) delay linear calibration solves the equation:
           delay(V_ij^model / V_ij^data) = delay(g_i) - delay(g_j)
                                         = tau_i - tau_j
       where tau is the delay that can be turned
       into a complex gain via: g = exp(i * 2pi * tau * freqs).

    4) Average amplitude linear calibration solves the equation:
            |V_ij^model / V_ij^data| = |g_avg|
 
    5) Tip-Tilt phase logarithmic calibration solves the equation
            angle(V_ij^model /  V_ij^data) = psi + dot(TT_Phi, B_ij)
        where psi is an overall gain phase scalar, 
        TT_Phi is the gain phase slope vector [radians / meter]
        and B_ij is the baseline vector between antenna i and j.

    Methods (1), (2) and (3) can be thought of as general bandpass solvers, whereas
    methods (3), (4) and (5) are methods that would be used for data that has already
    been redundantly calibrated.

    Be warned that the linearizations of the phase solvers suffer from phase wrapping
    pathologies, meaning that a delay calibration should generally precede a phs_logcal
    or a TT_phs_logcal solution.
    """
    def __init__(self, model, data, wgts=None, antpos=None, freqs=None, times=None, pols=None):
        """
        AbsCal object used to for phasing and scaling visibility data to an absolute reference model.

        The format of model, data and wgts is in the AbsCal dictionary format. This is a standard
        python dictionary or OrderedDictionary, with the convention that keys contain
        antennas-pairs + polarization, Ex. (1, 2, 'xx'),
        and values contain 2D complex ndarrays with [0] axis indexing time and [1] axis frequency.

        Optionally, a singe key can hold multiple polarizations, in which case the key loses
        its polarization, Ex. (1, 2), and the value becomes a 3D complex ndarray, with [2] axis
        indexing polarizations.

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

        antpos : type=dictionary, dict of antenna position vectors in ENU frame in meters.
                 origin of coordinates does not matter.
                 keys are antenna integers and values are 2D or 3D ndarray
                 position vectors in meters (topocentric coordinates),
                 with [0] index containing X (E-W) distance, and [1] index Y (N-S) distance
                 and [2] indexing Z (up-down) distance.
                 Can be generated from a pyuvdata.UVData instance via
                 ----
                 #!/usr/bin/env python
                 uvd = pyuvdata.UVData()
                 uvd.read_miriad(<filename>)
                 antenna_pos, ants = uvd.get_ENU_antpos()
                 antpos = dict(zip(ants, antenna_pos))
                 ----
                 This is needed only for Tip Tilt phase calibration.

        freqs : ndarray of frequency array, type=ndarray, dtype=float
            1d array containing visibility frequencies in Hz. Needed to write gain solutions
            out to calfits.
    
        times : ndarray of time array, type=ndarray, dtype=float
            1d array containing visibility times in Julian Date. Needed to write out to calfits.

        pols : list of polarizations, type=list, dtype=int
            list containing polarization integers in pyuvdata.UVData.polarization_array 
            format. Needed to write out to calfits. Can also be a single string
            Can be in {-5, -6, -7, -8} format or {'xx', 'yy', 'xy', 'yx'} format.
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
            if type(pols) is list or type(pols) is np.ndarray:
                if type(pols[0]) is str:
                    pols = map(lambda x: self.str2pol[x.lower()], pols)
                elif type(pols[0]) is int:
                    pols = pols
            elif type(pols) is str:
                pols = [self.str2pol[pols.lower()]]
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
            for k in self.keys:
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

        # setup baselines and antenna positions
        self.antpos = antpos
        self.antpos_arr = None
        self.bls = None
        if self.antpos is not None:
            self.bls = odict([(x, self.antpos[x[1]] - self.antpos[x[0]]) for x in self.keys])
            self.antpos_arr = np.array(map(lambda x: self.antpos[x], self.ants))
            self.antpos_arr -= np.median(self.antpos_arr, axis=0)

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
        can be accessed via the get functions
            self.ant_eta
            self.ant_eta_gain
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
            self.ant_phi
            self.ant_phi_gain
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

    def delay_lincal(self, medfilt=True, kernel=(1, 11), time_ax=0, freq_ax=1, verbose=True, time_avg=False):
        """
        Solve for per-antenna delay according to the equation
        by calling abscal_funcs.delay_lincal method.
        See abscal_funcs.delay_lincal for details.

        Currently only supports joint polarization solutions.

        Parameters:
        -----------
        medfilt : boolean, if True median filter data before fft

        kernel : size of median filter across (time, freq) axes, type=(int, int)

        time_avg : boolean, if True, average resultant antenna delays across time 

        Result:
        -------
        per-antenna delays and per-antenna delay gains
        can be accessed via the methods
            self.ant_dly
            self.ant_dly_gain
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
        fit = delay_lincal(model, data, medfilt=medfilt, df=df, kernel=kernel, verbose=verbose,
                           time_ax=time_ax, freq_ax=freq_ax)

        # turn into array
        self._ant_dly = np.array(map(lambda a: fit['tau{}'.format(a)], self.ants))

        # time avg
        if time_avg:
            Ntimes = self._ant_dly.shape[time_ax+1]
            self._ant_dly = np.moveaxis(np.median(self._ant_dly, axis=time_ax+1)[np.newaxis], 0, time_ax+1)
            self._ant_dly = np.repeat(self._ant_dly, Ntimes, axis=time_ax+1)

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
            self.abs_amp
            self.abs_amp_gain
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
            self.abs_psi
            self.abs_psi_gain
            self.TT_Phi
            self.TT_Phi_gain
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

    @property
    def ant_eta(self):
        """ return _ant_eta array """
        if hasattr(self, '_ant_eta'):
            return copy.copy(self._ant_eta)
        else:
            return None

    @property
    def ant_eta_gain(self):
        """ form complex gain from _ant_eta array """
        if hasattr(self, '_ant_eta'):
            return np.exp(self.ant_eta).astype(np.complex)
        else:
            return None

    @property
    def ant_phi(self):
        """ return _ant_phi array """
        if hasattr(self, '_ant_phi'):
            return copy.copy(self._ant_phi)
        else:
            return None

    @property
    def ant_phi_gain(self):
        """ form complex gain from _ant_phi array """
        if hasattr(self, '_ant_phi'):
            return np.exp(1j*self.ant_phi)
        else:
            return None

    @property
    def ant_dly(self):
        """ return _ant_dly array """
        if hasattr(self, '_ant_dly'):
            return copy.copy(self._ant_dly)
        else:
            return None

    @property
    def ant_dly_gain(self):
        """ form complex gain from _ant_dly array """
        if hasattr(self, '_ant_dly'):
            return np.exp(2j*np.pi*self.freqs*self.ant_dly)
        else:
            return None

    @property
    def abs_amp(self):
        """return _abs_amp array"""
        if hasattr(self, '_abs_amp'):
            return copy.copy(self._abs_amp)
        else:
            return None

    @property
    def abs_amp_gain(self):
        """form complex gain from _abs_amp array"""
        if hasattr(self, '_abs_amp'):
            return np.repeat(np.sqrt(self._abs_amp).astype(np.complex)[np.newaxis], len(self.ants), axis=0)
        else:
            return None

    @property
    def abs_psi(self):
        """return _abs_psi array"""
        if hasattr(self, '_abs_psi'):
            return copy.copy(self._abs_psi)
        else:
            return None

    @property
    def abs_psi_gain(self):
        """ form complex gain from _abs_psi array """
        if hasattr(self, '_abs_psi'):
            return np.repeat(np.exp(1j*self._abs_psi)[np.newaxis], len(self.ants), axis=0)
        else:
            return None

    @property
    def TT_Phi(self):
        """return _TT_Phi array"""
        if hasattr(self, '_TT_Phi'):
            return copy.copy(self._TT_Phi)
        else:
            return None

    @property
    def TT_Phi_gain(self):
        """ form complex gain from _TT_Phi array """
        if hasattr(self, '_TT_Phi'):
            return np.exp(1j*np.einsum("i...,hi->h...", self._TT_Phi, self.antpos_arr[:, :2]))
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