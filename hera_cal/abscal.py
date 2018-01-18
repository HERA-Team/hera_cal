"""
abscal.py
---------

Calibrate measured visibility
data to a visibility model using
linearizations of the (complex)
antenna-based calibration equation:

V_ij,xy^model = g_i_x * conj(g_j_y) * V_ij,xy^data.

Complex-valued parameters are broken into amplitudes and phases as:

V_ij,xy^model = exp(eta_ij,xy^model + i * phi_ij,xy^model)
g_i_x = exp(eta_i_x + i * phi_i_x)
g_j_y = exp(eta_j_y + i * phi_j_y)
V_ij,xy^data = exp(eta_ij,xy^data + i * phi_ij,xy^data)

where {i,j} index antennas and {x,y} are the polarization of
the i-th and j-th antenna respectively.
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
            log|V_ij^model / V_ij^data| = log|g_avg_i| + log|g_avg_j|
 
    5) Tip-Tilt phase logarithmic calibration solves the equation
            angle(V_ij^model /  V_ij^data) = psi + dot(TT_Phi, B_ij)
        where psi is an overall gain phase scalar, 
        TT_Phi is the gain phase slope vector [radians / meter]
        and B_ij is the baseline vector between antenna i and j.

    Methods (1), (2) and (3) can be thought of as general bandpass solvers, whereas
    methods (3), (4) and (5) are methods that would be used for data that has already
    been redundantly calibrated.

    Be warned that the linearizations of the phase solvers suffer from phase wrapping
    pathologies, meaning that a delay calibration and then average phase calibration should
    generally precede a phs_logcal or a TT_phs_logcal bandpass routine.
    """

    def __init__(self, model, data, wgts=None, antpos=None, freqs=None, pol_select=None,
                 model_ftype='miriad', data_ftype='miriad'):
        """
        AbsCal object used to for phasing and scaling visibility data to an absolute reference model.

        The format of model, data and wgts is in the AbsCal dictionary format. This is a standard
        python dictionary or OrderedDictionary, with the convention that keys contain
        antennas-pairs + polarization, Ex. (1, 2, 'xx'),
        and values contain 2D complex ndarrays with [0] axis indexing time and [1] axis frequency.

        Parameters:
        -----------
        model : visibility data of refence model, type=dictionary
                keys are antenna-pair + polarization tuples, Ex. (1, 2, 'xx').
                values are complex ndarray visibilities.
                these must be at least 2D arrays, with [0] axis indexing time
                and [1] axis indexing frequency. If the arrays are 3D arrays, the [2] axis
                should index polarization, in which case the key loses its pol entry, Ex. (1, 2).
 
                Optionally, model can be a path to a miriad file, or a pyuvdata.UVData object
                with a miriad file read-in, or a list of either ones.

        model_ftype : type=str, if model is a path to a file, this is its filetype
                      options=['miriad', 'uvfits']

        data : visibility data of measurements, type=dictionary
               keys are antenna pair + pol tuples (must match model), values are
               complex ndarray visibilities matching shape of model

                Optionally, model can be a path to a miriad file, or a pyuvdata.UVData object
                with a miriad file read-in, or a list of either ones. In this case, wgts, antpos, 
                freqs, times and pols will be overwritten with equivalent information from data.

        data_ftype : type=str, if data is a path to a file, this is its filetype
                     options=['miriad', 'uvfits']

        wgts : weights of data, type=dictionry, [default=None]
               keys are antenna pair + pol tuples (must match model), values are real floats
               matching shape of model and data

        antpos : type=dictionary, dict of antenna position vectors in ENU (topo) frame in meters.
                 origin of coordinates does not matter, but preferably are centered in the array.
                 keys are antenna integers and values are 2D ndarray position vectors,
                 with [0] index containing East-West distance, and [1] index North-South distance.
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
                1d array containing visibility frequencies in Hz.
                Needed for delay calibration.
    
        pol_select : list of polarizations you want to keep in data
                     type=list, dtype=str, Ex. ['xx', 'yy']
        """
        pols = None
        # check format of model
        if type(model) == list or type(model) == np.ndarray or type(model) == str or type(model) == UVData:
            (model, model_flags, model_antpos, model_ants, model_freqs,
             model_times, model_pols) = UVData2AbsCalDict(model, pop_autos=True, return_meta=True, pol_select=pol_select)

        # check format of data
        if type(data) == list or type(data) == np.ndarray or type(data) == str or type(data) == UVData:
            (data, flags, antpos, ants, freqs,
             times, pols) = UVData2AbsCalDict(data, pop_autos=True, return_meta=True, pol_select=pol_select)
            wgts = DataContainer(odict(map(lambda k: (k, (~flags[k]).astype(np.float)), flags.keys())))

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
        self.pol2str = {-5:"xx", -6:"yy", -7:"xy", -8:"yx"}

        # get pols is not defined, if so, make sure they are string format
        if pols is None:
            pols = np.unique(map(lambda k: k[2], self.keys))
        elif type(pols) == np.ndarray or type(pols) == list:
            if np.issubdtype(type(pols[0]), int):
                pols = map(lambda p: self.pol2str[p], pols)

        # convert to integer format
        self.pols = pols
        self.pols = map(lambda p: self.str2pol[p], self.pols)
        self.Npols = len(self.pols)

        # save pols in string format and get gain_pols
        self.polstrings = np.array(map(lambda p: self.pol2str[p], self.pols))
        self.gain_pols = np.unique(map(lambda p: [p[0], p[1]], self.polstrings))
        self.Ngain_pols = len(self.gain_pols)

        # setup weights
        if wgts is None:
            wgts = odict()
            for k in self.keys:
                wgts[k] = np.ones_like(data[k], dtype=np.float)
        self.wgts = wgts

        # setup ants
        self.ants = np.unique(np.concatenate(map(lambda k: k[:2], self.keys)))
        self.Nants = len(self.ants)

        # setup antenna positions
        self.antpos = antpos
        self.antpos_arr = None
        self.bls = None
        if self.antpos is not None:
            self.bls = odict([(x, self.antpos[x[0]] - self.antpos[x[1]]) for x in self.keys])
            self.antpos_arr = np.array(map(lambda x: self.antpos[x], self.ants))
            self.antpos_arr -= np.median(self.antpos_arr, axis=0)

        # setup gain solution keys
        self._gain_keys = map(lambda p: map(lambda a: (a, p), self.ants), self.gain_pols)
        self._flatten = lambda l: [item for sublist in l for item in sublist]

    def amp_logcal(self, verbose=True):
        """
        call abscal_funcs.amp_logcal() method. see its docstring for more details.

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
        self._ant_eta = odict(map(lambda k: (k, copy.copy(fit["eta_{}_{}".format(k[0], k[1])])), self._flatten(self._gain_keys)))
        self._ant_eta_arr = np.moveaxis(map(lambda pk: map(lambda k: self._ant_eta[k], pk), self._gain_keys), 0, -1)

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
        fit = phs_logcal(model, data, wgts=wgts, verbose=verbose)

        # form result array
        self._ant_phi = odict(map(lambda k: (k, copy.copy(fit["phi_{}_{}".format(k[0], k[1])])), self._flatten(self._gain_keys)))
        self._ant_phi_arr = np.moveaxis(map(lambda pk: map(lambda k: self._ant_phi[k], pk), self._gain_keys), 0, -1)

        # take time and freq average
        if avg:
            self._ant_phi = odict(map(lambda k: (k, np.ones_like(self._ant_phi[k])*np.angle(np.median(np.real(np.exp(1j*self._ant_phi[k])))+1j*np.median(np.imag(np.exp(1j*self._ant_phi[k]))))), self._flatten(self._gain_keys)))
            self._ant_phi_arr = np.moveaxis(map(lambda pk: map(lambda k: self._ant_phi[k], pk), self._gain_keys), 0, -1)

    def delay_lincal(self, medfilt=True, kernel=(1, 11), time_ax=0, freq_ax=1, verbose=True, time_avg=False,
                     solve_offsets=True):
        """
        Solve for per-antenna delay according to the equation
        by calling abscal_funcs.delay_lincal method.
        See abscal_funcs.delay_lincal for details.

        Parameters:
        -----------
        medfilt : boolean, if True median filter data before fft

        kernel : size of median filter across (time, freq) axes, type=(int, int)

        time_avg : boolean, if True, average resultant antenna delays across time 

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
        data = copy.copy(self.data)
        wgts = copy.deepcopy(self.wgts)

        # get freq channel width
        df = np.median(np.diff(self.freqs))

        # run delay_lincal
        fit = delay_lincal(model, data, wgts=wgts, solve_offsets=solve_offsets, medfilt=medfilt, df=df, kernel=kernel, verbose=verbose,
                           time_ax=time_ax, freq_ax=freq_ax)

        # time average
        if time_avg:
            k = self._flatten(self._gain_keys)[0]
            Ntimes = fit["tau_{}_{}".format(k[0], k[1])].shape[time_ax]
            for i, k in enumerate(self._flatten(self._gain_keys)):
                tau_key = "tau_{}_{}".format(k[0], k[1])
                tau_avg = np.moveaxis(np.median(fit[tau_key], axis=time_ax)[np.newaxis], 0, time_ax)
                fit[tau_key] = np.repeat(tau_avg, Ntimes, axis=time_ax)
                if solve_offsets:
                    phi_key = "phi_{}_{}".format(k[0], k[1])
                    gain = np.exp(1j*fit[phi_key])
                    real_avg = np.median(np.real(gain), axis=time_ax)
                    imag_avg = np.median(np.imag(gain), axis=time_ax)
                    phi_avg = np.moveaxis(np.angle(real_avg + 1j*imag_avg)[np.newaxis], 0, time_ax)
                    fit[phi_key] = np.repeat(phi_avg, Ntimes, axis=time_ax)

        # form result
        self._ant_dly = odict(map(lambda k: (k, copy.copy(fit["tau_{}_{}".format(k[0], k[1])])), self._flatten(self._gain_keys)))
        self._ant_dly_arr = np.moveaxis(map(lambda pk: map(lambda k: self._ant_dly[k], pk), self._gain_keys), 0, -1)

        if solve_offsets:
            self._ant_dly_phi = odict(map(lambda k: (k, copy.copy(fit["phi_{}_{}".format(k[0],k[1])])), self._flatten(self._gain_keys)))
            self._ant_dly_phi_arr = np.moveaxis(map(lambda pk: map(lambda k: self._ant_dly_phi[k], pk), self._gain_keys), 0, -1)

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
        wgts = copy.deepcopy(self.wgts)

        # run abs_amp_logcal
        fit = abs_amp_logcal(model, data, wgts=wgts, verbose=verbose)

        # form result
        self._abs_eta = odict(map(lambda k: (k, copy.copy(fit["eta_{}".format(k[1])])), self._flatten(self._gain_keys)))
        self._abs_eta_arr = np.moveaxis(map(lambda pk: map(lambda k: self._abs_eta[k], pk), self._gain_keys), 0, -1)

    def TT_phs_logcal(self, verbose=True, zero_psi=False, merge_pols=False):
        """
        call abscal_funcs.TT_phs_logcal() method. see its docstring for more details.

        Parameters:
        -----------
        zero_psi : type=boolean, set overall gain phase (psi) to identically zero in linsolve equations

        merge_pols : type=boolean, even if multiple polarizations are present in data, make free
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
        wgts = copy.deepcopy(self.wgts)
        antpos = self.antpos

        # run TT_phs_logcal
        fit = TT_phs_logcal(model, data, antpos, wgts=wgts, verbose=verbose, zero_psi=zero_psi, merge_pols=merge_pols)

        # manipulate if merge_pols
        if merge_pols:
            for i, gp in enumerate(self.gain_pols):
                fit['Phi_ew_{}'.format(gp)] = fit["Phi_ew"]
                fit['Phi_ns_{}'.format(gp)] = fit["Phi_ns"]
                fit.pop('Phi_ew')
                fit.pop('Phi_ns')

        # form result
        self._abs_psi = odict(map(lambda k: (k, copy.copy(fit["psi_{}".format(k[1])])), self._flatten(self._gain_keys)))
        self._abs_psi_arr = np.moveaxis(map(lambda pk: map(lambda k: self._abs_psi[k], pk), self._gain_keys), 0, -1)

        self._TT_Phi = odict(map(lambda k: (k, copy.copy(np.array([fit["Phi_ew_{}".format(k[1])], fit["Phi_ns_{}".format(k[1])]]))), self._flatten(self._gain_keys)))
        self._TT_Phi_arr = np.moveaxis(map(lambda pk: map(lambda k: np.array([self._TT_Phi[k][0], self._TT_Phi[k][1]]), pk), self._gain_keys), 0, -1)

    @property
    def ant_eta(self):
        """ return _ant_eta dict """
        if hasattr(self, '_ant_eta'):
            return copy.deepcopy(self._ant_eta)
        else:
            return None

    @property
    def ant_eta_gain(self):
        """ form complex gain from _ant_eta dict """
        if hasattr(self, '_ant_eta'):
            ant_eta = self.ant_eta
            return odict(map(lambda k: (k, np.exp(ant_eta[k]).astype(np.complex)), self._flatten(self._gain_keys)))
        else:
            return None

    @property
    def ant_eta_arr(self):
        """ return _ant_eta_arr array """
        if hasattr(self, '_ant_eta_arr'):
            return copy.copy(self._ant_eta_arr)
        else:
            return None

    @property
    def ant_eta_gain_arr(self):
        """ form complex gain from _ant_eta_arr """
        if hasattr(self, '_ant_eta_arr'):
            return np.exp(self.ant_eta_arr).astype(np.complex)
        else:
            return None

    @property
    def ant_phi(self):
        """ return _ant_phi dict """
        if hasattr(self, '_ant_phi'):
            return copy.deepcopy(self._ant_phi)
        else:
            return None

    @property
    def ant_phi_gain(self):
        """ form complex gain from _ant_phi dict """
        if hasattr(self, '_ant_phi'):
            ant_phi = self.ant_phi
            return odict(map(lambda k: (k, np.exp(1j*ant_phi[k])), self._flatten(self._gain_keys)))
        else:
            return None

    @property
    def ant_phi_arr(self):
        """ return _ant_phi_arr array """
        if hasattr(self, '_ant_phi_arr'):
            return copy.copy(self._ant_phi_arr)
        else:
            return None

    @property
    def ant_phi_gain_arr(self):
        """ form complex gain from _ant_phi_arr array """
        if hasattr(self, '_ant_phi_arr'):
            return np.exp(1j*self.ant_phi_arr)
        else:
            return None

    @property
    def ant_dly(self):
        """ return _ant_dly dict """
        if hasattr(self, '_ant_dly'):
            return copy.deepcopy(self._ant_dly)
        else:
            return None

    @property
    def ant_dly_gain(self):
        """ form complex gain from _ant_dly dict """
        if hasattr(self, '_ant_dly'):
            ant_dly = self.ant_dly
            return odict(map(lambda k: (k, np.exp(2j*np.pi*self.freqs.reshape(1, -1)*ant_dly[k])), self._flatten(self._gain_keys)))
        else:
            return None

    @property
    def ant_dly_arr(self):
        """ return _ant_dly_arr array """
        if hasattr(self, '_ant_dly_arr'):
            return copy.copy(self._ant_dly_arr)
        else:
            return None

    @property
    def ant_dly_gain_arr(self):
        """ form complex gain from _ant_dly_arr array """
        if hasattr(self, '_ant_dly_arr'):
            return np.exp(2j*np.pi*self.freqs.reshape(-1, 1)*self.ant_dly_arr)
        else:
            return None

    @property
    def ant_dly_phi(self):
        """ return _ant_dly_phi dict """
        if hasattr(self, '_ant_dly_phi'):
            return copy.deepcopy(self._ant_dly_phi)
        else:
            return None

    @property
    def ant_dly_phi_gain(self):
        """ form complex gain from _ant_dly_phi dict """
        if hasattr(self, '_ant_dly_phi'):
            ant_dly_phi = self.ant_dly_phi
            return odict(map(lambda k: (k, np.exp(1j*np.repeat(ant_dly_phi[k], self.Nfreqs, 1))), self._flatten(self._gain_keys)))
        else:
            return None

    @property
    def ant_dly_phi_arr(self):
        """ return _ant_dly_phi_arr array """
        if hasattr(self, '_ant_dly_phi_arr'):
            return copy.copy(self._ant_dly_phi_arr)
        else:
            return None

    @property
    def ant_dly_phi_gain_arr(self):
        """ form complex gain from _ant_dly_phi_arr array """
        if hasattr(self, '_ant_dly_phi_arr'):
            return np.exp(1j*np.repeat(self.ant_dly_phi_arr, self.Nfreqs, 2))
        else:
            return None

    @property
    def abs_eta(self):
        """return _abs_eta array"""
        if hasattr(self, '_abs_eta'):
            return copy.deepcopy(self._abs_eta)
        else:
            return None

    @property
    def abs_eta_gain(self):
        """form complex gain from _abs_eta dict"""
        if hasattr(self, '_abs_eta'):
            abs_eta = self.abs_eta
            return odict(map(lambda k: (k, np.exp(abs_eta[k]).astype(np.complex)), self._flatten(self._gain_keys)))
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
            return np.exp(self._abs_eta_arr).astype(np.complex)
        else:
            return None

    @property
    def abs_psi(self):
        """return _abs_psi array"""
        if hasattr(self, '_abs_psi'):
            return copy.deepcopy(self._abs_psi)
        else:
            return None

    @property
    def abs_psi_gain(self):
        """ form complex gain from _abs_psi array """
        if hasattr(self, '_abs_psi'):
            abs_psi = self.abs_psi
            return odict(map(lambda k: (k, np.exp(1j*abs_psi[k])), self._flatten(self._gain_keys)))
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
            return np.exp(1j*self._abs_psi_arr)
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
            return odict(map(lambda k: (k, np.exp(1j*np.einsum("i...,i->...", TT_Phi[k], self.antpos[k[0]][:2]))), self._flatten(self._gain_keys)))
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
            return np.exp(1j*np.einsum("hi...,hi->h...", self._TT_Phi_arr, self.antpos_arr[:, :2]))
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