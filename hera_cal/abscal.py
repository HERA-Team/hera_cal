l """
abscal.py
---------

Calibrate measured visibility
data to a visibility model using
linearizations of the (complex)
antenna-based calibration equation:

V_ij,xy^data = g_i_x * conj(g_j_y) * V_ij,xy^model.

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
            ln[abs(V_ij^data / V_ij^model)] = eta_i + eta_j

    2) per-antenna phase logarithmic calibration solves the equation:
           angle(V_ij^data / V_ij^model) = phi_i - phi_j

    3) delay linear calibration solves the equation:
           delay(V_ij^data / V_ij^model) = delay(g_i) - delay(g_j)
                                         = tau_i - tau_j
       where tau is the delay that can be turned
       into a complex gain via: g = exp(i * 2pi * tau * freqs).

    4) delay slope linear calibration solves the equation:
            delay(V_ij^data / V_ij^model) = dot(T_dly, B_ij)
        where T_dly is a delay slope in [ns / meter]
        and B_ij is the baseline vector between ant i and j.

    5) Average amplitude linear calibration solves the equation:
            log|V_ij^data / V_ij^model| = log|g_avg_i| + log|g_avg_j|
 
    6) Tip-Tilt phase logarithmic calibration solves the equation
            angle(V_ij^data /  V_ij^model) = psi + dot(TT_Phi, B_ij)
        where psi is an overall gain phase scalar, 
        TT_Phi is the gain phase slope vector [radians / meter]
        and B_ij is the baseline vector between antenna i and j.

    Methods (1), (2) and (3) can be thought of as general bandpass solvers, whereas
    methods (4), (5) and (6) are methods that would be used for data that has already
    been redundantly calibrated.

    Be warned that the linearizations of the phase solvers suffer from phase wrapping
    pathologies, meaning that a delay calibration should generally precede a
    phs_logcal or a TT_phs_logcal bandpass routine.
    """

    def __init__(self, model, data, wgts=None, antpos=None, freqs=None,
                 model_ftype='miriad', data_ftype='miriad', verbose=True):
        """
        AbsCal object used to for phasing and scaling visibility data to an absolute reference model.

        The format of model, data and wgts is in a dictionary format, with the convention that
        keys contain antennas-pairs + polarization, Ex. (1, 2, 'xx'), and values contain 2D complex
        ndarrays with [0] axis indexing time and [1] axis frequency.

        Parameters:
        -----------
        model : visibility data of refence model, type=DataContainer
                keys are antenna-pair + polarization tuples, Ex. (1, 2, 'xx').
                values are complex ndarray visibilities.
                these must be 2D arrays, with [0] axis indexing time
                and [1] axis indexing frequency.
 
                Optionally, model can be a path to a miriad or uvfits file, or a
                pyuvdata.UVData object, or a list of either.

        model_ftype : type=str, if model is a path to a file, this is its filetype
                      options=['miriad', 'uvfits']

        data : visibility data of measurements, type=DataContainer
               keys are antenna pair + pol tuples (must match model), values are
               complex ndarray visibilities matching shape of model

                Optionally, model can be a path to a miriad or uvfits file, or a
                pyuvdata.UVData object, or a list of either. In this case, wgts, antpos, 
                freqs, times and pols are overwritten with equivalent information from data object.

        data_ftype : type=str, if data is a path to a file, this is its filetype
                     options=['miriad', 'uvfits']

        wgts : weights of data, type=DataContainer, [default=None]
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
        """
        # set pols to None
        pols = None

        # check format of model if model is not a dictionary
        if type(model) == list or type(model) == np.ndarray or type(model) == str or type(model) == UVData:
            (model, model_flags, model_antpos, model_ants, model_freqs, model_lsts,
             model_times, model_pols) = io.load_vis(model, pop_autos=True, return_meta=True)

        # check format of data if data is not a dictionary
        if type(data) == list or type(data) == np.ndarray or type(data) == str or type(data) == UVData:
            (data, flags, data_antpos, data_ants, data_freqs, data_lsts,
             data_times, data_pols) = io.load_vis(data, pop_autos=True, return_meta=True)
            wgts = DataContainer(odict(map(lambda k: (k, (~flags[k]).astype(np.float)), flags.keys())))
            pols = data_pols
            freqs = data_freqs
            antpos = data_antpos

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
        self._ant_eta = odict(map(lambda k: (k, copy.copy(fit["eta_{}_{}".format(k[0], k[1])])), flatten(self._gain_keys)))
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
        self._ant_phi = odict(map(lambda k: (k, copy.copy(fit["phi_{}_{}".format(k[0], k[1])])), flatten(self._gain_keys)))
        self._ant_phi_arr = np.moveaxis(map(lambda pk: map(lambda k: self._ant_phi[k], pk), self._gain_keys), 0, -1)

        # take time and freq average
        if avg:
            self._ant_phi = odict(map(lambda k: (k, np.ones_like(self._ant_phi[k])*np.angle(np.median(np.real(np.exp(1j*self._ant_phi[k])))+1j*np.median(np.imag(np.exp(1j*self._ant_phi[k]))))), flatten(self._gain_keys)))
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
        data = copy.deepcopy(self.data)
        wgts = copy.deepcopy(self.wgts)

        # get freq channel width
        df = np.median(np.diff(self.freqs))

        # run delay_lincal
        fit = delay_lincal(model, data, wgts=wgts, solve_offsets=solve_offsets, medfilt=medfilt, df=df, kernel=kernel, verbose=verbose,
                           time_ax=time_ax, freq_ax=freq_ax)

        # time average
        if time_avg:
            k = flatten(self._gain_keys)[0]
            Ntimes = fit["tau_{}_{}".format(k[0], k[1])].shape[time_ax]
            for i, k in enumerate(flatten(self._gain_keys)):
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
        self._ant_dly = odict(map(lambda k: (k, copy.copy(fit["tau_{}_{}".format(k[0], k[1])])), flatten(self._gain_keys)))
        self._ant_dly_arr = np.moveaxis(map(lambda pk: map(lambda k: self._ant_dly[k], pk), self._gain_keys), 0, -1)

        if solve_offsets:
            self._ant_dly_phi = odict(map(lambda k: (k, copy.copy(fit["phi_{}_{}".format(k[0],k[1])])), flatten(self._gain_keys)))
            self._ant_dly_phi_arr = np.moveaxis(map(lambda pk: map(lambda k: self._ant_dly_phi[k], pk), self._gain_keys), 0, -1)

    def delay_slope_lincal(self, medfilt=True, kernel=(1, 11), time_ax=0, freq_ax=1, verbose=True, time_avg=False,
                            four_pol=False):
        """
        Solve for an array-wide delay slope (a subset of the omnical degeneracies) by calling 
        abscal_funcs.delay_slope_lincal method. See abscal_funcs.delay_slope_lincal for details.

        Parameters:
        -----------
        medfilt : boolean, if True median filter data before fft

        kernel : size of median filter across (time, freq) axes, type=(int, int)

        time_ax : the time axis index of the data

        freq_ax : the freq axis index of the data

        verbose : type=boolean, if True print feedback to stdout

        time_avg : boolean, if True, average resultant delay slope across time 

        four_pol : boolean, if True, form a joint polarization solution

        Result:
        -------
        delays slopes, per-antenna delay gains, per-antenna phase + phase gains
        can be accessed via the methods
            self.dly_slope
            self.dly_slope_gain
            self.dly_slope_arr
            self.dly_slope_gain_arr
        """
        # check for freq data
        if self.freqs is None:
            raise AttributeError("cannot delay_slope_lincal without self.freqs array")

        # assign data
        model = self.model
        data = copy.deepcopy(self.data)
        wgts = copy.deepcopy(self.wgts)
        antpos = self.antpos

        # get freq channel width
        df = np.median(np.diff(self.freqs))

        # run delay_slope_lincal
        fit = delay_slope_lincal(model, data, antpos, wgts=wgts, medfilt=medfilt, df=df, kernel=kernel, verbose=verbose,
                                  time_ax=time_ax, freq_ax=freq_ax, four_pol=four_pol)

        # separate pols if four_pol
        if four_pol:
            for i, gp in enumerate(self.gain_pols):
                fit['T_ew_{}'.format(gp)] = fit["T_ew"]
                fit['T_ns_{}'.format(gp)] = fit["T_ns"]
                fit.pop('T_ew')
                fit.pop('T_ns')

        # time average
        if time_avg:
            k = flatten(self._gain_keys)[0]
            Ntimes = fit["T_ew_{}".format(k[1])].shape[time_ax]
            for i, k in enumerate(flatten(self._gain_keys)):
                ew_key = "T_ew_{}".format(k[1])
                ns_key = "T_ns_{}".format(k[1])
                ew_avg = np.moveaxis(np.median(fit[ew_key], axis=time_ax)[np.newaxis], 0, time_ax)
                ns_avg = np.moveaxis(np.median(fit[ns_key], axis=time_ax)[np.newaxis], 0, time_ax)
                fit[ew_key] = np.repeat(ew_avg, Ntimes, axis=time_ax)
                fit[ns_key] = np.repeat(ns_avg, Ntimes, axis=time_ax)

        # form result
        self._dly_slope = odict(map(lambda k: (k, copy.copy(np.array([fit["T_ew_{}".format(k[1])], fit["T_ns_{}".format(k[1])]]))), flatten(self._gain_keys)))
        self._dly_slope_arr = np.moveaxis(map(lambda pk: map(lambda k: np.array([self._dly_slope[k][0], self._dly_slope[k][1]]), pk), self._gain_keys), 0, -1)

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
        self._abs_eta = odict(map(lambda k: (k, copy.copy(fit["eta_{}".format(k[1])])), flatten(self._gain_keys)))
        self._abs_eta_arr = np.moveaxis(map(lambda pk: map(lambda k: self._abs_eta[k], pk), self._gain_keys), 0, -1)

    def TT_phs_logcal(self, verbose=True, zero_psi=False, four_pol=False):
        """
        call abscal_funcs.TT_phs_logcal() method. see its docstring for more details.

        Parameters:
        -----------
        zero_psi : type=boolean, set overall gain phase (psi) to identically zero in linsolve equations

        four_pol : type=boolean, even if multiple polarizations are present in data, make free
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
        fit = TT_phs_logcal(model, data, antpos, wgts=wgts, verbose=verbose, zero_psi=zero_psi, four_pol=four_pol)

        # manipulate if four_pol
        if four_pol:
            for i, gp in enumerate(self.gain_pols):
                fit['Phi_ew_{}'.format(gp)] = fit["Phi_ew"]
                fit['Phi_ns_{}'.format(gp)] = fit["Phi_ns"]
                fit.pop('Phi_ew')
                fit.pop('Phi_ns')

        # form result
        self._abs_psi = odict(map(lambda k: (k, copy.copy(fit["psi_{}".format(k[1])])), flatten(self._gain_keys)))
        self._abs_psi_arr = np.moveaxis(map(lambda pk: map(lambda k: self._abs_psi[k], pk), self._gain_keys), 0, -1)

        self._TT_Phi = odict(map(lambda k: (k, copy.copy(np.array([fit["Phi_ew_{}".format(k[1])], fit["Phi_ns_{}".format(k[1])]]))), flatten(self._gain_keys)))
        self._TT_Phi_arr = np.moveaxis(map(lambda pk: map(lambda k: np.array([self._TT_Phi[k][0], self._TT_Phi[k][1]]), pk), self._gain_keys), 0, -1)

    @property
    def ant_eta(self):
        """ return _ant_eta dict, containing per-antenna amplitude solution """
        if hasattr(self, '_ant_eta'):
            return copy.deepcopy(self._ant_eta)
        else:
            return None

    @property
    def ant_eta_gain(self):
        """ form complex gain from _ant_eta dict """
        if hasattr(self, '_ant_eta'):
            ant_eta = self.ant_eta
            return odict(map(lambda k: (k, np.exp(ant_eta[k]).astype(np.complex)), flatten(self._gain_keys)))
        else:
            return None

    @property
    def ant_eta_arr(self):
        """ return _ant_eta in ndarray format """
        if hasattr(self, '_ant_eta_arr'):
            return copy.copy(self._ant_eta_arr)
        else:
            return None

    @property
    def ant_eta_gain_arr(self):
        """ return _ant_eta_gain in ndarray format """
        if hasattr(self, '_ant_eta_arr'):
            return np.exp(self.ant_eta_arr).astype(np.complex)
        else:
            return None

    @property
    def ant_phi(self):
        """ return _ant_phi dict, containing per-antenna phase solution """
        if hasattr(self, '_ant_phi'):
            return copy.deepcopy(self._ant_phi)
        else:
            return None

    @property
    def ant_phi_gain(self):
        """ form complex gain from _ant_phi dict """
        if hasattr(self, '_ant_phi'):
            ant_phi = self.ant_phi
            return odict(map(lambda k: (k, np.exp(1j*ant_phi[k])), flatten(self._gain_keys)))
        else:
            return None

    @property
    def ant_phi_arr(self):
        """ return _ant_phi in ndarray format """
        if hasattr(self, '_ant_phi_arr'):
            return copy.copy(self._ant_phi_arr)
        else:
            return None

    @property
    def ant_phi_gain_arr(self):
        """ return _ant_phi_gain in ndarray format """
        if hasattr(self, '_ant_phi_arr'):
            return np.exp(1j*self.ant_phi_arr)
        else:
            return None

    @property
    def ant_dly(self):
        """ return _ant_dly dict, containing per-antenna delay solution """
        if hasattr(self, '_ant_dly'):
            return copy.deepcopy(self._ant_dly)
        else:
            return None

    @property
    def ant_dly_gain(self):
        """ form complex gain from _ant_dly dict """
        if hasattr(self, '_ant_dly'):
            ant_dly = self.ant_dly
            return odict(map(lambda k: (k, np.exp(2j*np.pi*self.freqs.reshape(1, -1)*ant_dly[k])), flatten(self._gain_keys)))
        else:
            return None

    @property
    def ant_dly_arr(self):
        """ return _ant_dly in ndarray format """
        if hasattr(self, '_ant_dly_arr'):
            return copy.copy(self._ant_dly_arr)
        else:
            return None

    @property
    def ant_dly_gain_arr(self):
        """ return ant_dly_gain in ndarray format """
        if hasattr(self, '_ant_dly_arr'):
            return np.exp(2j*np.pi*self.freqs.reshape(-1, 1)*self.ant_dly_arr)
        else:
            return None

    @property
    def ant_dly_phi(self):
        """ return _ant_dly_phi dict, containing a single phase solution per antenna """
        if hasattr(self, '_ant_dly_phi'):
            return copy.deepcopy(self._ant_dly_phi)
        else:
            return None

    @property
    def ant_dly_phi_gain(self):
        """ form complex gain from _ant_dly_phi dict """
        if hasattr(self, '_ant_dly_phi'):
            ant_dly_phi = self.ant_dly_phi
            return odict(map(lambda k: (k, np.exp(1j*np.repeat(ant_dly_phi[k], self.Nfreqs, 1))), flatten(self._gain_keys)))
        else:
            return None

    @property
    def ant_dly_phi_arr(self):
        """ return _ant_dly_phi in ndarray format """
        if hasattr(self, '_ant_dly_phi_arr'):
            return copy.copy(self._ant_dly_phi_arr)
        else:
            return None

    @property
    def ant_dly_phi_gain_arr(self):
        """ return _ant_dly_phi_gain in ndarray format """
        if hasattr(self, '_ant_dly_phi_arr'):
            return np.exp(1j*np.repeat(self.ant_dly_phi_arr, self.Nfreqs, 2))
        else:
            return None

    @property
    def dly_slope(self):
        """ return _dly_slope dict, containing the delay slope across the array """
        if hasattr(self, '_dly_slope'):
            return copy.deepcopy(self._dly_slope)
        else:
            return None

    @property
    def dly_slope_gain(self):
        """ form a per-antenna complex gain from _dly_slope dict and the antpos dictionary attached to the class"""
        if hasattr(self, '_dly_slope'):
            # get dly_slope dictionary
            dly_slope = self.dly_slope
            # turn delay slope into per-antenna complex gains, while iterating over self._gain_keys
            return odict(map(lambda k: (k, np.exp(2j*np.pi*self.freqs.reshape(1, -1)*np.einsum("i...,i->...", dly_slope[k], self.antpos[k[0]][:2]))), flatten(self._gain_keys)))
        else:
            return None

    def custom_dly_slope_gain(self, gain_keys, antpos):
        """
        return dly_slope_gain with custom gain keys and antenna positions

        gain_keys : type=list, list of unique (ant, pol). Ex. [(0, 'x'), (1, 'x'), (0, 'y'), (1, 'y')]
        antpos : type=dictionary, contains antenna position vectors. keys are ant integer, values are ant position vectors
        """
        if hasattr(self, '_dly_slope'):
            # get dly slope dictionary
            dly_slope = self.dly_slope[self._gain_keys[0][0]]
            # turn delay slope into per-antenna complex gains, while iterating over gain_keys
            return odict(map(lambda k: (k, np.exp(2j*np.pi*self.freqs.reshape(1, -1)*np.einsum("i...,i->...", dly_slope, antpos[k[0]][:2]))), gain_keys))
        else:
            return None

    @property
    def dly_slope_arr(self):
        """ return _dly_slope_arr array """
        if hasattr(self, '_dly_slope_arr'):
            return copy.copy(self._dly_slope_arr)
        else:
            return None

    @property
    def dly_slope_gain_arr(self):
        """ form complex gain from _dly_slope_arr array """
        if hasattr(self, '_dly_slope_arr'):
            return np.exp(2j*np.pi*self.freqs.reshape(-1, 1)*np.einsum("hi...,hi->h...", self._dly_slope_arr, self.antpos_arr[:, :2]))
        else:
            return None

    @property
    def dly_slope_ant_dly_arr(self):
        """ form antenna delays from _dly_slope_arr array """
        if hasattr(self, '_dly_slope_arr'):
            return np.einsum("hi...,hi->h...", self._dly_slope_arr, self.antpos_arr[:, :2])
        else:
            return None

    @property
    def abs_eta(self):
        """return _abs_eta dict"""
        if hasattr(self, '_abs_eta'):
            return copy.deepcopy(self._abs_eta)
        else:
            return None

    @property
    def abs_eta_gain(self):
        """form complex gain from _abs_eta dict"""
        if hasattr(self, '_abs_eta'):
            abs_eta = self.abs_eta
            return odict(map(lambda k: (k, np.exp(abs_eta[k]).astype(np.complex)), flatten(self._gain_keys)))
        else:
            return None

    def custom_abs_eta_gain(self, gain_keys):
        """
        return abs_eta_gain with custom gain keys

        gain_keys : type=list, list of unique (ant, pol). Ex. [(0, 'x'), (1, 'x'), (0, 'y'), (1, 'y')]
        """
        if hasattr(self, '_abs_eta'):
            abs_eta = self.abs_eta[self._gain_keys[0][0]]
            return odict(map(lambda k: (k, np.exp(abs_eta).astype(np.complex)), gain_keys))
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
        """return _abs_psi dict"""
        if hasattr(self, '_abs_psi'):
            return copy.deepcopy(self._abs_psi)
        else:
            return None

    @property
    def abs_psi_gain(self):
        """ form complex gain from _abs_psi array """
        if hasattr(self, '_abs_psi'):
            abs_psi = self.abs_psi
            return odict(map(lambda k: (k, np.exp(1j*abs_psi[k])), flatten(self._gain_keys)))
        else:
            return None

    def custom_abs_psi_gain(self, gain_keys):
        """
        return abs_psi_gain with custom gain keys

        gain_keys : type=list, list of unique (ant, pol). Ex. [(0, 'x'), (1, 'x'), (0, 'y'), (1, 'y')]
        """
        if hasattr(self, '_abs_psi'):
            abs_psi = self.abs_psi[self._gain_keys[0][0]]
            return odict(map(lambda k: (k, np.exp(1j*abs_psi)), gain_keys))
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
            return odict(map(lambda k: (k, np.exp(1j*np.einsum("i...,i->...", TT_Phi[k], self.antpos[k[0]][:2]))), flatten(self._gain_keys)))
        else:
            return None

    def custom_TT_Phi_gain(self, gain_keys, antpos):
        """
        return TT_Phi_gain with custom gain keys and antenna positions

        gain_keys : type=list, list of unique (ant, pol). Ex. [(0, 'x'), (1, 'x'), (0, 'y'), (1, 'y')]
        antpos : type=dictionary, contains antenna position vectors. keys are ant integer, values are ant positions
        """
        if hasattr(self, '_TT_Phi'):
            TT_Phi = self.TT_Phi[self._gain_keys[0][0]]
            return odict(map(lambda k: (k, np.exp(1j*np.einsum("i...,i->...", TT_Phi, antpos[k[0]][:2]))), gain_keys))
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


def abscal_arg_parser():
    """
    argparser for general abscal run. By default no calibration is performed: the user
    needs to specify which calibration steps they want via the delay_cal, avg_phs_cal,
    delay_slope_cal, abs_amp_cal, TT_phs_cal, gen_amp_cal and gen_phs_cal flags. To learn
    more about these calibration steps, read the doc-string of the abscal_run() function
    in abscal.py, and the docstring of the AbsCal() class in abscal.py.
    """
    a = argparse.ArgumentParser(description="command-line drive script for hera_cal.abscal module")
    a.add_argument("--data_files", type=str, nargs='*', help="list of file paths of data to-be-calibrated.", required=True)
    a.add_argument("--model_files", type=str, nargs='*', help="list of data-overlapping miriad files for visibility model.", required=True)
    a.add_argument("--calfits_infiles", type=str, nargs='*', help="list of calfits files to multiply with abscal solution before writing to disk.")
    a.add_argument("--output_calfits_fname", type=str, nargs='*', default=[], help="names of output calfits files.")
    a.add_argument("--outdir", type=str, default=None, help="output directory")
    a.add_argument("--overwrite", default=False, action='store_true', help="overwrite output calfits file if it exists.")
    a.add_argument("--silence", default=False, action='store_true', help="silence output from abscal while running.")
    a.add_argument("--data_is_omni_solution", default=False, action='store_true', help='assume input data file is an omnical visibility solution')
    a.add_argument("--all_antenna_gains", default=False, action='store_true', help='if True, use full antenna list in data file to make gains')
    a.add_argument("--delay_cal", default=False, action='store_true', help='perform antenna delay calibration')
    a.add_argument("--avg_phs_cal", default=False, action='store_true', help='perform antenna avg phase calibration')
    a.add_argument("--delay_slope_cal", default=False, action='store_true', help='perform delay slope calibration')    
    a.add_argument("--abs_amp_cal", default=False, action='store_true', help='perform absolute amplitude calibration')
    a.add_argument("--TT_phs_cal", default=False, action='store_true', help='perform Tip-Tilt phase slope calibration')
    a.add_argument("--gen_amp_cal", default=False, action='store_true', help='perform general antenna amplitude bandpass calibration')
    a.add_argument("--gen_phs_cal", default=False, action='store_true', help='perform general antenna phase bandpass calibration')
    a.add_argument("--max_dlst", default=0.005, type=float, help="maximum allowed LST difference in model rephasing, otherwies model is flagged.")
    return a


def omni_abscal_arg_parser():
    """
    argparser specifically for abscal on omnni-calibrated data. The calibration steps exposed to the user
    include: delay_slope_cal, abs_amp_cal and TT_phs_cal. To learn more about these steps, read the doc-string of the
    abscal_run() function in abscal.py, and the docstring of the AbsCal() class in abscal.py.
    """
    a = argparse.ArgumentParser(description="command-line drive script for hera_cal.abscal module")
    a.add_argument("--data_files", type=str, nargs='*', help="list of file paths of data to-be-calibrated.", required=True)
    a.add_argument("--model_files", type=str, nargs='*', help="list of data-overlapping miriad files for visibility model.", required=True)
    a.add_argument("--calfits_infiles", type=str, nargs='*', help="list of calfits files to multiply with abscal solution before writing to disk.")
    a.add_argument("--output_calfits_fname", type=str, nargs='*', default=[], help="names of output calfits files.")
    a.add_argument("--outdir", type=str, default=None, help="output directory")
    a.add_argument("--overwrite", default=False, action='store_true', help="overwrite output calfits file if it exists.")
    a.add_argument("--silence", default=False, action='store_true', help="silence output from abscal while running.")
    a.add_argument("--data_is_omni_solution", default=False, action='store_true', help='assume input data file is an omnical visibility solution (still beta testing optimal weighting)')
    a.add_argument("--cal_shared_antennas_only", default=False, action='store_true', help='if True, only calibrate antennas present in both data and model.')
    a.add_argument("--delay_slope_cal", default=False, action='store_true', help='perform delay slope calibration')    
    a.add_argument("--abs_amp_cal", default=False, action='store_true', help='perform absolute amplitude calibration')
    a.add_argument("--TT_phs_cal", default=False, action='store_true', help='perform Tip-Tilt phase slope calibration')
    a.add_argument("--max_dlst", default=0.005, type=float, help="maximum allowed LST difference in model rephasing, otherwies model is flagged.")
    return a


def abscal_run(data_files, model_files, calfits_infiles=None, verbose=True, overwrite=False, write_calfits=True,
               output_calfits_fname=None, return_gains=False, return_object=False, outdir=None,
               match_red_bls=False, tol=1.0, reweight=False, rephase_model=True, all_antenna_gains=False,
               delay_cal=False, avg_phs_cal=False, delay_slope_cal=False, abs_amp_cal=False,
               TT_phs_cal=False, gen_amp_cal=False, gen_phs_cal=False, latitude=-30.72152, max_dlst=0.005, history=''):
    """
    run AbsCal on a set of time-contiguous data files, using time-contiguous model files that cover
    the data_files across LST.

    Parameters that control calibration steps are:

    delay_cal -> avg_phs_cal -> delay_slope_cal -> abs_amp_cal -> TT_phs_cal - > gen_amp_cal -> gen_phs_cal

    which are run in that order if any of these parameters are set to True. Calibration steps are
    run and then directly applied to the data before proceeding to the next calibration step. To
    learn more about these steps, see the docstring of the following functions in abscal_funcs.py:
    
    delay_cal : delay_lincal()
    avg_phs_cal : phs_logcal(avg=True)
    delay_slope_cal : delay_slope_lincal()
    abs_amp_cal : abs_amp_logcal()
    TT_phs_cal : TT_phs_logcal()
    gen_amp_cal : amp_logcal()
    gen_phs_cal : phs_logcal()

    Parameters:
    -----------
    data_files : type=list of strings, path(s) to data miriad file(s)
                a list of paths to miriad file(s) containing complex
                visibility data, or a path itself

    model_files : type=list of strings, path(s) to model miriad files(s)
                a list of paths to miriad file(s) containing complex
                visibility data, or a path itself

    calfits_infiles : type=list of strings, path(s) to calfits files(s)
                a list of paths to calfits file(s) containing gain solutions
                to multiply with abscal gain solution before writing to file.
                These files should correspond one-to-one with data_files. History,
                quality and flags are also propagated to final output calfits files.

    verbose : type=boolean, if True print output to stdout

    overwrite : type=boolean, if True, overwite output files

    write_calfits : type=boolean, if True, write out gains as calfits file

    output_calfits_fname : type=list of strings, filenames (not full path) of output calfits files 

    outdir : type=str, path to output directory

    return_gains : type=boolean, if True, return AbsCal gain dictionary

    return_object : type=boolean, if True, return AbsCal object

    match_red_bls : type=boolean, match unique data baselines to model baselines based on redundancy

    tol : type=float, baseline match tolerance in units of baseline vectors (e.g. meters)

    reweight : type=boolean, reweight unique baseline data based on redundancy

    rephase_model : type=boolean, rephase nearest neighbor model pixels onto data lst grid

    all_antenna_gains : type=boolean, if True, use full antenna list in data file to make gains,
                rather than just antennas present in the data. It is not possible
                to run delay_cal, avg_phs_cal, gen_phs_cal and gen_amp_cal when all_antenna_gains is True.

    delay_cal : type=boolean, if True, perform delay calibration

    delay_slope_cal : type=boolean, if True, perform delay slope calibration

    avg_phs_cal : type=boolean, if True, perform average phase calibration

    abs_amp_cal : type=boolean, if True, perform absolute gain calibration

    TT_phs_cal : type=boolean, if True, perform Tip-Tilt phase calibration

    gen_amp_cal : type=boolean, if True, perform general amplitude bandpass calibration

    gen_phs_cal : type=boolean, if True, perform general phase bandpass calibration

    latitude : type=float, latitude of array in degrees North

    max_dlst : type=float, maximum allowed LST difference in model rephasing, otherwies model is flagged.

    Result:
    -------
    if return_gains: return (gains dictionary)
    if return_object: return (AbsCal instance)
    if return_gains and return_objects: return (gains dictionary, AbsCal instance)
    if write_calfits: writes a calfits file with gain solutions
    """
    # only load model files needed to create LST overlap w/ data files
    # and reject data files that have no LST overlap w/ any of model files
    all_model_files = []
    for df in data_files:
        all_model_files.extend(match_times(df, model_files))

    model_files = sorted(set(all_model_files))

    # check length of model files
    nomodelfiles = False
    if len(model_files) == 0:
        echo("no model files overlap with data files in LST", verbose=verbose)
        nomodelfiles = True

    # load model files
    if nomodelfiles == False:
        echo ("loading model files", type=1, verbose=verbose)
        (model, model_flags, model_antpos, model_ants, model_freqs, model_times, model_lsts,
            model_pols) = io.load_vis(model_files, pop_autos=True, return_meta=True)
        antpos = model_antpos
        model_lsts[model_lsts < model_lsts[0]] += 2*np.pi

    # iterate over data files
    gains = []
    echo("loading data files", type=1, verbose=verbose)
    for i, dfile in enumerate(data_files):

        # check output filepath
        if write_calfits:
            # configure filename
            if output_calfits_fname == []:
                output_calfits_file = os.path.basename(dfile) + '.abscal.calfits'
            else:
                output_calfits_file = output_calfits_fname[i]
            if outdir is None:
                outdir = os.path.dirname(dfile)
            output_calfits_file = os.path.join(outdir, output_calfits_file)

            # check path
            if os.path.exists(output_calfits_file) and overwrite == False:
                raise IOError("{} exists, not overwriting".format(output_calfits_file))

        # load data and configure weights
        echo("loading {}".format(dfile), type=1, verbose=verbose)
        (data, data_flags, data_antpos, data_ants, data_freqs, data_times, data_lsts,
            data_pols) = io.load_vis(dfile, pop_autos=True, return_meta=True, pick_data_ants=False)
        bls = odict(map(lambda k: (k, data_antpos[k[0]] - data_antpos[k[1]]), data.keys()))
        Ntimes = len(data_times)
        Nfreqs = len(data_freqs)
        data_lsts[data_lsts < data_lsts[0]] += 2*np.pi

        # get data ants
        total_data_antpos = copy.deepcopy(data_antpos)
        data_ants = np.unique(map(lambda k: k[:2], data.keys()))
        data_antpos = odict(map(lambda k: (k, data_antpos[k]), data_ants))

        # get wgts
        wgts = DataContainer(odict(map(lambda k: (k, (~data_flags[k]).astype(np.float)), data_flags.keys())))

        # ensure nomodelfiles is False
        if nomodelfiles == False:
            # match redundant baselines
            if match_red_bls:
                data = match_red_baselines(data, data_antpos, model, model_antpos, tol=tol, verbose=verbose)
                antpos = model_antpos

            # rephase model to match data lst grid
            if rephase_model:
                new_model, new_flags = rephase_vis(model, model_lsts, data_lsts, bls, data_freqs, 
                                                   flags=model_flags, latitude=latitude, max_dlst=max_dlst)

                # set wgts to zero wheree model is flagged
                for k in new_flags.keys():
                    wgts[k][new_flags[k]] *= 0
            else:
                new_model = model

            # reweight according to redundancy
            if reweight:
                wgts = mirror_data_to_red_bls(wgts, model_antpos, tol=tol, weights=True)

            # instantiate class
            AC = AbsCal(new_model, data, wgts=wgts, antpos=antpos, freqs=data_freqs)
            total_gain_keys = flatten(map(lambda p: map(lambda k: (k, p), total_data_antpos.keys()), AC.gain_pols))

            gain_list = []

            if delay_cal:
                if all_antenna_gains:
                    raise ValueError("can't run delay_cal when all_antenna_gains is True")
                AC.delay_lincal(verbose=verbose, time_avg=True)
                result_gains = merge_gains((AC.ant_dly_gain, AC.ant_dly_phi_gain))
                cal_flags = odict(map(lambda k: (k, np.zeros_like(result_gains[k], np.bool)), result_gains.keys()))
                apply_cal.recalibrate_in_place(AC.data, AC.wgts, result_gains, cal_flags, gain_convention='divide')
                gain_list.append(AC.ant_dly_gain)
                gain_list.append(AC.ant_dly_phi_gain)

            if avg_phs_cal:
                if delay_cal == False:
                    echo("it is recommended to run a delay_cal before avg_phs_cal", verbose=verbose)
                if all_antenna_gains:
                    raise ValueError("can't run avg_phs_cal when all_antenna_gains is True")
                AC.phs_logcal(avg=True, verbose=verbose)
                cal_flags = odict(map(lambda k: (k, np.zeros_like(AC.ant_phi_gain[k], np.bool)), AC.ant_phi_gain.keys()))
                apply_cal.recalibrate_in_place(AC.data, AC.wgts, AC.ant_phi_gain, cal_flags, gain_convention='divide')
                gain_list.append(AC.ant_phi_gain)

            if delay_slope_cal:
                AC.delay_slope_lincal(verbose=verbose, time_avg=True)
                cal_flags = odict(map(lambda k: (k, np.zeros_like(AC.dly_slope_gain[k], np.bool)), AC.dly_slope_gain.keys()))
                apply_cal.recalibrate_in_place(AC.data, AC.wgts, AC.dly_slope_gain, cal_flags, gain_convention='divide')
                if all_antenna_gains:
                    gain_list.append(AC.custom_dly_slope_gain(total_gain_keys, total_data_antpos))
                else:
                    gain_list.append(AC.dly_slope_gain)

            if abs_amp_cal:
                AC.abs_amp_logcal(verbose=verbose)
                cal_flags = odict(map(lambda k: (k, np.zeros_like(AC.abs_eta_gain[k], np.bool)), AC.abs_eta_gain.keys()))
                apply_cal.recalibrate_in_place(AC.data, AC.wgts, AC.abs_eta_gain, cal_flags, gain_convention='divide')
                if all_antenna_gains:
                    gain_list.append(AC.custom_abs_eta_gain(total_gain_keys))
                else:
                    gain_list.append(AC.abs_eta_gain)

            if TT_phs_cal:
                if delay_slope_cal == False:
                    echo("it is recommended to run a delay_slope_cal before TT_phs_cal", verbose=verbose)
                AC.TT_phs_logcal(verbose=verbose)
                cal_flags = odict(map(lambda k: (k, np.zeros_like(AC.TT_Phi_gain[k], np.bool)), AC.TT_Phi_gain.keys()))
                apply_cal.recalibrate_in_place(AC.data, AC.wgts, AC.TT_Phi_gain, cal_flags, gain_convention='divide')
                if all_antenna_gains:
                    gain_list.append(AC.custom_TT_Phi_gain(total_gain_keys, total_data_antpos))
                    gain_list.append(AC.custom_abs_psi_gain(total_gain_keys))
                else:
                    gain_list.append(AC.abs_psi_gain)
                    gain_list.append(AC.TT_Phi_gain)

            if gen_amp_cal:
                if all_antenna_gains:
                    raise ValueError("can't run gen_amp_cal when all_antenna_gains is True")
                AC.amp_logcal(verbose=verbose)
                cal_flags = odict(map(lambda k: (k, np.zeros_like(AC.ant_eta_gain[k], np.bool)), AC.ant_eta_gain.keys()))
                apply_cal.recalibrate_in_place(AC.data, AC.wgts, AC.ant_eta_gain, cal_flags, gain_convention='divide')
                gain_list.append(AC.ant_eta_gain)

            if gen_phs_cal:
                if delay_cal == False and delay_slope_cal == False:
                    echo("it is recommended to run a delay_cal or delay_slope_cal before gen_phs_cal", verbose=verbose)
                if all_antenna_gains:
                    raise ValueError("can't run gen_phs_cal when all_antenna_gains is True")
                AC.phs_logcal(verbose=verbose)
                cal_flags = odict(map(lambda k: (k, np.zeros_like(AC.ant_phi_gain[k], np.bool)), AC.ant_phi_gain.keys()))
                apply_cal.recalibrate_in_place(AC.data, AC.wgts, AC.ant_phi_gain, cal_flags, gain_convention='divide')
                gain_list.append(AC.ant_phi_gain)

            # collate gains
            if len(gain_list) == 0:
                raise ValueError("abscal_run executed without any calibration arguments set to True")
            gain_dict = merge_gains(gain_list)
            flag_dict = odict(map(lambda k: (k, np.zeros((Ntimes, Nfreqs), np.bool)), gain_dict.keys()))

        # make blank gains if no modelfiles
        else:
            gain_pols = set(flatten(map(lambda p: [p[0], p[1]], data_pols)))
            gain_dict = map(lambda p: map(lambda a: ((a,p), np.ones((Ntimes, Nfreqs), np.complex)), data_ants), gain_pols)
            gain_dict = odict(flatten(gain_dict))
            flag_dict = odict(map(lambda k: (k, np.ones((Ntimes, Nfreqs), np.bool)), gain_dict.keys()))

        # write to file
        if write_calfits:
            # load calfits file if provided
            if calfits_infiles is not None:
                cal_in = UVCal()
                cal_in.read_calfits(calfits_infiles[i])
                out_gains, out_flags = io.load_cal(cal_in)
                for k in out_gains.keys():
                    if k in gain_dict:
                        out_gains[k] *= gain_dict[k]
                        out_flags[k] += flag_dict[k]
                    else:
                        out_flags[k] += np.ones(out_flags[k], np.bool)
                io.update_cal(cal_in, output_calfits_file,  gains=out_gains, flags=out_flags, add_to_history=history, clobber=overwrite)
            # write a calfits from scratch
            else:
                io.write_cal(output_calfits_file, gain_dict, data_freqs, data_times, flags=flag_dict, quality=None, 
                     total_qual=None, return_uvc=False, overwrite=overwrite, history=history)

        # append gain dict to gains
        gains.append(gain_dict)

    # form return tuple
    return_obj = ()

    # return gains if desired
    if return_gains:
        return_obj += (gains,)

    if return_object:
        return_obj += (AC,)

    # return
    if return_gains or return_object:
        return return_obj


