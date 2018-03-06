import numpy as np
import hera_cal.io as io
from pyuvdata import UVCal, UVData

def recalibrate_in_place(data, data_flags, new_gains, cal_flags, old_gains=None, gain_convention = 'divide'):
    '''Update data and data_flags in place, taking out old calibration solutions, putting in
    new calibration solutions, and updating flags from those calibration solutions. Previously 
    flagged data is left unmodified. Missing antennas from either the new gains or (if it's not None),
    the old gains are automatically flagged in the data.
    
    Arguments:
        data: DataContainer containing baseline-pol complex visibility data. This is modified in place.
        data_flags: DataContainer containing data flags. This is modified in place.
        new_gains: Dictionary of complex calibration gains to apply with keys like (1,'x')
        cal_flags: Dictionary with keys like (1,'x') of per-antenna boolean flags to update data_flags
            if either antenna in a visibility is flagged.
        old_gains: Dictionary of complex calibration gains to take out with keys like (1,'x').
            Default of None implies that the data is raw (i.e. uncalibrated).
        gain_convention: str, either 'divide' or 'multiply'. 'divide' means V_obs = gi gj* V_true,
            'multiply' means V_true = gi gj* V_obs.
    '''
    for (i,j,pol) in data.keys():
        if not np.all(data_flags[(i,j,pol)]):
            # if any antenna is missing from the gains, the data is flagged
            if new_gains.has_key((i,pol[0])) and new_gains.has_key((j,pol[1])) and (old_gains == None
                or (old_gains.has_key((i,pol[0])) and old_gains.has_key((j,pol[1])))):
                    gigj_new = new_gains[(i,pol[0])] * np.conj(new_gains[(j,pol[1])])
                    if old_gains is not None:
                        gigj_old = old_gains[(i,pol[0])] * np.conj(old_gains[(j,pol[1])])
                    else:
                        gigj_old = np.ones_like(gigj_new)
                    # update all the data, assuming it wasn't flagged to begin with
                    if gain_convention == 'divide':
                        data[(i,j,pol)][~data_flags[(i,j,pol)]] *= (gigj_old / gigj_new)[~data_flags[(i,j,pol)]]
                    elif gain_convention == 'multiply':
                        data[(i,j,pol)][~data_flags[(i,j,pol)]] *= (gigj_new / gigj_old)[~data_flags[(i,j,pol)]]
                    else:
                        raise KeyError("gain_convention must be either 'divide' or 'multiply'.")
                    # update data flags
                    data_flags[(i,j,pol)][cal_flags[(i, pol[0])]] = True
                    data_flags[(i,j,pol)][cal_flags[(j, pol[1])]] = True
            else:
                data_flags[(i,j,pol)] = np.ones_like(data_flags[(i,j,pol)]) #all flagged



def apply_cal(data_infilename, data_outfilename, new_calibration, old_calibration = None, flags_npz = None, 
              filetype = 'miriad', gain_convention = 'divide',  add_to_history = '', clobber = False, **kwargs):
    '''Update the calibration solution and flags on the data, writing to a new file. Takes out old calibration
    and puts in new calibration solution, including its flags. Also enables appending to history.

    Arguments:
        data_infilename: filename (or UVData object) of the data file to be updated.
        data_outfilename: filename of the resultant data file with the new calibration and flags.
        new_calibration: filename of the calfits file (or UVCal object) for the calibration to be applied,
            along with its new flags (if any)
        old_calibration: filename of the calfits file (or UVCal object) for the calibration to be unapplied.
            Default None means that the input data is raw (i.e. uncalibrated).
        flag_npz: optional path to npz file containing just flags to be ORed with flags in input data
        filetype: filename for the new file, either 'miriad' or 'uvfits'
        gain_convention: str, either 'divide' or 'multiply'. 'divide' means V_obs = gi gj* V_true,
            'multiply' means V_true = gi gj* V_obs.
        add_to_history: appends a string to the history of the output file. This will preceed combined histories
            of flags_npz (if applicable), new_calibration and, old_calibration (if applicable).
        clobber: if True, overwrites existing file at outfilename
        kwargs: dictionary mapping updated attributes to their new values.
            See pyuvdata.UVData documentation for more info.
    '''
    # load data, flags, and (optionally) npz flags
    uvd = UVData()
    if filetype == 'miriad':
        uvd.read_miriad(data_infilename)
    else:
        raise NotImplementedError('This function has not been implemented yet.')
    if flags_npz is not None:
        npz_flags = np.load(flags_npz)
        uvd.flag_array = np.logical_or(npz_flags['flag_array'], uvd.flag_array)
        add_to_history += ' FLAGS_NPZ_HISTORY: ' + str(npz_flags['history'])
    data, data_flags = io.load_vis(uvd)
    
    # load new calibration solution
    uvc = UVCal()
    uvc.read_calfits(new_calibration)
    add_to_history += ' NEW_CALFITS_HISTORY: ' + uvc.history
    new_gains, new_flags = io.load_cal(uvc)

    # load old calibration solution
    if old_calibration is not None:
        if isinstance(old_calibration, UVCal):
            old_uvc = old_calibration
        else:        
            old_uvc = UVCal()
            old_uvc.read_calfits(old_calibration)
        add_to_history += ' OLD_CALFITS_HISTORY: ' + old_uvc.history
        old_calibration, _ = io.load_cal(old_uvc)

    recalibrate_in_place(data, data_flags, new_gains, new_flags, old_gains=old_calibration, gain_convention=gain_convention)
    io.update_vis(data_infilename, data_outfilename, filetype_in=filetype, filetype_out=filetype, data=data, 
                  flags=data_flags, add_to_history=add_to_history, clobber=clobber, **kwargs)
