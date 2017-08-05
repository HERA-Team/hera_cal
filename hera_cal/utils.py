def get_HERA_aa(freqs,calfile='hera_cm',**kwargs):
    # create an aipy AntennaArray object using positions and hookup info in M&C.
    #
    # Inputs:
    # freqs = numpy array of frequencies in GHz
    #   - used to compute uvws by some aipy functions
    #   - for applications using antenna locations in meters, a nominal input
    #        value of np.array([0.15]) is probably fine
    # calfile = python library containing get_aa function.
    #   - default value is hera_cal.hera_cm handles import from M&C
    #   - don't include the .py
    #   - needs to be in your python path
    # array_epoch_jd = julian date of desired configuration
    #   - if not input uses default date set in calfile, hera_cal.hera_cm
    # locations_file = antenna location csv file exported from m&c
    #   - default file included in hera_cal
    #   - create a new one with hera_mc/scripts/write_antenna_location_file.py
    #

    #sometimes the input is set to None, which implies default
    if calfile is None:
        calfile='hera_cm'

    exec('from {calfile} import get_aa'.format(calfile=calfile))
    if calfile!='hera_cm':
        #array position is set based on user input cal file
        return get_aa(freqs)
    else:
        #use the time and position file aware get_aa
        return get_aa(freqs,**kwargs)
