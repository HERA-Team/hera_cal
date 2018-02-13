#! /usr/bin/env python
import os
import argparse
import pyuvdata
from hera_cal.version import version_info
import pyuvdata.utils as uvutils
import numpy as np

parser = argparse.ArgumentParser(description='Extract HERA hex antennas from data '
                                 'file, and save with new extension.')
parser.add_argument('--extension', type=str, help='Extension to be appended to '
                    'filename for output. Default="HH".', default='HH')
parser.add_argument('--filetype', type=str, help='Input and output file type. '
                    'Allowed values are "miriad" (default), and "uvfits".',
                    default='miriad')
parser.add_argument('--fixuvws', action='store_true', help='Optional flag to '
                    'use antenna positions to replace uvws.')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='Optional flag to overwrite output file if it exists.')
parser.add_argument('files', metavar='files', type=str, nargs='+',
                    help='Files to be processed.')
parser.add_argument('--ex_ants_file', type=str, help='Text file with list of antennas'
                    ' which are excluded downstream in RTP. Generally, these are '
                    'antennas which are being actively commissioned, or known as bad.'
                    ' Note these values are only stored in the history, not actually '
                    'flagged at this step.',
                    default=None)
args = parser.parse_args()

for filename in args.files:
    uv = pyuvdata.UVData()
    if args.filetype == 'miriad':
        uv.read_miriad(filename)
    elif args.filetype == 'uvfits':
        uv.read_uvfits(filename)
    else:
        raise ValueError('Unrecognized file type ' + str(args.filetype))
    st_type_str = uv.extra_keywords.pop('st_type').replace('\x00', '')
    st_type_list = st_type_str[1:-1].split(', ')
    ind = [i for i, x in enumerate(st_type_list) if x == 'herahex' or x == 'heraringa' or x == 'heraringb']
    uv.select(antenna_nums=uv.antenna_numbers[ind])
    st_type_list = list(np.array(st_type_list)[np.array(ind, dtype=int)])
    uv.extra_keywords['st_type'] = '[' + ', '.join(st_type_list) + ']'
    uv.history += ' Hera Hex antennas selected'
    if args.fixuvws:
        antpos = uv.antenna_positions + uv.telescope_location
        antpos = uvutils.ENU_from_ECEF(antpos.T, *uv.telescope_location_lat_lon_alt).T
        antmap = -np.ones(np.max(uv.antenna_numbers) + 1, dtype=int)
        for i, ant in enumerate(uv.antenna_numbers):
            antmap[ant] = i
        uv.uvw_array = antpos[antmap[uv.ant_2_array], :] - antpos[antmap[uv.ant_1_array], :]
        uv.history += ' and uvws corrected'
    uv.history += ' with hera_cal/scripts/extract_hh.py, hera_cal version: ' +\
                  str(version_info) + '.'
    if args.ex_ants_file:
        ex_ants = np.loadtxt(args.ex_ants_file, dtype=int)
        ex_ants = [str(ant) for ant in ex_ants if ant in uv.get_ants()]
        uv.history += ' Antennas to exclude in RTP: ' + ','.join(ex_ants) + '.'
    if args.filetype == 'miriad':
        base, ext = os.path.splitext(filename)
        uv.write_miriad(base + '.' + args.extension + ext, clobber=args.overwrite)
    else:
        base, ext = os.path.splitext(filename)
        uv.write_uvfits(base + args.extension + ext, clobber=args.overwrite)
    del(uv)  # Reset for next loop
