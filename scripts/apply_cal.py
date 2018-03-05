import argparse
from hera_cal import apply_cal as ac
import sys

a = argparse.ArgumentParser(description="Apply (and optionally, also unapply) a calfits file to visibility file.")
a.add_argument("infile", type=str, help="path to visibility data file to calibrate")
a.add_argument("outfile", type=str, help="path to new visibility results file")
a.add_argument("new_cal", type=str, help="path to new calibration calfits file")
a.add_argument("--old_cal", default=None, help="path to old calibration calfits file (to unapply)")
a.add_argument("--filetype", type=str, default='miriad', help='filetype of input and output data files')
a.add_argument("--gain_convention", type=str, default='divide', 
              help="'divide' means V_obs = gi gj* V_true, 'multiply' means V_true = gi gj* V_obs.")
a.add_argument("--clobber", default=False, action="store_true", help='overwrites existing file at outfile')

args = a.parse_args(sys.argv[1:])
#TODO: incorporate add to history


ac.apply_cal(args.infile, args.outfile, args.new_cal, old_calibration=args.old_cal, filetype=args.filetype, 
             gain_convention=args.gain_convention, clobber=args.clobber)