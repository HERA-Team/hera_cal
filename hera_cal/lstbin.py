"""
lstbin.py
=========

Bin complex visibility data by LST
"""
from abscal_funcs import *


def lst_align(data_fname, model_fnames=None, dLST=0.00299078, output_fname=None, outdir=None,
              overwrite=False, verbose=True, write_miriad=True, output_data=False,
              match='nearest', **interp2d_kwargs):
    """
    Interpolate complex visibilities to align time integrations with an LST grid.
    If output_fname is not provided, write interpolated data as
    input filename + "L.hour.decimal" miriad file. The LST grid can be created from
    scratch using the dLST parameter, or an LST grid can be imported from a model file.

    Parameters:
    -----------


    match : type=str, LST-bin matching method, options=['nearest','forward','backward']

    """
    # try to load model
    if model_fnames is not None:
        uvm = UVData()
        uvm.read_miriad(model_fnames)
        model_lsts = np.unique(uvm.lst_array) * 12 / np.pi
        model_freqs = np.unique(uvm.freq_array)
    else:
        # generate LST array
        model_lsts = np.arange(0, 24, dLST)
        model_freqs = None

    # load file
    echo("loading {}".format(data_fname), verbose=verbose)
    uvd = UVData()
    uvd.read_miriad(data_fname)

    # get data
    data, flags = UVData2AbsCalDict(uvd, pop_autos=False)

    # get data lst and freq arrays
    data_lsts, data_freqs = np.unique(uvd.lst_array) * 12 / np.pi, np.unique(uvd.freq_array)
    Ntimes = len(data_lsts)

    # get closest lsts
    sort = np.argsort(np.abs(model_lsts - data_lsts[0]))[:2]
    if match == 'nearest':
        start = sort[0]
    elif match == 'forward':
        start = np.max(sort)
    elif match == 'backward':
        start = np.min(sort)

    # create lst grid
    lst_indices = np.arange(start, start+Ntimes)
    model_lsts = model_lsts[lst_indices]

    # specify freqs
    if model_freqs is None:
        model_freqs = data_freqs
    Nfreqs = len(model_freqs)

    # interpolate data
    echo("interpolating data", verbose=verbose)
    interp_data, interp_flags = interp2d_vis(data, data_lsts, data_freqs, model_lsts, model_freqs, **interp2d_kwargs)
    Nbls = len(interp_data)

    # reorder into arrays
    uvd_data = np.array(interp_data.values())
    uvd_data = uvd_data.reshape(-1, 1, Nfreqs, 1)
    uvd_flags = np.array(interp_flags.values()).astype(np.bool)
    uvd_flags = uvd_flags.reshape(-1, 1, Nfreqs, 1)
    uvd_keys = np.repeat(np.array(interp_data.keys()).reshape(-1, 1, 2), Ntimes, axis=1).reshape(-1, 2)
    uvd_bls = np.array(map(lambda k: uvd.antnums_to_baseline(k[0], k[1]), uvd_keys))
    uvd_times = np.array(map(lambda x: utils.JD2LST.LST2JD(x, np.median(np.floor(uvd.time_array)), uvd.telescope_location_lat_lon_alt_degrees[1]), model_lsts))
    uvd_times = np.repeat(uvd_times[np.newaxis], Nbls, axis=0).reshape(-1)
    uvd_lsts = np.repeat(model_lsts[np.newaxis], Nbls, axis=0).reshape(-1)
    uvd_freqs = model_freqs.reshape(1, -1)

    # assign to uvdata object
    uvd.data_array = uvd_data
    uvd.flag_array = uvd_flags
    uvd.baseline_array = uvd_bls
    uvd.ant_1_array = uvd_keys[:, 0]
    uvd.ant_2_array = uvd_keys[:, 1]
    uvd.time_array = uvd_times
    uvd.lst_array = uvd_lsts * np.pi / 12
    uvd.freq_array = uvd_freqs
    uvd.Nfreqs = Nfreqs

    # write miriad
    if write_miriad:
        # check output
        if outdir is None:
            outdir = os.path.dirname(data_fname)
        if output_fname is None:
            output_fname = os.path.basename(data_fname) + 'L.{:07.4f}'.format(model_lsts[0])
        output_fname = os.path.join(outdir, output_fname)
        if os.path.exists(output_fname) and overwrite is False:
            raise IOError("{} exists, not overwriting".format(output_fname))

        # write to file
        echo("saving {}".format(output_fname), verbose=verbose)
        uvd.write_miriad(output_fname, clobber=True)

    # output data and flags
    if output_data:
        return interp_data, interp_flags, model_lsts, model_freqs


def lstbin_arg_parser():
    a = argparse.ArgumentParser()
    a.add_argument("--data_files", type=str, nargs='*', help="list of miriad files of data to-be-calibrated.", required=True)
    a.add_argument("--model_files", type=str, nargs='*', default=[], help="list of data-overlapping miriad files for visibility model.", required=True)
    a.add_argument("--calfits_fname", type=str, default=None, help="name of output calfits file.")
    a.add_argument("--overwrite", default=False, action='store_true', help="overwrite output calfits file if it exists.")
    a.add_argument("--silence", default=False, action='store_true', help="silence output from abscal while running.")
    a.add_argument("--zero_psi", default=False, action='store_true', help="set overall gain phase 'psi' to zero in linsolve equations.")
    return a






