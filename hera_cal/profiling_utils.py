import importlib
from pathlib import Path
from argparse import ArgumentParser

def run_func_with_profiling(fnc, args, **kwargs):
    if args.profile:
        from line_profiler import LineProfiler
        if 'output_file_select' in kwargs:
            label = '.'.join(kwargs['output_file_select'])
            output_file = Path(f"lst_bin_profile_{label}.prof")
        else:
            output_file = Path('lst_bin_profile.prof')

        print(f"Profiling the LST-binning. Output to {output_file}")

        profiler = LineProfiler()

        profiler.add_function(fnc)

        # Now add any user-defined functions that they want to be profiled.
        # Functions must be sent in as "path.to.module:function_name" or
        # "path.to.module:Class.method".
        for fnc in args.profile_funcs.split(","):
            module = importlib.import_module(fnc.split(":")[0])
            _fnc = module
            if ":" not in fnc:
                profiler.add_module(_fnc)
            else:
                for att in fnc.split(":")[-1].split("."):
                    _fnc = getattr(_fnc, att)
                profiler.add_function(_fnc)

            
    if args.profile:
        profiler.runcall(fnc, **kwargs)
    else:
        fnc(**kwargs)

    if args.profile:
        pth = Path(args.profile_output)
        if not pth.parent.exists():
            pth.parent.mkdir(parents=True)

        with open(pth, "w") as fl:
            profiler.print_stats(stream=fl, stripzeros=True)

def add_profiling_args(parser: ArgumentParser):
    grp = parser.add_argument_group(title="Options for line-profiling")

    grp.add_argument("--profile", action="store_true",
                        help="Line-Profile the script")
    grp.add_argument("--profile-funcs", type=str,
                        help="List of functions to profile")
    grp.add_argument("--profile-output", type=str, help="Output file for profiling info.")