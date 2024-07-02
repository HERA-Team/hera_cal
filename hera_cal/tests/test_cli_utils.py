import pytest
import logging
from hera_cal._cli_tools import parse_args, run_with_profiling, filter_kwargs
import sys
from argparse import ArgumentParser
import numpy as np
from scipy import linalg
from line_profiler import LineProfiler

logger = logging.getLogger(__name__)


# The following simply mocks the runcall method of LineProfiler
# so that the `enable_by_count()` method is not called on it.
# For some reason, calling this screws with coverage.py.
def simplerun(self, func, *args, **kwargs):
    return func(*args, **kwargs)


LineProfiler.runcall = simplerun


@pytest.fixture(scope="function")
def a():
    a = ArgumentParser()
    a.add_argument("thing")
    return a


def do_some_stuff(thing):
    logger.info(f"Doing some stuff with {thing}")
    np.zeros(10000)
    linalg.inv(np.eye(1000))


def test_cli_logging(capsys, a):
    sys.argv = [sys.argv[0], "things", '--log-level', 'INFO']
    args = parse_args(a)
    print("EFFECTIVE LEVEL: ", logging.getLevelName(logger.getEffectiveLevel()))

    run_with_profiling(do_some_stuff, args, thing=args.thing)

    assert "Doing some stuff with things" in capsys.readouterr().out


@pytest.mark.parametrize(
    "param",
    [
        ["--log-plain-tracebacks"], ["--log-absolute-time"], ['--log-no-mem'], ['--log-mem-backend', 'psutil'], ['--log-show-path']
    ]
)
def test_cli_args(capsys, a, param):
    sys.argv = [sys.argv[0], "things"] + param
    args = parse_args(a)
    print(args)
    run_with_profiling(do_some_stuff, args, thing=args.thing)


def test_cli_logging_level(capsys, a):
    sys.argv = [sys.argv[0], "things", '--log-level', 'WARNING']
    args = parse_args(a)

    run_with_profiling(do_some_stuff, args, thing=args.thing)

    assert "Doing some stuff with things" not in capsys.readouterr().out


def test_cli_profiling(tmp_path_factory, a):
    profile = tmp_path_factory.mktemp("data") / "profile.txt"

    sys.argv = [sys.argv[0], "things", '--log-level', 'INFO', '--profile', '--profile-output', str(profile)]
    args = parse_args(a)

    run_with_profiling(do_some_stuff, args, thing=args.thing)

    assert profile.exists()


def test_cli_profile_funcs(tmp_path_factory, a):
    profile = tmp_path_factory.mktemp("data") / "profile.txt"

    sys.argv = [sys.argv[0], "things", '--log-level', 'INFO', '--profile', '--profile-output', str(profile), '--profile-funcs', 'numpy,scipy.linalg:inv']
    args = parse_args(a)

    run_with_profiling(do_some_stuff, args, thing=args.thing)

    assert profile.exists()


def test_cli_filter_kwargs(a):
    sys.argv = [sys.argv[0], "things", '--log-level', 'WARNING']
    args = parse_args(a)
    aa = filter_kwargs(vars(args))
    print("YUP DID THAT")
    print(aa['thing'])
    assert "log_level" not in aa
