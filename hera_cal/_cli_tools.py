"""Useful helper functions and argparsers for scripts.

This module contains functions that add groups of arguments to an argparse.ArgumentParser.
For instance, it adds a group of arguments that determine how logging proceeds, and
also a group of arguments that determine if line-profiling is run, and how.

What This Module Adds to the Logging Experience
===============================================
See the :func:`setup_logger` function for details on what is added to the logger by
this module. Note that this function must be called for logging to be altered at all
(see the "how to use" section below for details).

Note that logging in python is only used if you actually use the ``logging`` module
and make logging statements. To get the most out of this, do the following in all
``hera_cal`` modules::

    import logging
    logger = logging.getLogger(__name__)

Then, in the body of the module, add logging statements instead of standard ``print``
statements::

    logger.info("This is an informative message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.debug("This is a debug message")

By setting the ``--log-level`` argument to a script (that is set up according to the
guidelines in this module), you can control how much actually is printed out on any
given run. Furthermore, the logged messages include extra information, such as the
current time, the log level of the message, and optionally other info. For example,
this module adds the ability to have a column with the current and peak memory usage.
Thus, for example, instead of simply having a message that prints out as::

    This is an informative message
    This is a warning message

You get::

    00:00:54 INFO     01.650 MB | 04.402 MB This is an informative message
    00:00:55 WARNING  01.650 MB | 04.402 MB This is a warning message

The memory-printing feature is useful for a cheap way to see if a script is leaking
memory, and understanding how much memory scripts are consuming.

What This Module Does for Line-Profiling
========================================
Line-profiling is a way to see how much time is spent in each line of code. This is
useful for identifying bottlenecks in code. This module adds the ability to run a
script with line-profiling, and to save the results to a human-readable file.
Importantly, it also adds the ability to specify from the command line which functions
are included in the line-profilng. See :func:`run_with_profiling` for details.


How To Use This Module In Your Script
=====================================
This module is intended to be imported into scripts that use argparse. For instance,
say you have written a script called ``script.py``, with the following contents::

    import argparse

    # An argument parser for the script. Could be constructed from an imported function.
    parser = argparse.ArgumentParser()
    parser.add_argument("foo", type=int)

    # A function intended to do the work. Usually in hera_cal this is some imported
    # function like ``load_delay_filter_and_write``
    def run_script(**kwargs):
        print(kwargs)

    # Parse arguments and run the script
    if __name__ == "__main__":
        args = parser.parse_args()
        kwargs = var(args)
        run_script(**kwargs)

You can add better logging options and line-profiling options to this script simply by
applying ``parse_args`` to the ``args``, and running the main function through the
``run_with_profiling`` function::

    import argparse
    from hera_cal._cli_tools import parse_args, run_with_profiling, filter_kwargs

    # An argument parser for the script. Could be constructed from an imported function.
    parser = argparse.ArgumentParser()
    parser.add_argument("foo", type=int)

    # A function intended to do the work. Usually in hera_cal this is some imported
    # function like ``load_delay_filter_and_write``
    def run_script(**kwargs):
        print(kwargs)

    # Parse arguments and run the script
    if __name__ == "__main__":
        args = parse_args(parser)         # Adds the logging/profiling arguments
        kwargs = filter_kwargs(var(args)) # Filters out the logging/profiling arguments
        run_with_profiling(run_script, args, **kwargs)  # Runs the script with profiling


How to Use This Module Interactively
====================================
This module is generally meant to be used directly in scripts, but you may want to use
some of the logging features in an interactive session. To do this, simply import
``setup_logger`` from this module, and call it::

    >>> from hera_cal._cli_tools import setup_logger
    >>> setup_logger(level="DEBUG", show_time_as_diff=True)

Then, any logging statements in hera_cal code (or your own code in the interactive
session) will have the desired logging behavior.

"""
from __future__ import annotations

import math
import tracemalloc as tr
from datetime import datetime
from rich._log_render import FormatTimeCallable
from rich.console import Console, ConsoleRenderable
from rich.containers import Renderables
from rich.logging import RichHandler
from rich.table import Table
from rich.text import Text, TextType
from string import Template
from typing import Iterable, Literal, Callable
import logging
from argparse import ArgumentParser, Namespace
import importlib
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)


def setup_logger(
    level: str = 'INFO',
    width: int = 160,
    show_time_as_diff: bool = True,
    rich_tracebacks: bool = True,
    show_mem: bool = True,
    mem_backend: Literal["tracemalloc", "psutil"] = "tracemalloc",
    show_path: bool = False,
):
    """Setup the logger for hera_cal scripts.

    Parameters
    ----------
    level : str, optional
        The logging level to use. Only messages at or above this level will be printed.
        Options are "DEBUG", "INFO", "WARNING", "ERROR", and "CRITICAL".
    width : int, optional
        The width of the on-screen text before wrapping.
    show_time_as_diff : bool, optional
        If True, show the time since the last message. If False, show the absolute time.
    rich_tracebacks : bool, optional
        If True, show tracebacks with rich formatting. If False, show tracebacks with
        plain formatting.
    show_mem : bool, optional
        If True, show the current and peak memory usage in the log messages.
    mem_backend : {"tracemalloc", "psutil"}, optional
        The backend to use for measuring memory usage. "tracemalloc" is the default, but
        "psutil" is more accurate.
    show_path : bool, optional
        If True, show the path to the file where the log message was generated on each
        log line.
    """
    cns = Console(width=width)

    logging.basicConfig(
        format="%(message)s",
        level=level,
        handlers=[
            RicherHandler(
                console=cns,
                rich_tracebacks=rich_tracebacks,
                tracebacks_show_locals=True,
                show_path=show_path,
                show_time_as_diff=show_time_as_diff,
                show_mem_usage=show_mem,
                mem_backend=mem_backend,
            )
        ],
        force=True,
    )


class DeltaTemplate(Template):
    delimiter = "%"


def _strfdelta(tdelta, fmt):
    days = tdelta.days
    hours, rem = divmod(tdelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    d = {
        "D": f"{days:02d}",
        "H": f"{hours + 24 * days:02d}",
        "h": f"{hours:02d}",
        "M": f"{minutes:02d}",
        "S": f"{seconds:02d}",
    }

    t = DeltaTemplate(fmt)
    return t.substitute(**d)


def fmt_bytes(x: int):
    """Format a number in bytes."""
    order = int(math.log(x, 1024))
    x /= 1024**order

    if x >= 100:
        order += 1
        x /= 1024

    unit = [" B", "KB", "MB", "GB", "TB"][order]
    return f"{x:06.3f} {unit}"


class LogRender:
    def __init__(
        self,
        show_time: bool = True,
        show_level: bool = False,
        show_path: bool = True,
        time_format: str | FormatTimeCallable = "[%x %X]",
        omit_repeated_times: bool = True,
        level_width: int | None = 8,
        show_mem_usage: bool = True,
        mem_backend: Literal["tracemalloc", "psutil"] = "tracemalloc",
        show_time_as_diff: bool = False,
        delta_time_format: str = "%H:%M:%S",
    ) -> None:
        """
        A class for rendering a log message.

        Parameters
        ----------
        show_time : bool, optional
            If True, show the time of the log message.
        show_level : bool, optional
            If True, show the level of the log message.
        show_path : bool, optional
            If True, show the path to the file where the log message was generated.
        time_format : str or callable, optional
            The format to use for the time. If a callable, it should take a datetime
            object and return a string.
        omit_repeated_times : bool, optional
            If True, don't show the time if it is the same as the previous log message.
        level_width : int or None, optional
            The width of the level column. If None, the level column will be omitted.
        show_mem_usage : bool, optional
            If True, show the current and peak memory usage in the log messages.
        mem_backend : {"tracemalloc", "psutil"}, optional
            The backend to use for measuring memory usage. "tracemalloc" is the default, but
            "psutil" is more accurate.
        show_time_as_diff : bool, optional
            If True, show the time since the last message. If False, show the absolute time.
        delta_time_format : str, optional
            The format to use for the time since the last message.
        """
        self.show_time = show_time
        self.show_level = show_level
        self.show_path = show_path
        self.time_format = time_format
        self.omit_repeated_times = omit_repeated_times
        self.level_width = level_width
        self._last_time: Text | None = None
        self._first_time: datetime | None = None
        self.delta_time_format = delta_time_format

        self.show_mem_usage = show_mem_usage
        self.mem_backend = mem_backend
        if mem_backend == "tracemalloc":
            if not tr.is_tracing():
                tr.start()

        elif mem_backend == "psutil":
            import psutil

            self._pr = psutil.Process
        else:
            raise ValueError(f"Invalid memory backend: {mem_backend}")

        self.show_time_as_diff = show_time_as_diff

    @classmethod
    def from_rich(
        cls,
        rich_log_render: LogRender,
        show_mem_usage: bool = True,
        mem_backend: Literal["tracemalloc", "psutil"] = "tracemalloc",
        show_time_as_diff: bool = False,
        delta_time_format: str = "%H:%M:%S",
    ):
        """
        Create a RichLog instance from a RichLog instance.

        :param rich_log_render:
            A RichLog instance.
        :param show_mem_usage:
            Whether to show
        """
        return cls(
            show_time=rich_log_render.show_time,
            show_level=rich_log_render.show_level,
            show_path=rich_log_render.show_path,
            time_format=rich_log_render.time_format,
            omit_repeated_times=rich_log_render.omit_repeated_times,
            level_width=rich_log_render.level_width,
            show_mem_usage=show_mem_usage,
            mem_backend=mem_backend,
            show_time_as_diff=show_time_as_diff,
            delta_time_format=delta_time_format,
        )

    def __call__(
        self,
        console: Console,
        renderables: Iterable[ConsoleRenderable],
        log_time: datetime | None = None,
        time_format: str | FormatTimeCallable | None = None,
        level: TextType = "",
        path: str | None = None,
        line_no: int | None = None,
        link_path: str | None = None,
    ) -> Table:

        output = Table.grid(padding=(0, 1))
        output.expand = True
        if self.show_time:
            output.add_column(style="log.time")
        if self.show_level:
            output.add_column(style="log.level", width=self.level_width)

        if self.show_mem_usage:
            output.add_column()

        output.add_column(ratio=1, style="log.message", overflow="fold")

        if self.show_path and path:
            output.add_column(style="log.path")

        row = []
        if self.show_time:
            log_time = log_time or console.get_datetime()
            if self._first_time is None:
                self._first_time = log_time

            if not self.show_time_as_diff:
                time_format = time_format or self.time_format
                if callable(time_format):
                    log_time_display = time_format(log_time)
                else:
                    log_time_display = Text(log_time.strftime(time_format))
                if log_time_display == self._last_time and self.omit_repeated_times:
                    row.append(Text(" " * len(log_time_display)))
                else:
                    row.append(log_time_display)
                    self._last_time = log_time_display
            else:
                time_diff = _strfdelta(
                    log_time - self._first_time, self.delta_time_format
                )
                row.append(time_diff)

        if self.show_level:
            row.append(level)

        if self.show_mem_usage:
            if self.mem_backend == "psutil":
                m = self._pr().memory_info().rss
                row.append(fmt_bytes(m))
            elif self.mem_backend == "tracemalloc":
                m, p = tr.get_traced_memory()
                row.append(f"{fmt_bytes(m)} | {fmt_bytes(p)}")

        row.append(Renderables(renderables))
        if self.show_path and path:
            path_text = Text()
            path_text.append(
                path, style=f"link file://{link_path}" if link_path else ""
            )
            if line_no:
                path_text.append(":")
                path_text.append(
                    f"{line_no}",
                    style=f"link file://{link_path}#{line_no}" if link_path else "",
                )
            row.append(path_text)

        output.add_row(*row)
        return output


class RicherHandler(RichHandler):
    def __init__(
        self,
        *args,
        show_mem_usage: bool = True,
        mem_backend: Literal["tracemalloc", "psutil"] = "tracemalloc",
        show_time_as_diff: bool = False,
        delta_time_format: str = "%H:%M:%S",
        **kwargs,
    ):
        """
        A RichHandler that adds memory usage and time difference to the log.

        See https://rich.readthedocs.io/en/stable/logging.html for details on the base
        class.

        Parameters
        ----------
        show_mem_usage
            Whether to show memory usage in the log.
        mem_backend
            Which backend to use for memory usage. Either "tracemalloc" or "psutil".
        show_time_as_diff
            Whether to show the time as a difference from the first log.
        delta_time_format
            Format string for the time difference.

        Other Parameters
        ----------------
        **kwargs
            Passed to the base class.
        """
        super().__init__(*args, **kwargs)
        self._log_render = LogRender.from_rich(
            self._log_render,
            show_mem_usage=show_mem_usage,
            mem_backend=mem_backend,
            show_time_as_diff=show_time_as_diff,
            delta_time_format=delta_time_format,
        )


def add_logging_args(parser: ArgumentParser):
    """
    Add logging arguments to an argparse parser.

    All arguments are optional and have sensible defaults. All arguments begin
    with "log-" so they can be easily identified.
    """
    grp = parser.add_argument_group(title="Options for logging")

    grp.add_argument(
        "--log-level", type=str, default='INFO',
        choices=['INFO', 'ERROR', 'WARNING', 'CRITICAL', "DEBUG"],
        help="logging level to display. "
    )
    grp.add_argument(
        "--log-width", type=int, default=160,
        help="width of logging output"
    )
    grp.add_argument(
        "--log-plain-tracebacks", action='store_true',
        help="use plain instead of rich tracebacks"
    )
    grp.add_argument(
        "--log-absolute-time", action='store_true',
        help="show logger time as absolute instead of relative to start"
    )
    grp.add_argument(
        "--log-no-mem", action='store_true', help='do not show memory usage'
    )
    grp.add_argument(
        "--log-mem-backend", type=str, default='tracemalloc', choices=['tracemalloc', 'psutil'],
    )
    grp.add_argument(
        "--log-show-path", action='store_true', help='show path of code in log msg'
    )


def init_logger_from_args(args):
    """Call :func:`setup_logger` with arguments from an argparse parser."""
    setup_logger(
        width=args.log_width,
        level=args.log_level,
        rich_tracebacks=not args.log_plain_tracebacks,
        show_time_as_diff=not args.log_absolute_time,
        mem_backend=args.log_mem_backend,
        show_mem=not args.log_no_mem,
        show_path=args.log_show_path,
    )


def _add_profile_funcs(profiler, profile_funcs):
    for fnc in profile_funcs.split(","):
        module = importlib.import_module(fnc.split(":")[0])
        _fnc = module
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if ":" not in fnc:
                profiler.add_module(_fnc)
            else:
                for att in fnc.split(":")[-1].split("."):
                    _fnc = getattr(_fnc, att)
                profiler.add_function(_fnc)


def run_with_profiling(function: Callable, args: Namespace, *posargs, **kwargs):
    """Run a function with profiling if the user has requested it.

    Only runs profiling if `args.profile` is True, and doesn't even import
    ``line_profiler`` if it's not.

    Parameters
    ----------
    function
        The function to run.
    args
        The namespace object returned by ``ArgumentParser.parse_args()``. This must
        have a ``profile`` attribute that is True if profiling is requested, as well
        as a ``profile_output`` attribute that is the path to the output file,
        and a ``profile_funcs`` attribute that is a comma-separated list of functions to
        profile. Use :func:`add_profiling_args` to add these arguments to your parser.
    *posargs
        Positional arguments to pass to ``function``.
    **kwargs
        Keyword arguments to pass to ``function``.
    """
    if args.profile:
        from line_profiler import LineProfiler

        logger.info(f"Profiling {function.__name__}. Output to {args.profile_output}")

        profiler = LineProfiler()

        profiler.add_function(function)

        # Now add any user-defined functions that they want to be profiled.
        # Functions must be sent in as "path.to.module:function_name" or
        # "path.to.module:Class.method".
        if args.profile_funcs:
            _add_profile_funcs(profiler, args.profile_funcs)

        pth = Path(args.profile_output)
        if not pth.parent.exists():
            pth.parent.mkdir(parents=True)

        out = profiler.runcall(function, *posargs, **kwargs)

        with open(pth, "w") as fl:
            profiler.print_stats(stream=fl, stripzeros=True)

        return out

    else:
        return function(*posargs, **kwargs)


def add_profiling_args(parser: ArgumentParser):
    """
    Add profiling arguments to an argparse parser.

    All arguments are optional and have sensible defaults. All arguments begin with
    "profile-" so they can be easily identified.
    """
    grp = parser.add_argument_group(title="Options for line-profiling")

    grp.add_argument(
        "--profile", action="store_true", help="Line-Profile the script"
    )
    grp.add_argument(
        "--profile-funcs", type=str, default='', help="List of functions to profile"
    )
    grp.add_argument(
        "--profile-output", type=str, help="Output file for profiling info."
    )


def parse_args(parser: ArgumentParser):
    """Convenience function for setting up CLI goodies from this module.

    This function adds both profiling and logging arguments to the parser, parses the
    args, and sets up the logger. It returns the parsed args.
    """
    add_profiling_args(parser)
    add_logging_args(parser)
    args = parser.parse_args()
    init_logger_from_args(args)
    return args


def filter_kwargs(kwargs: dict) -> dict:
    """Filter out kwargs that are used for logging and profiling."""
    return {
        k: v for k, v in kwargs.items() if (
            k != "profile" and
            not k.startswith("profile_") and
            not k.startswith("log_")
        )
    }
