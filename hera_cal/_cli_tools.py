"""Useful helper functions and argparsers for scripts."""
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
from typing import Iterable, Literal
import logging
from argparse import ArgumentParser

logger = logging.getLogger(__name__)

def setup_logger(level: str = 'INFO', width: int=160, show_time_as_diff: bool=True, rich_tracebacks: bool=True):    
    cns = Console(width=width)

    logging.basicConfig(
        format="%(message)s",
        level=level,
        handlers=[
            RicherHandler(
                console=cns,
                rich_tracebacks=rich_tracebacks,
                tracebacks_show_locals=True,
                show_path=False,
                show_time_as_diff=show_time_as_diff,
            )
        ],
    )
    

class DeltaTemplate(Template):
    delimiter = "%"


def strfdelta(tdelta, fmt):
    days = tdelta.days
    hours, rem = divmod(tdelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    d = {
        "D": f"{days:02d}",
        "H": f"{hours + 24*days:02d}",
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

    if x >= 100.0:
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
                time_diff = strfdelta(
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
        super().__init__(*args, **kwargs)
        self._log_render = LogRender.from_rich(
            self._log_render,
            show_mem_usage=show_mem_usage,
            mem_backend=mem_backend,
            show_time_as_diff=show_time_as_diff,
            delta_time_format=delta_time_format,
        )


def add_logging_args(parser: ArgumentParser):
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

def init_logger_from_args(args):
    setup_logger(
        width=args.log_width,
        level=args.log_level,
        rich_tracebacks=not args.log_plain_tracebacks,
        show_time_as_diff=not args.log_absolute_time,
    )

import importlib
from pathlib import Path
from argparse import ArgumentParser

def run_with_profiling(function, args, **kwargs):
    if args.profile:
        from line_profiler import LineProfiler

        logger.info(f"Profiling {function.__name__}. Output to {args.profile_output}")

        profiler = LineProfiler()

        profiler.add_function(function)

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
        profiler.runcall(function, **kwargs)
    else:
        function(**kwargs)

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

def parse_args(parser: ArgumentParser):
    add_profiling_args(parser)
    add_logging_args(parser)
    args = parser.parse_args()
    init_logger_from_args(args)
    return args

def filter_kwargs(kwargs: dict):
    return {k: v for k, v in kwargs.items() if not k.startswith("profile") and not k.startswith("log")}