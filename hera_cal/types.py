from typing import Literal

Antpair = tuple[int, int]
Pol = Literal['xx', 'yy', 'ee', 'nn', 'xy', 'yx', 'en', 'ne']
Baseline = tuple[int, int, Pol]
