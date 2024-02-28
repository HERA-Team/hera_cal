from .. import io
from pathlib import Path
import pytest


def test_apply_calfile_rules(tmpdir_factory):
    direc = tmpdir_factory.mktemp("test_apply_calfile_rules")

    datas = [Path(direc / f"data{i}.uvh5") for i in range(3)]
    for d in datas:
        d.touch()

    cals = [Path(direc / f"data{i}.calfile") for i in range(3)]
    for c in cals:
        c.touch()

    data_files, calfiles = io.apply_calfile_rules(
        [[str(d) for d in datas]],
        calfile_rules=[(".uvh5", ".calfile")],
        ignore_missing=False,
    )
    assert len(data_files[0]) == 3
    assert len(calfiles[0]) == 3

    cals[-1].unlink()
    with pytest.raises(IOError, match="does not exist"):
        io.apply_calfile_rules(
            [[str(d) for d in datas]],
            calfile_rules=[(".uvh5", ".calfile")],
            ignore_missing=False,
        )

    with pytest.warns(UserWarning, match="Calibration file .* does not exist"):
        data_files, calfiles = io.apply_calfile_rules(
            [[str(d) for d in datas]],
            calfile_rules=[(".uvh5", ".calfile")],
            ignore_missing=True,
        )
    assert len(data_files[0]) == 2
    assert len(calfiles[0]) == 2


def test_get_where_inpainted(tmp_path_factory):
    tmp: Path = tmp_path_factory.mktemp("get_where_inpainted")

    fls = []
    for outer in ["abc", "def"]:
        these = []
        for filename in outer:
            (tmp / f"{filename}.uvh5").touch()
            (tmp / f"{filename}.where_inpainted.h5").touch()
            these.append(tmp / f"{filename}.uvh5")
        fls.append(these)

    out = io._get_where_inpainted_files(
        fls, [(".uvh5", ".where_inpainted.h5")]
    )

    assert len(out) == 2
    assert len(out[0]) == 3
    assert len(out[1]) == 3

    with pytest.raises(IOError, match="Where inpainted file"):
        io._get_where_inpainted_files(fls, [(".uvh5", ".non_existent.h5")])


def test_configure_inpainted_mode():
    flg, inp = io._configure_inpainted_mode(
        output_flagged=True, output_inpainted=True, where_inpainted_files=[]
    )
    assert flg
    assert inp

    flg, inp = io._configure_inpainted_mode(
        output_flagged=True, output_inpainted=True, where_inpainted_files=["a_file.h5"]
    )
    assert flg
    assert inp

    flg, inp = io._configure_inpainted_mode(
        output_flagged=True, output_inpainted=False, where_inpainted_files=[]
    )
    assert flg
    assert not inp

    flg, inp = io._configure_inpainted_mode(
        output_flagged=True, output_inpainted=False, where_inpainted_files=["a_file.h5"]
    )
    assert flg
    assert not inp

    flg, inp = io._configure_inpainted_mode(
        output_flagged=True, output_inpainted=None, where_inpainted_files=[]
    )
    assert flg
    assert not inp

    flg, inp = io._configure_inpainted_mode(
        output_flagged=True, output_inpainted=None, where_inpainted_files=["a_file.h5"]
    )
    assert flg
    assert inp

    flg, inp = io._configure_inpainted_mode(
        output_flagged=False, output_inpainted=True, where_inpainted_files=[]
    )
    assert not flg
    assert inp

    with pytest.raises(ValueError, match="Both output_inpainted and output_flagged"):
        io._configure_inpainted_mode(
            output_flagged=False, output_inpainted=False, where_inpainted_files=[]
        )
