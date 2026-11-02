"""
Removing, reordering and sorting must move measurements with their frames.

All three used to rebuild through a `{path: data}` dict, which collapses two
equal paths onto one FramePhotometryData -- after which both rows referenced
the SAME object and a measurement written to one appeared in the other.
`add_files` de-duplicating by path is the only reason it never fired.

`remove_files` also only clamped the selection when it ran off the end, so
removing a row BELOW the current one slid the selection onto a different file.
And `sort_by_time` assigned whatever list it was handed, so a path that was not
in the model produced an empty record and a path left out lost its data.
"""
import pytest

from models.multi_photometry_model import MultiPhotometryModel


def model(n=5):
    m = MultiPhotometryModel()
    m.add_files([f"/data/f{i}.fits" for i in range(n)])
    for i in range(n):
        d = m._frame_data[i]
        d.selected_source = (10.0 * i, 20.0 * i)
        d.mag = 15.0 + i
    return m


def tags(m):
    """(filename, mag) per row, in list order."""
    return [(m._frame_data[i].fname, m._frame_data[i].mag)
            for i in range(len(m.files))]


def test_removing_carries_the_right_measurements():
    m = model()
    m.remove_files([1, 3])
    assert [f for f, _ in tags(m)] == ["f0.fits", "f2.fits", "f4.fits"]
    assert [g for _, g in tags(m)] == [15.0, 17.0, 19.0]
    assert [m._frame_data[i].index for i in range(3)] == [0, 1, 2]


def test_removing_a_row_below_the_selection_keeps_the_same_file():
    m = model()
    m.set_current_index(3)                 # f3.fits
    m.remove_files([1])
    assert m.get_current_data().fname == "f3.fits"


def test_removing_the_selected_row_lands_in_range():
    m = model()
    m.set_current_index(4)
    m.remove_files([4])
    assert m._current_index == 3
    assert m.get_current_data().fname == "f3.fits"


def test_removing_everything_clears_the_selection():
    m = model(2)
    m.remove_files([0, 1])
    assert m.files == []
    assert m._current_index == -1


def test_reorder_carries_measurements_and_the_selection():
    m = model()
    m.set_current_index(1)
    m.reorder_files([4, 3, 2, 1, 0])
    assert [f for f, _ in tags(m)] == [f"f{i}.fits" for i in (4, 3, 2, 1, 0)]
    assert [g for _, g in tags(m)] == [19.0, 18.0, 17.0, 16.0, 15.0]
    assert m.get_current_data().fname == "f1.fits"


def test_reorder_rejects_anything_that_is_not_a_permutation():
    m = model()
    with pytest.raises(ValueError):
        m.reorder_files([0, 1, 2])            # too short
    with pytest.raises(ValueError):
        m.reorder_files([0, 0, 1, 2, 3])      # duplicated


def test_no_two_rows_ever_share_one_data_object():
    """The aliasing hazard, exercised directly."""
    m = model()
    m.reorder_files([2, 0, 4, 1, 3])
    m.remove_files([0])
    objs = [id(m._frame_data[i]) for i in range(len(m.files))]
    assert len(set(objs)) == len(objs)
    m._frame_data[0].mag = -99.0
    assert [d.mag for i, d in sorted(m._frame_data.items()) if i][0] != -99.0


def test_sort_by_time_requires_a_permutation():
    m = model(3)
    with pytest.raises(ValueError):
        m.sort_by_time(["/data/f0.fits", "/data/nope.fits", "/data/f2.fits"])
    with pytest.raises(ValueError):
        m.sort_by_time(["/data/f0.fits", "/data/f1.fits"])     # one dropped


def test_sort_by_time_preserves_data():
    m = model(3)
    m.sort_by_time(["/data/f2.fits", "/data/f0.fits", "/data/f1.fits"])
    assert [f for f, _ in tags(m)] == ["f2.fits", "f0.fits", "f1.fits"]
    assert [g for _, g in tags(m)] == [17.0, 15.0, 16.0]
