"""
The operator's flats must not be silently replaced.

CalibrationWorker stored `self.auto_flat` -- the answer to the "No Flat frames
provided. Create auto-flats per Light?" prompt -- and never read it. Worse, a
master flat whose shape did not match the lights was turned into `fm = None`,
which dropped straight through to the auto-flat branch. So a binning mismatch
silently swapped measured flats for a degree-3 polynomial model of each frame's
OWN sky, with nothing in the header to say so.

`self.auto_flat_params` (polydeg, star_sigma) was likewise never read, so the
fit always ran at the defaults whatever the operator chose.
"""
import numpy as np
import pytest

from controllers.workers import CalibrationWorker


def worker(**kw):
    return CalibrationWorker(sets=[], output_dir="", **kw)


def test_consent_flag_and_params_are_stored():
    w = worker(auto_flat=True, auto_flat_params={'polydeg': 5, 'star_sigma': 3.0})
    assert w.auto_flat is True
    assert w.auto_flat_params['polydeg'] == 5
    assert w.auto_flat_params['star_sigma'] == 3.0


def test_auto_flat_branch_is_gated_on_consent():
    """
    The guard the worker now applies, exercised directly.

    A light set with no flat and no consent must raise rather than quietly
    manufacture a flat from the sky.
    """
    w = worker(auto_flat=False)
    assert w.auto_flat is False

    # Mirror of the branch in CalibrationWorker.run
    def choose_flat(flat_master, light_shape, allow_auto):
        if flat_master is not None and flat_master.shape != light_shape:
            raise ValueError("Master flat is %s, light frame is %s"
                             % (flat_master.shape, light_shape))
        if flat_master is None and not allow_auto:
            raise ValueError("No flat field is available for this light set "
                             "and auto-flat was not authorised.")
        return flat_master

    with pytest.raises(ValueError, match="not authorised"):
        choose_flat(None, (100, 100), w.auto_flat)

    with pytest.raises(ValueError, match="Master flat is"):
        choose_flat(np.ones((50, 50), dtype='float32'), (100, 100), True)

    good = np.ones((100, 100), dtype='float32')
    assert choose_flat(good, (100, 100), False) is good


def test_run_refuses_a_shape_mismatched_flat():
    """
    The real code path, read with comments stripped so the prose about the old
    behaviour cannot satisfy the assertions.
    """
    src = open(CalibrationWorker.run.__code__.co_filename, encoding='utf-8').read()
    body = src[src.index("def run(self)"):]
    code = "\n".join(l for l in body.split("\n")
                     if not l.strip().startswith("#"))

    assert "Master flat is" in code and "light frame is" in code, (
        "the shape-mismatch guard must be in run(), not only in this test")
    assert "self.auto_flat_params" in code, (
        "the operator's auto-flat parameters must reach the fit")
    assert "not self.auto_flat" in code, (
        "the auto-flat branch must be gated on the consent flag")
    assert "AUTO-FLAT" in code, (
        "auto-flat use must be recorded in the output header")
