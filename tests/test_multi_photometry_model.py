"""
Data-integrity tests for MultiPhotometryModel:
  - set_zeropoint_for_file must not fabricate mag=zp when instr_mag is None
  - duplicate file paths must not alias onto one FramePhotometryData
  - the exported photometry table carries the measurement provenance
    (aperture + camera parameters) needed to reproduce a light curve
"""
from models.multi_photometry_model import MultiPhotometryModel


def make_model(paths):
    m = MultiPhotometryModel()
    m.add_files(paths)
    return m


def test_set_zeropoint_with_missing_instr_mag_yields_none():
    m = make_model(["a.fits"])
    # Photometry ran but instr_mag failed (e.g. exptime missing)
    m._frame_data[0].aperture_result = {'instr_mag': None, 'flux': 123.0}
    m.set_zeropoint_for_file("a.fits", 25.0)
    assert m._frame_data[0].mag is None, (
        "mag must be None when instr_mag is None -- not silently equal to the zeropoint")


def test_set_zeropoint_with_valid_instr_mag():
    m = make_model(["a.fits"])
    m._frame_data[0].aperture_result = {'instr_mag': -7.5}
    m.set_zeropoint_for_file("a.fits", 25.0)
    assert m._frame_data[0].mag == 17.5


def test_add_files_deduplicates_and_returns_new_indices():
    m = MultiPhotometryModel()
    added1 = m.add_files(["x.fits", "y.fits"])
    assert added1 == [0, 1]
    # Re-adding an existing path plus one new one: only the new one is added
    added2 = m.add_files(["x.fits", "z.fits"])
    assert added2 == [2]
    assert m.files == ["x.fits", "y.fits", "z.fits"]


def test_photometry_table_carries_per_frame_camera_parameters():
    """A bulk run detects each frame's mode; the table must report THAT frame's
    gain/read noise, not the UI defaults, or the SNR cannot be reproduced."""
    m = make_model(["low.fits"])
    m.gain, m.read_noise, m.dark_noise = 0.93, 3.18, 4.15   # UI is set to High
    m.update_frame_data(0, selected_source=(10.0, 20.0), aperture_radius=21.0,
                        r_best=18.5, mag=17.0, mag_err=0.02, flux=1000.0, snr=50.0,
                        aperture_result={'bkg_mean': 12.5, 'aperture_area': 1385.4},
                        exptime=58.14, camera_mode="Low", gain_used=18.7,
                        read_noise_used=30.1, dark_noise_used=4.15)
    row = m.get_photometry_table()[0]
    assert row.camera_mode == "Low"
    assert row.gain == 18.7 and row.read_noise == 30.1
    assert row.aperture == 21.0 and row.r_best == 18.5
    assert row.bkg_mean == 12.5 and row.aperture_area == 1385.4
    assert row.exptime == 58.14


def test_photometry_table_falls_back_to_ui_values_when_mode_unknown():
    """Interactive photometry sets no per-frame mode: report the UI settings
    and label them, rather than leaving the columns blank."""
    m = make_model(["a.fits"])
    m.gain, m.read_noise, m.dark_noise = 0.93, 3.18, 4.15
    m.inner_coef, m.outer_coef = 2.5, 4.0
    m.exptime_override = 30.0
    m.update_frame_data(0, selected_source=(1.0, 2.0), aperture_radius=15.0,
                        mag=18.0, aperture_result={'bkg_mean': 3.0})
    row = m.get_photometry_table()[0]
    assert row.camera_mode == "manual"
    assert row.gain == 0.93 and row.read_noise == 3.18 and row.dark_noise == 4.15
    assert row.inner_coef == 2.5 and row.outer_coef == 4.0
    assert row.exptime == 30.0
    assert row.aperture_area is None       # not measured -> stays empty, never 0


def test_no_aliasing_after_remove_with_unique_paths():
    m = make_model(["x.fits", "y.fits", "z.fits"])
    m._frame_data[0].mag = 10.0
    m._frame_data[2].mag = 30.0
    m.remove_files([1])
    # After rebuild, each remaining frame keeps its own data object
    assert m._frame_data[0].mag == 10.0
    assert m._frame_data[1].mag == 30.0
    assert m._frame_data[0] is not m._frame_data[1]
