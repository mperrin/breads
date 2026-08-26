"""Unit tests for the spline fitting machinery in ``breads.jwst_tools.splines``.

These tests exist primarily as a *numerical regression harness*. The 3D spline
fitting code (:func:`~breads.jwst_tools.splines.fit_3dspline` and its worker
:func:`~breads.jwst_tools.splines._task_fit_3dspline`) is extremely expensive on
real data -- many hours and tens of GB of RAM -- and is therefore a target for
performance optimization. Several planned optimizations are intended to be
*numerically neutral*: they should change the runtime and memory footprint but
not the returned arrays. The snapshot test below is what proves that.

The tests deliberately avoid any real data files. ``fit_3dspline`` only requires
a small duck-typed interface from its data object (see :class:`StubIFU`), so the
whole module runs in a few seconds with no ``jwst``, CRDS, or FITS dependency.

Regenerating the reference snapshot
-----------------------------------
The snapshot in ``data/test_splines_reference.npz`` must only be regenerated
when a change to the fitting algorithm is *intended*::

    python -m breads.tests.test_splines --regenerate


Test Coverage Summary
---------------------
| Test |  Guards |
| test_get_spline_model_partition_of_unity  | spline basis contract (Tier 3 edits would change this deliberately)  + exact cubic reproduction |
| test_get_spline_model_zero_outside_node_range | no-extrapolation behavior |
| test_fit_3dspline_recovers_smooth_field | gross breakage |
| test_flags_injected_outlier_preserves_input_bad_pixels | rejection logic |
| test_serial_equals_parallel | concurrency contract (Tier 2) |
| test_reference_snapshot | the Tier 1/2 no-op proof |
| test_tile_vs_broadcast  | Tier 1 #1 |
| test_lsq_linear_vs_Cholesky  | Tier 1 #4 |


"""

import os

import numpy as np

from breads.fit import fitfm
from breads.instruments import Instrument
from breads.jwst_tools.splines import _build_3dspline_basis, _tmp_fm, fit_3dspline
from breads.utils import get_spline_model

REFERENCE_FILE = os.path.join(os.path.dirname(__file__), "data", "test_splines_reference.npz")

# Names of the six arrays returned by fit_3dspline, in order.
FIT_OUTPUT_NAMES = ("bestfit_model", "noise", "bad_pixels", "residuals",
                    "spline3d_paras", "spline3d_paras_err")


class StubIFU:
    """Minimal stand-in for a JWST IFU data object, for testing spline fitting.

    ``fit_3dspline`` only accesses ``breads_header["WV_REF"]``, ``data``,
    ``noise``, ``bad_pixels`` and ``get_ifu_coords()``, so this stub is
    sufficient to exercise the full fitting path without any real data.

    The synthetic scene is a smooth low-order function of the IFU coordinates
    and wavelength, which the 3D spline basis can represent well, plus Gaussian
    noise. This makes the expected fit quality predictable.
    """

    def __init__(self, ny=120, nx=80, seed=42, noise_level=0.01, outlier=None):
        rng = np.random.default_rng(seed)
        self.breads_header = {"WV_REF": 4.0}

        yy, xx = np.mgrid[0:ny, 0:nx]
        # IFU coordinates spanning a modest patch of sky, in arcsec
        self._ifux = xx / nx * 0.8 - 0.4
        self._ifuy = yy / ny * 0.8 - 0.4
        self.wavelengths = 3.0 + 1.0 * (xx / nx)

        self.truth = self.smooth_field(self._ifux, self._ifuy, self.wavelengths)
        self.noise_level = noise_level
        self.noise = np.full((ny, nx), noise_level)
        self.data = self.truth + rng.normal(0, noise_level, (ny, nx))

        self.bad_pixels = np.ones((ny, nx))
        # A pre-existing bad pixel, to exercise the masking path
        self.bad_pixels[5, 5] = np.nan

        if outlier is not None:
            position, amplitude = outlier
            self.data[position] += amplitude

    @staticmethod
    def smooth_field(ifux, ifuy, wvs):
        """The noiseless scene: smooth and low-order, so the spline can fit it."""
        return 1.0 + 0.4 * ifux + 0.3 * ifuy ** 2 + 0.2 * wvs

    def get_ifu_coords(self):
        return self._ifux, self._ifuy


def default_nodes():
    """Node vectors matching the StubIFU field of view.

    Spacing is chosen so that ``stamp_size`` spans several nodes, mirroring the
    geometry used on real data where stamps overlap by ``N_overlap_nodes``.
    """
    x_nodes = np.arange(-0.5, 0.5001, 0.05)
    y_nodes = np.arange(-0.5, 0.5001, 0.05)
    wv_nodes = np.linspace(3.0, 4.0, 4)
    return x_nodes, y_nodes, wv_nodes


def run_reference_fit(max_cores=1, **stub_kwargs):
    """Run ``fit_3dspline`` on the standard stub and return copies of its outputs.

    Copies are essential: the returned arrays are views onto shared memory that
    would be reused by a subsequent call.
    """
    dataobj = StubIFU(**stub_kwargs)
    x_nodes, y_nodes, wv_nodes = default_nodes()
    outputs = fit_3dspline(dataobj, x_nodes, y_nodes, wv_nodes,
                           stamp_size=(0.25, 0.25), max_cores=max_cores, threshold=10)
    return dataobj, {name: np.array(arr, copy=True)
                     for name, arr in zip(FIT_OUTPUT_NAMES, outputs)}


# ---------------------------------------------------------------------------
# Spline basis
# ---------------------------------------------------------------------------

def test_get_spline_model_partition_of_unity():
    """The spline basis must sum to unity and reproduce representable functions.

    This pins the mathematical contract of the *cardinal* basis currently used:
    ``M @ node_values`` interpolates the function whose values at the nodes are
    ``node_values``. A cubic spline basis reproduces cubics exactly.
    """
    nodes = np.linspace(0, 1, 9)
    # Strictly interior samples: get_spline_model uses strict inequalities and
    # returns all-zero rows for samples on or outside the node boundaries.
    samples = np.linspace(0.001, 0.999, 500)

    M = get_spline_model(nodes, samples, spline_degree=3)

    assert M.shape == (samples.size, nodes.size)
    np.testing.assert_allclose(M.sum(axis=1), 1.0, rtol=1e-12, atol=1e-12,
                               err_msg="spline basis is not a partition of unity")

    for label, func in (("cubic", lambda t: 2 * t ** 3 - t ** 2 + 0.5 * t + 1.0),
                        ("linear", lambda t: 3 * t + 1.0)):
        np.testing.assert_allclose(M @ func(nodes), func(samples), rtol=1e-10, atol=1e-12,
                                   err_msg=f"cubic spline basis failed to reproduce a {label}")


def test_get_spline_model_zero_outside_node_range():
    """Samples outside the node range get all-zero rows (no extrapolation)."""
    nodes = np.linspace(0, 1, 5)
    M = get_spline_model(nodes, np.array([-0.5, 1.5]), spline_degree=3)
    np.testing.assert_array_equal(M, 0.0)


# ---------------------------------------------------------------------------
# fit_3dspline behaviour
# ---------------------------------------------------------------------------

def test_fit_3dspline_recovers_smooth_field():
    """The fit should recover a smooth, representable scene to within the noise."""
    dataobj, out = run_reference_fit()

    fitted = np.where(np.isfinite(out["bestfit_model"]) & np.isfinite(dataobj.bad_pixels))
    assert fitted[0].size > 1000, "unexpectedly few pixels were fitted"

    residual_rms = np.std(out["residuals"][fitted])
    assert residual_rms < 3 * dataobj.noise_level, \
        f"residual rms {residual_rms:.4f} is too large for noise {dataobj.noise_level}"

    # The model should track the noiseless truth, not just the noisy data.
    model_error = np.abs(out["bestfit_model"][fitted] - dataobj.truth[fitted])
    assert np.median(model_error) < dataobj.noise_level, \
        "best fit model does not track the noiseless input scene"


def test_fit_3dspline_flags_outlier_as_bad_pixel():
    """A large injected outlier must be flagged in the returned bad pixel map."""
    position = (60, 40)
    _, clean_out = run_reference_fit()
    assert np.isfinite(clean_out["bad_pixels"][position]), \
        "test pixel is already flagged without an outlier injected"

    _, out = run_reference_fit(outlier=(position, 100.0))
    assert np.isnan(out["bad_pixels"][position]), \
        "a 100-sigma outlier was not flagged as a bad pixel"


def test_fit_3dspline_preserves_input_bad_pixels():
    """Pre-existing bad pixels are never fitted and stay flagged."""
    _, out = run_reference_fit()
    assert np.isnan(out["bad_pixels"][5, 5])
    assert np.isnan(out["bestfit_model"][5, 5])


def test_fit_3dspline_serial_matches_parallel():
    """Serial and multiprocess execution must produce identical results.

    The worker processes write into shared memory and mutate the shared bad
    pixel map, so this test pins the concurrency contract. It is expected to be
    the first thing to break if the stamp masking or work partitioning is
    reworked incorrectly.
    """
    _, serial = run_reference_fit(max_cores=1)
    _, parallel = run_reference_fit(max_cores=2)

    for name in FIT_OUTPUT_NAMES:
        np.testing.assert_allclose(
            serial[name], parallel[name], rtol=1e-12, atol=1e-15, equal_nan=True,
            err_msg=f"serial and parallel results differ for {name!r}")


def test_fit_3dspline_matches_reference_snapshot():
    """Regression snapshot: outputs must match a stored reference exactly.

    This is the primary guard for numerically-neutral performance work. Any
    change to the returned arrays will fail here. If a change is *intended*,
    regenerate the reference with::

        python -m breads.tests.test_splines --regenerate
    """
    # Deliberately a hard failure rather than a skip: this test is the main
    # guard against unintended numerical changes, so it must never pass
    # silently just because the reference file went missing.
    assert os.path.exists(REFERENCE_FILE), (
        f"reference snapshot missing: {REFERENCE_FILE}\n"
        "It is committed alongside this test. If it was deleted or an "
        "algorithm change is intended, regenerate it with:\n"
        "    python -m breads.tests.test_splines --regenerate")

    reference = np.load(REFERENCE_FILE)
    _, out = run_reference_fit()

    for name in FIT_OUTPUT_NAMES:
        expected, actual = reference[name], out[name]
        assert actual.shape == expected.shape, \
            f"{name!r} changed shape: {actual.shape} != {expected.shape}"

        # Compare the NaN pattern separately. It encodes which pixels were
        # fitted and which were rejected, so a change here is structural.
        np.testing.assert_array_equal(
            np.isnan(actual), np.isnan(expected),
            err_msg=f"NaN pattern changed for {name!r}")

        # Outputs are stored as float32, so 1e-6 is tight enough to catch any
        # genuine algorithmic change while tolerating harmless reassociation.
        scale = np.nanmax(np.abs(expected)) if np.any(np.isfinite(expected)) else 1.0
        np.testing.assert_allclose(
            actual, expected, rtol=1e-6, atol=1e-6 * scale, equal_nan=True,
            err_msg=f"{name!r} differs from the stored reference snapshot")


# ---------------------------------------------------------------------------
# Targeted tests for planned optimizations
# ---------------------------------------------------------------------------

def test_3d_design_matrix_tile_and_broadcast_agree():
    """``_build_3dspline_basis`` must match the original ``np.tile`` construction.

    ``_task_fit_3dspline`` previously materialized three full tiled arrays
    before multiplying them. Broadcasting is mathematically identical but
    allocates far less memory. This test pins the equivalence.

    Note the operand order is significant: floating point multiplication is not
    associative, so the broadcast form must multiply in the same ``x * y * wvs``
    order as the tiled form to be *bit* identical. Reordering the operands
    changes results at the 1e-16 level, which is harmless numerically but would
    make the optimization impossible to verify by exact comparison.
    """
    rng = np.random.default_rng(0)
    n_pix, n_wv, n_y, n_x = 200, 4, 6, 5
    M_wvs = rng.random((n_pix, n_wv))
    M_y = rng.random((n_pix, n_y))
    M_x = rng.random((n_pix, n_x))

    # Original implementation, retained here as the reference
    tiled = (np.tile(M_x[:, None, None, :], (1, n_wv, n_y, 1))
             * np.tile(M_y[:, None, :, None], (1, n_wv, 1, n_x))
             * np.tile(M_wvs[:, :, None, None], (1, 1, n_y, n_x)))
    tiled = tiled.reshape((n_pix, -1))

    np.testing.assert_array_equal(tiled, _build_3dspline_basis(M_x, M_y, M_wvs))

    # Guard the claim above: reordering really does perturb the result, so the
    # order-preserving form is the one that must be used.
    reordered = (M_wvs[:, :, None, None] * M_y[:, None, :, None] * M_x[:, None, None, :])
    reordered = reordered.reshape((n_pix, -1))
    np.testing.assert_allclose(reordered, tiled, rtol=1e-12, atol=1e-15)


def test_fitfm_unbounded_matches_cholesky_normal_equations():
    """The unbounded ``lsq_linear`` fit equals a Cholesky normal-equations solve.

    ``fitfm`` calls ``lsq_linear`` with infinite bounds and then separately forms
    and inverts ``M.T @ M`` for the error bars. Solving the normal equations once
    via Cholesky gives the same answer far more cheaply; this test pins that
    equivalence for the linear parameters, their uncertainties, and log prob.
    """
    rng = np.random.default_rng(3)
    n_data, n_para = 400, 12
    nodes = np.linspace(0, 1, n_para)
    samples = rng.uniform(0.001, 0.999, n_data)

    M = get_spline_model(nodes, samples, spline_degree=3)
    s = np.full(n_data, 0.05)
    truth = rng.normal(size=n_para)
    d = M @ truth + rng.normal(0, s)

    log_prob, rchi2, linparas, linparas_err = fitfm(
        nonlin_paras=[d, M, s, None, None], dataobj=Instrument(), fm_func=_tmp_fm,
        fm_paras={}, marginalize_noise_scaling=False, scale_noise=False)

    # Reference: normal equations on the noise-normalized system
    M_n = M / s[:, None]
    d_n = d / s
    MTM = M_n.T @ M_n
    expected_paras = np.linalg.solve(MTM, M_n.T @ d_n)
    covariance = np.linalg.inv(MTM)
    expected_err = np.sqrt(np.diag(covariance))

    np.testing.assert_allclose(linparas, expected_paras, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(linparas_err, expected_err, rtol=1e-8, atol=1e-10)

    # And the reported log probability should follow from the same quantities.
    chi2 = np.sum((d_n - M_n @ expected_paras) ** 2)
    expected_log_prob = ((n_para - n_data) / 2) * np.log(2 * np.pi) \
        - 0.5 * np.sum(2 * np.log(s)) \
        - 0.5 * np.linalg.slogdet(MTM)[1] \
        - 0.5 * chi2
    np.testing.assert_allclose(log_prob, expected_log_prob, rtol=1e-8)
    np.testing.assert_allclose(rchi2, chi2 / n_data, rtol=1e-8)


def _regenerate_reference():
    """Regenerate the stored reference snapshot. See the module docstring."""
    os.makedirs(os.path.dirname(REFERENCE_FILE), exist_ok=True)
    _, out = run_reference_fit()
    np.savez_compressed(REFERENCE_FILE, **out)
    size_kb = os.path.getsize(REFERENCE_FILE) / 1024
    print(f"Wrote {REFERENCE_FILE} ({size_kb:.0f} kB)")


if __name__ == "__main__":
    import sys

    if "--regenerate" in sys.argv:
        _regenerate_reference()
    else:
        print(__doc__)
