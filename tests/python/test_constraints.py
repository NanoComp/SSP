"""Tests for the minimum-lengthscale (geometric) constraints.

The reference behavior is the one described in R. Arrieta, G. Romano, and
S. G. Johnson, "Hyperparameter-free minimum-lengthscale constraints for topology
optimization," arXiv:2507.16108 (2025): with the derived hyperparameters, the
constraint is violated (positive) when the physical lengthscale of a feature
falls below the target lengthscale and satisfied (nonpositive) otherwise.
"""

import unittest

import numpy as np
from jax import grad, jit, value_and_grad
from jax import numpy as jnp
from jax.experimental import enable_x64

from ssp_topopt import (
    conic_filter,
    constraint_solid,
    constraint_void,
    ssp1_bilinear,
    ssp2,
    tanh_projection,
)
from ssp_topopt.constraints import solid_threshold, void_threshold


class TestThresholdFunctions(unittest.TestCase):
    """The conic-filter threshold points of Qian and Sigmund (2013)."""

    def test_known_values(self):
        # Eqs. (9) and (12) of Arrieta et al. (2025).
        self.assertAlmostEqual(float(solid_threshold(0.0)), 0.5)
        self.assertAlmostEqual(float(solid_threshold(1.0)), 0.75)
        self.assertAlmostEqual(float(solid_threshold(2.0)), 1.0)
        self.assertAlmostEqual(float(solid_threshold(3.0)), 1.0)

        self.assertAlmostEqual(float(void_threshold(0.0)), 0.5)
        self.assertAlmostEqual(float(void_threshold(1.0)), 0.25)
        self.assertAlmostEqual(float(void_threshold(2.0)), 0.0)
        self.assertAlmostEqual(float(void_threshold(3.0)), 0.0)

    def test_thresholds_are_complementary_and_monotonic(self):
        ratios = np.linspace(0.0, 3.0, 61)
        eta_e = np.asarray(solid_threshold(ratios))
        eta_d = np.asarray(void_threshold(ratios))

        np.testing.assert_allclose(eta_e + eta_d, 1.0, atol=1e-6)
        self.assertTrue(np.all(np.diff(eta_e) >= -1e-7))
        self.assertTrue(np.all(np.diff(eta_d) <= 1e-7))
        self.assertTrue(np.all((eta_e >= 0.5 - 1e-7) & (eta_e <= 1.0 + 1e-7)))
        self.assertTrue(np.all((eta_d >= -1e-7) & (eta_d <= 0.5 + 1e-7)))


class LengthscaleFixture(unittest.TestCase):
    """A stripe geometry whose physical lengthscale can be measured directly."""

    def setUp(self):
        self.lx = 0.5
        self.ly = 0.5
        self.resolution = 100
        self.target_length = 0.1
        self.filter_radius = self.target_length
        self.beta = np.inf
        self.eta_i = 0.5
        self.nx = int(np.round(self.lx * self.resolution)) + 1
        self.ny = int(np.round(self.ly * self.resolution)) + 1

        coords = np.linspace(-self.lx / 2, self.lx / 2, self.nx)
        self.x = coords[:, None] * np.ones((1, self.ny))

    def _filter_and_project(self, rho, projection=ssp2):
        rho_filtered = conic_filter(
            jnp.asarray(rho), self.filter_radius, self.lx, self.ly, self.resolution
        )
        rho_projected = projection(
            rho_filtered, self.beta, self.eta_i, self.resolution
        )
        return rho_filtered, rho_projected

    def _feature_width(self, rho_projected, solid=True):
        """Width of the central feature of the projected design, in physical units."""
        profile = np.asarray(rho_projected)[:, self.ny // 2]
        mask = profile > 0.5 if solid else profile < 0.5
        return float(np.count_nonzero(mask)) / self.resolution


class TestSolidConstraint(LengthscaleFixture):
    def test_detects_thin_solid_features(self):
        """A stripe thinner than the target violates the solid constraint."""
        for latent_width in (0.06, 0.08):
            with self.subTest(latent_width=latent_width):
                rho = (np.abs(self.x) <= latent_width / 2).astype(float)
                rho_filtered, rho_projected = self._filter_and_project(rho)

                width = self._feature_width(rho_projected)
                self.assertGreater(width, 0.0)
                self.assertLess(width, self.target_length)

                value = float(
                    constraint_solid(
                        rho_filtered,
                        rho_projected,
                        self.resolution,
                        self.target_length,
                    )
                )
                self.assertGreater(value, 0.0)

    def test_accepts_thick_solid_features(self):
        """A stripe at or above the target satisfies the solid constraint."""
        for latent_width in (0.10, 0.14, 0.20):
            with self.subTest(latent_width=latent_width):
                rho = (np.abs(self.x) <= latent_width / 2).astype(float)
                rho_filtered, rho_projected = self._filter_and_project(rho)

                width = self._feature_width(rho_projected)
                self.assertGreaterEqual(width, self.target_length)

                value = float(
                    constraint_solid(
                        rho_filtered,
                        rho_projected,
                        self.resolution,
                        self.target_length,
                    )
                )
                self.assertLessEqual(value, 0.0)

    def test_uniform_solid_design_is_feasible(self):
        """A fully solid design has no interfaces and no lengthscale violation."""
        rho_filtered = jnp.ones((self.nx, self.ny))
        rho_projected = jnp.ones((self.nx, self.ny))

        value = float(
            constraint_solid(
                rho_filtered, rho_projected, self.resolution, self.target_length
            )
        )
        self.assertAlmostEqual(value, -1.0, places=6)


class TestVoidConstraint(LengthscaleFixture):
    def test_detects_thin_void_features(self):
        """A gap thinner than the target violates the void constraint."""
        for latent_width in (0.06, 0.08):
            with self.subTest(latent_width=latent_width):
                rho = (np.abs(self.x) > latent_width / 2).astype(float)
                rho_filtered, rho_projected = self._filter_and_project(rho)

                width = self._feature_width(rho_projected, solid=False)
                self.assertGreater(width, 0.0)
                self.assertLess(width, self.target_length)

                value = float(
                    constraint_void(
                        rho_filtered,
                        rho_projected,
                        self.resolution,
                        self.target_length,
                    )
                )
                self.assertGreater(value, 0.0)

    def test_accepts_thick_void_features(self):
        """A gap at or above the target satisfies the void constraint."""
        for latent_width in (0.10, 0.14, 0.20):
            with self.subTest(latent_width=latent_width):
                rho = (np.abs(self.x) > latent_width / 2).astype(float)
                rho_filtered, rho_projected = self._filter_and_project(rho)

                width = self._feature_width(rho_projected, solid=False)
                self.assertGreaterEqual(width, self.target_length)

                value = float(
                    constraint_void(
                        rho_filtered,
                        rho_projected,
                        self.resolution,
                        self.target_length,
                    )
                )
                self.assertLessEqual(value, 0.0)

    def test_uniform_void_design_is_feasible(self):
        rho_filtered = jnp.zeros((self.nx, self.ny))
        rho_projected = jnp.zeros((self.nx, self.ny))

        value = float(
            constraint_void(
                rho_filtered, rho_projected, self.resolution, self.target_length
            )
        )
        self.assertAlmostEqual(value, -1.0, places=6)


class TestAnyProjectionOrder(LengthscaleFixture):
    """The constraints only see `rho_projected`, so any SSP order works."""

    def _design(self):
        rng = np.random.default_rng(42)
        return rng.random((self.nx, self.ny))

    def test_gradient_flows_through_each_projection(self):
        projections = {
            "tanh_projection": tanh_projection,
            "ssp1_bilinear": ssp1_bilinear,
            "ssp2": ssp2,
        }
        rho = self._design()

        for name, projection in projections.items():
            for constraint in (constraint_solid, constraint_void):
                with self.subTest(projection=name, constraint=constraint.__name__):

                    def objective(rho_flat, projection=projection, constraint=constraint):
                        rho_design = rho_flat.reshape((self.nx, self.ny))
                        rho_filtered = conic_filter(
                            rho_design,
                            self.filter_radius,
                            self.lx,
                            self.ly,
                            self.resolution,
                        )
                        if projection is tanh_projection:
                            # A finite beta keeps the plain tanh projection smooth.
                            rho_projected = projection(rho_filtered, 8.0, self.eta_i)
                        else:
                            rho_projected = projection(
                                rho_filtered, self.beta, self.eta_i, self.resolution
                            )
                        return constraint(
                            rho_filtered,
                            rho_projected,
                            self.resolution,
                            self.target_length,
                        )

                    value, gradient = value_and_grad(objective)(
                        jnp.asarray(rho.ravel())
                    )

                    self.assertTrue(np.isfinite(float(value)))
                    gradient = np.asarray(gradient)
                    self.assertTrue(np.isfinite(gradient).all())
                    self.assertGreater(np.linalg.norm(gradient), 0.0)

    def test_jit_matches_eager(self):
        rho = self._design()
        rho_filtered, rho_projected = self._filter_and_project(rho)

        def constraints(rho_filtered, rho_projected):
            return (
                constraint_solid(
                    rho_filtered, rho_projected, self.resolution, self.target_length
                ),
                constraint_void(
                    rho_filtered, rho_projected, self.resolution, self.target_length
                ),
            )

        eager = [float(v) for v in constraints(rho_filtered, rho_projected)]
        compiled = [
            float(v) for v in jit(constraints)(rho_filtered, rho_projected)
        ]

        np.testing.assert_allclose(compiled, eager, rtol=1e-5)


class TestGradientsAgainstFiniteDifferences(unittest.TestCase):
    """Compare reverse-mode gradients with central finite differences.

    Double precision is required here: the normalized constraint is O(1/epsilon),
    so differencing it in single precision is dominated by roundoff.
    """

    def test_adjoints_match_finite_differences(self):
        with enable_x64():
            lx = ly = 0.4
            resolution = 50
            target_length = 0.15
            nx = int(np.round(lx * resolution)) + 1
            ny = int(np.round(ly * resolution)) + 1

            rng = np.random.default_rng(0)
            rho = rng.random((nx, ny))
            rho_filtered = conic_filter(
                jnp.asarray(rho), target_length, lx, ly, resolution
            )
            rho_projected = ssp2(rho_filtered, np.inf, 0.5, resolution)
            perturbation = rng.standard_normal((nx, ny))
            step = 1e-6

            for constraint in (constraint_solid, constraint_void):
                for argument in ("rho_filtered", "rho_projected"):
                    with self.subTest(
                        constraint=constraint.__name__, argument=argument
                    ):

                        def scalar(value, argument=argument, constraint=constraint):
                            if argument == "rho_filtered":
                                return constraint(
                                    value, rho_projected, resolution, target_length
                                )
                            return constraint(
                                rho_filtered, value, resolution, target_length
                            )

                        base = (
                            rho_filtered
                            if argument == "rho_filtered"
                            else rho_projected
                        )
                        adjoint = float(
                            np.sum(np.asarray(grad(scalar)(base)) * perturbation)
                        )
                        finite_difference = (
                            float(scalar(base + step * perturbation))
                            - float(scalar(base - step * perturbation))
                        ) / (2 * step)

                        self.assertAlmostEqual(
                            adjoint / finite_difference, 1.0, places=5
                        )


class TestArgumentValidation(unittest.TestCase):
    def setUp(self):
        self.rho_filtered = jnp.full((8, 8), 0.5)
        self.rho_projected = jnp.zeros((8, 8))

    def test_negative_target_length(self):
        with self.assertRaises(ValueError):
            constraint_solid(self.rho_filtered, self.rho_projected, 10, -0.1)

    def test_nonpositive_conic_radius(self):
        with self.assertRaises(ValueError):
            constraint_void(
                self.rho_filtered, self.rho_projected, 10, 0.1, conic_radius=0.0
            )

    def test_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            constraint_solid(self.rho_filtered, jnp.zeros((8, 4)), 10, 0.1)

    def test_non_2d_input(self):
        with self.assertRaises(ValueError):
            constraint_solid(jnp.full((4, 4, 4), 0.5), jnp.zeros((4, 4, 4)), 10, 0.1)


if __name__ == "__main__":
    unittest.main()
