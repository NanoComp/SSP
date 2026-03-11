import unittest

import nlopt
import numpy as np
from jax import grad, value_and_grad
from jax import numpy as jnp

from ssp import conic_filter, get_conic_radius_from_eta_e, ssp_first_order


def figure_of_merit(x: jnp.ndarray) -> float:
    """Simple convex objective used in the example."""
    return jnp.mean((jnp.abs(x) ** 2).flatten())


def full_system(x: jnp.ndarray, beta: float, eta_i: float, resolution: int) -> float:
    """Projection followed by scalar objective."""
    return figure_of_merit(ssp_first_order(x, beta, eta_i, resolution))


class TestSSPFirstOrder(unittest.TestCase):
    def setUp(self):
        self.lx = 2.0
        self.ly = 2.0
        self.resolution = 50
        self.eta_i = 0.5
        self.eta_e = 0.75
        self.lengthscale = 0.25
        self.beta = np.inf
        self.filter_radius = get_conic_radius_from_eta_e(self.lengthscale, self.eta_e)
        self.nx = int(np.round(self.lx * self.resolution) + 1)
        self.ny = int(np.round(self.ly * self.resolution) + 1)
        self.seed = 42

    def _random_design(self) -> np.ndarray:
        np.random.seed(self.seed)
        return np.random.rand(self.nx, self.ny)

    def _filter_and_project(self, rho_design: np.ndarray):
        rho_filtered = conic_filter(
            rho_design,
            self.filter_radius,
            self.lx,
            self.ly,
            self.resolution,
        )
        rho_projected = ssp_first_order(
            rho_filtered,
            self.beta,
            self.eta_i,
            self.resolution,
        )
        return rho_filtered, rho_projected

    def test_smoke_pipeline(self):
        """Smoke test: import package and run tiny filter + projection pipeline."""
        rho = self._random_design()
        _, rho_projected = self._filter_and_project(rho)

        rho_projected_np = np.asarray(rho_projected)

        self.assertEqual(rho_projected_np.shape, rho.shape)
        self.assertTrue(np.isfinite(rho_projected_np).all())
        self.assertGreaterEqual(rho_projected_np.min(), -1e-7)
        self.assertLessEqual(rho_projected_np.max(), 1.0 + 1e-7)

    def test_nonzero_gradient_at_infinite_beta(self):
        """For beta=inf, verify gradient is nonzero as in the example."""
        rho = self._random_design()
        rho_filtered, _ = self._filter_and_project(rho)

        d_fom = grad(full_system)
        df_drho = d_fom(rho_filtered, self.beta, self.eta_i, self.resolution)
        grad_norm = float(np.linalg.norm(np.asarray(df_drho).ravel(), ord=2))

        self.assertGreater(grad_norm, 0.0)

    def test_ccsa_optimization_reduces_fom_to_tolerance(self):
        """Run a mini CCSA optimization and verify FOM reaches target tolerance."""
        rho0 = self._random_design()

        def optimization_objective(rho_flat: jnp.ndarray) -> jnp.ndarray:
            rho_design = rho_flat.reshape((self.nx, self.ny))
            _, rho_projected = self._filter_and_project(rho_design)
            return figure_of_merit(rho_projected)

        objective_and_grad = value_and_grad(optimization_objective)

        def nlopt_objective(x: np.ndarray, grad_out: np.ndarray) -> float:
            fom_value, gradient = objective_and_grad(jnp.asarray(x))
            if grad_out.size > 0:
                grad_out[:] = np.asarray(gradient, dtype=float)
            return float(fom_value)

        opt = nlopt.opt(nlopt.LD_CCSAQ, self.nx * self.ny)
        opt.set_lower_bounds(np.zeros(self.nx * self.ny))
        opt.set_upper_bounds(np.ones(self.nx * self.ny))
        opt.set_min_objective(nlopt_objective)
        opt.set_maxeval(15)

        x_opt = opt.optimize(rho0.ravel())
        final_fom = float(opt.last_optimum_value())

        self.assertEqual(x_opt.size, self.nx * self.ny)
        self.assertLessEqual(final_fom, 1e-10)


if __name__ == "__main__":
    unittest.main()
