"""
This is a simple example that demonstrates how to use the original "SSP1" algorithm
for subpixel-smoothed projection, combined with bilinear interpolation and conic smoothing.
We set up a simplistic gradient-based optimization problem that attempts
to drive the mean of the output to zero. We set β=∞ and show the gradient is nonzero.
"""

import time

import nlopt
import numpy as np
from jax import grad, jit, value_and_grad
from jax import numpy as jnp
from matplotlib import pyplot as plt

from ssp import conic_filter, get_conic_radius_from_eta_e, ssp_first_order


def figure_of_merit(x: jnp.ndarray) -> float:
    """A simple, convex reduction mean as the FOM.

    FOM = 1/(n*m)ΣΣ|x|^2
    """
    return jnp.mean((jnp.abs(x) ** 2).flatten())


def full_system(x: jnp.ndarray, beta, eta_i, resolution) -> float:
    """Include the projection and FOM"""
    return figure_of_merit(ssp_first_order(x, beta, eta_i, resolution))


def main():
    """Run the example and optimization."""
    
    # --------------------------------------------- #
    # Visualize the SSP transformation
    # --------------------------------------------- #

    # First set up a random initial condition. We'll use dimensionless units for everything.
    lx = 2.0
    ly = 2.0
    resolution = 50
    eta_i = 0.5
    eta_e = 0.75
    lengthscale = 0.25
    filter_radius = get_conic_radius_from_eta_e(lengthscale, eta_e)
    nx = int(np.round(lx * resolution) + 1)
    ny = int(np.round(ly * resolution) + 1)

    np.random.seed(42)
    rho = np.random.rand(nx, ny)
    beta = np.inf

    rho_filtered = conic_filter(rho, filter_radius, lx, ly, resolution)
    rho_projected = ssp_first_order(rho_filtered, beta, eta_i, resolution)

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(rho_filtered, vmin=0, vmax=1, cmap="binary")
    plt.title("Input")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(rho_projected, vmin=0, vmax=1, cmap="binary")
    plt.colorbar()
    plt.title("SSP Output")
    plt.tight_layout()
    plt.savefig("projection.png")

    # --------------------------------------------- #
    # Visualize the SSP gradient for β=∞
    # --------------------------------------------- #

    d_fom = grad(full_system)
    df_drho = d_fom(rho_filtered, beta, eta_i, resolution)
    max_val = jnp.max(df_drho)
    min_val = jnp.min(df_drho)
    vmax_vmin = max([abs(max_val), abs(min_val)])

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(rho_projected, vmin=0, vmax=1, cmap="binary")
    plt.title("SSP Output")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(df_drho, vmin=-vmax_vmin, vmax=vmax_vmin, cmap="RdBu")
    plt.colorbar()
    plt.title("Gradient")
    plt.tight_layout()
    plt.savefig("gradient.png")


    # --------------------------------------------- #
    # Shape optimization via nlopt
    # --------------------------------------------- #

    def optimization_objective(rho_flat: jnp.ndarray) -> jnp.ndarray:
        rho_design = rho_flat.reshape((nx, ny))
        rho_design_filtered = conic_filter(rho_design, filter_radius, lx, ly, resolution)
        rho_design_projected = ssp_first_order(
            rho_design_filtered, beta, eta_i, resolution
        )
        return figure_of_merit(rho_design_projected)

    objective_and_grad = jit(value_and_grad(optimization_objective))

    iteration_history = []
    time_history = []
    fom_history = []
    start_time = time.perf_counter()

    def nlopt_objective(x: np.ndarray, grad_out: np.ndarray) -> float:
        fom_value, gradient = objective_and_grad(jnp.asarray(x))

        if grad_out.size > 0:
            grad_out[:] = np.asarray(gradient, dtype=float)

        iteration = len(fom_history) + 1
        elapsed = time.perf_counter() - start_time
        fom_scalar = float(fom_value)

        iteration_history.append(iteration)
        time_history.append(elapsed)
        fom_history.append(fom_scalar)

        print(f"iter={iteration:03d} time={elapsed:8.3f}s FOM={fom_scalar:.6e}")
        return fom_scalar

    opt = nlopt.opt(nlopt.LD_CCSAQ, nx * ny)
    opt.set_lower_bounds(np.zeros(nx * ny))
    opt.set_upper_bounds(np.ones(nx * ny))
    opt.set_min_objective(nlopt_objective)
    opt.set_maxeval(25)

    x_opt = opt.optimize(rho.ravel())
    final_fom = opt.last_optimum_value()

    rho_opt = x_opt.reshape((nx, ny))
    rho_opt_filtered = conic_filter(rho_opt, filter_radius, lx, ly, resolution)
    rho_opt_projected = np.asarray(
        ssp_first_order(rho_opt_filtered, beta, eta_i, resolution)
    )

    print("\nOptimization complete")
    print(f"status={opt.last_optimize_result()} final_fom={final_fom:.6e}")

    print("\nIteration log (time in seconds):")
    for iteration, elapsed, fom in zip(iteration_history, time_history, fom_history):
        print(f"iter={iteration:03d} time={elapsed:8.3f}s FOM={fom:.6e}")

    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.semilogy(iteration_history, fom_history, marker="o", linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("FOM")
    plt.title("FOM vs Iteration")

    plt.subplot(1, 2, 2)
    im = plt.imshow(rho_opt_projected, vmin=0, vmax=1, cmap="binary")
    plt.title("Final SSP Projected Design")
    plt.colorbar(im)

    plt.tight_layout()
    plt.savefig("optimization.png")


if __name__ == "__main__":
    main()
