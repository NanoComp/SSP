"""Minimum-lengthscale (geometric) constraints combined with SSP.

The constraints of `ssp_topopt.constraints` measure whether the solid or void
features of a design are thinner than a target lengthscale. They are formulated
in terms of the filtered and projected densities only, so they work with any
order of subpixel smoothing; this example uses `ssp2`.

The example has two parts:

1. A sweep over a simple stripe geometry, which shows that the solid constraint
   changes sign at the target lengthscale (compare fig. 1c of Arrieta et al.).
2. A two-stage optimization that matches a target pattern containing bars that
   are thinner than the target lengthscale. The first stage is unconstrained and
   happily reproduces the sub-lengthscale bars; the second stage turns the
   constraints on and repairs the design, at a small cost in the figure of merit.

Ref: R. Arrieta, G. Romano, and S. G. Johnson, "Hyperparameter-free
minimum-lengthscale constraints for topology optimization," arXiv.org e-Print
archive, 2507.16108, July 2025.
"""

import time

import nlopt
import numpy as np
from jax import jit, value_and_grad
from jax import numpy as jnp
from matplotlib import pyplot as plt

from ssp_topopt import conic_filter, constraint_solid, constraint_void, ssp2

BETA = np.inf
ETA_I = 0.5


def filter_and_project(rho, filter_radius, lx, ly, resolution):
    """The usual filter-then-project pipeline."""
    rho_filtered = conic_filter(rho, filter_radius, lx, ly, resolution)
    rho_projected = ssp2(rho_filtered, BETA, ETA_I, resolution)
    return rho_filtered, rho_projected


def feature_width(rho_projected, resolution, solid=True):
    """Width of the central feature of a stripe design, in physical units."""
    profile = np.asarray(rho_projected)[:, rho_projected.shape[1] // 2]
    mask = profile > 0.5 if solid else profile < 0.5
    return float(np.count_nonzero(mask)) / resolution


def constraint_sweep():
    """Evaluate the constraints for stripes of varying width."""
    lx = ly = 0.5
    resolution = 100
    target_length = 0.1
    filter_radius = target_length

    nx = int(np.round(lx * resolution)) + 1
    ny = int(np.round(ly * resolution)) + 1
    coords = np.linspace(-lx / 2, lx / 2, nx)
    x = coords[:, None] * np.ones((1, ny))

    latent_widths = np.linspace(0.04, 0.24, 21)
    widths = []
    solid_values = []
    void_values = []

    # Note that the void constraint also fires for very thin stripes: once the
    # stripe is thin enough that it barely projects to solid, the crest of the
    # filtered density is a nearly-solid feature sitting inside the void region.
    print("Constraint sweep over stripe widths " f"(target lengthscale {target_length})")
    for latent_width in latent_widths:
        # A binary stripe of width `latent_width`, which the filter and the
        # projection turn into a stripe of some (smaller) physical width.
        rho = jnp.asarray((np.abs(x) <= latent_width / 2).astype(float))
        rho_filtered, rho_projected = filter_and_project(
            rho, filter_radius, lx, ly, resolution
        )

        width = feature_width(rho_projected, resolution)
        solid_value = float(
            constraint_solid(rho_filtered, rho_projected, resolution, target_length)
        )
        void_value = float(
            constraint_void(rho_filtered, rho_projected, resolution, target_length)
        )

        widths.append(width)
        solid_values.append(solid_value)
        void_values.append(void_value)
        print(
            f"  latent width={latent_width:.3f} physical width={width:.3f} "
            f"solid={solid_value:+.3e} void={void_value:+.3e}"
        )

    # Feasible designs have an exactly vanishing violation, so clip to a floor to
    # keep them visible on a log scale.
    floor = 1e-2
    plt.figure(figsize=(5, 3.5))
    plt.axhspan(floor / 2, 1.0, color="tab:green", alpha=0.1, label="feasible")
    plt.plot(
        widths, np.maximum(np.asarray(solid_values) + 1, floor), marker="o", label="solid"
    )
    plt.plot(
        widths, np.maximum(np.asarray(void_values) + 1, floor), marker="s", label="void"
    )
    plt.axhline(1.0, color="k", linestyle=":", label="threshold")
    plt.axvline(target_length, color="r", linestyle="--", label="target lengthscale")
    plt.yscale("log")
    plt.ylim(bottom=floor / 2)
    plt.xlabel("physical width of the stripe")
    plt.ylabel("constraint / threshold")
    plt.legend(fontsize=8)
    plt.title("Constraint value vs feature width")
    plt.tight_layout()
    plt.savefig("constraint_sweep.png")


def optimization_demo():
    """Repair a design with sub-lengthscale features using the constraints."""
    lx, ly = 1.2, 0.6
    resolution = 50
    target_length = 0.12
    filter_radius = target_length

    nx = int(np.round(lx * resolution)) + 1
    ny = int(np.round(ly * resolution)) + 1
    num_vars = nx * ny
    coords = np.linspace(-lx / 2, lx / 2, nx)

    # A target pattern of vertical bars, two of which are thinner than the
    # target lengthscale and therefore cannot be manufactured.
    target = np.zeros((nx, ny))
    position = -lx / 2 + 0.08
    for width in (0.24, 0.16, 0.08, 0.04):
        target[(coords >= position) & (coords <= position + width), :] = 1.0
        position += width + 0.12
    target_jnp = jnp.asarray(target)

    def pipeline(rho_flat):
        return filter_and_project(
            rho_flat.reshape((nx, ny)), filter_radius, lx, ly, resolution
        )

    def figure_of_merit(rho_flat):
        _, rho_projected = pipeline(rho_flat)
        return jnp.mean((rho_projected - target_jnp) ** 2)

    def solid_constraint(rho_flat):
        rho_filtered, rho_projected = pipeline(rho_flat)
        return constraint_solid(
            rho_filtered, rho_projected, resolution, target_length
        )

    def void_constraint(rho_flat):
        rho_filtered, rho_projected = pipeline(rho_flat)
        return constraint_void(rho_filtered, rho_projected, resolution, target_length)

    objective_and_grad = jit(value_and_grad(figure_of_merit))
    constraints_and_grad = {
        "solid": jit(value_and_grad(solid_constraint)),
        "void": jit(value_and_grad(void_constraint)),
    }

    fom_history = []
    constraint_history = {"solid": [], "void": []}

    def run_stage(x_init, constrained, maxeval):
        def nlopt_objective(x, grad_out):
            value, gradient = objective_and_grad(jnp.asarray(x))
            if grad_out.size > 0:
                grad_out[:] = np.asarray(gradient, dtype=float)
            fom_history.append(float(value))
            for name, value_and_grad_fn in constraints_and_grad.items():
                constraint_history[name].append(
                    float(value_and_grad_fn(jnp.asarray(x))[0])
                )
            return float(value)

        opt = nlopt.opt(nlopt.LD_CCSAQ, num_vars)
        opt.set_lower_bounds(np.zeros(num_vars))
        opt.set_upper_bounds(np.ones(num_vars))
        opt.set_min_objective(nlopt_objective)

        if constrained:
            for value_and_grad_fn in constraints_and_grad.values():

                def nlopt_constraint(x, grad_out, fn=value_and_grad_fn):
                    value, gradient = fn(jnp.asarray(x))
                    if grad_out.size > 0:
                        grad_out[:] = np.asarray(gradient, dtype=float)
                    return float(value)

                # The constraints are normalized, so the feasible region is
                # exactly where they are nonpositive.
                opt.add_inequality_constraint(nlopt_constraint, 0.0)

        opt.set_maxeval(maxeval)
        start = time.perf_counter()
        x_opt = opt.optimize(np.asarray(x_init, dtype=float))
        elapsed = time.perf_counter() - start

        label = "constrained" if constrained else "unconstrained"
        print(
            f"  {label}: {maxeval} evaluations in {elapsed:6.1f}s "
            f"FOM={fom_history[-1]:.4e} "
            f"solid={constraint_history['solid'][-1]:+.3e} "
            f"void={constraint_history['void'][-1]:+.3e}"
        )
        return x_opt

    print(f"\nTwo-stage optimization (target lengthscale {target_length})")
    rng = np.random.default_rng(0)
    x_init = 0.5 * np.ones(num_vars) + 0.01 * rng.standard_normal(num_vars)

    stage1_evals = 120
    x_stage1 = run_stage(x_init, constrained=False, maxeval=stage1_evals)
    # The constrained stage needs a few hundred iterations: CCSA has to first
    # drag the design back into the feasible region and then re-minimize.
    x_stage2 = run_stage(x_stage1, constrained=True, maxeval=400)

    _, projected_stage1 = pipeline(jnp.asarray(x_stage1))
    _, projected_stage2 = pipeline(jnp.asarray(x_stage2))

    plt.figure(figsize=(9, 3))
    for index, (image, title) in enumerate(
        (
            (target, "target"),
            (np.asarray(projected_stage1), "stage 1: unconstrained"),
            (np.asarray(projected_stage2), "stage 2: lengthscale constrained"),
        )
    ):
        plt.subplot(1, 3, index + 1)
        plt.imshow(image.T, vmin=0, vmax=1, cmap="binary", origin="lower")
        plt.title(title, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("constraint_designs.png")

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.semilogy(fom_history, linewidth=1.5)
    plt.axvline(stage1_evals, color="k", linestyle="--", linewidth=1)
    plt.xlabel("iteration")
    plt.ylabel("FOM")
    plt.title("Objective history", fontsize=9)

    plt.subplot(1, 2, 2)
    for name, values in constraint_history.items():
        # Shift by one so that the (normalized) constraint can be shown on a log
        # scale: values below one are feasible.
        plt.semilogy(np.asarray(values) + 1, linewidth=1.5, label=name)
    plt.axhline(1.0, color="k", linestyle=":", label="threshold")
    plt.axvline(stage1_evals, color="k", linestyle="--", linewidth=1)
    plt.xlabel("iteration")
    plt.ylabel("constraint / threshold")
    plt.title("Constraint history", fontsize=9)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("constraint_history.png")
    plt.show()


def main():
    constraint_sweep()
    optimization_demo()


if __name__ == "__main__":
    main()
