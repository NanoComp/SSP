"""Minimum-lengthscale (geometric) constraints for topology optimization.

These are the geometric constraints of Zhou et al. (2015) combined with the
hyperparameter-free thresholds derived by Arrieta et al. (2025). The constraints
penalize solid (or void) features whose lengthscale falls below a target value,
and are formulated purely in terms of the filtered density and the projected
density. Consequently they work with *any* order of subpixel smoothing --
`ssp1_bilinear`, `ssp2`, or even a plain `tanh_projection` -- since the
projection only enters through `rho_projected`.

The solid constraint reads

    g_s = (1/N) Σ_i I_s,i [min(rho_filtered_i - eta_e, 0)]^2 ,
    I_s = rho_projected * exp(-c |∇ rho_filtered|^2) ,

and the void constraint is the complementary expression

    g_v = (1/N) Σ_i I_v,i [min(eta_d - rho_filtered_i, 0)]^2 ,
    I_v = (1 - rho_projected) * exp(-c |∇ rho_filtered|^2) ,

where the structural functions I_s and I_v single out the "inflection regions"
of the design (the interior of a feature, where the filtered density is
stationary), and eta_e, eta_d are the eroded/dilated threshold points of the
conic filter. Following Arrieta et al., the decay rate is c = 64 R^2 and the
constraint threshold is eps = 1e-8, where R is the conic filter radius; the
constraint is well behaved for target_length / R roughly in [0.25, 1.5].

Refs:

R. Arrieta, G. Romano, and S. G. Johnson, "Hyperparameter-free minimum-lengthscale
constraints for topology optimization," arXiv.org e-Print archive, 2507.16108,
July 2025.

M. Zhou, B. S. Lazarov, F. Wang, and O. Sigmund, "Minimum length scale in topology
optimization by geometric constraints," Computer Methods in Applied Mechanics and
Engineering, vol. 293, pp. 266-282, 2015.

X. Qian and O. Sigmund, "Topological design of electromechanical actuators with
robustness toward over- and under-etching," Computer Methods in Applied Mechanics
and Engineering, vol. 253, pp. 237-251, 2013.
"""

from typing import Optional

from jax import numpy as jnp

from .utils import ArrayLikeType, gradient

# Hyperparameters derived in section 4.1 of Arrieta et al. (2025). The decay rate
# is expressed relative to the square of the conic filter radius, i.e. the actual
# decay rate is c = DEFAULT_CONSTRAINT_DECAYRATE * conic_radius**2.
DEFAULT_CONSTRAINT_THRESHOLD = 1e-8
DEFAULT_CONSTRAINT_DECAYRATE = 64.0


def solid_threshold(lengthscale_ratio: float):
    """The eroded threshold point eta_e of a conic filter.

    Ref: Eq. (9) of Arrieta et al. (2025), originally from Qian and Sigmund (2013).

    Args:
        lengthscale_ratio: the ratio of the target lengthscale to the conic
            filter radius, which must be nonnegative.

    Returns:
        The threshold point in the range [1/2, 1].
    """
    x = jnp.asarray(lengthscale_ratio)
    return jnp.where(
        x < 1,
        x**2 / 4 + 1 / 2,
        jnp.where(x < 2, -(x**2) / 4 + x, 1.0),
    )


def void_threshold(lengthscale_ratio: float):
    """The dilated threshold point eta_d of a conic filter.

    Ref: Eq. (12) of Arrieta et al. (2025), originally from Qian and Sigmund (2013).

    Args:
        lengthscale_ratio: the ratio of the target lengthscale to the conic
            filter radius, which must be nonnegative.

    Returns:
        The threshold point in the range [0, 1/2].
    """
    x = jnp.asarray(lengthscale_ratio)
    return jnp.where(
        x < 1,
        1 / 2 - x**2 / 4,
        jnp.where(x < 2, 1 + x**2 / 4 - x, 0.0),
    )


def _geometric_constraint(
    rho_filtered: ArrayLikeType,
    rho_projected: ArrayLikeType,
    resolution: float,
    target_length: float,
    conic_radius: Optional[float],
    constraint_threshold: float,
    constraint_decayrate: float,
    solid: bool,
):
    """Shared implementation of the solid and void lengthscale constraints."""
    if target_length < 0:
        raise ValueError("The target lengthscale must be nonnegative.")

    if conic_radius is None:
        conic_radius = target_length
    if conic_radius <= 0:
        raise ValueError("The conic filter radius must be positive.")

    rho_filtered = jnp.asarray(rho_filtered)
    rho_projected = jnp.asarray(rho_projected)
    if rho_filtered.ndim != 2:
        raise ValueError(
            f"Only 2D designs are supported, got {rho_filtered.ndim} dimensions."
        )
    if rho_filtered.shape != rho_projected.shape:
        raise ValueError(
            "rho_filtered and rho_projected must have the same shape, got "
            f"{rho_filtered.shape} and {rho_projected.shape}."
        )

    decayrate = constraint_decayrate * conic_radius**2

    # The gradient of the filtered density vanishes in the interior of a feature,
    # so this term restricts the constraint to those "inflection regions".
    rho_filtered_grad = gradient(rho_filtered, resolution)
    rho_filtered_grad_normsq = jnp.sum(rho_filtered_grad**2, axis=-1)
    extremal_region = jnp.exp(-decayrate * rho_filtered_grad_normsq)

    if solid:
        eta_m = solid_threshold(target_length / conic_radius)
        inflection_region = rho_projected * extremal_region
        # Only densities below the eroded threshold, i.e. features that are too
        # thin to survive an erosion by the target lengthscale, are penalized.
        beyond_threshold = jnp.minimum(rho_filtered - eta_m, 0.0)
    else:
        eta_m = void_threshold(target_length / conic_radius)
        inflection_region = (1 - rho_projected) * extremal_region
        beyond_threshold = jnp.minimum(eta_m - rho_filtered, 0.0)

    violation = jnp.mean(inflection_region * beyond_threshold**2)

    # Normalize so that the constraint is satisfied when it is nonpositive.
    return violation / constraint_threshold - 1


def constraint_solid(
    rho_filtered: ArrayLikeType,
    rho_projected: ArrayLikeType,
    resolution: float,
    target_length: float,
    conic_radius: Optional[float] = None,
    constraint_threshold: float = DEFAULT_CONSTRAINT_THRESHOLD,
    constraint_decayrate: float = DEFAULT_CONSTRAINT_DECAYRATE,
):
    """Calculate a solid minimum-lengthscale constraint function.

    This technique takes smoothed data, e.g. from filtering, `rho_filtered` and
    binary data, e.g. from projection, `rho_projected` both defined on the same
    grid, and measures whether features in the solid region, i.e. where
    `rho_projected` takes values of 1, violate the minimum `target_length`.

    The returned value is normalized by the constraint threshold, so the
    constraint is satisfied when the value is nonpositive and violated when it is
    positive. It can therefore be handed directly to a nonlinear optimizer such
    as `nlopt` as an inequality constraint. The unnormalized constraint value of
    Eq. (7) of Arrieta et al. (2025) is `(value + 1) * constraint_threshold`.

    Any projection may be used to produce `rho_projected`, including
    `ssp1_bilinear`, `ssp2`, and `tanh_projection`.

    Args:
        rho_filtered: the (2D) filtered design parameters, e.g. from
            `conic_filter`.
        rho_projected: the (2D) projected design parameters, e.g. from `ssp2`.
        resolution: resolution of the design grid.
        target_length: the minimum lengthscale to impose on the solid region.
        conic_radius: the radius of the conic filter used to obtain
            `rho_filtered`. Defaults to `target_length`, which is the
            recommended choice.
        constraint_threshold: the threshold that separates feasible from
            infeasible designs. May be tuned if feasible designs still don't
            meet the target lengthscale.
        constraint_decayrate: the decay rate of the structural function outside
            of the inflection region, relative to `conic_radius**2`.

    Returns:
        The normalized constraint value, which is nonpositive when the design
        satisfies the minimum lengthscale.

    Example:
        >>> rho_filtered = conic_filter(rho, filter_radius, lx, ly, resolution)
        >>> rho_projected = ssp2(rho_filtered, beta, eta_i, resolution)
        >>> constraint_solid(rho_filtered, rho_projected, resolution, filter_radius)
    """
    return _geometric_constraint(
        rho_filtered,
        rho_projected,
        resolution,
        target_length,
        conic_radius,
        constraint_threshold,
        constraint_decayrate,
        solid=True,
    )


def constraint_void(
    rho_filtered: ArrayLikeType,
    rho_projected: ArrayLikeType,
    resolution: float,
    target_length: float,
    conic_radius: Optional[float] = None,
    constraint_threshold: float = DEFAULT_CONSTRAINT_THRESHOLD,
    constraint_decayrate: float = DEFAULT_CONSTRAINT_DECAYRATE,
):
    """Calculate a void minimum-lengthscale constraint function.

    This technique takes smoothed data, e.g. from filtering, `rho_filtered` and
    binary data, e.g. from projection, `rho_projected` both defined on the same
    grid, and measures whether features in the void region, i.e. where
    `rho_projected` takes values of 0, violate the minimum `target_length`.

    The returned value is normalized by the constraint threshold, so the
    constraint is satisfied when the value is nonpositive and violated when it is
    positive. It can therefore be handed directly to a nonlinear optimizer such
    as `nlopt` as an inequality constraint. The unnormalized constraint value of
    Eq. (10) of Arrieta et al. (2025) is `(value + 1) * constraint_threshold`.

    Any projection may be used to produce `rho_projected`, including
    `ssp1_bilinear`, `ssp2`, and `tanh_projection`.

    Args:
        rho_filtered: the (2D) filtered design parameters, e.g. from
            `conic_filter`.
        rho_projected: the (2D) projected design parameters, e.g. from `ssp2`.
        resolution: resolution of the design grid.
        target_length: the minimum lengthscale to impose on the void region.
        conic_radius: the radius of the conic filter used to obtain
            `rho_filtered`. Defaults to `target_length`, which is the
            recommended choice.
        constraint_threshold: the threshold that separates feasible from
            infeasible designs. May be tuned if feasible designs still don't
            meet the target lengthscale.
        constraint_decayrate: the decay rate of the structural function outside
            of the inflection region, relative to `conic_radius**2`.

    Returns:
        The normalized constraint value, which is nonpositive when the design
        satisfies the minimum lengthscale.

    Example:
        >>> rho_filtered = conic_filter(rho, filter_radius, lx, ly, resolution)
        >>> rho_projected = ssp2(rho_filtered, beta, eta_i, resolution)
        >>> constraint_void(rho_filtered, rho_projected, resolution, filter_radius)
    """
    return _geometric_constraint(
        rho_filtered,
        rho_projected,
        resolution,
        target_length,
        conic_radius,
        constraint_threshold,
        constraint_decayrate,
        solid=False,
    )
