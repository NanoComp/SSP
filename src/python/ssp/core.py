"""
These are the core algorithms and routines behind SSP.
"""

from jax import numpy as jnp

from .utils import ArrayLikeType, tanh_projection

def ssp1_bilinear(
    rho_filtered: ArrayLikeType,
    beta: float,
    eta: float,
    resolution: float,
):
    """Project using the original SSP1 algorithm with bilinear interpolation.

    This technique integrates out the discontinuity within the projection
    function, allowing the user to smoothly increase β from 0 to ∞ without
    losing the gradient. Effectively, a level set is created, and from this
    level set, SSP1 bilinear subpixel smoothing is applied to the interfaces (if
    any are present).

    In order for this to work, the input array must already be smooth (e.g. by
    filtering).

    While the original approach involves numerical quadrature, this approach
    performs a "trick" by assuming that the user is always infinitely projecting
    (β=∞). In this case, the expensive quadrature simplifies to an analytic
    fill-factor expression. When to use this fill factor requires some careful
    logic.

    For one, we want to make sure that the user can indeed project at any level
    (not just infinity). So in these cases, we simply check if in interface is
    within the pixel. If not, we revert to the standard filter plus project
    technique.

    If there is an interface, we want to make sure the derivative remains
    continuous both as the interface leaves the cell, *and* as it crosses the
    center. To ensure this, we need to account for the different possibilities.

    Ref: A. M. Hammond, A. Oskooi, I. M. Hammond, M. Chen, S. E. Ralph, and 
    S. G. Johnson, “Unifying and accelerating level-set and density-based topology 
    optimization by subpixel-smoothed projection,” Optics Express, vol. 33, 
    pp. 33620-33642, July 2025. Editor's Pick.

    Args:
        rho_filtered: The (2D) input design parameters (already filered e.g.
            with a conic filter).
        beta: The thresholding parameter in the range [0, inf]. Determines the
            degree of binarization of the output.
        eta: The threshold point in the range [0, 1].
        resolution: resolution of the design grid (not the Meep grid
            resolution).
    Returns:
        The projected and smoothed output.

    Example:
        >>> Lx = 2; Ly = 2
        >>> resolution = 50
        >>> eta_i = 0.5; eta_e = 0.75
        >>> lengthscale = 0.1
        >>> filter_radius = get_conic_radius_from_eta_e(lengthscale, eta_e)
        >>> Nx = onp.round(Lx * resolution) + 1
        >>> Ny = onp.round(Ly * resolution) + 1
        >>> rho = onp.random.rand(Nx, Ny)
        >>> beta = npa.inf
        >>> rho_filtered = conic_filter(rho, filter_radius, Lx, Ly, resolution)
        >>> rho_projected = smoothed_projection(rho_filtered, beta, eta_i, resolution)
    """
    # TODO [alechammond] The current implementation is ported from meep 
    # and only supports 2D inputs. We'll want to generalize this to 
    # arbitrary dimensions.

    # Note that currently, the underlying assumption is that the smoothing
    # kernel is a circle, which means dx = dy.
    dx = dy = 1 / resolution
    R_smoothing = 0.55 * dx

    rho_projected = tanh_projection(rho_filtered, beta=beta, eta=eta)

    # Compute the spatial gradient (using finite differences) of the *filtered*
    # field, which will always be smooth and is the key to our approach. This
    # gradient essentially represents the normal direction pointing the the
    # nearest inteface.
    rho_filtered_grad = jnp.gradient(rho_filtered)
    rho_filtered_grad_helper = (rho_filtered_grad[0] / dx) ** 2 + (
        rho_filtered_grad[1] / dy
    ) ** 2

    # Note that a uniform field (norm=0) is problematic, because it creates
    # divide by zero issues and makes backpropagation difficult, so we sanitize
    # and determine where smoothing is actually needed. The value where we don't
    # need smoothings doesn't actually matter, since all our computations our
    # purely element-wise (no spatial locality) and those pixels will instead
    # rely on the standard projection. So just use 1, since it's well behaved.
    nonzero_norm = jnp.abs(rho_filtered_grad_helper) > 0

    rho_filtered_grad_norm = jnp.sqrt(
        jnp.where(nonzero_norm, rho_filtered_grad_helper, 1)
    )
    rho_filtered_grad_norm_eff = jnp.where(nonzero_norm, rho_filtered_grad_norm, 1)

    # The distance for the center of the pixel to the nearest interface
    d = (eta - rho_filtered) / rho_filtered_grad_norm_eff

    # Only need smoothing if an interface lies within the voxel. Since d is
    # actually an "effective" d by this point, we need to ignore values that may
    # have been sanitized earlier on.
    needs_smoothing = nonzero_norm & (jnp.abs(d) < R_smoothing)

    # The fill factor is used to perform the SSP1 bilinear subpixel smoothing.
    # We use the (2D) analytic expression that comes when assuming the smoothing
    # kernel is a circle. Note that because the kernel contains some
    # expressions that are sensitive to NaNs, we have to use the "double where"
    # trick to avoid the Nans in the backward trace. This is a common problem
    # with array-based AD tracers, apparently. See here:
    # https://github.com/google/jax/issues/1052#issuecomment-5140833520
    d_R = d / R_smoothing
    F = jnp.where(
        needs_smoothing, 0.5 - 15 / 16 * d_R + 5 / 8 * d_R**3 - 3 / 16 * d_R**5, 1.0
    )
    # F(-d)
    F_minus = jnp.where(
        needs_smoothing, 0.5 + 15 / 16 * d_R - 5 / 8 * d_R**3 + 3 / 16 * d_R**5, 1.0
    )

    # Determine the upper and lower bounds of materials in the current pixel (before projection).
    rho_filtered_minus = rho_filtered - R_smoothing * rho_filtered_grad_norm_eff * F
    rho_filtered_plus = (
        rho_filtered + R_smoothing * rho_filtered_grad_norm_eff * F_minus
    )

    # Finally, we project the extents of our range.
    rho_minus_eff_projected = tanh_projection(rho_filtered_minus, beta=beta, eta=eta)
    rho_plus_eff_projected = tanh_projection(rho_filtered_plus, beta=beta, eta=eta)

    # Only apply smoothing to interfaces
    rho_projected_smoothed = (
        1 - F
    ) * rho_minus_eff_projected + F * rho_plus_eff_projected
    return jnp.where(
        needs_smoothing,
        rho_projected_smoothed,
        rho_projected,
    )
