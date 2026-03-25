"""
Helper functions and other auxilliary routines commonly used when implementing SSP.

Most of these routines were ported directly from meep. Since autograd is no longer 
maintained, but jax _is_ maintained, we swap out autograd for jax. 

TODO: Most of these routines are only compatible with 2D and need extra work in order 
to generalize to arbitrary dimensions.
"""

from typing import Union, List, Tuple
import numpy as np
from interpax import interp2d


# Use jax as our autograd engine
from jax import numpy as jnp
from jax import vmap,grad
import jax

ArrayLikeType = Union[List, Tuple, np.ndarray]


def _centered(arr: np.ndarray, newshape: ArrayLikeType) -> np.ndarray:
    """Formats the output of an FFT to center the zero-frequency component.

    A helper function borrowed from SciPy:
    https://github.com/scipy/scipy/blob/v1.4.1/scipy/signal/signaltools.py#L263-L270

    Args:
        arr: output array from an FFT operation.
        newshape: 1d array with two elements (integers) specifying the dimensions
            of the array to be returned.

    Returns:
        The input array with the zero-frequency component as the central element.
    """
    newshape = np.asarray(newshape)
    currshape = np.array(arr.shape)
    startind = (currshape - newshape) // 2
    endind = startind + newshape
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]

    return arr[tuple(myslice)]


def _quarter_to_full_kernel(arr: np.ndarray, pad_to: np.ndarray) -> np.ndarray:
    """Constructs the full kernel from its nonnegative quadrant.

    Args:
        arr: 2d input array representing the nonnegative quadrant of a
            filter kernel with C4v symmetry.
        pad_to: 1d array with two elements (integers) specifying the size
            of the zero padding.

    Returns:
        The complete kernel.
    """
    pad_size = pad_to - 2 * np.array(arr.shape) + 1

    top = np.zeros((pad_size[0], arr.shape[1]))
    bottom = np.zeros((pad_size[0], arr.shape[1] - 1))
    middle = np.zeros((pad_to[0], pad_size[1]))

    top_left = arr[:, :]
    top_right = jnp.flipud(arr[1:, :])
    bottom_left = jnp.fliplr(arr[:, 1:])
    bottom_right = jnp.flipud(
        jnp.fliplr(arr[1:, 1:])
    )  # equivalent to flip, but flip is incompatible with autograd

    return jnp.concatenate(
        (
            jnp.concatenate((top_left, top, top_right)),
            middle,
            jnp.concatenate((bottom_left, bottom, bottom_right)),
        ),
        axis=1,
    )


def _edge_pad(arr: np.ndarray, pad: np.ndarray) -> np.ndarray:
    """Border-pads the edges of an array.

    Used to preprocess the design weights prior to convolution with the filter.
    Border padding an image will set the value of each padded pixel equal to
    the value of the nearest pixel in the image. Used to implement feature-
    preserving convolution filters that prevent unwanted edge effects.

    Args:
        arr: 2d array whose borders contain the values to use for padding
        pad: 2x2 array of integers indicating the size
            of the borders to pad the array with.

    Returns:
        A 2d array with border padding.
    """
    # fill sides
    left = jnp.tile(arr[0, :], (pad[0][0], 1))
    right = jnp.tile(arr[-1, :], (pad[0][1], 1))
    top = jnp.tile(arr[:, 0], (pad[1][0], 1)).transpose()
    bottom = jnp.tile(arr[:, -1], (pad[1][1], 1)).transpose()

    # fill corners
    top_left = jnp.tile(arr[0, 0], (pad[0][0], pad[1][0]))
    top_right = jnp.tile(arr[-1, 0], (pad[0][1], pad[1][0]))
    bottom_left = jnp.tile(arr[0, -1], (pad[0][0], pad[1][1]))
    bottom_right = jnp.tile(arr[-1, -1], (pad[0][1], pad[1][1]))

    if pad[0][0] > 0 and pad[0][1] > 0 and pad[1][0] > 0 and pad[1][1] > 0:
        return jnp.concatenate(
            (
                jnp.concatenate((top_left, top, top_right)),
                jnp.concatenate((left, arr, right)),
                jnp.concatenate((bottom_left, bottom, bottom_right)),
            ),
            axis=1,
        )
    elif pad[0][0] == 0 and pad[0][1] == 0 and pad[1][0] > 0 and pad[1][1] > 0:
        return jnp.concatenate((top, arr, bottom), axis=1)
    elif pad[0][0] > 0 and pad[0][1] > 0 and pad[1][0] == 0 and pad[1][1] == 0:
        return jnp.concatenate((left, arr, right), axis=0)
    elif pad[0][0] == 0 and pad[0][1] == 0 and pad[1][0] == 0 and pad[1][1] == 0:
        return arr
    else:
        raise ValueError("At least one of the padding numbers is invalid.")


def convolve_design_weights_and_kernel(
    x: np.ndarray, h: np.ndarray, periodic_axes: ArrayLikeType = None
) -> np.ndarray:
    """Convolves the design weights with the kernel.

    Uses a 2d FFT to perform the convolution operation. This approach is
    typically faster than a direct calculation. It also preserves the shape
    of the input and output arrays. The design weights are border-padded
    prior to the FFT to preserve features on the edges of the design region.

    Args:
        x: 2d design weights.
        h: filter kernel. Must be same size as `x`
        periodic_axes: list of axes (x, y = 0, 1) that are to be treated as
            periodic. Default is None (all axes are non-periodic).

    Returns:
        The convolution of the design weights with the kernel as a 2d array.
    """
    (sx, sy) = x.shape

    if periodic_axes is None:
        h = _quarter_to_full_kernel(h, 3 * np.array([sx, sy]))
        x = _edge_pad(x, ((sx, sx), (sy, sy)))
    else:
        (kx, ky) = h.shape

        npx = int(
            np.ceil((2 * kx - 1) / sx)
        )  # 2*kx-1 is the size of a complete kernel in the x direction
        npy = int(
            np.ceil((2 * ky - 1) / sy)
        )  # 2*ky-1 is the size of a complete kernel in the y direction
        if npx % 2 == 0:
            npx += 1  # Ensure npx is an odd number
        if npy % 2 == 0:
            npy += 1  # Ensure npy is an odd number

        periodic_axes = np.array(periodic_axes)
        # Repeat the design pattern in periodic directions according to
        # the kernel size
        x = jnp.tile(
            x, (npx if 0 in periodic_axes else 1, npy if 1 in periodic_axes else 1)
        )

        jnpdx = 0 if 0 in periodic_axes else sx
        jnpdy = 0 if 1 in periodic_axes else sy
        x = _edge_pad(
            x, ((jnpdx, jnpdx), (jnpdy, jnpdy))
        )  # pad only in nonperiodic directions
        h = _quarter_to_full_kernel(
            h,
            np.array(
                [
                    npx * sx if 0 in periodic_axes else 3 * sx,
                    npy * sy if 1 in periodic_axes else 3 * sy,
                ]
            ),
        )

    h = h / jnp.sum(h)  # Normalize the kernel

    return _centered(
        jnp.real(jnp.fft.ifft2(jnp.fft.fft2(x) * jnp.fft.fft2(h))), (sx, sy)
    )


def _get_resolution(resolution: ArrayLikeType) -> tuple:
    """Converts input design-grid resolution to the acceptable format.

    Args:
        resolution: number of list of numbers representing design-grid
                    resolution, allowing anisotropic resolution.

    Returns:
        A two-element tuple composed of the resolution in x and y directions.
    """
    if isinstance(resolution, (tuple, list, np.ndarray)):
        if len(resolution) == 2:
            return resolution
        elif len(resolution) == 1:
            return resolution[0], resolution[0]
        else:
            raise ValueError(
                "The dimension of the design-grid resolution is incorrect."
            )
    elif isinstance(resolution, (int, float)):
        return resolution, resolution
    else:
        raise ValueError("The input for design-grid resolution is invalid.")


def mesh_grid(
    radius: float,
    Lx: float,
    Ly: float,
    resolution: ArrayLikeType,
    periodic_axes: ArrayLikeType = None,
) -> tuple:
    """Obtains the numbers of grid points and the coordinates of the grid
    of the design region.

    Args:
        radius: filter radius (in Meep units).
        Lx: length of design region in X direction (in Meep units).
        Ly: length of design region in Y direction (in Meep units).
        resolution: resolution of the design grid (not the Meep grid
            resolution).
        periodic_axes: list of axes (x, y = 0, 1) that are to be treated as
            periodic. Default is None (all axes are non-periodic).

    Returns:
        A four-element tuple composed of the numbers of grid points and
        the coordinates of the grid.
    """
    resolution = _get_resolution(resolution)
    Nx = int(round(Lx * resolution[0])) + 1
    Ny = int(round(Ly * resolution[1])) + 1

    if Nx <= 1 and Ny <= 1:
        raise AssertionError(
            "The grid size is improper. Check the size and resolution of the design region."
        )

    xv = np.arange(0, Lx / 2, 1 / resolution[0]) if resolution[0] > 0 else [0]
    yv = np.arange(0, Ly / 2, 1 / resolution[1]) if resolution[1] > 0 else [0]

    # If the design weights are periodic in a direction,
    # the size of the kernel in that direction needs to be adjusted
    # according to the filter radius.
    if periodic_axes is not None:
        periodic_axes = np.array(periodic_axes)
        if 0 in periodic_axes:
            xv = (
                jnp.arange(0, jnp.ceil(2 * radius / Lx) * Lx / 2, 1 / resolution[0])
                if resolution[0] > 0
                else [0]
            )
        if 1 in periodic_axes:
            yv = (
                jnp.arange(0, jnp.ceil(2 * radius / Ly) * Ly / 2, 1 / resolution[1])
                if resolution[1] > 0
                else [0]
            )

    X, Y = np.meshgrid(xv, yv, sparse=True, indexing="ij")
    return Nx, Ny, X, Y

def conic_filter(
    x: np.ndarray,
    radius: float,
    Lx: float,
    Ly: float,
    resolution: ArrayLikeType,
    periodic_axes: ArrayLikeType = None,
) -> np.ndarray:
    """A linear conic (or "hat") filter.

    Ref: B.S. Lazarov, F. Wang, & O. Sigmund, Length scale and
    manufacturability in density-based topology optimization.
    Archive of Applied Mechanics, 86(1-2), pp. 189-218 (2016).

    Args:
        x: 2d design weights.
        radius: filter radius (in Meep units).
        Lx: length of design region in X direction (in Meep units).
        Ly: length of design region in Y direction (in Meep units).
        resolution: resolution of the design grid (not the Meep grid
            resolution).
        periodic_axes: list of axes (x, y = 0, 1) that are to be treated as
            periodic. Default is None (all axes are non-periodic).

    Returns:
        The filtered design weights.
    """
    Nx, Ny, X, Y = mesh_grid(radius, Lx, Ly, resolution, periodic_axes)
    x = x.reshape(Nx, Ny)  # Ensure the input is 2d
    h = jnp.where(
        X**2 + Y**2 < radius**2, (1 - np.sqrt(abs(X**2 + Y**2)) / radius), 0
    )
    return convolve_design_weights_and_kernel(x, h, periodic_axes)


def gradient(x:             np.ndarray,
             resolution:    float,
             method:        str = 'cubic2'
             ):

      resolution = _get_resolution(resolution)
      dx         = 1/resolution[0]
      dy         = 1/resolution[1]
      Nx,Ny      = x.shape
      Lx         = dx*(Nx-1)
      Ly         = dy*(Ny-1)
      x_coords   = jnp.linspace(-Lx/2, Lx/2, Nx, endpoint = True)
      y_coords   = jnp.linspace(-Ly/2, Ly/2, Ny, endpoint = True)
   
      interpolator = lambda p: interp2d(*p,x_coords,y_coords, x,method=method)

      Xf, Yf = jnp.meshgrid(x_coords, y_coords, indexing="ij")
      points = jnp.stack((Xf.ravel(),Yf.ravel()), axis=-1)

      return vmap(grad(interpolator))(points).reshape((Nx,Ny,2))


def hessian(x:             np.ndarray,
             resolution:    float,
             method:        str = 'cubic2'
             ):

      resolution = _get_resolution(resolution)
      dx         = 1/resolution[0]
      dy         = 1/resolution[1]
      Nx,Ny      = x.shape
      Lx         = dx*(Nx-1)
      Ly         = dy*(Ny-1)
      x_coords   = jnp.linspace(-Lx/2, Lx/2, Nx, endpoint = True)
      y_coords   = jnp.linspace(-Ly/2, Ly/2, Ny, endpoint = True)
   
      interpolator = lambda p: interp2d(*p,x_coords,y_coords, x,method=method)

      Xf, Yf = jnp.meshgrid(x_coords, y_coords, indexing="ij")
      points = jnp.stack((Xf.ravel(),Yf.ravel()), axis=-1)

      return vmap(jax.hessian(interpolator))(points).reshape((Nx,Ny,2,2))





def tanh_projection(x: np.ndarray, beta: float, eta: float) -> np.ndarray:
    """Sigmoid projection filter.

    Ref: F. Wang, B. S. Lazarov, & O. Sigmund, On projection methods,
    convergence and robust formulations in topology optimization.
    Structural and Multidisciplinary Optimization, 43(6), pp. 767-784 (2011).

    Args:
        x: 2d design weights to be filtered.
        beta: thresholding parameter in the range [0, inf]. Determines the
            degree of binarization of the output.
        eta: threshold point in the range [0, 1].

    Returns:
        The filtered design weights.
    """

    if beta == 0:
        # No projection
        return x

    if beta == jnp.inf:
        # Note that backpropagating through here can produce NaNs. So we
        # manually specify the step function to keep the gradient clean.
        return jnp.where(x > eta, 1.0, 0.0)
    else:
        return (jnp.tanh(beta * eta) + jnp.tanh(beta * (x - eta))) / (
            jnp.tanh(beta * eta) + jnp.tanh(beta * (1 - eta))
        )

def get_conic_radius_from_eta_e(b, eta_e):
    """Calculates the corresponding filter radius given the minimum length scale (b)
    and the desired eroded threshold point (eta_e).

    Args:
        b : float
            Desired minimum length scale.
        eta_e : float
            Eroded threshold point (1-eta)

    Returns:
        The conic filter radius.

    References
        [1] Qian, X., & Sigmund, O. (2013). Topological design of electromechanical actuators with
        robustness toward over-and under-etching. Computer Methods in Applied
        Mechanics and Engineering, 253, 237-251.
        [2] Wang, F., Lazarov, B. S., & Sigmund, O. (2011). On projection methods, convergence and
        robust formulations in topology optimization. Structural and Multidisciplinary
        Optimization, 43(6), 767-784.
        [3] Lazarov, B. S., Wang, F., & Sigmund, O. (2016). Length scale and manufacturability in
        density-based topology optimization. Archive of Applied Mechanics, 86(1-2), 189-218.
    """
    if (eta_e >= 0.5) and (eta_e < 0.75):
        return b / (2 * np.sqrt(eta_e - 0.5))
    elif (eta_e >= 0.75) and (eta_e <= 1):
        return b / (2 - 2 * np.sqrt(1 - eta_e))
    else:
        raise ValueError(
            "The erosion threshold point (eta_e) must be between 0.5 and 1."
        )