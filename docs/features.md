# Supported Features

The Julia package lives in [`src/julia/SSP`](../src/julia/SSP) and the Python package in
[`src/python/ssp_topopt`](../src/python/ssp_topopt). The table below tracks what each one
currently implements; **please keep it up to date when adding or removing functionality.**

| Feature | Julia (`SSP`) | Python (`ssp_topopt`) |
| --- | --- | --- |
| Conic ("hat") filter | ✅ `conic_filter` | ✅ `conic_filter` |
| Filter radius from an eroded threshold point | ❌ | ✅ `get_conic_radius_from_eta_e` |
| Plain tanh projection | ❌ (internal only) | ✅ `tanh_projection` |
| First-order subpixel smoothing (SSP1), linear interpolation | ✅ `ssp1_linear` | ✅ `ssp1_bilinear` |
| First-order subpixel smoothing (SSP1), cubic interpolation | ✅ `ssp1` | ❌ |
| Second-order subpixel smoothing (SSP2), differentiable through topology changes | ✅ `ssp2` | ✅ `ssp2` |
| Finite and infinite projection strength (0 ≤ β ≤ ∞) | ✅ | ✅ |
| Dilation/erosion of the projected contour | ✅ `dilation_distance` argument | ❌ |
| Minimum-lengthscale constraints for solid and void | ✅ `constraint_solid`, `constraint_void` | ❌ |
| Lengthscale constraints compatible with any SSP order | ✅ (constraints act on `rho_filtered`/`rho_projected`) | ❌ |
| Reverse-mode automatic differentiation | ✅ hand-written adjoints, exposed to Zygote.jl and friends through a ChainRulesCore.jl extension | ✅ through JAX (`grad`, `jit`, `vmap`) |
| Dimensionality | N-dimensional code paths (only 2D is currently tested) | 2D only |
| Periodic filter axes | ❌ | ✅ `periodic_axes` argument of `conic_filter` |
| Low-level `init`/`solve!`/`adjoint_solve!` API with reduced allocations | ✅ | ❌ |
| Explicit control over padding/boundary conditions, kernels, interpolation, and projection target points | ✅ (low-level API) | ❌ |

## Known limitations

* The subpixel fill factor is the analytic expression for a *circular* smoothing kernel, so
  both implementations assume an isotropic grid (`dx == dy`). Julia asserts that all grid
  steps are equal; Python takes a single scalar `resolution` in the projection routines
  (`conic_filter` does accept an anisotropic `resolution`).
* Only 2D usage is covered by the tests and examples in this repository, even though the
  Julia routines are written generically over the number of dimensions.
* The high-level Julia `conic_filter` always pads by replicating the boundary values.
  Other padding styles (`FillPadding`, `Inner`) are only reachable through the low-level API.

## Not yet supported

Contributions welcome — these are known gaps rather than fundamental limitations:

* Python: cubic-interpolation SSP1, dilation/erosion, and minimum-lengthscale constraints.
* Python: a low-level API with reusable workspaces.
* Julia: `get_conic_radius_from_eta_e`-style helpers and periodic filter axes.
* Both: validated 3D usage and anisotropic grid spacings in the projection.
