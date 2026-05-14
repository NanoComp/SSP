# SSP

A Smoothed Subpixel Projection (SSP) package for topology optimization in Julia.
Supports N-dimensional data and reverse-mode automatic differentiation with minimal allocations.

## Usage

This package provides a high-level API nearly identical to its python cousin.
It provides a `conic_filter` routine for smoothing design variables as well as
three projections: `ssp1_linear`, `ssp1`, and `ssp2`.
We also provide length scale constraint functions `constraint_solid` and `constraint_void`.

### Example

```julia
using SSP: conic_filter, ssp2

Nx = Ny = 100
grid = (
    range(-1, 1, length=Nx),
    range(-1, 1, length=Ny),
)

design_vars = let a = 0.5, b = 0.499
    # Cassini oval
    [((x^2 + y^2)^2 - 2a^2 * (x^2 - y^2) + a^4 - b^4) + 0.5 for (x, y) in Iterators.product(grid...)]
end
radius = 0.1
beta = Inf
eta = 0.5

mapping = let radius=radius, grid=grid, beta=beta, eta=eta
    function (design_vars)
        rho_filtered = conic_filter(design_vars, radius, grid)
        rho_projected = ssp2(rho_filtered, beta, eta, grid)
        return rho_projected
    end
end

mapping(design_vars)
```

This API also supports reverse-mode automatic differentaion through packages
that rely on ChainRulesCore.jl, such as Zygote.jl. Therefore, calculating
gradients is straightforward:

```julia
using Zygote

fom = let mapping=mapping
    function (x)
        rho_projected = mapping(x)
        return sum(abs2, rho_projected) / length(rho_projected)
    end
end
Zygote.withgradient(fom, design_vars)
```

For usage of the constraint functions, see `SSP/examples/julia/ssp_constrained.jl`.

### Low-level API

We provide a SciML-like API that allows reduced allocations and finer control
over algorithm details such as boundary conditions and padding for interpolation
and the selection of projection points. See `SSP/examples/julia/ssp2_example.jl`
for how to use this API.
