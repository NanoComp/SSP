using SSP: conic_filter, ssp2

using Random
using CairoMakie
using CairoMakie: colormap
using NLopt
using Zygote


Nx = Ny = 100
grid = (
    range(-1, 1, length=Nx),
    range(-1, 1, length=Ny),
)
# Random.seed!(42)
# design_vars = rand(Nx, Ny)
# design_vars = [sinpi(x) * sinpi(y) for (x, y) in Iterators.product(grid...)]
design_vars = let a = 0.5, b = 0.499
    # Cassini oval
    [((x^2 + y^2)^2 - 2a^2 * (x^2 - y^2) + a^4 - b^4) + 0.5 for (x, y) in Iterators.product(grid...)]
end
radius = 0.1
beta = Inf
eta = 0.5
dilations = [-0.1, 0.0, 0.1]

ssp_projections = map(dilations) do dilation
    rho_filtered = conic_filter(design_vars, radius, grid)
    rho_projected = ssp2(rho_filtered, beta, eta, grid, dilation)
    return rho_projected
end

let
    fig = Figure(size = (1200, 400))
    for (i, (dilation, rho_projected)) in enumerate(zip(dilations, ssp_projections))
        ax = Axis(fig[1,2i-1]; title = "dilation = $dilation", aspect=DataAspect())
        h = heatmap!(grid..., rho_projected; colormap=colormap("grays"))
        Colorbar(fig[1,2i], h)
    end
    save("dilation_comparison.png", fig)
end