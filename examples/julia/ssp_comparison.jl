using SSP: conic_filter, ssp1_linear, ssp1, ssp2

using Random
using CairoMakie
using CairoMakie: colormap
using NLopt


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
ssp_algs = (ssp1_linear, ssp1, ssp2)

ssp_projections = map(ssp_algs) do ssp
    rho_filtered = conic_filter(design_vars, radius, grid)
    rho_projected = ssp(rho_filtered, beta, eta, grid)
    return rho_projected
end

let
    fig = Figure(size = (1200, 400))
    for (i, (ssp, rho_projected)) in enumerate(zip(ssp_algs, ssp_projections))
        ax = Axis(fig[1,2i-1]; title = "$(string(nameof(ssp))) projection", aspect=DataAspect())
        h = heatmap!(grid..., rho_projected; colormap=colormap("grays"))
        Colorbar(fig[1,2i], h)
    end
    save("projection_comparison.png", fig)
end
