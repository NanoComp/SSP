using SSP: conic_filter, ssp1_linear, ssp1, ssp2

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

function figure_of_merit(rho_projected)
    sum(abs2, rho_projected) / length(rho_projected)
end

ssp_projection_gradients = map(ssp_algs) do ssp
    design_vars_gradient = Zygote.gradient(design_vars) do design_vars
        rho_filtered = conic_filter(design_vars, radius, grid)
        rho_projected = ssp(rho_filtered, beta, eta, grid)
        return figure_of_merit(rho_projected)
    end
    return design_vars_gradient[1]
end

let
    fig = Figure(size = (1200, 400))
    for (i, (ssp, rho_projected_gradient)) in enumerate(zip(ssp_algs, ssp_projection_gradients))
        ax = Axis(fig[1,2i-1]; title = "$(string(nameof(ssp))) projection gradient", aspect=DataAspect())
        h = heatmap!(grid..., rho_projected_gradient; colormap=colormap("RdBu"))
        Colorbar(fig[1,2i], h)
    end
    save("projection_gradient_comparison.png", fig)
end

ssp_optimization_histories = map(ssp_algs) do ssp
    opt = NLopt.Opt(:LD_CCSAQ, length(design_vars))
    evaluation_history = Float64[]
    my_objective_fn = let evaluation_history=evaluation_history, design_vars=design_vars
        function (x, grad)
            fom, adj_design = Zygote.withgradient(x) do x
                rho_filtered = conic_filter(reshape(x, size(design_vars)), radius, grid)
                rho_projected = ssp(rho_filtered, beta, eta, grid)
                return figure_of_merit(rho_projected)
            end
            if !isempty(grad)
                copy!(grad, vec(adj_design[1]))
            end
            push!(evaluation_history, fom)
            return fom
        end
    end
    NLopt.min_objective!(opt, my_objective_fn)
    NLopt.maxeval!(opt, 50)
    fmax, xmax, ret = NLopt.optimize(opt, vec(design_vars))
    return evaluation_history
end

let
    fig = Figure()
    ax = Axis(fig[1,1]; title = "Optimization history", yscale=log10)
    for (i, (ssp, evaluation_history)) in enumerate(zip(ssp_algs, ssp_optimization_histories))
        scatterlines!(ax, evaluation_history; label=string(nameof(ssp)))
    end
    Legend(fig[1,2], ax)
    save("evaluation_history_comparison.png", fig)
end
