using SSP: conic_filter, ssp2, constraint_void, constraint_solid

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


function figure_of_merit(rho_projected)
    sum(abs2, rho_projected) / length(rho_projected)
end

opt = NLopt.Opt(:LD_CCSAQ, length(design_vars))
evaluation_history = Float64[]
my_objective_fn = let evaluation_history=evaluation_history, design_vars=design_vars
    function (x, grad)
        fom, adj_design = Zygote.withgradient(x) do x
            rho_filtered = conic_filter(reshape(x, size(design_vars)), radius, grid)
            rho_projected = ssp2(rho_filtered, beta, eta, grid)
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
constraint_history = Float64[]
my_constraint_fn = let constraint_history=constraint_history, design_vars=design_vars
    function (result, x, grad)
        result[1], adj_design_s = Zygote.withgradient(x) do x
            rho_filtered = conic_filter(reshape(x, size(design_vars)), radius, grid)
            rho_projected = ssp2(rho_filtered, beta, eta, grid)
            return constraint_solid(rho_filtered, rho_projected, grid, radius)
        end
        if !isempty(grad)
            copy!(view(grad, :, 1), vec(adj_design_s[1]))
        end
        result[2], adj_design_v = Zygote.withgradient(x) do x
            rho_filtered = conic_filter(reshape(x, size(design_vars)), radius, grid)
            rho_projected = ssp2(rho_filtered, beta, eta, grid)
            return constraint_void(rho_filtered, rho_projected, grid, radius)
        end
        if !isempty(grad)
            copy!(view(grad, :, 2), vec(adj_design_v[1]))
        end
        append!(constraint_history, result)
        return 
    end
end
num_constraints = 2
constraint_tols = fill(0.0, num_constraints)
NLopt.inequality_constraint!(opt, my_constraint_fn, constraint_tols)
NLopt.maxeval!(opt, 100)
fmax, xmax, ret = NLopt.optimize(opt, vec(design_vars))

let
    fig = Figure()
    ax = Axis(fig[1,1]; title = "Optimization history", yscale=log10, limits = (nothing, (1e-16, 1e1)))
    scatterlines!(ax, evaluation_history)
    ax2 = Axis(fig[1,2]; title = "Constraint history")
    scatterlines!(ax2, constraint_history[1:2:end]; label="solid")
    scatterlines!(ax2, constraint_history[2:2:end]; label="void")
    Legend(fig[1, 3], ax2)
    save("constraint_history.png", fig)


    rho_filtered = conic_filter(reshape(xmax, size(design_vars)), radius, grid)
    rho_projected = ssp2(rho_filtered, beta, eta, grid)
    fig = Figure()
    ax = Axis(fig[1,1]; title="optimized projected density")
    h = heatmap!(grid..., rho_projected; colormap=colormap("grays"), colorrange=(0,1))
    Colorbar(fig[1,2], h)
    save("constraint_solution.png", fig)

end
