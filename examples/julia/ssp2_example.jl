using SSP: init, solve!, adjoint_solve!
using SSP: Kernel, Pad, Convolve, Project
using .Kernel: conickernel
using .Pad: FillPadding, BoundaryPadding, Inner, PaddingProblem, DefaultPaddingAlgorithm
using .Convolve: DiscreteConvolutionProblem, FFTConvolution
using .Project: ProjectionProblem, SSP1_linear, SSP1, SSP2

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

kernel = conickernel(grid, radius)

padprob = PaddingProblem(;
    data = design_vars,
    boundary = BoundaryPadding(size(kernel) .- 1, size(kernel) .- 1),
    # boundary = FillPadding(1.0, size(kernel) .- 1, size(kernel) .- 1),
)
padalg = DefaultPaddingAlgorithm()
padsolver = init(padprob, padalg)
padsol = solve!(padsolver)

convprob = DiscreteConvolutionProblem(;
    data = padsol.value,
    kernel,
)

convalg = FFTConvolution()
convsolver = init(convprob, convalg)
convsol = solve!(convsolver)

depadprob = PaddingProblem(;
    data = convsol.value,
    boundary = Inner(size(kernel) .- 1, size(kernel) .- 1),
)
depadalg = DefaultPaddingAlgorithm()
depadsolver = init(depadprob, depadalg)
depadsol = solve!(depadsolver)

filtered_design_vars = depadsol.value

# projection points need not be the same as design variable grid
target_grid = (
    range(-1, 1, length=Nx * 1),
    range(-1, 1, length=Ny * 1),
)
target_points = vec(collect(Iterators.product(target_grid...)))
projprob = ProjectionProblem(;
    rho_filtered=filtered_design_vars,
    grid,
    target_points,
    beta = Inf,
    eta = 0.5,
)
# projalg = SSP1_linear()
# projalg = SSP1()
projalg = SSP2()
projsolver = init(projprob, projalg)
projsol = solve!(projsolver)

projected_design_vars = projsol.value

let
    fig = Figure()
    ax1 = Axis(fig[1,1]; title = "design variables", aspect=DataAspect())
    h1 = heatmap!(grid..., design_vars; colormap=colormap("grays"))
    Colorbar(fig[1,2], h1)

    ax2 = Axis(fig[1,3]; title = "SSP2 output", aspect=DataAspect())
    h2 = heatmap!(target_grid..., reshape(projected_design_vars, length.(target_grid)); colormap=colormap("grays"))
    Colorbar(fig[1,4], h2)
    save("design.png", fig)
end

function fom(data, grid)
    return sum(abs2, data) / length(data)
end
obj = fom(projected_design_vars, grid)

function adjoint_fom(adj_fom, data, grid)
    adjoint_fom!(similar(data), adj_fom, data, grid)
end
function adjoint_fom!(adj_data, adj_fom, data, grid)
    adj_data .= (adj_fom / length(data)) .* 2 .* data
    return adj_data
end

adj_rho_projected = adjoint_fom(1.0, projected_design_vars, grid)

adj_projsol = (; value=adj_rho_projected)
adj_projprob = adjoint_solve!(projsolver, adj_projsol, projsol.tape)
adj_depadsol = (; value=adj_projprob.rho_filtered)
adj_depadprob = adjoint_solve!(depadsolver, adj_depadsol, depadsol.tape)
adj_convsol = (; value=adj_depadprob.data)
adj_convprob = adjoint_solve!(convsolver, adj_convsol, convsol.tape)
adj_padsol = (; value=adj_convprob.data)
adj_padprob = adjoint_solve!(padsolver, adj_padsol, padsol.tape)
adj_design_vars = adj_padprob.data

let
    fig = Figure()
    ax1 = Axis(fig[1,1]; title = "SSP2 output", aspect=DataAspect())
    h1 = heatmap!(ax1, target_grid..., reshape(projected_design_vars, length.(target_grid)); colormap=colormap("grays"))
    Colorbar(fig[1,2], h1)

    ax2 = Axis(fig[1,3]; title = "design variables gradient", aspect=DataAspect())
    h2 = heatmap!(ax2, grid..., adj_design_vars; colormap=colormap("RdBu"))
    Colorbar(fig[1,4], h2)
    save("design_gradient.png", fig)
end

fom_withgradient = let grid=grid, padsolver=padsolver, convsolver=convsolver, depadsolver=depadsolver, projsolver=projsolver, adj_rho_projected=adj_rho_projected
    function (design_vars)

        padsolver.data = design_vars
        padsol = solve!(padsolver)
        convsolver.data = padsol.value
        convsol = solve!(convsolver)
        depadsolver.data = convsol.value
        depadsol = solve!(depadsolver)
        projsolver.rho_filtered = depadsol.value
        projsol = solve!(projsolver)

        _fom = fom(projsol.value, grid)
        adjoint_fom!(adj_rho_projected, 1.0, projsol.value, grid)

        adj_projsol = (; value=adj_rho_projected)
        adj_projprob = adjoint_solve!(projsolver, adj_projsol, projsol.tape)
        adj_depadsol = (; value=adj_projprob.rho_filtered)
        adj_depadprob = adjoint_solve!(depadsolver, adj_depadsol, depadsol.tape)
        adj_convsol = (; value=adj_depadprob.data)
        adj_convprob = adjoint_solve!(convsolver, adj_convsol, convsol.tape)
        adj_padsol = (; value=adj_convprob.data)
        adj_padprob = adjoint_solve!(padsolver, adj_padsol, padsol.tape)
        adj_design_vars = adj_padprob.data
        return _fom, adj_design_vars
    end
end

h = 1e-6
Random.seed!(0)
perturb = h * randn(size(design_vars))
fom_ph, = fom_withgradient(design_vars + perturb)
fom_mh, = fom_withgradient(design_vars - perturb)
dfomdh_fd = (fom_ph - fom_mh) / 2h

fom_val, adj_design_vars = fom_withgradient(design_vars)
dfomdh = 2sum(adj_design_vars .* perturb) / 2h
@show dfomdh_fd dfomdh

opt = NLopt.Opt(:LD_CCSAQ, length(design_vars))
evaluation_history = Float64[]
my_objective_fn = let fom_withgradient=fom_withgradient, evaluation_history=evaluation_history, design_vars=design_vars
    function (x, grad)
        val, adj_design = fom_withgradient(reshape(x, size(design_vars)))
        if !isempty(grad)
            copy!(grad, vec(adj_design))
        end
        push!(evaluation_history, val)
        return val
    end
end
NLopt.min_objective!(opt, my_objective_fn)
NLopt.maxeval!(opt, 50)
fmax, xmax, ret = NLopt.optimize(opt, vec(design_vars))

let
    padsolver.data = reshape(xmax, size(design_vars))
    padsol = solve!(padsolver)
    convsolver.data = padsol.value
    convsol = solve!(convsolver)
    depadsolver.data = convsol.value
    depadsol = solve!(depadsolver)
    projsolver.rho_filtered = depadsol.value
    projsol = solve!(projsolver)

    fig = Figure()
    ax1 = Axis(fig[1,1]; title = "Objective history", yscale=log10, limits = (nothing, (1e-16, 1e1)))
    h1 = scatterlines!(ax1, evaluation_history)

    ax2 = Axis(fig[1,2]; title = "Final SSP2 design", aspect=DataAspect())
    h2 = heatmap!(target_grid..., reshape(projsol.value, length.(target_grid)); colormap=colormap("grays"))
    Colorbar(fig[1,3], h2)
    save("optimization.png", fig)
end
