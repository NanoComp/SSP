using SSP: init, solve!, adjoint_solve!
using SSP: Kernel, Pad, Convolve, Project, Constrain
using .Kernel: conickernel
using .Pad: FillPadding, BoundaryPadding, Inner, PaddingProblem, DefaultPaddingAlgorithm
using .Convolve: DiscreteConvolutionProblem, FFTConvolution
using .Project: ProjectionProblem, SSP2
using .Constrain: LengthConstraintProblem, GeometricConstraints, solid, void


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
    # boundary = BoundaryPadding(size(kernel) .- 1, size(kernel) .- 1),
    boundary = FillPadding(1.0, size(kernel) .- 1, size(kernel) .- 1),
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
    range(-1, 1, length=Nx * 2),
    range(-1, 1, length=Ny * 2),
)
target_points = vec(collect(Iterators.product(target_grid...)))
projprob = ProjectionProblem(;
    rho_filtered=filtered_design_vars,
    grid,
    target_points,
    beta = Inf,
    eta = 0.5
)
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

constraintprob = LengthConstraintProblem(;
    rho_filtered = filtered_design_vars,
    grid = grid,
    rho_projected = projected_design_vars,
    target_points,
    material = void,
    target_length = radius,
)
constraintalg = GeometricConstraints()
constraintsolver = init(constraintprob, constraintalg)
constraintsol = solve!(constraintsolver)
@show constraintsol # adjoint_solve!(constraint_solver, 1.0, constraintsol.tape)

function fom(data, grid)
    return sum(abs2, data) * prod(step, grid)
end
obj = fom(projected_design_vars, grid)

function adjoint_fom(adj_fom, data, grid)
    adjoint_fom!(similar(data), adj_fom, data, grid)
end
function adjoint_fom!(adj_data, adj_fom, data, grid)
    adj_data .= adj_fom .* 2 .* data .* prod(step, grid)
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

constraint_withgradient = let grid=grid, padsolver=padsolver, convsolver=convsolver, depadsolver=depadsolver, projsolver=projsolver
    function (design_vars, constraintsolver)

        padsolver.data = design_vars
        padsol = solve!(padsolver)
        convsolver.data = padsol.value
        convsol = solve!(convsolver)
        depadsolver.data = convsol.value
        depadsol = solve!(depadsolver)
        projsolver.rho_filtered = depadsol.value
        projsol = solve!(projsolver)
        constraintsolver.rho_filtered = depadsol.value
        constraintsolver.rho_projected = projsol.value
        constraintsol = solve!(constraintsolver)

        adj_constraintsol = (; value=1.0)
        adj_constraintprob = adjoint_solve!(constraintsolver, adj_constraintsol, constraintsol.tape)
        adj_projsol = (; value=adj_constraintprob.rho_projected)
        adj_projprob = adjoint_solve!(projsolver, adj_projsol, projsol.tape)
        adj_depadsol = (; value=adj_projprob.rho_filtered + adj_constraintprob.rho_filtered)
        adj_depadprob = adjoint_solve!(depadsolver, adj_depadsol, depadsol.tape)
        adj_convsol = (; value=adj_depadprob.data)
        adj_convprob = adjoint_solve!(convsolver, adj_convsol, convsol.tape)
        adj_padsol = (; value=adj_convprob.data)
        adj_padprob = adjoint_solve!(padsolver, adj_padsol, padsol.tape)
        adj_design_vars = adj_padprob.data

        return constraintsol.value, adj_design_vars
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

constraint_ph, = constraint_withgradient(design_vars + perturb, constraintsolver)
constraint_mh, = constraint_withgradient(design_vars - perturb, constraintsolver)
dconstraintdh_fd = (constraint_ph - constraint_mh) / 2h

# only agrees in the first digit
_, adj_design_vars_constraint = constraint_withgradient(design_vars, constraintsolver)
dconstraintdh = 2sum(adj_design_vars_constraint .* perturb) / 2h
@show dconstraintdh_fd dconstraintdh

let
    fig = Figure()
    ax1 = Axis(fig[1,1]; title = "SSP2 output", aspect=DataAspect())
    h1 = heatmap!(ax1, target_grid..., reshape(projected_design_vars, length.(target_grid)); colormap=colormap("grays"))
    Colorbar(fig[1,2], h1)

    ax2 = Axis(fig[1,3]; title = "design variables gradient", aspect=DataAspect())
    h2 = heatmap!(ax2, grid..., adj_design_vars_constraint; colormap=colormap("RdBu"))
    Colorbar(fig[1,4], h2)
    save("constraint_gradient.png", fig)
end


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
NLopt.max_objective!(opt, my_objective_fn)
constraint_history = []
my_constraint_fn = let constraint_withgradient=constraint_withgradient, constraint_history=constraint_history, design_vars=design_vars, constraintsolver=constraintsolver
    function (results, x, grad)
        map!(results, [(1, void), (2, solid)]) do (i, material)
            constraintsolver.material = material
            val, adj_design = constraint_withgradient(reshape(x, size(design_vars)), constraintsolver)
            if !isempty(grad)
                copy!(view(grad, :, i), vec(adj_design))
            end
            return val
        end
        push!(constraint_history, copy(results))
        return
    end
end
constraint_tols = zeros(2) # void & solid
NLopt.inequality_constraint!(opt, my_constraint_fn, constraint_tols)
NLopt.maxeval!(opt, 30)
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
    ax1 = Axis(fig[1,1]; title = "Objective history", yscale=log10)
    h1 = scatterlines!(ax1, evaluation_history)

    ax2 = Axis(fig[1,2]; title = "Final SSP2 design", aspect=DataAspect())
    h2 = heatmap!(target_grid..., reshape(projsol.value, length.(target_grid)); colormap=colormap("grays"))
    Colorbar(fig[1,3], h2)

    ax3 = Axis(fig[1,4]; title = "Constraint history")
    h3 = scatterlines!(ax3, getindex.(constraint_history, 1))
    h3 = scatterlines!(ax3, getindex.(constraint_history, 2))
    save("optimization.png", fig)
end
