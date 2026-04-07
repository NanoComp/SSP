function conic_filter_withsolver(data, radius, grid)
    kernel = Kernel.conickernel(grid, radius)

    padprob = Pad.PaddingProblem(;
        data,
        boundary = Pad.BoundaryPadding(size(kernel) .- 1, size(kernel) .- 1),
    )
    padalg = Pad.DefaultPaddingAlgorithm()
    padsolver = init(padprob, padalg)
    padsol = solve!(padsolver)

    convprob = Convolve.DiscreteConvolutionProblem(;
        data = padsol.value,
        kernel,
    )

    convalg = Convolve.FFTConvolution()
    convsolver = init(convprob, convalg)
    convsol = solve!(convsolver)

    depadprob = Pad.PaddingProblem(;
        data = convsol.value,
        boundary = Pad.Inner(size(kernel) .- 1, size(kernel) .- 1),
    )
    depadalg = Pad.DefaultPaddingAlgorithm()
    depadsolver = init(depadprob, depadalg)
    depadsol = solve!(depadsolver)

    return depadsol.value, (padsolver, convsolver, depadsolver)
end

"""
    conic_filter(data, radius, grid)

Apply a conic filter of given `radius` to the `data` values on the given `grid`, i.e. a tuple of ranges.
Boundary padding is applied to the data in order to preserve features at the edges of the design region.
"""
conic_filter(args...) = conic_filter_withsolver(args...)[1]

function conic_filter_rrule(adj_depad_value, padsolver, convsolver, depadsolver)
    adj_depadsol = (; value=adj_depad_value)
    adj_depadprob = adjoint_solve!(depadsolver, adj_depadsol, nothing)
    adj_convsol = (; value=adj_depadprob.data)
    adj_convprob = adjoint_solve!(convsolver, adj_convsol, nothing)
    adj_padsol = (; value=adj_convprob.data)
    adj_padprob = adjoint_solve!(padsolver, adj_padsol, nothing)
    return adj_padprob.data
end

function ssp_withsolver(alg, rho_filtered, beta, eta, grid)
    target_points = vec(collect(Iterators.product(grid...)))
    prob = Project.ProjectionProblem(;
        rho_filtered,
        grid,
        target_points,
        beta,
        eta,
    )
    solver = init(prob, alg)
    sol = solve!(solver)
    return reshape(sol.value, size(rho_filtered)), solver
end

"""
    ssp1_linear(rho_filtered, beta, eta, grid)

Project using the original [SSP1 algorithm] [1] with linear interpolation.

This technique takes smoothed data, e.g. from filtering, `rho_filtered` defined on a `grid` and projects it to an almost-everywhere binary design, when `beta=Inf`, with smoothed values within a grid spacing of the level-set `eta`.

For finite `beta`, this projection is similar to a more simple tanh projection.

At `beta=Inf`, this projection is not differentiable through topology changes, although practical examples may continue to do well.

[1]: A. M. Hammond, A. Oskooi, I. M. Hammond, M. Chen, S. E. Ralph, and S. G. Johnson, [“Unifying and accelerating level-set and density-based topology optimization by subpixel-smoothed projection,”](http://doi.org/10.1364/OE.563512) Optics Express, vol. 33, pp. 33620–33642, July 2025. Editor's Pick.
"""
ssp1_linear(args...; kws...) = ssp_withsolver(Project.SSP1_linear(; kws...), args...)[1]

"""
    ssp1(rho_filtered, beta, eta, grid)

Project using the original [SSP1 algorithm] [1] with cubic interpolation.

This technique takes smoothed data, e.g. from filtering, `rho_filtered` defined on a `grid` and projects it to an almost-everywhere binary design, when `beta=Inf`, with smoothed values within a grid spacing of the level-set `eta`.

For finite `beta`, this projection is similar to a more simple tanh projection.

At `beta=Inf`, this projection is not differentiable through topology changes, although practical examples may continue to do well.

[1]: A. M. Hammond, A. Oskooi, I. M. Hammond, M. Chen, S. E. Ralph, and S. G. Johnson, [“Unifying and accelerating level-set and density-based topology optimization by subpixel-smoothed projection,”](http://doi.org/10.1364/OE.563512) Optics Express, vol. 33, pp. 33620–33642, July 2025. Editor's Pick.
"""
ssp1(args...; kws...) = ssp_withsolver(Project.SSP1(; kws...), args...)[1]

"""
    ssp2(rho_filtered, beta, eta, grid)

Project using the improved [SSP2 algorithm] [1] with cubic interpolation.

This technique takes smoothed data, e.g. from filtering, `rho_filtered` defined on a `grid` and projects it to an almost-everywhere binary design, when `beta=Inf`, with smoothed values within a grid spacing of the level-set `eta`.

For finite `beta`, this projection is similar to a more simple tanh projection.

At `beta=Inf`, this projection is differentiable through topology changes because it employs a regularization using the Hessian of `rho_filtered`.

[1]: G. Romano, R. Arrieta, and S. G. Johnson, [“Differentiating through binarized topology changes: Second-order subpixel-smoothed projection,”](http://arxiv.org/abs/2601.10737) arXiv.org e-Print archive, 2601.10737, January 2026.
"""
ssp2(args...; kws...) = ssp_withsolver(Project.SSP2(; kws...), args...)[1]

function ssp_rrule(adj_rho_projected, solver)
    adj_sol = (; value=adj_rho_projected)
    adj_prob = adjoint_solve!(solver, adj_sol, nothing)
    return adj_prob.rho_filtered
end

function lengthconstraint_withsolver(material, rho_filtered, rho_projected, grid, target_length, conic_radius=target_length;
                                     constraint_threshold=Constrain.GeometricConstraints_default_constraint_threshold)
    target_points = vec(collect(Iterators.product(grid...)))
    prob = Constrain.LengthConstraintProblem(;
        rho_filtered,
        grid,
        rho_projected,
        target_points,
        material,
        target_length,
        conic_radius
    )
    alg = Constrain.GeometricConstraints(; constraint_threshold)
    solver = init(prob, alg)
    sol = solve!(solver)
    return sol.value, solver
end

"""
    constraint_solid(rho_filtered, rho_projected, grid, target_length)

Calculate a solid length-scale constraint function using the [geometric constraint algorithm] [1].

This technique takes smoothed data, e.g. from filtering, `rho_filtered` and binary data, e.g. from projection, `rho_projected` both defined on the same `grid` and calculates whether features in the solid region, i.e. where `rho_projected` takes values of 1, violate the minimum `target_length`.

[1]: R. Arrieta, G. Romano, and S. G. Johnson, [“Hyperparameter-free minimum-lengthscale constraints for topology optimization,”](http://arxiv.org/abs/2507.16108) arXiv.org e-Print archive, 2507.16108, July 2025.
"""
constraint_solid(args...; kws...) = lengthconstraint_withsolver(Constrain.solid, args...; kws...)[1]

"""
    constraint_void(rho_filtered, rho_projected, grid, target_length)

Calculate a void length-scale constraint function using the [geometric constraint algorithm] [1].

This technique takes smoothed data, e.g. from filtering, `rho_filtered` and binary data, e.g. from projection, `rho_projected` both defined on the same `grid` and calculates whether features in the void region, i.e. where `rho_projected` takes values of 0, violate the minimum `target_length`.

[1]: R. Arrieta, G. Romano, and S. G. Johnson, [“Hyperparameter-free minimum-lengthscale constraints for topology optimization,”](http://arxiv.org/abs/2507.16108) arXiv.org e-Print archive, 2507.16108, July 2025.
"""
constraint_void(args...; kws...) = lengthconstraint_withsolver(Constrain.void, args...; kws...)[1]

function lengthconstraint_rrule(adj_constraint, solver)
    adj_sol = (; value=adj_constraint)
    adj_prob = adjoint_solve!(solver, adj_sol, nothing)
    return adj_prob
end