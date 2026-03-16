module Project

using FastInterpolations: cubic_interp!, cubic_adjoint, DerivOp
import SSP: init, solve!, adjoint_solve!

Base.@kwdef struct ProjectionProblem{D,G,T}
    data::D
    grid::G
    target_points::T
end

mutable struct ProjectionSolver{D,G,T,A,C}
    data::D
    const grid::G
    const target_points::T
    alg::A
    cacheval::C
end

Base.@kwdef struct SSP2{B,E}
    beta::B
    eta::E
end

function init(prob::ProjectionProblem, alg::SSP2)
    (; data, grid, target_points) = prob
    # TODO refactor these workspaces as views of a single array
    rho_filtered = similar(data, length(target_points))
    rho_filtered_gradient = similar(data, length(target_points), length(grid))
    rho_filtered_hessian = similar(data, length(target_points), length(grid), length(grid))
    rho_projected = similar(data, length(target_points))
    adj_data = similar(data)
    adj_data_tmp = similar(data)
    adj_rho_filtered = similar(rho_filtered)
    adj_rho_filtered_gradient = similar(rho_filtered_gradient)
    adj_rho_filtered_hessian = similar(rho_filtered_hessian)
    cacheval = (; rho_filtered, rho_filtered_gradient, rho_filtered_hessian, rho_projected, adj_data, adj_data_tmp, adj_rho_filtered, adj_rho_filtered_gradient, adj_rho_filtered_hessian)
    return ProjectionSolver(data, grid, target_points, alg, cacheval)
end

function solve!(solver::ProjectionSolver)
    proj_solve!(solver, solver.alg)
end

function proj_solve!(solver, alg::SSP2)

    (; data, grid, target_points, cacheval) = solver
    (; rho_filtered, rho_filtered_gradient, rho_filtered_hessian, rho_projected) = cacheval

    interpolatevaluegradienthessian!!!(rho_filtered, rho_filtered_gradient, rho_filtered_hessian, grid, data, target_points)
    dx_all = step.(grid)
    @assert allequal(dx_all)
    dx = first(dx_all)
    R_smoothing = 0.55 * dx
    for (i, rho_f, rho_g, rho_h) in zip(eachindex(rho_projected), rho_filtered, eachslice(rho_filtered_gradient; dims=1), eachslice(rho_filtered_hessian; dims=1))
        # the calculation of the norm is not local in memory, but this is
        # because the interpolation is done as SoA whereas here we use AoS
        rho_g_normsq = sum(abs2, rho_g)
        rho_h_normsq = sum(abs2, rho_h)
        rho_p, tape = smoothed_projection(rho_f, rho_g_normsq, rho_h_normsq, R_smoothing, alg.beta, alg.eta)
        rho_projected[i] = rho_p
    end
    return (; value=rho_projected, tape=nothing)
end

function interpolatevaluegradienthessian!!!(interp_output, interp_gradient_output, interp_hessian_output, grid, data, target_points)
    cubic_interp!(interp_output, grid, data, target_points)
    ndim = Val{ndims(data)}()
    dims = ntuple(n -> Val{n}(), ndim) # construct a type-stable tuple / range because this gets unrolled
    dograd = function (::Val{dim}) where {dim}
        # construct deriv tuple in a type-stable way by using a function barrier
        deriv = ntuple(n -> DerivOp{n == dim ? 1 : 0}(), ndim)
        out = view(interp_gradient_output, :, dim)
        cubic_interp!(out, grid, data, target_points; deriv)
    end
    foreach(dograd, dims)
    dohess = function (::Val{dim1}, ::Val{dim2}) where {dim1, dim2}
        deriv = ntuple(n -> DerivOp{(n == dim1 ? 1 : 0) + (n == dim2 ? 1 : 0)}(), ndim)
        out = view(interp_hessian_output, :, dim1, dim2)
        cubic_interp!(out, grid, data, target_points; deriv)
    end
    foreach(dim1 -> foreach(dim2 -> dohess(dim1, dim2), dims), dims)
    return
end


function tanh_projection(x, beta, eta)
    u = beta * eta
    v = beta * (1 - eta)
    tanhu = tanh(u)
    tanhv = tanh(v)
    den = tanhu + tanhv
    if iszero(den) # this implies β → 0^+ is causing underflow when this function should approach the identity
        (x * oneunit(beta)) / one(den) # multiply/divide to get right units and type stability
    else
        (tanhu + tanh(beta * (x - eta))) / den
    end
end

function adjoint_tanh_projection(adj_out, x, beta, eta)
    if isinf(beta)
        zero(adj_out)
    else
        den = tanh(beta * eta) + tanh(beta * (1 - eta))
        if iszero(den) # this implies β → 0^+ is causing underflow when this function should approach zero
            adj_out * zero(beta)
        else
            adj_out * beta * sech(beta * (x - eta))^2 / den
        end
    end
end


function smoothed_projection(rho_filtered, rho_filtered_gradient_normsq, rho_filtered_hessian_normsq, R_smoothing, beta, eta)
    rho_projected = tanh_projection(rho_filtered, beta, eta)

    den_helper = rho_filtered_gradient_normsq + R_smoothing^2 * rho_filtered_hessian_normsq

    nonzero_norm = abs(den_helper) > 0

    den_eff = sqrt(ifelse(nonzero_norm, den_helper, oneunit(den_helper)))
    d = (eta - rho_filtered) / den_eff
    d_R = d / R_smoothing

    needs_smoothing = nonzero_norm & (abs(d_R) < one(d_R))
    F_plus  = ifelse(needs_smoothing, 1//2 + d_R * evalpoly(d_R^2, (-15//16, 5//8, -3//16)), one(d_R))
    F_minus = ifelse(needs_smoothing, 1//2 + d_R * evalpoly(d_R^2, (15//16, -5//8, 3//16)),  one(d_R))

    rho_filtered_minus = rho_filtered - R_smoothing * den_eff * F_plus
    rho_filtered_plus  = rho_filtered + R_smoothing * den_eff * F_minus

    rho_minus_eff_projected = tanh_projection(rho_filtered_minus, beta, eta)
    rho_plus_eff_projected  = tanh_projection(rho_filtered_plus,  beta, eta)

    rho_projected_smoothed = (1 - F_plus) * rho_minus_eff_projected + F_plus * rho_plus_eff_projected
    tape = (; F_plus, F_minus, d, d_R, rho_projected, den_helper, den_eff, nonzero_norm, needs_smoothing, rho_filtered_minus, rho_filtered_plus, rho_minus_eff_projected, rho_plus_eff_projected)
    return ifelse(needs_smoothing, rho_projected_smoothed, rho_projected), tape
end


function adjoint_solve!(solver::ProjectionSolver, adj_sol, tape)
    adjoint_proj_solve!(solver, solver.alg, adj_sol, tape)
end

function adjoint_proj_solve!(solver, alg::SSP2, adj_sol, tape)

    (; data, grid, target_points, cacheval) = solver
    (; rho_filtered, rho_filtered_gradient, rho_filtered_hessian, adj_data, adj_data_tmp, adj_rho_filtered, adj_rho_filtered_gradient, adj_rho_filtered_hessian) = cacheval

    # We do not keep a tape and need to repeat the forward calculation
    interpolatevaluegradienthessian!!!(rho_filtered, rho_filtered_gradient, rho_filtered_hessian, grid, data, target_points)

    dx_all = step.(grid)
    @assert allequal(dx_all)
    dx = first(dx_all)
    R_smoothing = 0.55 * dx
    for (i, adj_proj, rho_f, rho_g, rho_h) in zip(eachindex(adj_rho_filtered), adj_sol, rho_filtered, eachslice(rho_filtered_gradient; dims=1), eachslice(rho_filtered_hessian; dims=1))
        # the calculation of the norm is not local in memory, but this is
        # because the interpolation is done as SoA whereas here we use AoS
        rho_g_normsq = sum(abs2, rho_g)
        rho_h_normsq = sum(abs2, rho_h)
        rho_p, tape = smoothed_projection(rho_f, rho_g_normsq, rho_h_normsq, R_smoothing, alg.beta, alg.eta)
        adj_rho_f, adj_rho_g_normsq, adj_rho_h_normsq = adjoint_smoothed_projection(adj_proj, tape, rho_f, rho_g_normsq, rho_h_normsq, R_smoothing, alg.beta, alg.eta)
        adj_rho_filtered[i] = adj_rho_f
        adj_rho_filtered_gradient[i, :] .= 2adj_rho_g_normsq .* rho_g
        adj_rho_filtered_hessian[i, :, :] .= 2adj_rho_h_normsq .* rho_h
    end
    adjoint_interpolatevaluegradienthessian!!(adj_data, adj_data_tmp, adj_rho_filtered, adj_rho_filtered_gradient, adj_rho_filtered_hessian, grid, target_points)

    return (; data=adj_data, grid=nothing, target_points=nothing)
end

function adjoint_interpolatevaluegradienthessian!!(adj_data, adj_data_tmp, adj_interp_output, adj_interp_gradient_output, adj_interp_hessian_output, grid, target_points)
    adj_op = cubic_adjoint(grid, target_points)
    adj_op(adj_data, adj_interp_output)
    ndim = Val{ndims(adj_data)}()
    dims = ntuple(n -> Val{n}(), ndim) # construct a type-stable tuple / range because this gets unrolled
    dograd = function (::Val{dim}) where {dim}
        # construct deriv tuple in a type-stable way by using a function barrier
        deriv = ntuple(n -> DerivOp{n == dim ? 1 : 0}(), ndim)
        adj_op_grad = cubic_adjoint(grid, target_points; deriv)
        adj_grad = view(adj_interp_gradient_output, :, dim)
        adj_op_grad(adj_data_tmp, adj_grad)
        adj_data .+= adj_data_tmp
    end
    foreach(dograd, dims)
    dohess = function (::Val{dim1}, ::Val{dim2}) where {dim1, dim2}
        deriv = ntuple(n -> DerivOp{(n == dim1 ? 1 : 0) + (n == dim2 ? 1 : 0)}(), ndim)
        adj_op_hess = cubic_adjoint(grid, target_points; deriv)
        adj_hess = view(adj_interp_hessian_output, :, dim1, dim2)
        adj_op_hess(adj_data_tmp, adj_hess)
        adj_data .+= adj_data_tmp
    end
    foreach(dim1 -> foreach(dim2 -> dohess(dim1, dim2), dims), dims)
    return
end


function adjoint_smoothed_projection(adj_rho_projected_maybe_smoothed, tape, rho_filtered, rho_filtered_gradient_normsq, rho_filtered_hessian_normsq, R_smoothing, beta, eta)
    (; F_plus, F_minus, d, d_R, rho_projected, den_helper, den_eff, nonzero_norm, needs_smoothing, rho_filtered_minus, rho_filtered_plus, rho_minus_eff_projected, rho_plus_eff_projected) = tape

    adj_rho_projected = ifelse(needs_smoothing, zero(adj_rho_projected_maybe_smoothed), adj_rho_projected_maybe_smoothed)
    adj_rho_filtered  = adjoint_tanh_projection(adj_rho_projected, rho_filtered, beta, eta)

    adj_rho_projected_smoothed  = ifelse(needs_smoothing, adj_rho_projected_maybe_smoothed, zero(adj_rho_projected_maybe_smoothed))
    adj_rho_plus_eff_projected  = adj_rho_projected_smoothed * F_plus
    adj_rho_minus_eff_projected = adj_rho_projected_smoothed * (1 - F_plus)

    adj_rho_filtered_minus = adjoint_tanh_projection(adj_rho_minus_eff_projected, rho_filtered_minus, beta, eta)
    adj_rho_filtered_plus  = adjoint_tanh_projection(adj_rho_plus_eff_projected,  rho_filtered_plus, beta, eta)
    adj_rho_filtered += adj_rho_filtered_minus + adj_rho_filtered_plus

    adj_F_plus = -adj_rho_filtered_minus * R_smoothing * den_eff + adj_rho_projected_smoothed * (rho_plus_eff_projected - rho_minus_eff_projected)
    adj_den_eff = R_smoothing * (adj_rho_filtered_plus * F_minus - adj_rho_filtered_minus * F_plus)
    adj_F_minus = adj_rho_filtered_plus * R_smoothing * den_eff
    adj_d_R = ifelse(needs_smoothing, adj_F_plus * evalpoly(d_R^2, (-15//16, 15//8, -15//16)) + adj_F_minus * evalpoly(d_R^2, (15//16, -15//8, 15//16)), zero(adj_F_plus) + zero(adj_F_minus))
    
    adj_d = adj_d_R / R_smoothing
    adj_rho_filtered -= adj_d / den_eff
    adj_den_eff -= adj_d * d / den_eff
    adj_den_helper = ifelse(nonzero_norm, adj_den_eff, zero(adj_den_eff)) / 2den_eff

    adj_rho_filtered_gradient_normsq = adj_den_helper
    adj_rho_filtered_hessian_normsq = adj_den_helper * R_smoothing^2

    return adj_rho_filtered, adj_rho_filtered_gradient_normsq, adj_rho_filtered_hessian_normsq
end
end
