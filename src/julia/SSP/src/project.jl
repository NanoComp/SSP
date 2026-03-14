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
    interp_output = similar(data, length(target_points))
    interp_gradient_output = similar(data, length(target_points), length(grid))
    interp_hessian_output = similar(data, length(target_points), length(grid), length(grid))
    adj_data = similar(data)
    adj_data_tmp = similar(data)
    cacheval = (; interp_output, interp_gradient_output, interp_hessian_output, adj_data, adj_data_tmp)
    return ProjectionSolver(data, grid, target_points, alg, cacheval)
end

function solve!(solver::ProjectionSolver)
    proj_solve!(solver, solver.alg)
end

function proj_solve!(solver, alg::SSP2)

    (; data, grid, target_points, cacheval) = solver
    (; interp_output, interp_gradient_output, interp_hessian_output) = cacheval

    interpolatevaluegradienthessian!!!(interp_output, interp_gradient_output, interp_hessian_output, grid, data, target_points)
    value, tape = smoothed_projection(interp_output, interp_gradient_output, interp_hessian_output, grid, alg.beta, alg.eta)
    return (; value, tape)
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


function smoothed_projection(rho_filtered_vals, rho_filtered_gradient_vals, rho_filtered_hessian_vals, grids, beta, eta)
    dx_all = step.(grids)
    @assert allequal(dx_all)
    dx = first(dx_all)
    R_smoothing = 0.55 * dx

    rho_projected = tanh_projection.(rho_filtered_vals, beta, eta)

    den_helper = vec(mapreduce(abs2, +, rho_filtered_gradient_vals; dims=2))
    den_helper .+= R_smoothing^2 .* vec(mapreduce(abs2, +, rho_filtered_gradient_vals; dims=(2, 3)))

    nonzero_norm = abs.(den_helper) .> 0

    den_norm = sqrt.(ifelse.(nonzero_norm, den_helper, one(eltype(den_helper))))
    den_eff = ifelse.(nonzero_norm, den_norm, one(eltype(den_norm)))
    d = (eta .- rho_filtered_vals) ./ den_eff
    needs_smoothing = nonzero_norm .& (abs.(d) .< R_smoothing)
    d_R = d ./ R_smoothing
    F_plus = ifelse.(needs_smoothing, 0.5 .- (15 / 16) .* d_R .+ (5 / 8) .* d_R .^ 3 .- (3 / 16) .* d_R .^ 5, 1.0)
    F_minus = ifelse.(needs_smoothing, 0.5 .+ (15 / 16) .* d_R .- (5 / 8) .* d_R .^ 3 .+ (3 / 16) .* d_R .^ 5, 1.0)
    rho_filtered_minus = rho_filtered_vals .- R_smoothing .* den_eff .* F_plus
    rho_filtered_plus = rho_filtered_vals .+ R_smoothing .* den_eff .* F_minus

    rho_minus_eff_projected = tanh_projection.(rho_filtered_minus, beta, eta)
    rho_plus_eff_projected = tanh_projection.(rho_filtered_plus, beta, eta)

    rho_projected_smoothed = (1 .- F_plus) .* rho_minus_eff_projected .+ F_plus .* rho_plus_eff_projected
    tape = (; F_plus, F_minus, d, d_R, rho_projected, den_helper, den_eff, den_norm, nonzero_norm, needs_smoothing, rho_filtered_minus, rho_filtered_plus, rho_minus_eff_projected, rho_plus_eff_projected)
    return ifelse.(needs_smoothing, rho_projected_smoothed, rho_projected), tape
end


function adjoint_solve!(solver::ProjectionSolver, adj_sol, tape)
    adjoint_proj_solve!(solver, solver.alg, adj_sol, tape)
end

function adjoint_proj_solve!(solver, alg::SSP2, adj_sol, tape)

    (; data, grid, target_points, cacheval) = solver
    (; interp_output, interp_gradient_output, interp_hessian_output, adj_data, adj_data_tmp) = cacheval

    # need to recompute interpolation since the data may have changed since the forward pass
    interpolatevaluegradienthessian!!!(interp_output, interp_gradient_output, interp_hessian_output, grid, data, target_points)

    adj_interp_output, adj_interp_gradient_output, adj_interp_hessian_output = adjoint_smoothed_projection(adj_sol, tape, interp_output, interp_gradient_output, interp_hessian_output, grid, alg.beta, alg.eta)
    adjoint_interpolatevaluegradienthessian!!(adj_data, adj_data_tmp, adj_interp_output, adj_interp_gradient_output, adj_interp_hessian_output, grid, target_points)

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

function adjoint_smoothed_projection(adj_rho_ssp_projected, tape, rho_filtered_vals, rho_filtered_gradient_vals, rho_filtered_hessian_vals, grids, beta, eta)
    (; F_plus, F_minus, d, d_R, rho_projected, den_helper, den_eff, den_norm, nonzero_norm, needs_smoothing, rho_filtered_minus, rho_filtered_plus, rho_minus_eff_projected, rho_plus_eff_projected) = tape
    dx_all = step.(grids)
    @assert allequal(dx_all)
    dx = first(dx_all)
    R_smoothing = 0.55 * dx

    adj_rho_projected = adj_rho_ssp_projected .* .!needs_smoothing
    adj_rho_filtered_vals = adjoint_tanh_projection.(adj_rho_projected, rho_filtered_vals, beta, eta)

    adj_rho_projected_smoothed = adj_rho_ssp_projected .* needs_smoothing
    adj_rho_plus_eff_projected = adj_rho_projected_smoothed .* F_plus
    adj_rho_minus_eff_projected = adj_rho_projected_smoothed .* (1 .- F_plus)
    adj_rho_filtered_minus = adjoint_tanh_projection.(adj_rho_minus_eff_projected, rho_filtered_minus, beta, eta)
    adj_rho_filtered_plus = adjoint_tanh_projection.(adj_rho_plus_eff_projected, rho_filtered_minus, beta, eta)
    adj_rho_filtered_vals .+= adj_rho_filtered_minus .+ adj_rho_filtered_plus
    adj_F_plus = .-adj_rho_filtered_minus .* R_smoothing .* den_eff .+ adj_rho_projected_smoothed .* (rho_plus_eff_projected .- rho_minus_eff_projected)
    adj_den_eff = R_smoothing .* (adj_rho_filtered_plus .* F_minus .- adj_rho_filtered_minus .* F_plus)
    adj_F_minus = adj_rho_filtered_plus .* R_smoothing .* den_eff
    adj_d_R = ifelse.(needs_smoothing, adj_F_plus .* ((-15 / 16) .+ (15 / 8) .* d_R .^ 2 - (15 / 16) .* d_R .^ 4) .+ adj_F_minus .* ((15 / 16) .- (15 / 8) .* d_R .^ 2 .+ (15 / 16) .* d_R .^ 4), 0.0)
    adj_d = adj_d_R ./ R_smoothing
    adj_rho_filtered_vals .-= adj_d ./ den_eff
    adj_den_eff .-= adj_d .* d ./ den_eff
    adj_den_norm = ifelse.(nonzero_norm, adj_den_eff, 0.0)
    adj_den_helper = ifelse.(nonzero_norm, adj_den_norm ./ (2 .* den_norm), 0.0)
    adj_rho_filtered_gradient_vals = 2 .* adj_den_helper .* rho_filtered_gradient_vals
    adj_rho_filtered_hessian_vals = R_smoothing^2 .* 2 .* adj_den_helper .* rho_filtered_hessian_vals

    return adj_rho_filtered_vals, adj_rho_filtered_gradient_vals, adj_rho_filtered_hessian_vals
end
end
