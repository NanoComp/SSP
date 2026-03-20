module Constrain

using ..Interpolate: InterpolationProblem, CubicInterp, ValueWithGradient
import SSP: init!, solve!, adjoint_solve!

Base.@kwdef struct Solid{T}
    val::T=1
end

Base.@kwdef struct Void{T}
    val::T=0
end

Base.@kwdef struct LengthConstraintProblem{D,G,DB,T,K}
    data_smooth::D
    grid::G
    data_binary::DB
    target_points::T
    kind::K
end

mutable struct LengthConstraintSolver{D,G,DB,T,K,A,C}
    data_smooth::D
    grid::G
    data_binary::DB
    target_points::T
    kind::K
    alg::A
    cacheval::C
end

Base.@kwdef struct GeometricConstraints{T,S}
    target_length::T
    conic_radius::T=target_length
    constraint_threshold::S=1e-8
end

function init!(prob::LengthConstraintProblem, alg::GeometricConstraints)
    (; data_smooth, grid, data_binary, target_points, kind) = prob
    
    interp_prob = InterpolationProblem(; data=data_smooth, grid, target_points)
    interp_alg = CubicInterp(; deriv=ValueWithGradient())
    interp_solver = init!(interp_prob, interp_alg)
    interp_sol = solve!(interp_solver)

    c = 64 * alg.conic_radius^2
    eta_e = materialthreshold(kind, alg.target_length/alg.conic_radius)

    adj_rho_filtered_value = similar(interp_sol.value)
    adj_rho_filtered_gradient = similar(interp_sol.gradient)
    adj_rho_projected = similar(data_binary)

    cacheval = (; interp_solver, c, eta_e, adj_rho_filtered_value, adj_rho_filtered_gradient, adj_rho_projected)
    return LengthConstraintSolver(data_smooth, grid, data_binary, target_points, kind, alg, cacheval)
end

# TODO decide if/how the threshold function should depend on the threshold value
function materialthreshold(::Solid, x)
    # assumption: x is unitless
    if zero(x) <= x < one(x)
        (x^2 / 2 + 1) / 2
    elseif one(x) <= x < 2*one(x)
        -x^2 / 4 + x
    elseif 2*one(x) <= x
        one(x)
    else
        throw(DomainError(x, "Threshold parameter must be positive"))
    end
end

function materialthreshold(::Void, x)
    # assumption: x is unitless
    if zero(x) <= x < one(x)
        (1 - x^2 / 2) / 2
    elseif one(x) <= x < 2*one(x)
        1 + x^2 / 4 - x
    elseif 2*one(x) <= x
        zero(x)
    else
        throw(DomainError(x, "Threshold parameter must be positive"))
    end
end

function solve!(solver::LengthConstraintSolver)
    constrain_solve!(solver, solver.alg)
end

function constrain_solve!(solver, alg::GeometricConstraints)

    (; data_smooth, data_binary, target_points, kind, cacheval) = solver
    (; interp_solver, c, eta_e) = cacheval

    interp_solver.data = data_smooth
    rho_filtered = solve!(interp_solver)

    constraint = zero(eltype(data_binary))
    for (rho_p, rho_f, rho_g) in zip(data_binary, rho_filtered.value, eachslice(rho_filtered.gradient; dims=1))
        rho_g_normsq = sum(abs2, rho_g)
        value, tape = localconstraint(rho_p, rho_f, rho_g_normsq, c, eta_e, kind)
        constraint += value
    end
    constraint = constraint / length(target_points) / alg.constraint_threshold - 1

    return (; value=constraint, tape=nothing)
end

function localconstraint(rho_projected, rho_filtered, rho_filtered_gradient_normsq, c, eta_e, kind)
    extremal_region = exp(-c * rho_filtered_gradient_normsq)
    if kind isa Solid
        inflection_region = rho_projected * extremal_region
        above_threshold = min(rho_filtered - eta_e, zero(eta_e))
    elseif kind isa Void
        inflection_region = (1 - rho_projected) * extremal_region
        above_threshold = min(eta_e - rho_filtered, zero(eta_e))
    else
        error("unknown kind of material")
    end
    sqabove_threshold = above_threshold ^ 2
    value = inflection_region * sqabove_threshold
    tape = (; extremal_region, inflection_region, above_threshold, sqabove_threshold)
    return value, tape
end

function adjoint_solve!(solver::LengthConstraintSolver, adj_sol, tape)
    adjoint_constrain_solve!(solver, solver.alg, adj_sol, tape)
end

function adjoint_constrain_solve!(solver, alg, adj_sol, tape)
    (; data_smooth, grid, data_binary, target_points, kind, cacheval) = solver
    (; interp_solver, c, eta_e, adj_rho_filtered_value, adj_rho_filtered_gradient, adj_rho_projected) = cacheval

    # We do not keep a tape and need to repeat the forward calculation
    interp_solver.data = data_smooth
    rho_filtered = solve!(interp_solver)

    adj_constraint = adj_sol / length(target_points) / alg.constraint_threshold
    for (i, rho_p, rho_f, rho_g) in zip(eachindex(adj_rho_filtered_value), data_binary, rho_filtered.value, eachslice(rho_filtered.gradient; dims=1))
        rho_g_normsq = sum(abs2, rho_g)
        value, tape = localconstraint(rho_p, rho_f, rho_g_normsq, c, eta_e, kind)
        adj_rho_p, adj_rho_f, adj_rho_g_normsq = adjoint_localconstraint(adj_constraint, tape, rho_p, rho_f, rho_g_normsq, c, eta_e, kind)
        adj_rho_projected[i] = adj_rho_p
        adj_rho_filtered_value[i] = adj_rho_f
        adj_rho_filtered_gradient[i, :] .= 2adj_rho_g_normsq .* rho_g
    end

    adj_rho_filtered = (; value=adj_rho_filtered_value, gradient=adj_rho_filtered_gradient)
    adj_interp_prob = adjoint_solve!(interp_solver, adj_rho_filtered, rho_filtered.tape)

    return (; data_smooth=adj_interp_prob.data, grid=nothing, data_binary=adj_rho_projected, target_points=nothing, kind=nothing)
end

function adjoint_localconstraint(adj_constraint, tape, rho_projected, rho_filtered, rho_filtered_gradient_normsq, c, eta_e, kind)

    (; extremal_region, inflection_region, above_threshold, sqabove_threshold) = tape

    adj_inflection_region = adj_constraint * sqabove_threshold
    adj_sqabove_threshold = adj_constraint * inflection_region
    adj_above_threshold = adj_sqabove_threshold * 2above_threshold
    if kind isa Solid
        adj_rho_projected = adj_inflection_region * extremal_region
        adj_extremal_region = adj_inflection_region * rho_projected
        adj_rho_filtered = ifelse(rho_filtered - eta_e < zero(eta_e), adj_above_threshold, zero(eta_e))
    elseif kind isa Void
        adj_rho_projected = -adj_inflection_region * extremal_region
        adj_extremal_region = adj_inflection_region * (1 - rho_projected)
        adj_rho_filtered = ifelse(eta_e - rho_filtered < zero(eta_e), -adj_above_threshold, zero(eta_e))
    else
        error("unknown kind of material")
    end
    adj_rho_filtered_gradient_normsq = adj_extremal_region * extremal_region * -c

    return adj_rho_projected, adj_rho_filtered, adj_rho_filtered_gradient_normsq
end

end