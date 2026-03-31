module Constrain

using ..Interpolate: InterpolationProblem, CubicInterp, ValueWithGradient
import SSP: init!, solve!, adjoint_solve!

public LengthConstraintProblem, GeometricConstraints, solid, void

@enum Material begin
    solid
    void
end

Base.@kwdef struct LengthConstraintProblem{D,G,DB,T,L}
    rho_filtered::D
    grid::G
    rho_projected::DB
    target_points::T
    material::Material
    target_length::L
    conic_radius::L=target_length
end

function Base.copy(prob::LengthConstraintProblem)
    newprob = LengthConstraintProblem(;
        rho_filtered = copy(prob.rho_filtered),
        grid = prob.grid,
        rho_projected = copy(prob.rho_projected),
        target_points = copy(prob.target_points),
        material = prob.material,
        target_length = prob.target_length,
        conic_radius = prob.conic_radius,
    )
    return newprob
end

mutable struct LengthConstraintSolver{D,G,DB,T,L,A,C}
    rho_filtered::D
    grid::G
    rho_projected::DB
    target_points::T
    material::Material
    target_length::L
    conic_radius::L
    alg::A
    cacheval::C
    
end

Base.@kwdef struct GeometricConstraints{S}
    constraint_threshold::S=1e-8
end

function init!(prob::LengthConstraintProblem, alg::GeometricConstraints)
    (; rho_filtered, grid, rho_projected, target_points, material, target_length, conic_radius) = prob
    
    interp_prob = InterpolationProblem(; data=rho_filtered, grid, target_points)
    interp_alg = CubicInterp(; deriv=ValueWithGradient())
    interp_solver = init!(interp_prob, interp_alg)
    interp_sol = solve!(interp_solver)

    adj_rho_filtered_value = similar(interp_sol.value)
    adj_rho_filtered_gradient = similar(interp_sol.gradient)
    adj_rho_projected = similar(rho_projected)

    cacheval = (; interp_solver, adj_rho_filtered_value, adj_rho_filtered_gradient, adj_rho_projected)
    return LengthConstraintSolver(rho_filtered, grid, rho_projected, target_points, material, target_length, conic_radius, alg, cacheval)
end

function materialthreshold(material, x)
    if material == solid
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
    elseif material == void
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
    else
        throw(ArgumentError("material must be solid or void"))
    end
end

function solve!(solver::LengthConstraintSolver)
    constrain_solve!(solver, solver.alg)
end

function constrain_solve!(solver, alg::GeometricConstraints)

    (; rho_filtered, rho_projected, target_points, material, target_length, conic_radius, cacheval) = solver
    (; interp_solver) = cacheval

    c = 64 * conic_radius^2
    eta_m = materialthreshold(material, target_length/conic_radius)

    interp_solver.data = rho_filtered
    rho_filtered = solve!(interp_solver)

    constraint = zero(eltype(rho_projected))
    for (rho_p, rho_f, rho_g) in zip(rho_projected, rho_filtered.value, eachslice(rho_filtered.gradient; dims=1))
        rho_g_normsq = sum(abs2, rho_g)
        value, tape = localconstraint(rho_p, rho_f, rho_g_normsq, c, eta_m, material)
        constraint += value
    end
    constraint = constraint / length(target_points) / alg.constraint_threshold - 1

    return (; value=constraint, tape=nothing)
end

function localconstraint(rho_projected, rho_filtered, rho_filtered_gradient_normsq, c, eta_m, material)
    extremal_region = exp(-c * rho_filtered_gradient_normsq)
    if material == solid
        inflection_region = rho_projected * extremal_region
        above_threshold = min(rho_filtered - eta_m, zero(eta_m))
    elseif material == void
        inflection_region = (1 - rho_projected) * extremal_region
        above_threshold = min(eta_m - rho_filtered, zero(eta_m))
    else
        error("unknown material of material")
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
    (; rho_filtered, grid, rho_projected, target_points, material, target_length, conic_radius, cacheval) = solver
    (; interp_solver, adj_rho_filtered_value, adj_rho_filtered_gradient, adj_rho_projected) = cacheval

    c = 64 * conic_radius^2
    eta_m = materialthreshold(material, target_length/conic_radius)

    # We do not keep a tape and need to repeat the forward calculation
    interp_solver.data = rho_filtered
    rho_filtered = solve!(interp_solver)

    adj_constraint = adj_sol.value / length(target_points) / alg.constraint_threshold
    for (i, rho_p, rho_f, rho_g) in zip(eachindex(adj_rho_filtered_value), rho_projected, rho_filtered.value, eachslice(rho_filtered.gradient; dims=1))
        rho_g_normsq = sum(abs2, rho_g)
        value, tape = localconstraint(rho_p, rho_f, rho_g_normsq, c, eta_m, material)
        adj_rho_p, adj_rho_f, adj_rho_g_normsq = adjoint_localconstraint(adj_constraint, tape, rho_p, rho_f, rho_g_normsq, c, eta_m, material)
        adj_rho_projected[i] = adj_rho_p
        adj_rho_filtered_value[i] = adj_rho_f
        adj_rho_filtered_gradient[i, :] .= 2adj_rho_g_normsq .* rho_g
    end

    adj_rho_filtered = (; value=adj_rho_filtered_value, gradient=adj_rho_filtered_gradient)
    adj_interp_prob = adjoint_solve!(interp_solver, adj_rho_filtered, rho_filtered.tape)

    return (; rho_filtered=adj_interp_prob.data, grid=nothing, rho_projected=adj_rho_projected, target_points=nothing, material=nothing)
end

function adjoint_localconstraint(adj_constraint, tape, rho_projected, rho_filtered, rho_filtered_gradient_normsq, c, eta_m, material)

    (; extremal_region, inflection_region, above_threshold, sqabove_threshold) = tape

    adj_inflection_region = adj_constraint * sqabove_threshold
    adj_sqabove_threshold = adj_constraint * inflection_region
    adj_above_threshold = adj_sqabove_threshold * 2above_threshold
    if material == solid
        adj_rho_projected = adj_inflection_region * extremal_region
        adj_extremal_region = adj_inflection_region * rho_projected
        adj_rho_filtered = ifelse(rho_filtered - eta_m < zero(eta_m), adj_above_threshold, zero(eta_m))
    elseif material == void
        adj_rho_projected = -adj_inflection_region * extremal_region
        adj_extremal_region = adj_inflection_region * (1 - rho_projected)
        adj_rho_filtered = ifelse(eta_m - rho_filtered < zero(eta_m), -adj_above_threshold, zero(eta_m))
    else
        error("unknown material of material")
    end
    adj_rho_filtered_gradient_normsq = adj_extremal_region * extremal_region * -c

    return adj_rho_projected, adj_rho_filtered, adj_rho_filtered_gradient_normsq
end

end