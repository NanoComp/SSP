module Project

using ..Interpolate: InterpolationProblem, LinearInterp, CubicInterp, ValueWithGradient, ValueWithGradientAndHessian
import SSP: init!, solve!, adjoint_solve!

public ProjectionProblem, SSP1_linear, SSP1, SSP2

"""
    ProjectionProblem(; rho_filtered, grid, target_points, beta=Inf, eta=1/2)

Define a problem for projecting smoothed data `rho_filtered` defined on a `grid`, i.e. a tuple of range, at a list of selected `target_points`, i.e. a vector of coordinate tuples.
The projection pushes `rho` values above `eta` towards 1 and `rho` values below `eta` towards 0 with a stiffnes parameter `beta`.
"""
Base.@kwdef struct ProjectionProblem{D,G,T,B}
    rho_filtered::D
    grid::G
    target_points::T
    beta::B = eltype(rho_filtered)(Inf)
    eta::B = eltype(rho_filtered)(1//2)
    # dilation/erosion distance = 0
end

function Base.copy(prob::ProjectionProblem)
    newprob = ProjectionProblem(;
        rho_filtered = copy(prob.rho_filtered),
        grid = prob.grid,
        target_points = copy(prob.target_points),
        beta = prob.beta,
        eta = prob.eta,
    )
    return newprob
end

mutable struct ProjectionSolver{D,G,T,B,A,C}
    rho_filtered::D
    const grid::G
    const target_points::T
    beta::B
    eta::B
    # dilation/erosion distance
    alg::A
    cacheval::C
end

Base.@kwdef struct SSPAlg{T,I}
    smoothing_radius::T=11//20
    interp::I
end

"""
    SSP1_linear(; smoothing_radius = 0.55)

Perform a smoothed subpixel projection employing linear interpolation of the filtered density that is not differentiable through topology changes.
The `smoothing_radius` keyword sets the radius of the smoothing kernel relative to the grid spacing.
"""
SSP1_linear(; kws...) = SSPAlg(; interp=LinearInterp(; deriv=ValueWithGradient()), kws...)

"""
    SSP1(; smoothing_radius = 0.55)

Perform a smoothed subpixel projection employing cubic interpolation of the filtered density that is not differentiable through topology changes.
The `smoothing_radius` keyword sets the radius of the smoothing kernel relative to the grid spacing.
"""
SSP1(; kws...) = SSPAlg(; interp=CubicInterp(; deriv=ValueWithGradient()), kws...)

"""
    SSP2(; smoothing_radius = 0.55)

Perform a smoothed subpixel projection employing cubic interpolation of the filtered density that is differentiable through topology changes.
The `smoothing_radius` keyword sets the radius of the smoothing kernel relative to the grid spacing.
"""
SSP2(; kws...) = SSPAlg(; interp=CubicInterp(; deriv=ValueWithGradientAndHessian()), kws...)

function init!(prob::ProjectionProblem, alg::SSPAlg)
    (; rho_filtered, grid, target_points, beta, eta) = prob

    interp_prob = InterpolationProblem(; data=rho_filtered, grid, target_points)
    interp_alg = alg.interp
    interp_solver = init!(interp_prob, interp_alg)
    rho_filtered_interp = solve!(interp_solver)

    rho_projected = similar(rho_filtered, length(target_points))

    adj_rho_filtered_interp = (;
        value = similar(rho_filtered_interp.value),
        gradient = similar(rho_filtered_interp.gradient),
        # Hessian information is used by SSP2 but not SSP1, hence it is optional
        (haskey(rho_filtered_interp, :hessian) ? (; hessian=similar(rho_filtered_interp.hessian)) : (;))...
    )

    cacheval = (; interp_solver, rho_projected, adj_rho_filtered_interp)# adj_rho_filtered_value, adj_rho_filtered_gradient, adj_rho_filtered_hessian)

    return ProjectionSolver(rho_filtered, grid, target_points, beta, eta, alg, cacheval)
end

function solve!(solver::ProjectionSolver)
    proj_solve!(solver, solver.alg)
end

function proj_solve!(solver, alg::SSPAlg)

    (; rho_filtered, grid, beta, eta, cacheval) = solver
    (; interp_solver, rho_projected) = cacheval

    interp_solver.data = rho_filtered
    rho_filtered_interp = solve!(interp_solver)

    dx_all = step.(grid)
    @assert allequal(dx_all)
    dx = first(dx_all)
    R_smoothing = alg.smoothing_radius * dx

    for (i, rho_f) in zip(eachindex(rho_projected), rho_filtered_interp.value)
        # the calculation of the norm is not local in memory, but this is
        # because the interpolation is done as SoA whereas here we use AoS
        rho_filtered_interp_derivs = (;
            gradient = view(rho_filtered_interp.gradient, i, :),
            (haskey(rho_filtered_interp, :hessian) ? (; hessian = view(rho_filtered_interp.hessian, i, :, :)) : (;))...
        )
        rho_filtered_interp_derivs_normsq = map(Base.Fix1(sum, abs2), rho_filtered_interp_derivs)
        rho_p, tape = smoothed_projection(rho_f, rho_filtered_interp_derivs_normsq, R_smoothing, beta, eta)
        rho_projected[i] = rho_p
    end
    return (; value=rho_projected, tape=nothing)
end

function tanh_projection(x, beta, eta)
    if isinf(beta)
        z = x - eta
        bz = beta * z
        if z > zero(z)
            one(bz)
        elseif z < zero(z)
            zero(bz)
        else
            one(bz) / 2
        end
    else
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


function smoothed_projection(rho_filtered, rho_filtered_derivs_normsq, R_smoothing, beta, eta)
    rho_projected = tanh_projection(rho_filtered, beta, eta)

    den_helper = if haskey(rho_filtered_derivs_normsq, :hessian)
        # SSP2
        rho_filtered_derivs_normsq.gradient + R_smoothing^2 * rho_filtered_derivs_normsq.hessian
    else
        # SSP1
        rho_filtered_derivs_normsq.gradient
    end

    nonzero_norm = abs(den_helper) > zero(den_helper)

    den_eff = sqrt(ifelse(nonzero_norm, den_helper, oneunit(den_helper)))
    d = (eta - rho_filtered) / den_eff
    d_R = d / R_smoothing

    needs_smoothing = nonzero_norm & (abs(d_R) < one(d_R))
    F_plus  = ifelse(needs_smoothing, 1//2 + d_R * evalpoly(d_R^2, (-15//16,  5//8, -3//16)), one(d_R))
    F_minus = ifelse(needs_smoothing, 1//2 + d_R * evalpoly(d_R^2, ( 15//16, -5//8,  3//16)), one(d_R))

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

function adjoint_proj_solve!(solver, alg::SSPAlg, adj_sol, tape)

    (; rho_filtered, grid, beta, eta, cacheval) = solver
    (; interp_solver, adj_rho_filtered_interp) = cacheval
    # (; interp_solver, adj_rho_filtered_value, adj_rho_filtered_gradient, adj_rho_filtered_hessian) = cacheval

    # We do not keep a tape and need to repeat the forward calculation
    interp_solver.data = rho_filtered
    rho_filtered_interp = solve!(interp_solver)

    dx_all = step.(grid)
    @assert allequal(dx_all)
    dx = first(dx_all)
    R_smoothing = alg.smoothing_radius * dx

    for (i, adj_proj, rho_f) in zip(eachindex(adj_rho_filtered_interp.value), adj_sol.value, rho_filtered_interp.value)
        # the calculation of the norm is not local in memory, but this is
        # because the interpolation is done as SoA whereas here we use AoS
        rho_filtered_interp_derivs = (;
            gradient = view(rho_filtered_interp.gradient, i, :),
            (haskey(rho_filtered_interp, :hessian) ? (; hessian = view(rho_filtered_interp.hessian, i, :, :)) : (;))...
        )
        rho_filtered_interp_derivs_normsq = map(Base.Fix1(sum, abs2), rho_filtered_interp_derivs)
        rho_p, tape = smoothed_projection(rho_f, rho_filtered_interp_derivs_normsq, R_smoothing, beta, eta)
        adj_rho_f, adj_rho_derivs_normsq = adjoint_smoothed_projection(adj_proj, tape, rho_f, rho_filtered_interp_derivs_normsq, R_smoothing, beta, eta)
        adj_rho_filtered_interp.value[i] = adj_rho_f
        adj_rho_filtered_interp.gradient[i, :] .= 2adj_rho_derivs_normsq.gradient .* rho_filtered_interp_derivs.gradient
        if haskey(rho_filtered_interp, :hessian)
            adj_rho_filtered_interp.hessian[i, :, :] .= 2adj_rho_derivs_normsq.hessian .* rho_filtered_interp_derivs.hessian
        end
    end

    adj_interp_prob = adjoint_solve!(interp_solver, adj_rho_filtered_interp, rho_filtered_interp.tape)

    return (; rho_filtered=adj_interp_prob.data, grid=nothing, target_points=nothing)
end

function adjoint_smoothed_projection(adj_rho_projected_maybe_smoothed, tape, rho_filtered, rho_filtered_derivs_normsq, R_smoothing, beta, eta)
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

    adj_rho_filtered_derivs_normsq = (;
        gradient = adj_den_helper,
        (haskey(rho_filtered_derivs_normsq, :hessian) ? (; hessian = adj_den_helper * R_smoothing^2) : (;))...
    )

    return adj_rho_filtered, adj_rho_filtered_derivs_normsq
end
end
