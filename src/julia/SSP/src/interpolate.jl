module Interpolate

using FastInterpolations: cubic_interp!, cubic_adjoint, CubicFit, DerivOp,
    linear_interp!, linear_adjoint
import SSP: init!, solve!, adjoint_solve!

public InterpolationProblem, LinearInterp, CubicInterp
public Value, ValueWithGradient, ValueWithGradientAndHessian

"""
    InterpolationProblem(; data, grid, target_points)

Define an interpolation problem of a multidimensional `data` array over a `grid`, i.e. a tuple of ranges, queried at a list of `target_points`, i.e. a vector of coordinate tuples or SVectors.
"""
Base.@kwdef struct InterpolationProblem{D,G,T}
    data::D
    grid::G
    target_points::T
end

function Base.copy(prob::InterpolationProblem)
    newprob = InterpolationProblem(;
        data = copy(prob.data),
        grid = prob.grid,
        target_points = copy(prob.target_points),
    )
    return newprob
end

mutable struct InterpolationSolver{D,G,T,A,C}
    data::D
    const grid::G
    const target_points::T
    alg::A
    cacheval::C
end

"""
    Value()

Singleton type to request the interpolation return the value of the interpolant at the target points as the `value` field of the solution.
"""
struct Value end

"""
    ValueWithGradient()

Singleton type to request the interpolation return the value and gradient of the interpolant at the target points as the `value` and `gradient` fields of the solution.
"""
struct ValueWithGradient end

"""
    ValueWithGradientandHessian()

Singleton type to request the interpolation return the value, gradient, and hessian of the interpolant at the target points as the `value`, `gradient`, and `hessian` fields of the solution.
"""
struct ValueWithGradientAndHessian end

"""
    InterpolationAlgorithm

Supertype for FastInterpolations.jl interpolations
"""
abstract type InterpolationAlgorithm end

"""
    LinearInterp(; deriv=Value())

Perform linear interpolation of the data with no extrapolation.
Optionally compute higher derivatives by modifying the `deriv` keyword.
"""
Base.@kwdef struct LinearInterp{D} <: InterpolationAlgorithm
    deriv::D=Value()
end

"""
    CubicInterp(; bc=CubicFit(), deriv=Value())

Perform cubic interpolation of the data with no extrapolation.
Uses the default `bc` from FastInterpolations.jl to determine the interpolation boundary conditions.
Optionally compute higher derivatives by modifying the `deriv` keyword.
"""
Base.@kwdef struct CubicInterp{B,D} <: InterpolationAlgorithm
    bc::B=CubicFit()
    deriv::D=Value()
end

function init!(prob::InterpolationProblem, alg::InterpolationAlgorithm)
    (; data, grid, target_points) = prob

    # TODO refactor these workspaces as views of a single array
    cacheval_value = if alg.deriv isa Value || alg.deriv isa ValueWithGradient || alg.deriv isa ValueWithGradientAndHessian
        interp_value = similar(data, length(target_points))
        adj_data = similar(data)
        adj_data_tmp = similar(data) # this is only used by higher derivatives
        adj_op = interp_adjoint_operator(alg, grid, target_points)
        (; interp_value, adj_data, adj_data_tmp, adj_op)
    else
        (;)
    end

    cacheval_gradient = if alg.deriv isa ValueWithGradient || alg.deriv isa ValueWithGradientAndHessian
        interp_gradient = similar(data, length(target_points), length(grid))
        (; interp_gradient)
    else
        (;)
    end

    cacheval_hessian = if alg.deriv isa ValueWithGradientAndHessian
        interp_hessian = similar(data, length(target_points), length(grid), length(grid))
        (; interp_hessian)
    else
        (;)
    end

    cacheval = (; cacheval_value..., cacheval_gradient..., cacheval_hessian...)
    return InterpolationSolver(data, grid, target_points, alg, cacheval)
end

function interp_adjoint_operator(::LinearInterp, grid, target_points)
    linear_adjoint(grid, target_points)
end

function interp_adjoint_operator(alg::CubicInterp, grid, target_points)
    cubic_adjoint(grid, target_points; bc=alg.bc)
end

function solve!(solver::InterpolationSolver)
    interp_solve!(solver, solver.alg)
end

function interp_solve!(solver, alg::InterpolationAlgorithm)
    (; data, grid, target_points, cacheval) = solver
    
    sol_value = if alg.deriv isa Value || alg.deriv isa ValueWithGradient || alg.deriv isa ValueWithGradientAndHessian
        let (; interp_value) = cacheval
            interp!(interp_value, alg, grid, data, target_points)
            (; value=interp_value)
        end
    else
        (;)
    end

    ndim = Val{ndims(data)}()
    dims = ntuple(n -> Val{n}(), ndim) # construct a type-stable tuple / range because this gets unrolled

    sol_gradient = if alg.deriv isa ValueWithGradient || alg.deriv isa ValueWithGradientAndHessian
        let (; interp_gradient) = cacheval
            dograd = function (::Val{dim}) where {dim}
                # construct deriv tuple in a type-stable way by using a function barrier
                deriv = ntuple(n -> DerivOp{n == dim ? 1 : 0}(), ndim)
                out = view(interp_gradient, :, dim)
                interp!(out, alg, grid, data, target_points; deriv)
            end
            foreach(dograd, dims)
            (; gradient=interp_gradient)
        end
    else
        (;)
    end

    sol_hessian = if alg.deriv isa ValueWithGradientAndHessian
        let (; interp_hessian) = cacheval
            dohess = function (::Val{dim1}, ::Val{dim2}) where {dim1, dim2}
                deriv = ntuple(n -> DerivOp{(n == dim1 ? 1 : 0) + (n == dim2 ? 1 : 0)}(), ndim)
                out = view(interp_hessian, :, dim1, dim2)
                interp!(out, alg, grid, data, target_points; deriv)
            end
            foreach(dim1 -> foreach(dim2 -> dohess(dim1, dim2), dims), dims)
            (; hessian=interp_hessian)
        end
    else
        (;)
    end

    # no tape because intermediate results are not needed for the adjoint
    sol = (; tape=nothing, sol_value..., sol_gradient..., sol_hessian...)
    return sol
end

function interp!(interp_value, alg::LinearInterp, grid, data, target_points; kws...)
    linear_interp!(interp_value, grid, data, target_points; kws...)
end

function interp!(interp_value, alg::CubicInterp, grid, data, target_points; kws...)
    cubic_interp!(interp_value, grid, data, target_points; bc=alg.bc, kws...)
end

function adjoint_solve!(solver::InterpolationSolver, adj_sol, tape)
    adjoint_interp_solve!(solver, solver.alg, adj_sol, tape)
end

function adjoint_interp_solve!(solver, alg::InterpolationAlgorithm, adj_sol, tape)
    (; data, grid, target_points, cacheval) = solver

    adj_prob_value = if alg.deriv isa Value || alg.deriv isa ValueWithGradient || alg.deriv isa ValueWithGradientAndHessian
        # create a local scope for cache variables so that they don't get boxed
        let (; adj_data, adj_data_tmp, adj_op) = cacheval
            adj_op(adj_data, adj_sol.value)
            (; data=adj_data)
        end
    else
        (;)
    end

    ndim = Val{ndims(data)}()
    dims = ntuple(n -> Val{n}(), ndim) # construct a type-stable tuple / range because this gets unrolled

    adj_prob_gradient = if alg.deriv isa ValueWithGradient || alg.deriv isa ValueWithGradientAndHessian
        let (; adj_data, adj_data_tmp, adj_op) = cacheval
            dograd = function (::Val{dim}) where {dim}
                deriv = ntuple(n -> DerivOp{n == dim ? 1 : 0}(), ndim)
                adj_grad = view(adj_sol.gradient, :, dim)
                adj_op(adj_data_tmp, adj_grad; deriv)
                adj_data .+= adj_data_tmp
            end
            foreach(dograd, dims)
            (;) # data adjoint is already accumulated into adj_prob_value.data
        end
    else
        (;)
    end

    adj_prob_hessian = if alg.deriv isa ValueWithGradientAndHessian
        let (; adj_data, adj_data_tmp, adj_op) = cacheval
            dohess = function (::Val{dim1}, ::Val{dim2}) where {dim1, dim2}
                deriv = ntuple(n -> DerivOp{(n == dim1 ? 1 : 0) + (n == dim2 ? 1 : 0)}(), ndim)
                adj_hess = view(adj_sol.hessian, :, dim1, dim2)
                adj_op(adj_data_tmp, adj_hess; deriv)
                adj_data .+= adj_data_tmp
            end
            foreach(dim1 -> foreach(dim2 -> dohess(dim1, dim2), dims), dims)
            (;) # data adjoint is already accumulated into adj_prob_value.data
        end
    else
        (;)
    end

    adj_prob = (; data=nothing, grid=nothing, target_points=nothing, adj_prob_value..., adj_prob_gradient..., adj_prob_hessian...)
    return adj_prob
end

end